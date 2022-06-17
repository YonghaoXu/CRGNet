import argparse
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torch.backends.cudnn as cudnn
from utils.tools import *
from dataset.zurich_dataset import ZurichDataSet
from model.Networks import BaseNet
import skimage.morphology as mpg
import lovasz_losses as L
import random

name_classes = np.array(['Roads','Buildings','Trees','Grass','Bare Soil','Water','Rails','Pools'], dtype=np.str)
epsilon = 1e-14

def init_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

def get_arguments():

    parser = argparse.ArgumentParser(description="CRGNet")
    
    #dataset
    parser.add_argument("--data_dir", type=str, default='/iarai/home/yonghao.xu/Data/Zurich/',
                        help="dataset path.")
    parser.add_argument("--train_list", type=str, default='./dataset/zurich_train.txt',
                        help="training list file.")
    parser.add_argument("--test_list", type=str, default='./dataset/zurich_test.txt',
                        help="test list file.")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="the index of the label ignored in the training.")
    parser.add_argument("--input_size_train", type=str, default='128,128',
                        help="width and height of input training images.")         
    parser.add_argument("--input_size_test", type=str, default='128,128',
                        help="width and height of input test images.")                
    parser.add_argument("--num_classes", type=int, default=8,
                        help="number of classes.")   
    parser.add_argument("--mode", type=int, default=1,
                        help="annotation type (0-full, 1-point).")
    parser.add_argument("--id", type=int, default=1,
                        help="annotator id).")                       

    #network
    parser.add_argument("--batch_size", type=int, default=64,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="base learning rate.")
    parser.add_argument("--num_steps", type=int, default=5000,
                        help="Number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=5000,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--restore_from", type=str, default='/iarai/home/yonghao.xu/PreTrainedModel/fcn8s_from_caffe.pth',
                        help="pretrained ResNet model.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--tau", type=float, default=0.95,
                        help="prob tau.")
    parser.add_argument("--lambda_con", type=float, default=1,
                        help="consistency weight.")

    #result
    parser.add_argument("--snapshot_dir", type=str, default='./Exp/',
                        help="where to save snapshots of the model.")

    return parser.parse_args()

modename = ['full','point']

def main():

    args = get_arguments()
    snapshot_dir = args.snapshot_dir+'Zurich/CRGNet_mode_'+modename[args.mode]+'_id_'+str(args.id)+'/time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)
    f = open(snapshot_dir+'ZurichSeg_log.txt', 'w')

    w, h = map(int, args.input_size_test.split(','))
    input_size_test = (w, h)
    w, h = map(int, args.input_size_train.split(','))
    input_size_train = (w, h)

    cudnn.enabled = True
    cudnn.benchmark = True
    init_seeds()

    # Create network
    model = BaseNet(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)

    new_params = model.state_dict().copy()
    for i,j in zip(saved_state_dict,new_params):
        if (i[0] !='f')&(i[0] != 's')&(i[0] != 'u'):
            new_params[j] = saved_state_dict[i]

    model.load_state_dict(new_params)
    
   
    model.train()
    model = model.cuda()

    src_loader = data.DataLoader(
                    ZurichDataSet(args.data_dir, args.train_list, max_iters=args.num_steps_stop*args.batch_size,
                    crop_size=input_size_train,set='train',mode=args.mode,id=args.id),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


    test_loader = data.DataLoader(
                    ZurichDataSet(args.data_dir, args.test_list,set='test'),
                    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    
    # interpolation for the probability maps and labels 
    interp_train = nn.Upsample(size=(input_size_train[1], input_size_train[0]), mode='bilinear')
    interp_test = nn.Upsample(size=(input_size_test[1], input_size_test[0]), mode='bilinear')
    sample_rate = int(input_size_train[1]/8)
    pool = torch.nn.MaxPool2d(8, stride=8)
    label_up_interp = nn.Upsample(size=(input_size_train[1], input_size_train[0]), mode='nearest')
   
    loss_hist = np.zeros((args.num_steps_stop,5))
    F1_best = 0.6

    # 8-connectivity neighborhood
    delta_r = np.array([-1,0,1,-1,1,-1,0,1])
    delta_c = np.array([1,1,1,0,0,-1,-1,-1])

    # morphological operations for removing the boundary of the expanded annotations
    se = mpg.square(3)
    se_label = mpg.square(8)


    L_seg = nn.CrossEntropyLoss(ignore_index=255)
    L_con = torch.nn.MSELoss()

    for batch_index, src_data in enumerate(src_loader):
        if batch_index==args.num_steps_stop:
            break
        tem_time = time.time()
        model.train()
        optimizer.zero_grad()
        
        adjust_learning_rate(optimizer,args.learning_rate,batch_index,args.num_steps)

        images, labels, _, name = src_data
        images = images.cuda()      
        pb_ori,pe_ori = model(images)   
        
        # prediction of the base classifier
        pb_output = interp_train(pb_ori)
        # prediction of the expanded classifier
        pe_output = interp_train(pe_ori)
        
        # initialize the expanded label matrix E
        E = torch.argmax(pb_ori,1)
        pb_cur,_ = torch.max(F.softmax(pb_ori,1),1)


        ind_expand = 0
        num_expand = 0

        labels_np = labels.numpy().astype('int64')
        labels_np[labels_np==255] = -1
        labels_down = np.zeros((args.batch_size,sample_rate,sample_rate)).astype('uint8')

        # limit the max iteration of region growing for acceleration
        max_iter = int(10*((float(batch_index) / args.num_steps) ** 0.9))
        
        # downsample the point-level annotations for acceleration 
        for id_img in range(args.batch_size):
            label_pool = pool(torch.from_numpy(labels_np[id_img,:,:]).unsqueeze(0).unsqueeze(0).float()).squeeze()
            label_pool[label_pool==-1] = 255
            labels_down[id_img] = label_pool.numpy().astype('uint8')
        
        # region grow per 20 iterations for acceleration 
        if (batch_index+1) % 20 == 0:
            for id_img in range(args.batch_size):
                label_cur = labels_down[id_img,:,:]
                label_new = label_cur.copy()
                cur_iter = 0
                is_grow = True
                while is_grow:
                    cur_iter += 1
                    label_inds = (label_cur<255)*1
                    erosion = mpg.erosion(label_inds,se)
                    
                    # only need to visit the 8-connectivity neighborhood of the boundary pixels in the annotation
                    label_inds = label_inds-erosion
                    rc_inds = np.where(label_inds>0)
                    update_count = 0

                    # visit the 8-connectivity neighborhood
                    for i in range(len(rc_inds[0])):
                        y_cur = label_cur[rc_inds[0][i],rc_inds[1][i]]
                        for j in range(len(delta_r)):
                            index_r = rc_inds[0][i] + delta_r[j]
                            index_c = rc_inds[1][i] + delta_c[j]
                            
                            valid = (index_r>=0)&(index_r<sample_rate)&(index_c>=0)&(index_c<sample_rate)
                            if valid:
                                if (label_new[index_r,index_c]==255):
                                    y_neighbor = E[id_img,index_r,index_c]
                                    p_neighbor = pb_cur[id_img,index_r,index_c]
                                    if (y_neighbor==y_cur)&(p_neighbor>args.tau):
                                        label_new[index_r,index_c] = y_cur
                                        update_count += 1
                    if update_count>0:
                        ind_expand += 1
                        label_cur = label_new.copy()
                        num_expand += update_count
                        if cur_iter >= max_iter:
                            is_grow = False
                            labels_down[id_img,:,:] = label_cur
                    else:
                        is_grow = False
                        labels_down[id_img,:,:] = label_cur
        

        
        # Segmentation Loss
        labels = labels.cuda().long()
        L_seg_value = L_seg(pb_output, labels)
        _, predict_labels = torch.max(pb_output, 1)
        lbl_pred = predict_labels.detach().cpu().numpy()
        lbl_true = labels.detach().cpu().numpy()
        metrics_batch = []
        for lt, lp in zip(lbl_true, lbl_pred):
            _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=args.num_classes)
            metrics_batch.append(mean_iu)                
        miou = np.nanmean(metrics_batch, axis=0)  
    
        # Expansion Loss
        labels_downup = label_up_interp(torch.from_numpy(labels_down).float().unsqueeze(1)).squeeze().numpy().astype('uint8')
        # remove the boundary pixels in the expanded annotations to reduce noise
        for id_img in range(args.batch_size):
            labels_downup[id_img] = mpg.dilation(labels_downup[id_img],se_label)
        labels_downup = torch.from_numpy(labels_downup).float().cuda()
        pe_output = nn.functional.softmax(pe_output, dim=1)
        L_exp_value = L.lovasz_softmax(pe_output, labels_downup)
                   

        # Consistency Loss
        pb_output = nn.functional.softmax(pb_output, dim=1)
        L_con_value = L_con(pb_output, pe_output)

        total_loss = L_seg_value + L_exp_value + args.lambda_con * L_con_value
        
        loss_hist[batch_index,0] = L_seg_value.item()
        loss_hist[batch_index,1] = L_exp_value.item()
        loss_hist[batch_index,2] = L_con_value.item()
        loss_hist[batch_index,3] = miou
        
        total_loss.backward()
        optimizer.step()

        loss_hist[batch_index,-1] = time.time() - tem_time

        if (batch_index+1) % 10 == 0: 
            print('Iter %d/%d time: %.2f miou = %.1f L_seg = %.3f L_exp = %.3f L_con = %.3f'%(batch_index+1,args.num_steps,np.mean(loss_hist[batch_index-9:batch_index+1,-1]),np.mean(loss_hist[batch_index-9:batch_index+1,3])*100,np.mean(loss_hist[batch_index-9:batch_index+1,0]),np.mean(loss_hist[batch_index-9:batch_index+1,1]),np.mean(loss_hist[batch_index-9:batch_index+1,2])))
            f.write('Iter %d/%d time: %.2f miou = %.1f L_seg = %.3f L_exp = %.3f L_con = %.3f\n'%(batch_index+1,args.num_steps,np.mean(loss_hist[batch_index-9:batch_index+1,-1]),np.mean(loss_hist[batch_index-9:batch_index+1,3])*100,np.mean(loss_hist[batch_index-9:batch_index+1,0]),np.mean(loss_hist[batch_index-9:batch_index+1,1]),np.mean(loss_hist[batch_index-9:batch_index+1,2])))
            f.flush() 
            
        # evaluation per 100 iterations
        if (batch_index+1) % 100 == 0:            
            model.eval()
            TP_all = np.zeros((args.num_classes, 1))
            FP_all = np.zeros((args.num_classes, 1))
            TN_all = np.zeros((args.num_classes, 1))
            FN_all = np.zeros((args.num_classes, 1))
            n_valid_sample_all = 0
            F1 = np.zeros((args.num_classes, 1))
            IoU = np.zeros((args.num_classes, 1))
        
            for index, batch in enumerate(test_loader):  
                image, label,_, name = batch
                label = label.squeeze().numpy()

                img_size = image.shape[2:] 
                block_size = input_size_test
                min_overlap = 40

                # crop the test images into 128×128 patches
                y_end,x_end = np.subtract(img_size, block_size)
                x = np.linspace(0, x_end, int(np.ceil(x_end/np.float(block_size[1]-min_overlap)))+1, endpoint=True).astype('int')
                y = np.linspace(0, y_end, int(np.ceil(y_end/np.float(block_size[0]-min_overlap)))+1, endpoint=True).astype('int')

                test_pred = np.zeros(img_size)
                 
                for j in range(len(x)):    
                    for k in range(len(y)):            
                        r_start,c_start = (y[k],x[j])
                        r_end,c_end = (r_start+block_size[0],c_start+block_size[1])
                        image_part = image[0,:,r_start:r_end, c_start:c_end].unsqueeze(0).cuda()

                        with torch.no_grad():
                            pb,pe = model(image_part)

                        _,pred = torch.max(interp_test(nn.functional.softmax(pb,dim=1)+nn.functional.softmax(pe,dim=1)).detach(), 1)
                        pred = pred.squeeze().data.cpu().numpy()
                        
                        
                        if (j==0)and(k==0):
                            test_pred[r_start:r_end, c_start:c_end] = pred
                        elif (j==0)and(k!=0):
                            test_pred[r_start+int(min_overlap/2):r_end, c_start:c_end] = pred[int(min_overlap/2):,:]
                        elif (j!=0)and(k==0):
                            test_pred[r_start:r_end, c_start+int(min_overlap/2):c_end] = pred[:,int(min_overlap/2):]
                        elif (j!=0)and(k!=0):
                            test_pred[r_start+int(min_overlap/2):r_end, c_start+int(min_overlap/2):c_end] = pred[int(min_overlap/2):,int(min_overlap/2):]
            
                
                print(index+1, '/', len(test_loader), ': Testing ', name)

                # evaluate one image
                TP,FP,TN,FN,n_valid_sample = eval_image(test_pred.reshape(-1),label.reshape(-1),args.num_classes)
                TP_all += TP
                FP_all += FP
                TN_all += TN
                FN_all += FN
                n_valid_sample_all += n_valid_sample

            OA = np.sum(TP_all)*1.0 / n_valid_sample_all
            for i in range(args.num_classes):
                P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
                R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
                F1[i] = 2.0*P*R / (P + R + epsilon)
                IoU[i] = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)
            

            for i in range(args.num_classes):
                f.write('===>' + name_classes[i] + ': %.2f\n'%(F1[i] * 100))
                print('===>' + name_classes[i] + ': %.2f'%(F1[i] * 100))
            mF1 = np.mean(F1)            
            mIoU = np.mean(IoU)
                        
            f.write('===> mean F1: %.2f mean IoU: %.2f OA: %.2f\n'%(mF1*100,mIoU*100,OA*100))
            print('===> mean F1: %.2f mean IoU: %.2f OA: %.2f'%(mF1*100,mIoU*100,OA*100))
           
            if mF1>F1_best:
                F1_best = mF1
                # save the models        
                f.write('Save Model\n') 
                print('Save Model')                     
                model_name = 'Zurich_batch'+repr(batch_index+1)+'mF1_'+repr(int(mF1*10000))+'.pth'
                torch.save(model.state_dict(), os.path.join(
                    snapshot_dir, model_name))
 
    f.close()
    

if __name__ == '__main__':
    main()
