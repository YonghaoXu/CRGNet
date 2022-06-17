import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from model.Networks import BaseNet
from dataset.zurich_dataset import ZurichDataSet
import os
from utils.tools import *
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

epsilon = 1e-14

def get_arguments():
    
    parser = argparse.ArgumentParser(description="CRGNet")
    parser.add_argument("--data_dir", type=str, default='/iarai/home/yonghao.xu/Data/Zurich/',
                        help="dataset path.")
    parser.add_argument("--test_list", type=str, default='./dataset/zurich_test.txt',
                        help="test list file.")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="the index of the label ignored in the training.")  
    parser.add_argument("--input_size_test", type=str, default='128,128',
                        help="width and height of input test images.")   
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")              
    parser.add_argument("--num_classes", type=int, default=8,
                        help="number of classes.")
    parser.add_argument("--restore-from", type=str, default='/iarai/home/yonghao.xu/Code/CRGNet/Exp/Zurich/SelfTrain_id_1/time0314_1208/Zurich_batch1400mF1_7383.pth',
                        help="restored model.")   
    parser.add_argument("--snapshot_dir", type=str, default='./Result/Zurich/',
                        help="path to save result.")
    return parser.parse_args()

def main():
    
    args = get_arguments()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    f = open(args.snapshot_dir+'Test_Zurich.txt', 'w')
    
    w, h = map(int, args.input_size_test.split(','))
    input_size_test = (w, h)

    cudnn.enabled = True
    cudnn.benchmark = True
    model = BaseNet(num_classes=args.num_classes)

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()
   
    test_loader = data.DataLoader(
                    ZurichDataSet(args.data_dir, args.test_list,set='test'),
                    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

   
    interp_test = nn.Upsample(size=(input_size_test[1], input_size_test[0]), mode='bilinear')

    name_classes = np.array(['Roads','Buildings','Trees','Grass','Bare Soil','Water','Rails','Pools'], dtype=np.str)
    TP_all = np.zeros((args.num_classes, 1))
    FP_all = np.zeros((args.num_classes, 1))
    TN_all = np.zeros((args.num_classes, 1))
    FN_all = np.zeros((args.num_classes, 1))
    n_valid_sample_all = 0
    F1 = np.zeros((args.num_classes, 1))
    IoU = np.zeros((args.num_classes, 1))
   
    TP_all_crf = np.zeros((args.num_classes, 1))
    FP_all_crf = np.zeros((args.num_classes, 1))
    TN_all_crf = np.zeros((args.num_classes, 1))
    FN_all_crf = np.zeros((args.num_classes, 1))
    n_valid_sample_all_crf = 0
    F1_crf = np.zeros((args.num_classes, 1))
    IoU_crf = np.zeros((args.num_classes, 1))

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
        test_porb = np.zeros((args.num_classes,image.shape[2],image.shape[3]))
            
        for j in range(len(x)):    
            for k in range(len(y)):            
                r_start,c_start = (y[k],x[j])
                r_end,c_end = (r_start+block_size[0],c_start+block_size[1])
                image_part = image[0,:,r_start:r_end, c_start:c_end].unsqueeze(0).cuda()
            
                with torch.no_grad():
                    pb,pe = model(image_part)
                src_output = interp_test(nn.functional.softmax(pb,dim=1)+nn.functional.softmax(pe,dim=1))
                _,pred = torch.max(src_output.detach(), 1)

                pred = pred.squeeze().data.cpu().numpy()
                src_output = src_output.cpu().detach().numpy().squeeze()
                
                if (j==0)and(k==0):
                    test_pred[r_start:r_end, c_start:c_end] = pred
                    test_porb[:,r_start:r_end, c_start:c_end] = src_output
                elif (j==0)and(k!=0):
                    test_pred[r_start+int(min_overlap/2):r_end, c_start:c_end] = pred[int(min_overlap/2):,:]
                    test_porb[:,r_start+int(min_overlap/2):r_end, c_start:c_end] = src_output[:,int(min_overlap/2):,:]
                elif (j!=0)and(k==0):
                    test_pred[r_start:r_end, c_start+int(min_overlap/2):c_end] = pred[:,int(min_overlap/2):]
                    test_porb[:,r_start:r_end, c_start+int(min_overlap/2):c_end] = src_output[:,:,int(min_overlap/2):]
                elif (j!=0)and(k!=0):
                    test_pred[r_start+int(min_overlap/2):r_end, c_start+int(min_overlap/2):c_end] = pred[int(min_overlap/2):,int(min_overlap/2):]
                    test_porb[:,r_start+int(min_overlap/2):r_end, c_start+int(min_overlap/2):c_end] = src_output[:,int(min_overlap/2):,int(min_overlap/2):]
    
        print(index+1, '/', len(test_loader), ': Testing ', name)
        
        TP,FP,TN,FN,n_valid_sample = eval_image(test_pred.reshape(-1),label.reshape(-1),args.num_classes)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid_sample_all += n_valid_sample
      
        test_pred = np.asarray(test_pred, dtype=np.uint8)

        output_col = index2bgr_z(test_pred)
        plt.imsave('%s/%s_CRGNet.png' % (args.snapshot_dir, name[0].split('.')[0]),output_col)
    
        # CRF
        im = np.ascontiguousarray(np.moveaxis(image[0].cpu().numpy(),0,-1).astype('uint8'))
        unary = unary_from_softmax(test_porb)
        unary = np.ascontiguousarray(unary)

        d = dcrf.DenseCRF2D(img_size[1], img_size[0], args.num_classes)
        d.setUnaryEnergy(unary)

        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im, compat=10)
      
        Q = d.inference(5)
        test_pred_crf = np.argmax(Q, axis=0).reshape((img_size[0], img_size[1]))

        TP,FP,TN,FN,n_valid_sample = eval_image(test_pred_crf.reshape(-1),label.reshape(-1),args.num_classes)
        TP_all_crf += TP
        FP_all_crf += FP
        TN_all_crf += TN
        FN_all_crf += FN
        n_valid_sample_all_crf += n_valid_sample
        
        test_pred_crf = np.asarray(test_pred_crf, dtype=np.uint8)

        output_col_crf = index2bgr_z(test_pred_crf)
        plt.imsave('%s/%s_CRGNet_crf.png' % (args.snapshot_dir, name[0].split('.')[0]),output_col_crf)


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

    OA = np.sum(TP_all_crf)*1.0 / n_valid_sample_all_crf
    for i in range(args.num_classes):
        P = TP_all_crf[i]*1.0 / (TP_all_crf[i] + FP_all_crf[i] + epsilon)
        R = TP_all_crf[i]*1.0 / (TP_all_crf[i] + FN_all_crf[i] + epsilon)
        F1_crf[i] = 2.0*P*R / (P + R + epsilon)
        IoU_crf[i] = TP_all_crf[i]*1.0 / (TP_all_crf[i] + FP_all_crf[i] + FN_all_crf[i] + epsilon)

    for i in range(args.num_classes):
        f.write('===>' + name_classes[i] + ': %.2f\n'%(F1_crf[i] * 100))
        print('===>' + name_classes[i] + ': %.2f'%(F1_crf[i] * 100))
    mF1 = np.mean(F1_crf)
    mIoU = np.mean(IoU_crf)
    f.write('===> mean F1: %.2f mean IoU: %.2f OA: %.2f\n'%(mF1*100,mIoU*100,OA*100))
    print('===> mean F1: %.2f mean IoU: %.2f OA: %.2f'%(mF1*100,mIoU*100,OA*100))
    f.close()

if __name__ == '__main__':
    main()
