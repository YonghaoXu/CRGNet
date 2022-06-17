import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
from model.Networks import BaseNet
from dataset.vaihingen_dataset import VaihingenDataSet
import os
from PIL import Image
from utils.tools import *

def get_arguments():
    
    parser = argparse.ArgumentParser(description="CRGNet")
    parser.add_argument("--data_dir", type=str, default='/iarai/home/yonghao.xu/Data/Vaihingen/',
                        help="dataset path.")
    parser.add_argument("--test_list", type=str, default='./dataset/vaihingen_train.txt',
                        help="test list file.")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="the index of the label ignored in the training.")  
    parser.add_argument("--input_size_test", type=str, default='128,128',
                        help="width and height of input test images.")   
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for multithread dataloading.")              
    parser.add_argument("--num_classes", type=int, default=5,
                        help="number of classes.")
    parser.add_argument("--restore-from", type=str, default='./Vaihingen_batch3600mF1_6666.pth',
                        help="restored model.")   
    parser.add_argument("--snapshot_dir", type=str, default='/iarai/home/yonghao.xu/Data/Vaihingen/point/pseudo',
                        help="path to save pseudo labels.")
    parser.add_argument("--id", type=int, default=1,
                        help="annotator id).")
    return parser.parse_args()

def main():
    
    args = get_arguments()
    snapshot_dir = args.snapshot_dir+str(args.id)+'/'
    
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    
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
                    VaihingenDataSet(args.data_dir, args.test_list,set='test'),
                    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    interp_test = nn.Upsample(size=(input_size_test[1], input_size_test[0]), mode='bilinear')
    
    for index, batch in enumerate(test_loader):  
        image, label,_, name = batch
        label = label.squeeze().numpy()
        image_size = image.shape[2:] 

        block_size = input_size_test
        min_overlap = 40

        y_end,x_end = np.subtract(image_size, block_size)
        x = np.linspace(0, x_end, int(np.ceil(x_end/np.float(block_size[1]-min_overlap)))+1, endpoint=True).astype('int')
        y = np.linspace(0, y_end, int(np.ceil(y_end/np.float(block_size[0]-min_overlap)))+1, endpoint=True).astype('int')

        test_pred = np.zeros(image_size)
            
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
    
        test_pred = np.asarray(test_pred, dtype=np.uint8)
        output = Image.fromarray(test_pred)
        output.save('%s/%s.png' % (snapshot_dir, name[0].split('.')[0]))
        
if __name__ == '__main__':
    main()
