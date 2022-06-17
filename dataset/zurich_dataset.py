import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as tf

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def random_crop(image,mask,crop_size=(512,512)):
    not_valid = True
    while not_valid:
        i, j, h, w = transforms.RandomCrop.get_params(image,output_size=crop_size)
        image_crop = tf.crop(image,i,j,h,w)
        mask_crop = tf.crop(mask,i,j,h,w)
        label = np.asarray(mask_crop, np.float32)
        if np.sum(label.reshape(-1)<255)>0:
            not_valid = False
    return image_crop,mask_crop 

def random_crop_pseudo(image,mask,pseudo,crop_size=(512,512)):
    not_valid = True
    while not_valid:
        i, j, h, w = transforms.RandomCrop.get_params(image,output_size=crop_size)
        image_crop = tf.crop(image,i,j,h,w)
        mask_crop = tf.crop(mask,i,j,h,w)
        pseudo_crop = tf.crop(pseudo,i,j,h,w)
        label = np.asarray(mask_crop, np.float32)
        if np.sum(label.reshape(-1)<255)>0:
            #print(np.sum(label.reshape(-1)<255))
            not_valid = False
    return image_crop,mask_crop,pseudo_crop 


class ZurichDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None,crop_size=(512, 512), mean=IMG_MEAN, scale=False, mirror=False, ignore_label=255,set='train',mode=0,id=None):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.set = set
        self.mode = mode
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []

        if mode==0:#Full supervision
            for name in self.img_ids:
                img_file = osp.join(self.root, "img/%s" % name)
                label_file = osp.join(self.root, "gt/%s" % name.replace('tif','png'))
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })
        elif mode==1:#Point-level supervision
            for name in self.img_ids:
                img_file = osp.join(self.root, "img/%s" % name)
                label_file = osp.join(self.root, "point/"+"an"+str(id)+"/%s" % name.replace('tif','png'))
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })        
        elif mode==2:#Self-Train
            for name in self.img_ids:                
                img_file = osp.join(self.root, "img/%s" % name)
                label_file = osp.join(self.root, "point/"+"an"+str(id)+"/%s" % name.replace('tif','png'))
                pseudo_file = osp.join(self.root, "point/"+"pseudo"+str(id)+"/%s" % name.replace('tif','png'))
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "pseudo": pseudo_file,
                    "name": name
                })
            
        

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        if self.mode < 2:
            image = Image.open(datafiles["img"]).convert('RGB')
            label = Image.open(datafiles["label"])
            name = datafiles["name"]

            if self.set=='train':
                image,label = random_crop(image,label,self.crop_size)
                
            image = np.asarray(image, np.float32)
            label = np.asarray(label, np.float32)

            size = image.shape
            image = image[:, :, ::-1]
            image -= self.mean
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        else:
            image = Image.open(datafiles["img"]).convert('RGB')
            label = Image.open(datafiles["label"])
            pseudo = Image.open(datafiles["pseudo"])
            name = datafiles["name"]

            image,label,pseudo = random_crop_pseudo(image,label,pseudo,self.crop_size)
                

            image = np.asarray(image, np.float32)
            label = np.asarray(label, np.float32)
            pseudo = np.asarray(pseudo, np.float32)

            size = image.shape
            image = image[:, :, ::-1]
            image -= self.mean
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), pseudo.copy(), name
          
