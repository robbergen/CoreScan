import os
import glymur
import cv2
import numpy as np
import torch 
import torch.nn as nn

# from torchvision.io import read_image
from torch.utils.data import Dataset
# from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class_min_dict = {"amphibole":(52,82,52),
              "apophyllite":(0,255,0),
              "aspectral":(209,209,209),
              "biotite":(128,0,0),
              "carbonate":(0,255,255),
              "carbonate-actinolite":(44,109,0),
              "chlorite":(0,192,0),
              "clinochlore":(45,95,45),
              "dickite":(148,138,84),
              "epidote":(188,255,55),
              "iron carbonate":(185,255,255),
              "iron oxide":(255,154,0),
              "gypsum":(213,87,171),
              "kaolinite":(191,183,143),
              "montmorillonite":(175,175,255),
                "NA":(0,0,0),
              "nontronite":(105,105,255),
              "phlogopite":(88, 0, 0),
              "prehnite":(70, 70, 220),
              "sericite":(58,102,156),
              "silica":(166,166,166),
              "tourmaline":(255,0,0),
                "UNK1":(83, 141, 213),
                "UNK2":(155, 187, 89),
                "UNK3":(0, 108, 105),
              "vermiculite":(95, 100, 200)
             }

class_RGB_dict = {v: k for k, v in class_min_dict.items()}



def create_mask(img,mineral,class_dict):
    #Input images should be channels x height x width
    mask = np.zeros(img.shape[1:])

    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            if tuple(img[:,i,j]) == class_dict[mineral]:
                mask[i,j] = 1
    return mask   

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, label_dir, mineral_type, class_dict, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_list = sorted([f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(img_dir,f))])
        self.label_list = sorted([f for f in os.listdir(self.label_dir) if os.path.isfile(os.path.join(label_dir,f))])
        self.transform = transform
        self.target_transform = target_transform
        self.mineral_type = mineral_type
        self.class_dict = class_dict

    def __len__(self):
        return len(img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        label_path = os.path.join(self.label_dir, self.label_list[idx])
            
        #Get header info
        j2k = glymur.Jp2k(img_path)
        j2k2 = glymur.Jp2k(label_path)
        
        rgb_tiepoint = j2k.box[3].data['ModelTiePoint'][3:6]
        cm_tiepoint = j2k2.box[3].data['ModelTiePoint'][3:6]
        rgb_res = j2k.box[3].data['ModelPixelScale'][0]
        cm_res = rgb_res*10
        
        #Read images
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        
        #Resize and transpose
        rgb_box = (rgb_tiepoint,rgb_tiepoint+[img.shape[1]*rgb_res,-img.shape[0]*rgb_res,0])
        cm_box = (cm_tiepoint,cm_tiepoint+[label.shape[1]*cm_res,-label.shape[0]*cm_res,0])
        
        img = cv2.copyMakeBorder(img, -min(int((rgb_box[0][1]-cm_box[0][1])/cm_res),0),
                          max(int((rgb_box[1][1]-cm_box[1][1])/cm_res),0),
                          max(int((rgb_box[0][0]-cm_box[0][0])/cm_res),0),
                          -min(int((rgb_box[1][0]-cm_box[1][0])/cm_res),0),
                         cv2.BORDER_CONSTANT)
        label = cv2.copyMakeBorder(label, max(int((rgb_box[0][1]-cm_box[0][1])/cm_res),0),
                          -min(int((rgb_box[1][1]-cm_box[1][1])/cm_res),0),
                          -min(int((rgb_box[0][0]-cm_box[0][0])/cm_res),0),
                          max(int((rgb_box[1][0]-cm_box[1][0])/cm_res),0),
                         cv2.BORDER_CONSTANT)
            
        img = cv2.resize(img, (2048,128))
        label = cv2.resize(label, (2048,128))
    
        img = np.transpose(img,[2,0,1])
        label = np.transpose(label,[2,0,1])
        
        #Normalize images
        mean, std = np.mean(np.mean(img,axis=-1),axis=-1), np.std(np.std(img,axis=-1),axis=-1)
        mask = create_mask(label,self.mineral_type,self.class_dict)

        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        
        if self.transform:
            transform = transforms.Compose([transforms.Normalize(mean, std), self.transform])
            img = transform(img)
        else:
            transform = transforms.Normalize(mean, std)
            img = transform(img)
        if self.target_transform:
            mask = self.target_transform(label)
        return img, mask

def create_dataset(path, label_path, mineral, batch_size):
    dataset = CustomImageDataset(path,label_path,mineral,class_min_dict)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_dataloader = DataLoader(train_set, batch_size=batch_size)
    val_dataloader = DataLoader(val_set, batch_size=batch_size)
    return train_dataloader, val_dataloader


class CustomPatchDataset(Dataset):
    def __init__(self, img_dir, label_dir, mineral_type, class_dict, patch_size = (128,128), transform=None, target_transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_list = sorted([f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(img_dir,f))])
        self.label_list = sorted([f for f in os.listdir(self.label_dir) if os.path.isfile(os.path.join(label_dir,f))])
        self.transform = transform
        self.target_transform = target_transform
        self.mineral_type = mineral_type
        self.class_dict = class_dict
        self.patch_size = patch_size
        self.patches_per_img = (2048//self.patch_size[0])*(128//self.patch_size[1])

    def __len__(self):
        return len(self.img_list)*self.patches_per_img

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx//self.patches_per_img])
        label_path = os.path.join(self.label_dir, self.label_list[idx//self.patches_per_img])
            
        #Get header info
        j2k = glymur.Jp2k(img_path)
        j2k2 = glymur.Jp2k(label_path)
        
        rgb_tiepoint = j2k.box[3].data['ModelTiePoint'][3:6]
        cm_tiepoint = j2k2.box[3].data['ModelTiePoint'][3:6]
        rgb_res = j2k.box[3].data['ModelPixelScale'][0]
        cm_res = rgb_res*10
        
        #Read images
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        
        #Resize and transpose
        rgb_box = (rgb_tiepoint,rgb_tiepoint+[img.shape[1]*rgb_res,-img.shape[0]*rgb_res,0])
        cm_box = (cm_tiepoint,cm_tiepoint+[label.shape[1]*cm_res,-label.shape[0]*cm_res,0])
        
        img = cv2.copyMakeBorder(img, -min(int((rgb_box[0][1]-cm_box[0][1])/cm_res),0),
                          max(int((rgb_box[1][1]-cm_box[1][1])/cm_res),0),
                          max(int((rgb_box[0][0]-cm_box[0][0])/cm_res),0),
                          -min(int((rgb_box[1][0]-cm_box[1][0])/cm_res),0),
                         cv2.BORDER_CONSTANT)
        label = cv2.copyMakeBorder(label, max(int((rgb_box[0][1]-cm_box[0][1])/cm_res),0),
                          -min(int((rgb_box[1][1]-cm_box[1][1])/cm_res),0),
                          -min(int((rgb_box[0][0]-cm_box[0][0])/cm_res),0),
                          max(int((rgb_box[1][0]-cm_box[1][0])/cm_res),0),
                         cv2.BORDER_CONSTANT)
            
        img = cv2.resize(img, (2048,128))
        label = cv2.resize(label, (2048,128))
    
        img = np.transpose(img,[2,0,1])
        label = np.transpose(label,[2,0,1])

        #Create patches here and choose patch [(idx%self.patches_per_img)//(128//patch_size[1]),(idx%self.patches_per_img)%(128//patch_size[1])]
        patchy = (idx%self.patches_per_img)//(128//self.patch_size[1]) # for patch size 32 this ranges from 0-63
        patchx = (idx%self.patches_per_img)%(128//self.patch_size[1]) # for patch size 32 this ranges from 0-4
        img = img[:,patchx*self.patch_size[0]:(patchx+1)*self.patch_size[0],patchy*self.patch_size[1]:(patchy+1)*self.patch_size[1]]
        label = label[:,patchx*self.patch_size[0]:(patchx+1)*self.patch_size[0],patchy*self.patch_size[1]:(patchy+1)*self.patch_size[1]]
        
        #Normalize images
        mean, std = np.mean(np.mean(img,axis=-1),axis=-1), np.std(np.std(img,axis=-1),axis=-1)
        mask = create_mask(label,self.mineral_type,self.class_dict)

        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        
        if self.transform:
            transform = transforms.Compose([transforms.Normalize(mean, std), self.transform])
            img = transform(img)
        else:
            transform = transforms.Normalize(mean, std)
            img = transform(img)
        if self.target_transform:
            mask = self.target_transform(label)
        return img, mask

def create_patch_dataset(path, label_path, mineral, batch_size):
    dataset = CustomPatchDataset(path,label_path,mineral,class_min_dict)
    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.9*len(dataset)), len(dataset)-int(0.9*len(dataset))])
    train_dataloader = DataLoader(train_set, batch_size=batch_size)
    val_dataloader = DataLoader(val_set, batch_size=batch_size)
    return train_dataloader, val_dataloader