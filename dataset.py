""" train and test dataset

author jundewu transunet
"""
import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate



class REFUGEDataset(Dataset):
    def __init__(self, args, data_path , transform = None, transform_seg = None, mode = 'Train',plane = False):
        df = pd.read_csv(os.path.join(data_path, 'REFUGE','REFUGE1' + mode + '.csv'), encoding='gbk')
        self.name_list = df['imgName'].tolist()
        self.mask_list = df['maskName'].tolist()
        self.mmask_list = df['multimaskName'].tolist()
        self.label_list = df['label'].tolist()
        self.data_path = data_path

        self.transform_seg = transform_seg
        self.transform = transform

        if plane:
            self.name_list = [a.split('_')[0] + '.jpg' for a in self.mmask_list]

    @classmethod
    def to2(cls,mask):
        img_nd = np.array(mask)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        if img_nd.max() > 1:
            img_nd = img_nd / 255

        disc = img_nd.copy()
        disc[disc != 0] = 1
        cup = img_nd.copy()
        cup[cup != 1] = 0

        disc = disc[:,:,0]*255
        cup = cup[:,:,0]*255

        img_nd = np.dstack((disc,cup))

        # return Image.fromarray(disc.astype(np.uint8)),Image.fromarray(cup.astype(np.uint8))
        return img_nd
    
    @classmethod
    def allone(cls, disc,cup):
        disc = np.array(disc) / 255
        cup = np.array(cup) / 255
        return  np.clip(disc * 0.5 + cup,0,1)
    
    @classmethod
    def reversecolor(cls,seg):
        seg = 255 - np.array(seg)
        return seg

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        
        label = int(self.label_list[index])

        multiname = self.mmask_list[index].split('.')[0].split('_')[0]
        img_path = os.path.join(self.data_path, multiname + "_ci.jpg")
        msk_path = os.path.join(self.data_path, multiname + "_cm.bmp")

        img = Image.open(img_path).convert('RGB')
        true_mask = Image.open(msk_path).convert('L')
        true_mask = self.to2(true_mask)

        masks = []
        ones = []
        data_path = self.data_path
        for n in range(1,8):     # n:1-7
            cup_path = os.path.join(data_path, multiname + '_seg_cup_c_' + str(n) + '.png')
            disc_path = os.path.join(data_path, multiname + '_seg_disc_c_' + str(n) + '.png')

            cup = Image.open(cup_path).convert('L')
            disc = Image.open(disc_path).convert('L')

            one =  self.allone(disc, cup)

            if self.transform_seg:
                disc = self.transform_seg(disc)
                cup = self.transform_seg(cup)
                one = self.transform_seg(one)
            
            Mask = torch.cat((disc,cup),0)

            masks.append(Mask)
            ones.append(one)

        if self.transform:
            img = self.transform(img)

        if self.transform_seg:
            true_mask = self.transform_seg(true_mask)

        return img, true_mask, ones, masks, label, name.split('/')[-1]

    

