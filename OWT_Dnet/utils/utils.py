import random

import numpy as np
import torch
from PIL import Image
from skimage import exposure

import tifffile

#---------------------------------------------------------#
#   Image Processing
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image         = SIEM(np.array(image, np.float64))  
        return image 



import cv2

def resize_image(image, size):
    ih, iw, hhhh = image.shape
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)

    zero_array = np.zeros((h, w, hhhh), dtype=np.float32)

    start_x = (w - nw) // 2
    start_y = (h - nh) // 2
    zero_array[start_y:start_y+nh, start_x:start_x+nw] = image_resized

    new_image = np.transpose(zero_array, [2, 0, 1])


    return new_image, nw, nh

    

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

#---------------------------------------------------#
#   Stretching and Leveling
#---------------------------------------------------#
def SIEM(image):
    s1=image[ :, :, :2]  #s1,VV,VH
    s2=image[ :, :, 2:]  #s1,B2,B3,B8
    s1image_data4 = (s1 + 30) / 40   #-30，10
    s2image_data4 = (s2-0.1) / 0.3   #0.1，0.4
    s1 = np.clip(s1image_data4,0,1)
    s2 = np.clip(s2image_data4,0,1)

    s1image_data1 = linear_stretch(s1,2,98)
    s1image_data2 = linear_stretch(s1,5,95)  
    s1image_data3 = histogram_equalization(s1)
    s1image_data4 = linear_stretch(s1,1,99)
    
    s2image_data1 = linear_stretch(s2,2,98)
    s2image_data2 = linear_stretch(s2,5,95)  
    s2image_data3 = histogram_equalization(s2)
    s2image_data4 = linear_stretch(s2,1,99)
    
    image = np.concatenate([s1,s1image_data1,s1image_data2,s1image_data3,s1image_data4,s2,s2image_data1,s2image_data2,s2image_data3,s2image_data4], axis=2)

    return image

def linear_stretch(image, lower_percent=2, upper_percent=98):

    out = np.zeros_like(image, dtype=np.float32)
    for band in range(image.shape[2]):
        p2, p98 = np.percentile(image[:, :, band], (lower_percent, upper_percent))
        out[:, :, band] = np.clip((image[:, :, band] - p2) / (p98 - p2+1e-6), 0, 1)
    return out.astype(np.float32)



def histogram_equalization(image):
    out = np.zeros_like(image, dtype=np.float32)
    for band in range(image.shape[2]):
        out[:, :, band] = exposure.equalize_hist(image[:, :, band])
    return (out).astype(np.float32)



def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

