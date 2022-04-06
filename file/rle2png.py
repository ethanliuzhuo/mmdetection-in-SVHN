#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:53:20 2022

@author: ethan
"""

import numpy as np
import pandas as pd
import cv2
from PIL import Image

# 将图片编码为rle格式
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 将rle格式进行解码为图片
def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')
#%%

#train_mask = pd.read_csv('/Users/ethan/Downloads/train_mask.csv', sep='\t', names=['name', 'mask'])

# 读取第一张图，并将对于的rle解码为mask矩阵
#img = cv2.imread('train/'+ train_mask['name'].iloc[0])
#mask = rle_decode(train_mask['mask'].iloc[0])

#print(rle_encode(mask) == train_mask['mask'].iloc[0])

#%%

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((512, 512), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

#将目标路径下的rle文件中所包含的所有rle编码，保存到save_img_dir中去
def rle_2_img(train_rle_dir,save_img_dir):
    masks = pd.read_csv(train_rle_dir, sep='\t', names=['ImageId', 'EncodedPixels'])
    not_empty = pd.notna(masks.EncodedPixels)
    print(not_empty.sum(), 'masks in', masks[not_empty].ImageId.nunique(), 'images')
    print((~not_empty).sum(), 'empty images in', masks.ImageId.nunique(), 'total images')
    all_batchs = list(masks.groupby('ImageId'))
    train_images = []
    train_masks = []
    i = 0
    for img_id, mask in all_batchs[:]:
        c_mask = masks_as_image(mask['EncodedPixels'].values)
        c_mask[c_mask==1] = 255
        
        #im = Image.fromarray(c_mask)
        #im = im.convert("RGBA")
        #im.save(save_img_dir+img_id.split('.')[0] + '.png')
        cv2.imwrite(save_img_dir+img_id.split('.')[0] + '.png',c_mask)
        print(i,img_id.split('.')[0] + '.png')
        i += 1
        
    return train_images, train_masks



if __name__ == '__main__':
    rle_2_img('./train_mask.csv',
              './VOC2012/SegmentationClass/')