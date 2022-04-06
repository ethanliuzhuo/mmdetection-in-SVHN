# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:38:17 2021

@author: 17478
"""
import os
from termios import CIBAUD
import cv2
import numpy as np
from tqdm import tqdm
os.chdir("/home/mmdetection/data/archive/VOC2007/")
tr_img_pth = "JPEGImages/"

tr_img_files = [file for file in os.listdir(tr_img_pth)\
            if os.path.splitext(file)[1] in [".png",".jpg"]]
totalRGB = np.asarray([0,0,0],dtype=np.longlong)
totalVar = np.asarray([0,0,0],dtype=np.float64)
meanRGB = np.asarray([0,0,0],dtype=np.float64)
varRGB = np.asarray([0,0,0],dtype=np.float64)
sum = 0
for img_file in tqdm(tr_img_files,desc="calculating mean",mininterval=0.1):
    img_file = os.path.join(tr_img_pth,img_file)
    img = cv2.imread(img_file,-1)
    #print(img_file)
    try:
        totalRGB[0] += np.sum(img[:,:,0])
        totalRGB[1] += np.sum(img[:,:,1])
        totalRGB[2] += np.sum(img[:,:,2])
        sum += 1
    except:
        continue
img_size = img.shape[:2]
print(sum)

total_pixels = img_size[0]*img_size[1]*sum
#total_pixels = img_size[0]*img_size[1]*len(tr_img_files)

meanRGB[0] = totalRGB[0]/total_pixels
meanRGB[1] = totalRGB[1]/total_pixels
meanRGB[2] = totalRGB[2]/total_pixels
sum = 0
for img_file in tqdm(tr_img_files,desc="calculating var",mininterval=0.1):
    img_file = os.path.join(tr_img_pth,img_file)
    img = cv2.imread(img_file,-1)
    try:
        totalVar[0] += np.sum((img[:,:,0]-meanRGB[0])**2)
        totalVar[1] += np.sum((img[:,:,1]-meanRGB[1])**2)
        totalVar[2] += np.sum((img[:,:,2]-meanRGB[2])**2)
        sum += 1
    except:
        continue
total_pixels = img_size[0]*img_size[1]*sum

varRGB[0] = np.sqrt(totalVar[0]/total_pixels)
varRGB[1] = np.sqrt(totalVar[1]/total_pixels)
varRGB[2] = np.sqrt(totalVar[2]/total_pixels)
print("img_size:{}x{}".format(img_size[1],img_size[0]))
print("meanRGB:{}".format(meanRGB))
print("stdRGB:{}".format(varRGB))