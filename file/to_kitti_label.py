import os
from secrets import randbelow
import pandas as pd
"""
path = 'data/house/'
df = pd.read_csv(path + 'train_mask.csv',header=None)

print(df.head(5))
train_df = df.sample(frac = 0.95, random_state = 2022,axis = 0)
test_df = df[~df.index.isin(train_df.index)]

for i in df[0][:]:
    #print(i.split('\t'))
    name = i.split('\t')[0].split('.')[0]
    label = i.split('\t')[1]
    label = 'house ' + label
    path_txt = path + '/label/' +  name + '.txt'
    with open(path_txt,'w') as f:
        f.write(label)

#%%
train_name = []
for i in train_df[0][:]:
    #print(i.split('\t'))
    name = i.split('\t')[0].split('.')[0]
    train_name += [name]
with open(path + 'train.txt','w') as f:
    for i in train_name:
        f.write(i +'\n')

# %%
test_name = []
for i in test_df[0][:]:
    #print(i.split('\t'))
    name = i.split('\t')[0].split('.')[0]
    test_name += [name]

with open(path + 'val.txt','w') as f:
    for i in test_name:
        f.write(i +'\n')
"""
import os
from secrets import randbelow
import pandas as pd
import numpy as np
import cv2

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

train_mask = pd.read_csv('data/VOCdevkit/train_mask.csv', sep='\t', names=['name', 'mask'])
path = 'data/VOCdevkit/'
from tqdm import trange
from tqdm import tqdm

for i in trange(len(train_mask)):

    try:
        mask = rle_decode(train_mask['mask'].iloc[i])
    except:
        mask = np.zeros(512*512, dtype=np.uint8)
        mask = mask.reshape((512, 512), order='F')

    path_txt = path + '/label/' +  train_mask['name'].iloc[i].split('.')[0] + '.txt'
    np.savetxt(path_txt,mask, fmt="%i",delimiter=" ")
    


    
