# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 09:46:35 2022

@author: CAPE
"""

import time
import os
from PIL import Image
import cv2
import numpy as np

'''人为构造xml文件的格式'''
out0 ='''<annotation>
    <folder>%(folder)s</folder>
    <filename>%(name)s</filename>
    <path>%(path)s</path>
    <source>
        <database>None</database>
    </source>
    <size>
        <width>%(width)d</width>
        <height>%(height)d</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
'''
out1 = '''    <object>
        <name>%(class)s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%(xmin)d</xmin>
            <ymin>%(ymin)d</ymin>
            <xmax>%(xmax)d</xmax>
            <ymax>%(ymax)d</ymax>
        </bndbox>
    </object>
'''

out2 = '''</annotation>
'''

'''txt转xml函数'''
def translate(fdir,lists): 
    source = {}
    label = {}
    for jpg in lists:
        print(jpg)
        if jpg[-4:] == '.jpg':
            image= cv2.imread(jpg)#路径不能有中文
            h,w,_ = image.shape #图片大小
#            cv2.imshow('1',image)
#            cv2.waitKey(1000)
#            cv2.destroyAllWindows()
            
            fxml = jpg.replace('.jpg','.xml')
            fxml = open(fxml, 'w');
            imgfile = jpg.split('/')[-1]
            source['name'] = imgfile 
            source['path'] = jpg
            source['folder'] = 'VOC2007'

            source['width'] = w
            source['height'] = h
            
            fxml.write(out0 % source)
            txt = jpg.replace('.jpg','.txt')

            lines = np.loadtxt(txt)#读入txt存为数组
            # print(type(lines))
            # print(lines)
            # print(lines.shape)
            if lines.shape == (5,):
                label['class'] = str(int(lines[0])) #类别索引从0开始
                
                '''把txt上的数字（归一化）转成xml上框的坐标'''
                xmin = float(lines[1] - 0.5*lines[3])*w
                ymin = float(lines[2] - 0.5*lines[4])*h
                xmax = float(xmin + lines[3]*w)
                ymax = float(ymin + lines[4]*h)
                
                label['xmin'] = xmin
                label['ymin'] = ymin
                label['xmax'] = xmax
                label['ymax'] = ymax

                # if label['xmin']>=w or label['ymin']>=h or label['xmax']>=w or label['ymax']>=h:
                #     continue
                # if label['xmin']<0 or label['ymin']<0 or label['xmax']<0 or label['ymax']<0:
                #     continue
                    
                fxml.write(out1 % label)
            else:
                for box in lines:
                    # print(box.shape)
                    if box.shape != (5,):
                        box = lines
                        # print(box)
                        # print(box.shape)
                    '''把txt上的第一列（类别）转成xml上的类别
                       我这里是labelimg标1、2、3，对应txt上面的0、1、2'''
                    label['class'] = str(int(box[0])) #类别索引从0开始
                    
                    '''把txt上的数字（归一化）转成xml上框的坐标'''
                    xmin = float(box[1] - 0.5*box[3])*w
                    ymin = float(box[2] - 0.5*box[4])*h
                    xmax = float(xmin + box[3]*w)
                    ymax = float(ymin + box[4]*h)
                    
                    label['xmin'] = xmin
                    label['ymin'] = ymin
                    label['xmax'] = xmax
                    label['ymax'] = ymax
    
                    # if label['xmin']>=w or label['ymin']>=h or label['xmax']>=w or label['ymax']>=h:
                    #     continue
                    # if label['xmin']<0 or label['ymin']<0 or label['xmax']<0 or label['ymax']<0:
                    #     continue
                        
                    fxml.write(out1 % label)
            fxml.write(out2)

if __name__ == '__main__':
    file_dir = '/home/mmdetection/data/ship/train'#
    lists=[]
    for i in os.listdir(file_dir):
        if i[-3:]=='jpg':
            lists.append(file_dir+'/'+i)       
    #print(lists)
    translate(file_dir,lists)
    print('---------------Done!!!--------------')            
                
