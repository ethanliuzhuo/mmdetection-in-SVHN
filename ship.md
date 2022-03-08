# 海上船舶智能检测 Baseline
# 使用mmdetection(2.22版本)进行海上船舶智能检测 

数字中国建设峰会组委会[海上船舶智能检测 ](https://www.dcic-china.com/competitions/10022)使用[Swin Transformer](https://github.com/microsoft/Swin-Transformer)进行单类目标检测Baseline；

比赛题目在数字中国建设峰会组委会的[官网](https://www.dcic-china.com/competitions/10022)已经详细阐述，这里不再赘述；

## 1.安装使用mmdetection
### 1.1测试
在这里，因为mmdetection只官方适配于Linux河MacOS操作系统，对Windows（7、10）并不官方支持，如果需要在Windows系统配置mmdetection，可参考[这里](https://www.bilibili.com/video/av795876868/)，下面配置在Ubuntu18.04进行。

在进入官网安装教程前，首先需要安装CUDA10.2，CUdnn7.6.X, 具体教程在[这里](https://blog.csdn.net/qq_32408773/article/details/84112166)可以找到。安装完毕后，安装[Anaconda](https://www.anaconda.com/)，用于建立虚拟环境。

或者进入[Docker Hub](https://hub.docker.com/)安装CUDA10.2，CUdnn7.6.X的Docker。

在官网的安装教程中，有一些并不适合，在安装完Anaconda的终端命令行中，具体操作如下：

```bashrc
#创造虚拟环境
conda create -n open-mmlab python=3.8 -y 
#激活虚拟环境
conda activate open-mmlab

#从pytorch的官网下载安装pytorch，这里使用最新版1.10
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

#安装mmcv，是一个mmdet的基础包
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

#或者直接运行
pip install mmcv-full

#下载mmdetection，目前最新版是2.22, 如果后续更新想下载固定版本，直接去github搜版本号
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

#下载一些依赖包并安装
pip install -r requirements/build.txt
pip install -v -e . #别漏了最后一个点
```

下载编译过程较慢，至此mmdetection配置完成

### 1.2测试
先去mmdetection的[model zoo/faster_rcnn](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn)里下载faster_rcnn_r50_fpn的模型，放入 `./checkpoints `中，然后cd 至mmdetection

命令行执行:
```
python demo/image_demo.py demo/demo.jpg configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
```

![coco_test](https://github.com/open-mmlab/mmdetection/raw/master/resources/coco_test_12510.jpg)

然后就会输出图片，由此表示mmdetection配置完成


## 2.转化格式
将数据集下载后我们得到`training_dataset.zip`和`test_dataset.zip`等文件。解压以后得到文件夹`A`和`测试集`的文件夹。

`A`文件夹中，训练数据集中包括两类数据文件，第一类是.jpg格式的SAR影像文件，第二类是txt格式的船舶标注信息文本文件，两者通过相同的名称进行关联，名称命名规则可忽略。

- txt中第一位数字0代表船舶（因此本次竞赛只有单类）；
- 第二位数字计算公式为LA/L代表相对于切片大小的横坐标比例位置；
- 第三位数字计算公式为DA/D代表相对于切片大小的纵坐标比例位置；
- 第四位公式为l/L代表相对于切片大小的比例长度；
- 第五位公式为d/D代表相对于切片大小的比例宽度；
- 本次数据集中L和D均为256；

这种格式与我们常见的VOC Xml格式略有不同，为了方便起见，我们统一用脚本将此格式转化为VOC的标注格式。

txt转VOC的代码为(仅针对本数据集)：

```python
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

                    fxml.write(out1 % label)
            fxml.write(out2)

if __name__ == '__main__':
    file_dir = '/home/mmdetection/data/ship/A'
    lists=[]
    for i in os.listdir(file_dir):
        if i[-3:]=='jpg':
            lists.append(file_dir+'/'+i)       
    #print(lists)
    translate(file_dir,lists)
    print('---------------Done!!!--------------')            
```
保存的xml格式文件保存在原文件夹里。在LInux中，首先在这个数据集路径（我的路径是`cd /home/mmdetection/data/ship/`）下的构建一个新的文件夹`mkdir VOC2007`，然后`cd VOC2007`。在此路径下，新建三个文件夹`mkdir Annotations`,`mkdir ImageSets`,`mkdir ImageSets/Main`,`mkdir JPEGImages`。使用`cd ..`回去`ship`路径下，将图片和标注信息转移到新的VOC文件夹下面，`mv A/*.jpg VOC2007/JPEGImages/`（图片移动）和`mv A/*.xml VOC2007/Annotations/`（标注移动）.

在`VOC2007/ImageSets/Main`下生成`train.txt`,`val.txt`,`test.txt`,代码如下：

```python
import os
import pandas as pd
import random 
file_obj = open("VOC2007/ImageSets/Main/val.txt", 'w', encoding='utf-8')
file_obj2 = open("VOC2007/ImageSets/Main/train.txt", 'w', encoding='utf-8')
file_obj3 = open("VOC2007/ImageSets/Main/test.txt", 'w', encoding='utf-8')

jpg_list = os.listdir('VOC2007/JPEGImages')
jpg_list_test = os.listdir('test')

 
def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2
 
val, train = data_split(jpg_list, ratio=0.1, shuffle=True) #10%划分训练测试
print(len(train))
print(len(val))

    
for i in val:
    k = str(i.split('.')[0])
    file_obj.writelines(k)
    file_obj.write('\n')

file_obj.close()

for i in train:
    k = str(i.split('.')[0])
    file_obj2.writelines(k)
    file_obj2.write('\n')

file_obj2.close()

for i in jpg_list_test:
    k = str(i.split('.')[0])
    file_obj3.writelines(k)
    file_obj3.write('\n')

file_obj3.close()
```

最终数据存放格式为：
```brash
./ship
    └── VOC2007
         ├── Annotations  # 标注的VOC格式的xml标签文件
         ├── JPEGImages   # 数据集图片
         └── ImageSet
                └── Main
                     ├── test.txt   # 划分的测试集
                     ├── train.txt   # 划分的训练集
                     └── val.txt   # 划分的验证集
```

## 3.修改配置
### 3.1 修改data配置文件
#### 3.1.1 修改dataset配置文件

复制文件并重命名为`cp /mmdetection/configs/_base_/datasets/voc712.py /mmdetection/configs/_base_/datasets/mydata.py`，

因为我们使用的是VOC2007格式，并没有VOC2012,，因此只要把其中含有VOC2012路径注释掉，并修改路径和图像的均值以及大小。

修改过后内容如下：

```python
# dataset settings
dataset_type = 'VOCDataset' #数据集格式
data_root = 'data/ship/' #修改图片路径
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2, #batch size，显卡打的可以修改
    workers_per_gpu=4,
    train=dict(
        #type='RepeatDataset',  #注释掉，否则有错误
        #times=3,
        #dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/train.txt',
                #data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/'],#, data_root + 'VOC2012/'],
            pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/val.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline
        ))
evaluation = dict(interval=1, metric='mAP') #验证方法
```

#### 3.1.2 修改dataset名称

修改名称：`./mmdet/core/evaluation/class_names.py`
，注意别忘了加逗号。

将VOC名称修改如下:
```python
def voc_classes():
    return ['0',]
```

修改：`./mmdet/datasets/voc.py`，注意别忘了加逗号。

将VOC名称修改如下:
```python
    CLASSES = ('0',)

    PALETTE = [(106, 0, 228),] #调色板
```



### 3.2 修改模型配置文件

修改： `./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py`

将第一行的` ../_base_/datasets/mask_rcnn_r50_fpn.py`更改为` ../_base_/datasets/cascade_rcnn_r50_fpn.py`，我们不需要mask，因此换了一个主模型

将第二行的` ../_base_/datasets/coco_detection.py`更改为` ../_base_/datasets/voc0712.py`，从COCO数据集更改为VOC数据集格式

```python
_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/mydata.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
```

- 第10行修改成`type='CascadeRCNN'`
- 第31行`img_norm_cfg`修改成`img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)`
- 第36行把`with_mask=True`改成`with_mask=False`
- 第73行把`keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])`中的`'gt_masks'`删掉

### 3.3 修改类别数量
修改` ../_base_/datasets/cascade_rcnn_r50_fpn.py`

使用`Ctrl + F`，搜索`num_classes`，将所有的类别数从80改成10；

注意：类别有多少类为多少类，背景不计入类别

### 3.4 加载预训练模型

修改`configs/_base_/default_runtime.py`

其代码注释如下：
```python
checkpoint_config = dict(interval=1) #每一个epochs保存一次模型
# yapf:disable
log_config = dict(
    interval=100,  #每100步输出一次信息
    hooks=[
        dict(type='TextLoggerHook'),# 控制台输出信息的风格
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None #加载预训练模型的路径
resume_from = None #恢复上次保存的模型的路径，继续训练
workflow = [('train', 1),('val', 1)] #当前工作区名称
```


去mmdetection [model zoo](https://github.com/open-mmlab/mmdetection/tree/master/configs/swin)下载`mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth`，保存至`checkpoints`文件夹中，将路径复制到`load_from`，顶替None。这样就可以预训练模型配置。

### 3.5 修改学习策略
修改：`../configs/_base_/schedules/schedule_1x.py`

因为原来的学习速率为0.02，这对于该数据集来说太高了，将`lr`改为lr=0.0001，不需要太高

```python
optimizer = dict(
    _delete_=True,
    type='AdamW', #优化器
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[27, 33]) #第27、33轮衰减学习速率
runner = dict(max_epochs=36) #36轮
```

## 4.训练

单显卡训练命令：

`python tools/train.py ./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py`

多显卡训练命令:

`./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}`
如：`./tools/dist_train.sh  ./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py 2 --work-dir ship`

`--work-dir`代表模型保存路径

## 5.输出

经过36轮训练，数据可以达到mAP至0.88+

在预测前，需要将`测试集`的图片复制到`data/VOCdevkit/VOC2007/JPEGImages`中，并注意已经生成了好了txt格式

如果想用`tool/test.py`预测，得先要生成对应名字的xml放入标注的文件夹，如果不想用，则使用如下代码：


```python
from mmdet.apis import init_detector, inference_detector
import mmcv

import mmcv
import os
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
#from mmdet.apis import init_detector, inference_detector, show_result


config_file = '/home/mmdetection/ship/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'
checkpoint_file = '/home/mmdetection/ship/latest.pth'

model = init_detector(config_file,checkpoint_file, device='cuda:0')

img_dir = '/home/mmdetection/data/ship/VOC2007/JPEGImages/'
out_dir = '/home/mmdetection/data/ship/predict/'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

fp = open('/home/mmdetection/data/ship/VOC2007/ImageSets/Main/test.txt','r')
test_list = fp.readlines()

imgs=[]
for test_1 in test_list:
    test_1 = test_1.replace('\n','')
    name = img_dir + test_1 + '.jpg'
    imgs.append(name)

results = []

count = 0
for img in imgs:
    count += 1
    if count % 100 == 0:
        print('model is processing the {}/{} images.'.format(count,len(imgs)))
    result = inference_detector(model,img)
    results.append(result)

print('\nwriting results to {}'.format('ship.pkl'))
mmcv.dump(results,out_dir+'ship.pkl')

from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import pickle
import numpy as np

f = open('predict/ship.pkl','rb')
data = pickle.load(f)
print(len(data))

fp = open('/home/mmdetection/data/ship/VOC2007/ImageSets/Main/test.txt','r')
test_list = fp.readlines()

def convert(size, box):
    #print(box)
    box= (box[0],box[2],box[1],box[3])
    #print(box)
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)

i = 0
ship_code = {}
iou = 0.5
name_list = []
ship_all_box = []
for img in data[:]:
    #print(len(img[0]))
    #print(img[0])
    score_all = []
    ship_add =[]
    ship_box = []

    for number in img[0]:
        #print(number.tolist())
        #print(len(number))
        
        if len(number) != 0:
            #print(number)
            x1,y1,x2,y2,score = number[0],number[1],number[2],number[3],number[4]
            score_all += [score]
            if score > iou:
                #print(score)
                ship_add += [(x1,y1,x2,y2)]
                #ship_code[test_list[i]] = x1,y1,x2,y2
        else:
            ship_code[test_list[i]] = {}
    #print(ship_add)
    name = test_list[i].replace('\n','')
    
    for box in ship_add:
        #print(box)
        box_2 = convert((256,256), box)
        #print(box_2)
        ship_box += [str(box_2[0]) + ' ' +str(box_2[1]) + ' '+str(box_2[2]) + ' '+str(box_2[3])]
    
    ship_code[name] = ship_box
    if (np.array(score_all) <= iou).all():
        ship_code[name] = ''
    i += 1

import pandas as pd

ship_code_all = [ship_code]
df = pd.DataFrame(ship_code_all).T
df = df.reset_index(level=0)
df["index"] = pd.to_numeric(df["index"])
df = df.sort_values(by="index" , ascending=True)

ship_final_all = []
for i in df[0]:
    if len(i) > 1:
        final = ''
        k = 1
        for j in i:
            if k != len(i):
                final += str(j) + ';'
            else:
                final += str(j)
            k += 1
    elif len(i) == 1:
        print(i)
        print(i[0])
        final = i[0]
    else:
        final = ''
    ship_final_all += [final]
df[0] = ship_final_all

df.to_csv('predict/submission.csv',index= False,header=0)
```

最后提交，完成

## 错误提示

-  `AssertionError: The 'num_classes' (10) in Shared2FCBBoxHead of MMDataParallel does not matches the length of 'CLASSES' 80) in RepeatDataset` 错误原因：数据集的类别信息仍是coco类别80类，(我的数据集是10类)。在修改完 class_names.py 和 voc.py 之后要重新编译： `python setup.py install`；
-  `AssertionError: 'CLASSES' in ConcatDatasetshould be a tuple of str.Add comma if number of classes is 1 as CLASSES = (0,)` 错误原因：修改num_classes成自己的类别数，如果是一个类别，漏了逗号，需要写成CLASSES = (person,)，否则会出现错误；
-  `AttributeError: 'ConfigDict' object has no attribute 'pipeline'` 错误原因：这是官方文件中的bug, 是因为pascal_voc下这几个配置文件都调用了.\configs\_base_\voc0712.py, 而错误就发生在 .\configs\_base_\voc0712.py, 标红的那一块(左图35,36,37行, 其实就是把这三行删掉), 改成右图。
-  `FileNotFoundError: [Errno 2] No such file or directory: '*.png' `错误原因：`mmdetection/mmdet/datasets/xml_style.py`中的51行图片格式不对，改成相应的图片格式;







