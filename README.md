# mmdetection-in-SVHN
# 使用mmdetection(2.3版本)进行街景字符编码识别

天池比赛[街景字符编码识别](https://tianchi.aliyun.com/competition/entrance/531795/introduction)使用[DetectoRS](https://zhuanlan.zhihu.com/p/145897444)进行目标检测

比赛题目在天池的官网已经详细阐述，这里不再赘述；

## 1.安装使用mmdetection
### 1.1测试
在这里，因为mmdetection只官方适配于Linux河MacOS操作系统，对Windows（7、10）并不官方支持，如果需要在Windows系统配置mmdetection，可参考[这里](https://www.bilibili.com/video/av795876868/)，下面配置在Ubuntu18.04进行。

在进入官网安装教程前，首先需要安装CUDA，CUdnn, 具体教程在[这里](https://blog.csdn.net/qq_32408773/article/details/84112166)可以找到。安装完毕后，安装[Anaconda](https://www.anaconda.com/)，用于建立虚拟环境。

在官网的安装教程中，有一些并不适合，在安装完Anaconda的终端命令行中，具体操作如下：

```bashrc
#创造虚拟环境
conda create -n open-mmlab python=3.7 -y 
#激活虚拟环境
conda activate open-mmlab

#从pytorch的官网下载安装pytorch，这里最好使用1.3版本的，最新版本1.6有些错误
conda install pytorch=1.3 cudatoolkit=10.1 torchvision -c pytorch

#安装mmcv，是一个mmdet的基础包
pip install mmcv-full==latest+torch1.3.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
#或者直接运行
pip install mmcv-full

#下载mmdetection，目前最新版是2.3, 如果后续更新想下载固定版本，直接去github搜版本号
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
将数据集解压以后的标注格式为：`"filename": {"height": [a,b], "label": [a,b], "left": [a,b], "top": [a,b], "width": [a,b]}`

例如：`"009999.png": {"height": [20, 20, 20], "label": [1, 4, 3], "left": [35, 43, 53], "top": [11, 12, 11], "width": [11, 13, 13]}`

这种格式与我们常见的VOC Xml格式和COCO格式略有不同，为了方便起见，我们统一用脚本将此格式转化为VOC的标注格式
转VOC的代码为：
```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 19:51:31 2020

@author: ethan
"""

import matplotlib.image as image
import json
import os
#%%

with open("data/street_encoding/mchar_val.json",'r',encoding='utf-8') as json_file:
    data = json.load(json_file)
    
headstr = """\
<annotation>
    <folder>VOC2007</folder>
    <filename>%s</filename>
    <source>
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>220208496</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""
 
tailstr = '''\
</annotation>
'''
#%%
    
for i in data:
    # i = '000001.png'
    filename = str(int(i[:-4]) + 30000).zfill(6) + '.jpg'
    dir = 'data/street_encoding/VOC/test/VOCdevkit/VOC2007/JPEGImages/' + filename
    img = image.imread(dir)
    head = headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    objs = data[i]
    
    def write_xml(anno_path, head, objs, tail):
        f = open(anno_path, "w")
        f.write(head)
        for i in range(len(objs['label'])):
            label = objs['label'][i]
            xmin = objs['left'][i]
            ymin = objs['top'][i]
            xmax = objs['left'][i]+objs['width'][i]
            ymax = objs['height'][i]+objs['top'][i]
            f.write(objstr % (label, xmin, ymin,xmax ,  ymax))
        f.write(tail)
    anno_path = 'data/VOCdevkit/VOC2007/Annotations/' + str(int(i[:-4]) + 30000).zfill(6) + '.xml'
    write_xml(anno_path, head, objs, tail)
```

在此之前，先用代码将train和val的照片更改为jpg格式，并且将val从000000编码更改为030000（即在train后面的编码继续加，保证照片和xml名不重合）。

```python
imglist = os.listdir('data/VOCdevkit/VOC2007/JPEGImages')

for i in imglist:
    os.rename('data/VOCdevkit/VOC2007/JPEGImages/' + i, 'data/VOCdevkit/VOC2007/JPEGImages/' + i[:-3]+'jpg')
```

然后把
所有标注文件放在`data/VOCdevkit/VOC2007/Annotations`下，
所有照片放在`data/VOCdevkit/VOC2007/JPEGImages`下，
并在`data/VOCdevkit/VOC2007/ImageSets/Main`下生成`train.txt`,`val.txt`,`trainval.txt`,`test.txt`,代码如下：

```
import os
file_obj = open("data/VOCdevkit/VOC2007/ImageSets/Main/train.txt", 'w', encoding='utf-8')

for i in range(0,30000):
    k = str(i)
    other_url = k.zfill(6)
    print(other_url)
    file_obj.writelines(other_url)
    file_obj.write('\n')

file_obj.close()
```

最终数据存放格式为：
```
./data
└── VOCdevkit
    └── VOC2007
        ├── Annotations  # 标注的VOC格式的xml标签文件
        ├── JPEGImages   # 数据集图片
        ├── ImageSet
                └── Main
          ├── test.txt   # 划分的测试集
          ├── train.txt   # 划分的训练集
          ├── trainval.txt
          └── val.txt   # 划分的验证集
```

## 3.修改配置
### 3.1. 修改模型配置文件

修改： `./configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py`

将第三行的` ../_base_/datasets/coco_detection.py`更改为` ../_base_/datasets/voc0712.py`，从COCO数据集更改为VOC数据集

```
_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
```

### 3.2. 修改学习策略
修改：`../configs/_base_/schedules/schedule_1x.py`

因为原来的学习速率为0.02，这对于该数据集来说太高了，将`lr`改为lr=0.0001，不需要太高

### 3.3. 加载预训练模型
修改：`../_base_/default_runtime.py`
其代码注释如下：
```
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50, #每50个batch输出一次信息
    hooks=[
        dict(type='TextLoggerHook'), # 控制台输出信息的风格
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None  #加载预训练模型的路径
resume_from = None #恢复上次保存的模型的路径，继续训练
workflow = [('train', 1)] #当前工作区名称
```

去mmdetection [model zoo](http://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_cascade_rcnn_r50_1x_coco/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth)下载`detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth`，保存至`checkpoints`文件夹中，将路径复制到`load_from`，顶替None。这样就可以预训练模型配置。

### 3.4. 修改模型配置
修改`'../_base_/models/cascade_rcnn_r50_fpn.py'`

使用`Ctrl + F`，搜索`num_classes`，将所有的类别数从80改成10；

注意：类别有多少类为多少类，背景不计入类别

### 3.4. 修改类别名称
有三处地方需要修改

修改：`./mmdetection/configs/_base_/datasets/voc712.py`
因为我们使用的是VOC2007格式，并没有VOC2012,，因此只要把其中含有VOC2012路径注释掉

修改过后内容如下：
```
# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=3,
        # 把含有VOC2012的路径去掉
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/train.txt',
            ],
			# img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            img_prefix=[data_root + 'VOC2007/'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/val.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
```

注意：第32行的`samples_per_gpu`将2修改为1，因为该模型比较大，单张8G的显存只能使用每次1张，2张就会超出现存，如果是2080 Ti或者30XX 等11G或更多显存的显卡，可以调高至2或更高尝试；第33行`workers_per_gpu`可以不用怎么搞，一般`workers_per_gpu = 4`，5都可以



修改：`./mmdet/core/evaluation/class_names.py`

修改`voc_classes`，修改后内容如下：
```
def voc_classes():
    # return [
    #     'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    #     'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    #     'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    # ]
    return ['0','1','2','3','4','5','6','7','8','9']
```


修改：`./mmdet/datasets/voc.py`

将`CLASSES`修改为` CLASSES = ('0','1','2','3','4','5','6','7','8','9')`

## 4.训练

单显卡训练命令：

`python tools/train.py ./configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py`

多显卡训练命令:

`./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}`
如：`./tools/dist_train.sh  ./configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py 2`

## 5.输出

经过三轮训练，数据可以达到mAP至80+(一个epoch就要八小时累觉不爱)

在预测前，需要将`mchar_test_a`或者`mchar_test_b`的文件名修改为`040000.jpg 至 079999.jpg `，并生成`test.txt`保存至`data/VOCdevkit/VOC2007/ImageSets/Main`，把测试集图片放入`data/VOCdevkit/VOC2007/JPEGImages`中

或者不修改名称，将`data/VOCdevkit/VOC2007/JPEGImages`改为`data/VOCdevkit/VOC2007/JPEGImages_trainval`，把测试集图片放入`data/VOCdevkit/VOC2007/JPEGImages`中，省去了改名的麻烦。生成`test.txt`保存至`data/VOCdevkit/VOC2007/ImageSets/Main`。

测试命令：
`tools/test.py ./configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py ./work_dirs/detectors_cascade_rcnn_r50_1x_coco/latest.pth --out ./result.pkl`

测试结果保存至`result.pkl`中

如果需要看预测结果图片，提前创造一个文件夹为`test_img`，然后执行命令：
`tools/test.py ./configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py ./work_dirs/detectors_cascade_rcnn_r50_1x_coco/latest.pth --out ./result.pkl --show-dir ./test_img`
所有预测照片就保存在`test_img`文件夹中

最后将`result.pkl`转为要求格式：
```python
from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import pickle

f = open('result.pkl','rb')
data = pickle.load(f)
# print(data)

#%%
answer = []
for img in data:
    
    street_code = {}
    i = 0
    for number in img:
        if len(number) != 0:
            for j in number.tolist():
                if j[4] > 0.2: #置信率，仅需要预测概率大于0.2的预测结果，可以修改
                    # print(str(i) + ' ' + str(j))
                    street_code[j[0]] = i
        i +=1 
		
    keys = []
    for key,value in street_code.items():
        keys += [key]
    keys.sort()
    street = ''
	
    for k in keys:
        street += str(street_code[k])
    answer += [street]
import pandas as pd
import os
df = pd.DataFrame()

imglist = []
for i in range(40000):
    k = str(i)
    k = k.zfill(6)
    imglist  += [str(k)+'.jpg']

df['file_name'] = imglist
df['file_code'] = answer


df.to_csv('answer_020.csv',encoding = 'utf-8',index=False)
```

最后提交，完成






