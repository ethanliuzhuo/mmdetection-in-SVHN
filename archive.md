# 军用飞机目标检测 Baseline
# 使用mmdetection(2.22版本)进行军用飞机检测 

Kaggle [Military Aircraft Detection Dataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset)使用[Swin Transformer](https://github.com/microsoft/Swin-Transformer)进行多类目标检测Baseline；

数据集描述：
- VOC数据集格式 (xmin, ymin, xmax, ymax)
- 36 种飞机型号
('F15','F18','Mirage2000','US2','JAS39','RQ4','EF2000','C5','A400M','SR71','B1','C130','AG600','F14','C17','F35','B52','Su57','U2','Tu160','F22','B2','A10','F4','YF23','J20','F117','E2','XB70','Tu95','F16','V22','Rafale','MQ9','Mig31','Be200')

## 目录：
- 安装mmdetection
    - （不推荐）[官方方法](#jump1)
    - （推荐） [Docker方法](#jump2)
- [转化格式](#jump3)
- 修改配置
    - [性能版](#jump4)
    - (新手推荐)[简易版](#jump5)
- [训练](#jump6)
- [测试](#jump7)
- [错误提示](#jump8)
    
   
<span id="jump1"></span>
## 1.安装使用mmdetection

## 方法1：（不推荐）官方教程
### 1.1安装
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
pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html

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

<span id="jump2"></span>
## （推荐）方法2：Docker

如果熟悉Docker，可使用Docker创建，这样可以省去配置的麻烦。流程如下：

1. 在`239`服务器路径`/home/xxxx/Docker_images`路径下有一个`mmlab.tar`的压缩包，该压缩包为docker image文件，进入该文件夹；

2. 使用命令`docker import - {yourname} <  mmlab.tar`，创建一个新的docker image。其中yourname填写image名字，最好不要与前面的名字重复，如 `docker import - mmlabtest <  mmlab.tar`；

3. 第二步过程有点久，请耐心等待。有一串sha256:xxxxx的输出代表导入成功；

4. 输入`docker image ls`，查看IMGAE ID，复制记录下来；

5. 输入`docker run --name {your_container_name}  -p 88:66 -tdi --shm-size 64G --gpus all -v /home/xxx/xxx/:/home {IMGAE ID} /bin/bash`，生成Container。其中:
    - `--name`代表Container的名称,可自定义名字；
    - `-p 88:66`代表服务器端口映射至Docker端口（只要不和以前的Container端口冲突都可以自选）；
    - `--shm-size`代表共享内存，非常重要；
    - `-v` 代表本地服务器文件夹映射到Docker的文件夹，以后所有文件都会保存在这里 ，冒号前本地服务器路径，必须为绝对路径，冒号后Container挂载的路径，如`home/root/myname/:/home`
    - `{IMGAE ID}`输入第四步得到的ID； /bin/bash必填，启动项；
6. (可选) 输入`docker ps`，查询生成的`Container ID`;
7. 进入Docker,`docker exec -it {your_container_name} bash` 或者`docker exec -it {your_container_ID} bash`；
8.  进入主文件夹`cd home`，下载mmdetection文件夹`git clone https://github.com/open-mmlab/mmdetection.git`，然后`cd mmdetection`。执行编译程序`python setup.py install`；
9. `Finished processing dependencies for mmdet==2.22.0`代表编译成功；
10. 安装[官网](https://mmdetection.readthedocs.io/en/v2.21.0/get_started.html) 方法验证一次即可。有输出就行。

11. (参考)`docker image rm a780a9281059`删除image


<span id="jump3"></span>
## 2.转化格式

将数据集下载解压后我们得到`annotated`、`crop`和`dataset`文件夹。其中只有`dataset`文件夹中的数据集有用。

文件解释如下：
```bash
./dataset
    ├── 000ec980b5b17156a55093b4bd6004ab.csv #标注信息
    ├── 000ec980b5b17156a55093b4bd6004ab.jpg #图片
    ├── *
    ├── *
    ├── ffca28378a5df5113d498f59ed282a98.csv #标注信息
    └── ffca28378a5df5113d498f59ed282a98.jpg #图片
```

为了方便使用mmdetection进行训练、预测，我们需要将数据集的文件夹整理成VOC的文件夹的目录形式。

PASCAL VOC挑战赛 （The PASCAL Visual Object Classes ）是一个世界级的计算机视觉挑战赛, PASCAL全称：Pattern Analysis, Statical Modeling and Computational Learning，是一个由欧盟资助的网络组织。

详情可以参考[链接1](https://arleyzhang.github.io/articles/1dc20586/)和[链接2](https://zhuanlan.zhihu.com/p/33405410)，有详细的说明。

VOC的文件夹目录形式是这样：
```brash
./VOC2007
    ├── Annotations  # 标注的VOC格式的xml标签文件
    ├── JPEGImages   # 数据集图片
    └── ImageSet     #数据集划分
           └── Main
                ├── test.txt   # 划分的测试集
                ├── train.txt   #划分的训练集
                └── val.txt   # 划分的验证集
```

首先现在`data/archive`文件夹下创建文件夹`VOC2007`，使用命令`mkdir VOC2007`。然后`cd VOC2007`进入文件夹，用相同的方法创建`Annotations`、`JPEGImages`、`ImageSet`。最后别忘了进入`ImageSet`，再创建一个文件夹`Main`，非常重要。

该数据集已经提供class name和坐标，就是VOC格式的，因此无需转化。编写一个脚本，把csv的格式转成VOC的数据集xml格式。创建`csv_2_xml.py`文件放在`data/archive`文件夹下。


```python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 09:46:35 2022

@author: liuzhuo
"""
import pandas as pd
import numpy as np
import os
from rich.progress import track

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

path = '/dataset' #修改路径
csv_lists = os.listdir(path)
csv_list = [i for i in csv_lists if 'csv' in i] #只选择csv的格式

for csv in track(csv_list):
    """遍历"""
    df = pd.read_csv(path + csv) #读取csv
    
    source = {}
    label = {}
    
    jpg = (path + csv).replace('csv','jpg') #读取照片格式
    fxml = jpg.replace('.jpg','.xml') #xml文件路径
    
    fxml = open(fxml, 'w') #在同一个文件夹下创建文件
    imgfile = df['filename'][0] #图片名
    source['name'] = imgfile     
    source['path'] = jpg 
    source['folder'] = 'VOC2007' #固定格式
    
    source['width'] = df['width'][0] #图片宽
    source['height'] = df['height'][0] #图片高
    
    fxml.write(out0 % source) #写入

    for i in range(len(df)):
        label['class'] = df['class'][i]
        
        '''把csv上的数字（归一化）转成xml上框的坐标'''
        xmin = df['xmin'][i]
        ymin = df['ymin'][i]
        xmax = df['xmax'][i]
        ymax = df['ymax'][i]
        
        label['xmin'] = xmin
        label['ymin'] = ymin
        label['xmax'] = xmax
        label['ymax'] = ymax
            
        fxml.write(out1 % label)
    fxml.write(out2)
    fxml.close()
print('---------------Done!!!--------------')            
```

生成的xml文件现在依然在`dataset`中，现在将照片和标注转移至VOC文件夹中。在该目录下，使用命令`cp -r dataset/*.xml VOC2007/Annotations/`转移标注文件和命令`cp -r dataset/*.jpg VOC2007/JPEGImages/`。

在`VOC2007/ImageSets/Main`下，划分训练集和测试集，生成`train.txt`,`val.txt`,代码如下：

```python
import os
import pandas as pd
import random
print('注意路径')
file_obj = open("VOC2007/ImageSets/Main/val.txt", 'w', encoding='utf-8') #注意路径
file_obj2 = open("VOC2007/ImageSets/Main/train.txt", 'w', encoding='utf-8') #注意路径

jpg_list = os.listdir('VOC2007/JPEGImages') #注意路径
jpg_list_test = os.listdir('test') #注意路径

 
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
print('训练集有：',len(train))
print('测试集有：',len(val))

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
```

最终数据存放格式为：
```brash
./archive
    ├── dataset
        ├── **.jpg
        ├── **.csv
        └── *****
    └── VOC2007
        ├── Annotations  # 标注的VOC格式的xml标签文件
                ├── **.xml
                └── **.xml
        ├── JPEGImages   # 数据集图片
                ├── **.jpg
                └── **.jpg
        └── ImageSet     # 训练集划分
                └── Main
                     ├── train.txt 
                     └── val.txt
```

请检查文件夹是否如上，以及Annotations和JPEGImages的数量是否相等。

下面为了检查xml标注是否成功，用一个脚本进行可视化检查：
```python
import cv2
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import xml.etree.ElementTree as ET
from pylab import mpl
import matplotlib

import matplotlib.pyplot as plt


def draw_single_image(ann_path,img_path,save_path=None):
    """
    ann_path:指定xml的绝对路径
    img_path:指定xml的绝对路径
    save_path:如果不是None,那么将是结果图的保存路径；反之则画出来
    """
    img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
    if img is None or not img.any():
            raise '有空图'
    tree = ET.parse(ann_path)
    root = tree.getroot()
    result = root.findall("object")
    for obj in result:
        name = obj.find("name").text
        x1=int(obj.find('bndbox').find('xmin').text)
        y1=int(obj.find('bndbox').find('ymin').text)
        x2=int(obj.find('bndbox').find('xmax').text)
        y2=int(obj.find('bndbox').find('ymax').text)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
        print(name)
        cv2.putText(img,name,(x1,y1),cv2.FONT_ITALIC,2,(0,0,255),3) #各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    if save_path is None:
        imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20,10))
        plt.imshow(imgrgb)
    else:
        cv2.imencode('.jpg',img)[1].tofile(save_path)

root = '/home/mmdetection/data/archive/dataset' #
ann_path = os.path.join(root, '0a13a49b1de78259ec8b646e9cb748d8.xml')  # xml文件所在路径
pic_path = os.path.join(root, '0a13a49b1de78259ec8b646e9cb748d8.jpg')  # 样本图片路径

draw_single_image(ann_path,pic_path,save_path=None)
```

<img src="https://github.com/ethanliuzhuo/mmdetection-in-SVHN/blob/master/img/111.png" width="700px">


<span id="jump4"></span>
## 3.修改配置

### 3.1 修改数据读取配置文件
#### 3.1.1 修改dataset配置文件

复制文件并重命名为`cp /mmdetection/configs/_base_/datasets/voc712.py /mmdetection/configs/_base_/datasets/mydata.py`，

因为我们使用的是VOC2007格式，并没有VOC2012,，因此只要把其中含有VOC2012路径注释掉，并修改路径和图像的均值以及大小。

修改过后内容如下：
`/mmdetection/configs/_base_/datasets/mydata.py`

<span id="jump"></span>
```python
# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/archive/'  #在VOC2007上一级
img_norm_cfg = dict(
    mean=[319.104,299.816,277.655], std=[278.717,261.0871,246.235], to_rgb=True)
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
    samples_per_gpu=4, #batch_size,爆显存改这里
    workers_per_gpu=4,
    train=dict(
        #type='RepeatDataset',
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
        ann_file=data_root + 'VOC2007/ImageSets/Main/val.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline
        ))
evaluation = dict(interval=1, metric='mAP')
```

#### 3.1.2 修改dataset名称

修改名称：`./mmdet/core/evaluation/class_names.py`
，注意别忘了加逗号。

将`voc_classes`名称修改如下:
```python
def voc_classes():
    return [
        'F15','F18','Mirage2000','US2','JAS39','RQ4','EF2000','C5','A400M',
        'SR71','B1','C130','AG600','F14','C17','F35','B52','Su57','U2','Tu160',
        'F22','B2','A10','F4','YF23','J20','F117','E2','XB70','Tu95','F16',
        'V22','Rafale','MQ9','Mig31','Be200'
    ] 
```

修改：`./mmdet/datasets/voc.py`，添加CLASSES和PALETTE，PALETTE是调色板，输出画图的框的颜色，随意填RGB三原色。

将VOC名称修改如下:
```python
    CLASSES = ('F15','F18','Mirage2000','US2','JAS39','RQ4','EF2000','C5','A400M',
               'SR71','B1','C130','AG600','F14','C17','F35','B52','Su57','U2','Tu160',
               'F22','B2','A10','F4','YF23','J20','F117','E2','XB70','Tu95','F16',
               'V22','Rafale','MQ9','Mig31','Be200')

    PALETTE = [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
               (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
               (153, 69, 1), (120, 166, 157), (0, 182, 199), (0, 226, 252),
               (182, 182, 255), (0, 0, 230), (220, 20, 60), (163, 255, 0),
               (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88),(182, 192, 255), 
               (255, 0, 230), (225, 20, 60), (163, 255, 255),
               (0, 90, 0), (255, 95, 161), (0, 80, 101), (60, 130, 88),
               (196, 226, 255), (0, 65, 100), (0, 243, 142), (255, 45, 255),
               (124, 69, 1), (120, 1, 1), (0, 45, 199), (0, 5, 252)]
    
```



### 3.2 修改模型配置文件

我们选择swin-Transformation作为分类主干网络，选择CascadeRCNN作为检测主干网络。

修改： `./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py`

将第一行的` ../_base_/datasets/mask_rcnn_r50_fpn.py`更改为` ../_base_/datasets/cascade_rcnn_r50_fpn.py`，我们不需要mask，因此换了一个主模型

将第二行的` ../_base_/datasets/coco_detection.py`更改为` ../_base_/datasets/mydata.py`，从COCO数据集更改为VOC数据集格式

```python
_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/mydata.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
```

- 第10行修改成`type='CascadeRCNN'`，即检测模型主框架
- 第31行`img_norm_cfg`修改成`img_norm_cfg = dict(
    mean=[319.104,299.816,277.655], std=[278.717,261.0871,246.235], to_rgb=True)`
- 第36行把`with_mask=True`改成`with_mask=False`
- 第73行把`keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])`中的`'gt_masks'`删掉

### 3.3 修改类别数量
修改` ../_base_/datasets/cascade_rcnn_r50_fpn.py`

使用`Ctrl + F`，搜索`num_classes`，将所有的类别数从80改成36，即一共36类，非常重要 ；

注意：类别有多少类为多少类，背景不计入类别

（非常重要）最后，因为修改了类别，需要重新编译，覆盖以前的链接路径，`python setup.py install`。

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


去mmdetection [model zoo](https://github.com/open-mmlab/mmdetection/tree/master/configs/swin)下载`mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth`，保存至`checkpoints`文件夹中，将路径复制到`load_from`，顶替None。这样就可以预训练模型配置。不如不需要预训练模型，则用None代替。

`resume_from`表示如果训练中断，恢复上次保存的模型的路径，继续训练。

### 3.5 修改学习策略
修改：`../configs/_base_/schedules/schedule_1x.py`

因为原来的学习速率为0.02，这对于该数据集来说太高了，将`lr`改为lr=0.0001，不需要太高;

这里可以设置学习速率衰减和优化器，类型与Pytorch保持一致，文档可以参考Pytorch;

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
<span id="jump5"></span>
## （简易版）使用cascade_rcnn配置预测。
如果swin transformer的教程太难，我们可以使用cascade_rcnn来进行简单的预测。

参考`configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py`, 里面有四个文件路径，我们分别修改这四个文件：

1. `../_base_/models/cascade_rcnn_r50_fpn.py`中，找到`num_classes`，将其所有数字变成36，一共三处，并保存；

2. `../_base_/datasets/coco_detection.py`中，把`coco_detection`变成`voc0712.py`，从COCO数据集更改为VOC数据集格式,`voc0712.py`的内容在[3.1.1](#jump)中，并保存；

3. `../_base_/schedules/schedule_1x.py`，把学习速率`lr = 0.0001`该小即可，并保存；

4. `../_base_/default_runtime.py`中`load_from`替换成预训练模型路径。点击[这里](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_caffe_fpn_1x_coco/cascade_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.404_20200504_174853-b857be87.pth)下载预训练并保存到checkpoints里，复制路径到`load_from`并保存。

<span id="jump6"></span>
## 4.训练

单显卡训练命令：

`python tools/train.py ./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py --work-dir archive --gpu-id 0`

各参数依次是：训练脚本名，配置文件，模型和日志保存路径（没有会自己创建），制定GPU（可选，不填默认0）

多显卡训练命令:

`./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}`
如：`./tools/dist_train.sh  ./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py 4 --work-dir archive`

各参数依次是：训练脚本名，配置文件，GPU数量，模型和日志保存路径（没有会自己创建）

<span id="jump7"></span>
## 5.预测图片

经过36轮训练，数据可以达到mAP至0.77+，对于该数据集已经足够。

### 5.1  大规模图片预测
如果是测试验证集的图片，可以使用`tool/test.py`预测，

在预测前，需要将`验证集`的图片复制到`data/VOCdevkit/VOC2007/JPEGImages`中，并注意已经生成了好了txt格式

测试命令`python tools/test.py configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py archive/latest.pth  --out archive/result.pkl --show-dir archive --show-score-thr 0.5`

各参数依次是：测试脚本名，配置文件路径，模型路径，输出结果路径（BOX和概率信息），测试框好的图片保存路径，置信度（可选，不选默认0.3）

### 5.2  单图片预测

如果实在Jupyter Notebook等需要可视化界面时，使用：

```python
# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
# Check MMDetection installation
import mmdet
print(mmdet.__version__)
import cv2
# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

checkpoint = '/home/mmdetection/archive/latest.pth' #模型保存路径
config = '/home/mmdetection/archive/222.py' #配置文件路径
model = init_detector(config, checkpoint, device='cuda:0')

# test a single image
img = '/home/mmdetection/data/archive/dataset/819275df23bec1c0314d5399a1a37162.jpg'  #图片测试路径
result = inference_detector(model, img) #BOX和概率信息
imgs =  show_result_pyplot(model, img, result,score_thr =0.5)
```

如果只测试，但只需要保存图片至路径时：

```python
out_file = '/data/archive/test.jpg'
imgs = model.show_result(img, result,score_thr =0.7)
cv2.imwrite(out_file, imgs)
```
<img src="https://github.com/ethanliuzhuo/mmdetection-in-SVHN/blob/master/img/output.png" width="500px">

### 5.3  视频预测

视频本质是由一张张图片（帧）组成的长图片，因此如果对视频进行直接输出，需要用OpenCV对视频进行拆解，拆解一个个frame，然后转成图片进行预测，再组合成视频。

`python /home/mmdetection/demo/video_demo.py /home/mmdetection/data/archive/video/ag600.mp4 /home/mmdetection/archive/222.py /home/mmdetection/archive/latest.pth --score-thr 0.5 --out /home/mmdetection/data/archive/video/ag600_1.mp4 `

各参数依次是：视频检测脚本名，原视频路径，配置文件路径，模型路径，，置信度（可选，不选默认0.3），视频输出路径

因为拆解速度慢，需要需要时间。1分钟的视频大约需要3分钟预测输出。

如果想在Jupyter-Notebook中查看mp4视频，脚本在：

```python
from IPython.display import clear_output,  display, HTML
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2
import os

def show_video(video_path:str,small:int=2):
    if not os.path.exists(video_path):
        print("视频文件不存在")
    video = cv2.VideoCapture(video_path)
    current_time = 0
    while(True):
        try:
            clear_output(wait=True)
            ret, frame = video.read()
            if not ret:
                break
            lines, columns, _ = frame.shape
   
            if current_time == 0:
                current_time = time.time()
            else:
                last_time = current_time
                current_time = time.time()
                fps = 1. / (current_time - last_time)
                text = "FPS: %d" % int(fps)
                cv2.putText(frame, text , (0,100), cv2.FONT_HERSHEY_TRIPLEX, 3.65, (255, 0, 0), 2)
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (int(columns / small), int(lines / small)))

            img = Image.fromarray(frame)

            display(img)
            # 控制帧率
            time.sleep(0.001)
        except KeyboardInterrupt:
            video.release()
show_video('/home/mmdetection/data/archive/video/ag600_1.mp4')
```

<img src="https://github.com/ethanliuzhuo/mmdetection-in-SVHN/blob/master/img/f22_2.gif" width="400px">

<span id="jump8"></span>
## 错误提示

-  `AssertionError: The 'num_classes' (10) in Shared2FCBBoxHead of MMDataParallel does not matches the length of 'CLASSES' 80) in RepeatDataset` 错误原因：数据集的类别信息仍是coco类别80类，(我的数据集是10类)。在修改完 class_names.py 和 voc.py 之后要重新编译： `python setup.py install`；
-  `AssertionError: 'CLASSES' in ConcatDatasetshould be a tuple of str.Add comma if number of classes is 1 as CLASSES = (0,)` 错误原因：修改num_classes成自己的类别数，如果是一个类别，漏了逗号，需要写成CLASSES = (person,)，否则会出现错误；
-  `AttributeError: 'ConfigDict' object has no attribute 'pipeline'` 错误原因：这是官方文件中的bug, 是因为pascal_voc下这几个配置文件都调用了.\configs\_base_\voc0712.py, 而错误就发生在 .\configs\_base_\voc0712.py, 标红的那一块(左图35,36,37行, 其实就是把这三行删掉), 改成右图。
-  `FileNotFoundError: [Errno 2] No such file or directory: '*.png' `错误原因：`mmdetection/mmdet/datasets/xml_style.py`中的51行图片格式不对，改成相应的图片格式;
