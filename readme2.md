# mmsegmentation-in-IAIL(地表建筑物识别)

# 使用mmsegmentation(v0.21.1版本)进行地表建筑物识别

天池比赛街景字符编码识别使用DetectoRS进行目标检测

数据来源（[Inria Aerial Image Labeling](https://project.inria.fr/aerialimagelabeling/)），原来的每张图片非常大，所以进行进行拆分处理。

<img src="https://github.com/ethanliuzhuo/mmdetection-in-SVHN/blob/master/img/kit1.jpg" width="200px">
<img src="https://github.com/ethanliuzhuo/mmdetection-in-SVHN/blob/master/img/kit2.jpg" width="200px">

拆分的图片和比赛题目在天池的[官网](https://tianchi.aliyun.com/competition/entrance/531872/introduction)已经详细阐述，这里不再赘述；

## 1. 安装使用mmsegmentation
在这里，因为mmsegmentation只官方适配于Linux和MacOS操作系统，对Windows（7、10）并不官方支持，如果需要在Windows系统配置mmdetection，可参考[这里](https://www.bilibili.com/video/av795876868/)，下面配置在Ubuntu18.04进行。

在进入官网安装教程前，首先需要安装CUDA，Cudnn, 具体教程在[这里](https://blog.csdn.net/qq_32408773/article/details/84112166)可以找到。安装完毕后，安装[Anaconda](https://www.anaconda.com/)，用于建立虚拟环境。

在官网的安装教程中，有一些并不适合，在安装完Anaconda的终端命令行中，具体操作如下：

```bashrc
#创造虚拟环境
conda create -n open-mmlab python=3.7 -y 
#激活虚拟环境
conda activate open-mmlab

#从pytorch的官网下载安装pytorch，根据自己的CUDA 和 Cudnn的版本，选择相应的版本进行下载
conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch

#安装mmcv，是一个mmdse的基础包, CUDA 10.2为例
pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html

#或者直接运行
pip install mmcv-full

#下载mmsegmentation，目前最新版是0.21.1, 如果后续更新想下载固定版本，直接去github搜版本号
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation

#下载一些依赖包并安装
pip install -e .  # 或者 "python setup.py develop",别漏了最后一个点
```

下载编译过程较慢，至此mmsegmentation配置完成

## 1.1. 测试
```bashrc
import torch, torchvision
import mmseg
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
config_file = '/home/mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py' #修改路径
checkpoint_file = '/home/mmsegmentation/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth' #修改路径

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = '/home/mmsegmentation/demo/demo.png'
result = inference_segmentor(model, img)

# show the results
show_result_pyplot(model, img, result, get_palette('cityscapes'))
```

<img src="https://github.com/ethanliuzhuo/mmdetection-in-SVHN/blob/master/img/%E4%B8%8B%E8%BD%BD.png" width="400px">

然后就会输出图片，由此表示mmsegmentation配置完成.

## 2. 数据准备

在mmsegmentation文件夹下，创建文件夹`mkdir data`, `mkdir data/house`, `mkdir data/house/labels`，然后`cd data/house`;

将所有的下载好的压缩文件上传至这个文件夹，然后解压所有文件，比如`unzip train.zip`；

其中
  - `train_mask.csv`：存储图片的标注的rle编码；
  - `train`和`test`文件夹：存储训练集和测试集图片；

下面，我们将rle格式进行解码为图片；
首先，我们先把rle格式保存为txt格式，再将其储存为图片，当然也可以直接储存为图片，为了方便理解，先保存txt的字符格式；

在路径`/home/mmsegmentation`执行这段代码

```bashrc
import os
from secrets import randbelow
import pandas as pd
import numpy as np
import cv2
from tqdm import trange
from tqdm import tqdm
import time
from concurrent import futures


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

train_mask = pd.read_csv('data/house/train_mask.csv', sep='\t', names=['name', 'mask']) #读取csv
path = 'data/house/' #主路径

#输出txt文件
def out_txt(i):
    try:
        mask = rle_decode(train_mask['mask'].iloc[i]) #解码，必须是有标注的图片
    except:
        mask = np.zeros(512*512, dtype=np.uint8) #如果图片是空的，没有标注信息，则生成512*512大小的的纯背景图片
        mask = mask.reshape((512, 512), order='F')

        path_txt = path + '/labels/' +  train_mask['name'].iloc[i].split('.')[0] + '.txt'  #输出路径，记得提前设置文件夹mkdir data/house/labels
        np.savetxt(path_txt,mask, fmt="%i",delimiter=" ")  #保存

tasks, results = [], []
with futures.ThreadPoolExecutor(max_workers=5) as executor: #多线程计算
    for i in range(len(train_mask)):
        tasks.append(executor.submit(out_txt, i))
    for task in tqdm(futures.as_completed(tasks), total=len(tasks)): #查看进度
        results.append(task.result())
```

保存完txt文件以后，我们需要根据txt的信息，保存标注图片，以png的格式保存；

```bashrc
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm

# convert dataset annotation to semantic segmentation map
data_root = 'data/house'
img_dir = 'images'  #储存训练集图片的位置
ann_dir = 'labels'  #储存标注信息的位置
# define class and plaette for better visualization
classes = ('backgrade', 'house')   #填入label名称
palette = [[0, 0, 0], [255, 255, 255]] #可以改成其他颜色
    
for file in tqdm(mmcv.scandir(osp.join(data_root, ann_dir), suffix='.txt')):
      seg_map = np.loadtxt(osp.join(data_root, ann_dir, file)).astype(np.uint8)
      seg_img = Image.fromarray(seg_map).convert('P')
      seg_img.putpalette(np.array(palette, dtype=np.uint8))
      seg_img.save(osp.join(data_root, ann_dir, file.replace('.txt','.png')))
```
<img src="https://github.com/ethanliuzhuo/mmdetection-in-SVHN/blob/master/img/%E4%B8%8B%E8%BD%BD%20(1).png" width="400px">

测试图片，有输出即可；

```bashrc
import matplotlib.patches as mpatches
img = Image.open('data/house/labels/1A4O4TGL19.png')
plt.figure(figsize=(8, 6))
im = plt.imshow(np.array(img.convert('RGB')))

patches = [mpatches.Patch(color=np.array(palette[i])/255.,label=classes[i]) for i in range(2)]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize='large')
plt.show()
```

## 3. 划分训练、验证集和测试集
```bashrc
# 随机划分
split_dir = 'splits' #保存在该路径下
mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    osp.join(data_root, ann_dir), suffix='.png')]
#划分训练集
with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
    train_length = int(len(filename_list)*29/30)  #比例大小根据实际需要调整
    f.writelines(line + '\n' for line in filename_list[:train_length])
    
#划分验证集
with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
    f.writelines(line + '\n' for line in filename_list[train_length:]) #剩余的作为验证集
    
# 划分测试集
img_dir = 'test_a'
mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    osp.join(data_root, img_dir), suffix='.jpg')]

with open(osp.join(data_root, split_dir, 'test.txt'), 'w') as f:
    f.writelines(line + '\n' for line in filename_list)
```

最终数据存放格式为：
```bashrc
./mmsegmentation/data
                  └── house
                      ├──  images  #原始图片
                            ├── TGKBA2WTXG.jpg  
                            ├── *  
                            └── ZZXF6MOMGQ.jpg
                      ├──  labels  #标注信息
                            ├── Q6PNPF6EFT.png  
                            ├── Q6PNPF6EFT.txt
                            ├── * 
                            ├── WOQRR5XL8L.png
                            └── WOQRR5XL8L.txt
                      ├──  splits   #训练集划分
                            ├── test.txt  
                            ├── train.txt  
                            └── val.txt  
                      └──  test_a   #测试集
                            ├── QC9YNOG85A.jpg  
                            ├── *
                            └── WTNVWP6BSA.jpg
```

## 4. 修改配置

### 4.1. 修改 Dataset Classes 数据配置模型

在`mmsegmentation/mmseg/datasets`，这里保存了数种不同的数据集格式；现在我们以VOC的数据集格式为例，创建一个新的Dataset Class；
```bashrc
# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class HouseDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background', 'house') #类别名字，可以修改

    PALETTE = [[0, 0, 0], [255, 255, 255]]  #类别颜色，可以修改

    def __init__(self, split, **kwargs):
        super(HouseDataset, self).__init__(  # HouseDataset取名，可以修改成其他名称
            img_suffix='.jpg',   #img_suffix使用原始图片格式，这里是jpg图片
            seg_map_suffix='.png', split=split, **kwargs)  #seg_map_suffix使用标注图片格式，这里是png图片
        assert osp.exists(self.img_dir) and self.split is not None
```

这样我们就完成了对地表建筑物识别的类别名称设置了。设置好之后，保存在`mmseg/datasets/`目录下，命名为自己喜欢的名称，我取名为`voc2.py`。

另外还需要设置一下该目录下的`mmsegmentation/mmseg/datasets/__init__.py`文件，按照注释修改：
```bashrc
from .voc import PascalVOCDataset
from .voc2 import HouseDataset   #加入最后一行，HouseDataset，之前取数据集的名称

__all__ = [
    'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset',
    'PascalVOCDataset', 'ADE20KDataset', 'PascalContextDataset',
    'PascalContextDataset59', 'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset',
    'STAREDataset', 'DarkZurichDataset', 'NightDrivingDataset',
    'COCOStuffDataset', 'LoveDADataset', 'MultiImageMixDataset',
    'ISPRSDataset', 'PotsdamDataset','HouseDataset'   #加入名称
]
```

### 4.2. 修改 Dataset Config 数据配置文件
在`mmsegmentation/configs/_base_/datasets` 中，创建一个新文件或者修改`pascal_voc12.py` 中的内容；

以`pascal_voc12.py`为模板，我们修改成：

```bashrc
# dataset settings
dataset_type = 'HouseDataset'  #数据集名称，与__init__.py 新加入的名称一致
data_root = 'data/house/' #数据路径
img_norm_cfg = dict(
    #mean=[103.18, 108.86, 100.05], std=[51.54,46.80,44.87], to_rgb=True) #该数据集的方差和均值
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)  #原始的数据集的方差和均值
crop_size = (512, 512) #数据增强时裁剪的大小，可以修改成小一点
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)), #img_scale 修改成原始图片大小
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8, #batch size， 1,2,4，8,16 这样设置
    workers_per_gpu=8, #dataloader的线程数目，一般设2，4，8，根据CPU核数确定
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images', #图片路径
        ann_dir='labels', #标注信息路径
        split='splits/train.txt', #训练集路径
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        split='splits/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        split='splits/test.txt',
        pipeline=test_pipeline))
```

根据注释，修改内容；保存成新的文件`pascal_voc_my.py`至`mmsegmentation/configs/_base_/datasets`下；

注意！因为修改原始配置，所以需要执行`python setup.py develop`重新编译一次；

### 4.3. 修改模型Config配置文件

Config文件是train.py直接调用的config文件，模型可以根据自己需要进行选择，模型的效果都在官方文档中有。以`pspnet`为例，我们修改`mmsegmentation/configs/pspnet/pspnet_r50-d8_769x769_40k_cityscapes.py`这个Config文件，变成符合我们的数据集；

```bashrc
_base_ = [
    '../_base_/models/pspnet_r50-d8.py', #骨干模型路径
    '../_base_/datasets/pascal_voc_my.py', #修改成刚刚写的Dataset config文件路径
    '../_base_/default_runtime.py',  #这里保持不变，可以修改
    '../_base_/schedules/schedule_40k.py' #这里保持不变，可以修改
]
model = dict(
    decode_head=dict(align_corners=True),  #不变
    auxiliary_head=dict(align_corners=True), #不变
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513))) #不变

evaluation = dict(metric='mDice') #增加了验证方法，按照比赛的要求
```

进入`../_base_/models/pspnet_r50-d8.py·`修改，即`mmsegmentation/configs/_base_/models/pspnet_r50-d8.py`；
如果是单GPU训练，则将`norm_cfg = dict(type='SyncBN', requires_grad=True)`修改成`norm_cfg = dict(type='BN', requires_grad=True)，如果使用了SyncBN却只有一块可用的GPU，那可能会报类似AssertionError:Default process group is not initialized的错误。

然后修改`num_classes`，另`num_classes = 2`，分别位于24和37行，修改完成后保存；其他不需要改；

### 4.4. 可选配置修改

`../_base_/default_runtime.py`修改：

- 在`../_base_/default_runtime.py`中，可以选择修改`workflow`，[（'train'，1）]表示只有一个workflow，名为'train'的workflow执行一次。workflow按照总的时间段将模型分为若干个循环进行训练。这里可以改成`workflow = [('train', 1),('val', 1)]`;
- `load_from = None`是预训练加载模型的路径；
- `resume_from = None`是中断训练后，如果想继续训练，加载的模型路径；
-  `interval=100` 是多少步以后输入一次loss

`../_base_/schedule_40k.py`修改：

```bashrc
# optimizer
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0005) #优化器选择，学习速率
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False) #学习速率衰减方法
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)  #多少步长
checkpoint_config = dict(by_epoch=False, interval=1000) #多久保存一次模型
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True) #多久验证一次和验证方法
```

## 5. 训练

如果需要预训练模型，去mmlab相应的网站下载，比如我配置了pspnet_r50-d8_769x769_40k_cityscapes的文件，就需要去[这里](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet)下载相应的模型到`checkpoints`里。比如在`checkpoints`里使用`wget https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_769x769_40k_cityscapes/pspnet_r50-d8_769x769_40k_cityscapes_20200606_112725-86638686.pth`

单显卡训练命令：

`python tools/train.py configs/pspnet/pspnet_r50-d8_769x769_40k_cityscapes.py --work-dir house --load-from checkpoints/pspnet_r50-d8_769x769_40k_cityscapes_20200606_112725-86638686.pth`

- `train.py` 为训练命令；
- `configs/pspnet/pspnet_r50-d8_769x769_40k_cityscapes.py`为配置文件路径；
- `--work-dir`为模型保存路径，没有会自动生成；
- `--load-from`为预训练模型路径；

多显卡训练命令:

`./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}`
如：`./tools/dist_train.sh configs/pspnet/pspnet_r50-d8_769x769_40k_cityscapes.py 2 --work-dir house --load-from checkpoints/pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth`

在配置文件路径后面加入GPU 数量即可。

## 6. 预测

使用两个V100训练400000步大概需要一天半的时间，大约23个epoch。mDice在验证集的值为0.9371。

在预测前，需要将`test_a`的图片复制到`images`，即`cp data/house/test_a/* data/house/images`，否则路径不对；

预测命令： `python tools/test.py configs/pspnet/pspnet_r50-d8_769x769_40k_cityscapes.py house/latest.pth --out data/house/result.pkl --show-dir data/house/perdict`

- `configs`为配置文件；
- `house/latest.pth`为模型路径；
- `--out` 输出结果文件路径；
- `--show-dir` 输出结果图片路径，记得提前生成空的文件夹；

输出csv结果
```bashrc
import pickle
f = open('house2/result.pkl','rb') #结果保存路径
data = pickle.load(f)

mask = []
name = []
for i,image in enumerate(data):
    rle = rle_encode(image.astype('uint8')) #这个函数之前有，转为rle格式
    mask += [rle]
    
df = pd.DataFrame()
test_mask = pd.read_csv('data/house/test_a_samplesubmit.csv',sep='\t',names = ['name','mask']) #读取样本数据

name = test_mask['name'] 

df['name'] = name
mask_all = ['' for i in range(len(df))]

for i in range(len(mask)):
    ind  = df[df.name==filename_list[i]+ '.jpg'].index[0]
    mask_all[ind] = mask[i]
df['mask'] = mask_all

df.to_csv('data/house2/test_b.csv',encoding='utf-8',header = None,index=False,sep = '\t') #输出结果
```

最后提交，完成


