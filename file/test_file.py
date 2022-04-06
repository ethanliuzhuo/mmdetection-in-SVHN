from sys import path_importer_cache
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
config_file = "VOC/111myconfigs.py"
checkpoint_file = 'VOC/latest.pth'
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
img = '/home/mmsegmentation/data/house/JPEGImages/0A3B10OZ9S.jpg'
result = inference_segmentor(model, img)
show_result_pyplot(model, img, result, [[0,0,0],[255,255,255]])
#%%

from PIL import Image
img = Image.open('/home/mmsegmentation/data/house/SegmentationClass2/0A3B10OZ9S.png')
print(img.size)
img2 = Image.open('/home/mmsegmentation/data/house/JPEGImages/0A3B10OZ9S.jpg')
print(img.size)

#%%
img = Image.open('/home/mmsegmentation/data/VOCdevkit/VOC2012/JPEGImages/2007_000063.jpg')
print(img.size)
img2 = Image.open('/home//mmsegmentation/data/VOCdevkit/VOC2012/SegmentationClass/2007_000063.png')
print(img.size)
#%%
import matplotlib.image as mpimg
I = mpimg.imread('/home/mmsegmentation/data/house/SegmentationClass2/0A3B10OZ9S.png')
I1 = mpimg.imread('/home/mmsegmentation/data/house/JPEGImages/0A3B10OZ9S.jpg')

I2 = mpimg.imread('/home/mmsegmentation/data/VOCdevkit/VOC2012/JPEGImages/2007_000063.jpg')
I3 = mpimg.imread('/home//mmsegmentation/data/VOCdevkit/VOC2012/SegmentationClass/2007_000063.png')
print(I.shape)
print(I1.shape)

print(I2.shape)
print(I3.shape)