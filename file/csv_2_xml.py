
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

path = 'dataset//'
csv_lists = os.listdir(path)
csv_list = [i for i in csv_lists if 'csv' in i]

for csv in track(csv_list):
    df = pd.read_csv(path + csv)
    
    source = {}
    label = {}
    
    jpg = (path + csv).replace('csv','jpg')
    fxml = jpg.replace('.jpg','.xml')
    
    fxml = open(fxml, 'w');
    imgfile = df['filename'][0]
    source['name'] = imgfile      #####
    source['path'] = jpg
    source['folder'] = 'VOC2007'
    
    source['width'] = df['width'][0]
    source['height'] = df['height'][0]
    
    fxml.write(out0 % source)

    for i in range(len(df)):
        label['class'] = df['class'][i]
        
        '''把txt上的数字（归一化）转成xml上框的坐标'''
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
                