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
 
val, train = data_split(jpg_list, ratio=0.1, shuffle=True)
print(len(train))
print(len(val))

    
for i in val:
    k = str(i.split('.')[0])
    # other_url = k.zfill(6)
    #print(k)
    file_obj.writelines(k)
    file_obj.write('\n')

file_obj.close()

for i in train:
    k = str(i.split('.')[0])
    # other_url = k.zfill(6)
    #print(k)
    file_obj2.writelines(k)
    file_obj2.write('\n')

file_obj2.close()

for i in jpg_list_test:
    k = str(i.split('.')[0])
    # other_url = k.zfill(6)
    #print(k)
    file_obj3.writelines(k)
    file_obj3.write('\n')

file_obj3.close()