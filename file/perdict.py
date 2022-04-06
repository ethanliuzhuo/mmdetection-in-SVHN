from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import pickle

f = open('result2.pkl','rb')
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
                if j[4] >= 0.3: #置信率，仅需要预测概率大于0.2的预测结果，可以修改
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
    imglist  += [str(k)+'.png']

df['file_name'] = imglist
df['file_code'] = answer


df.to_csv('answer_020.csv',encoding = 'utf-8',index=False)