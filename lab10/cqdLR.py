#coding:utf-8
'''
Author:Don98
date:2018.5.29
platfrom : win32
version : 3.6.2
一些简单的说明。
学习率即rate设置为0.0001
使用了批量梯度下降
'''

import numpy as np
from matplotlib import pyplot as plt

rate = 0.0001

def get_data(folder_path):
    with open(folder_path ,'r') as f:
        data = f.readlines()
    label = []
    for i in range(len(data)):
        data[i] = data[i].strip('\n').split('\t')
        label.append(data[i].pop())
    return np.array(data,dtype = 'float64') , np.array(label,dtype = 'float64')
    
def get_w0(xnp , w ,ynp):
    w0 = []
    for i in range(xnp.shape[0]):
        w0.append(rate * (ynp[i] - 1 / ( 1 + np.e**(-np.dot(xnp[i],w)))))
    return np.array(w0,dtype = 'float64')
    
def to_val(w,valnp,ynp2):
    counter = 0
    for i in range(valnp.shape[0]):
        h = ynp2[i] - 1 / ( 1 + np.e**(-np.dot(valnp[i].T,w)))
        if abs(h) > 0.5:
            counter += 1
    # print("the error rate is %lf"% (counter / valnp.shape[0]))
    return counter
    
def to_upgrade(w , trainnp,ynp1):
    w0 = get_w0(trainnp ,w,ynp1)
    w = w + np.dot(w0.T,trainnp)
    return w
        
def work(trainnp , ynp,valnp,ynp2):
    w = np.array([1] * trainnp.shape[1],dtype = 'float64')
    wt = np.array([1] * trainnp.shape[1],dtype = 'float64')
    counter = 0
    while True:
        if counter % 1000 == 0:
            print(counter,1 - to_val(w,valnp,ynp2) / valnp.shape[0])
        if counter == 100000:
            break
        wt = to_upgrade(wt , trainnp,ynp1)
        a = to_val(w,trainnp,ynp1)
        b = to_val(wt,trainnp,ynp1)
        w = wt if a > b else w
        # print(a,b)
        counter += 1
    print('Accuracy:%lf'%(1 - to_val(w,valnp,ynp2) / valnp.shape[0]))
    
    
if __name__ == '__main__':
    train_folder = 'horseColicTraining.txt'
    val_folder = 'horseColicTest.txt'
    trainnp , ynp1 = get_data(train_folder)
    valnp , ynp2 = get_data(val_folder)
    w = work(trainnp, ynp1,valnp,ynp2)
