'''
Author : Don
date : 2018/5/2
version:python3.6.2
'''
import re
import numpy as np
import math
import glob
import os
import random

def download_data(folder_name):
    all_data = [];row_data = [];classfy_list = []
    lastpath = os.getcwd()
    os.chdir(folder_name+'/ham')
    ham = glob.glob('*txt')
    for i in ham:
        wordlist = [k.lower() for k in re.split('\W+',open(i,encoding="ISO-8859-1").read()) if len(k)> 2 ]
        all_data.extend(wordlist)
        row_data.append(wordlist)
        classfy_list.append(1)
    os.chdir(os.pardir+'/spam')
    spam = glob.glob('*txt')
    for i in ham:
        wordlist = [k.lower() for k in re.split('\W+',open(i,encoding="ISO-8859-1").read()) if len(k)> 2 ]
        all_data.extend(wordlist)
        row_data.append(wordlist)
        classfy_list.append(0)
    return all_data , row_data , classfy_list    
  
def ensure(testlist):
    trainlist = list(range(0,50))
    for i in testlist:
        trainlist.pop(trainlist.index(i))
    return trainlist
  
def get_word_vector(row_data):
    wordvector = []
    for i in row_data:
        wordvector += i
    return list(set(wordvector))
    
def get_num_vector(wordvector,row_data):
    result_vector = [0] * len(wordvector)
    for word in row_data:
        if word in wordvector:
            result_vector[row_data.index(word)] += 1
        else:
            print("Sorry sir , the word can't be found in the vocabulary , Please upgrade you it !")
    return result_vector
    
def train(trainnp,trainclassfy):
    row = len(trainnp)
    column = len(trainnp[0])
    pA = sum(trainclassfy) / float(row)
    p0 = np.ones(column);p1 = np.ones(column)
    p0all = 2.0 ; p1all = 2.0
    for i in range(row):
        if trainclassfy[i] == 1:
            p1 += trainnp[i] #to calculate the number of per word
            p1all += sum(trainnp[i])
        else:
            p0 += trainnp[i]
            p0all += sum(trainnp[i])
    p1vector = np.log(p0 / p0all)
    p0vector = np.log(p1 / p1all)
    return p0vector , p1vector , pA
    
def classfy(wordvector , p0vector , p1vector , pA):
    p0 = sum(wordvector * p0vector) + np.log(pA)
    p1 = sum(wordvector * p1vector) + np.log(pA)
    if p0 > p1:
        return 0
    else:
        return 1
    
    
def useNB(folder_name):
    all_data , row_datas , classfy_list = download_data(folder_name)
    testlist = random.sample(list(range(0,50)),10)#to creat 10 no-repeat num
    trainlist = ensure(testlist)
    wordvector = get_word_vector(row_datas)
    trainnp = []; trainclassfy = []
    for i in trainlist:
        trainnp.append(get_num_vector(wordvector,row_datas[i]))
        trainclassfy.append(classfy_list[i])
    p0vector,p1vector,pA = train(np.array(trainnp),np.array(trainclassfy))
    errornum = 0
    for i in testlist:
        wordlist = get_num_vector(wordvector,row_datas[i])
        if classfy(wordlist,p0vector,p1vector,pA) != classfy_list[i]:
            errornum += 1
            print("the ", row_datas[i]," is false")
    print("Finally , the errorrate is %lf" % (errornum / len(row_datas)))
    return errornum / len(row_datas)
    
if __name__ == '__main__':
    #Please input your path
    folder_name = './email' 
    a = 0
    path = os.getcwd()
    for i in range(10):
        a += useNB(folder_name)
        os.chdir(path)
    print("="*100 ,"\n" ,str(a / 10).center(100,'-'),"\n","="*100)