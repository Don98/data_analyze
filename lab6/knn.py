# -*- coding:utf-8 -*-
"""
项目名称：LAB6
作者：ABC
日期：2018.04.11
changed by Don
date 2018.04.17
"""

#导入必要的库
from numpy import *
import operator
import time
from os import listdir
import numpy as np
import csv
#创建数据集和标签
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#实现knn分类器
def classify(inputPoint,dataSet,labels,k):
    """
        inputPoint:待判断的点
        dataSet:数据集合
        labels:标签向量，维数和dataSet行数相同，比如labels[1]代表dataSet[1]的类别
        k:邻居数目
        输出：inputPoint的标签
    """
    dataSetSize = dataSet.shape[0]     #已知分类的数据集（训练集）的行数
    #先用tile函数将输入点拓展成与训练集相同维数的矩阵，再计算欧氏距离
    #使用tile函数将inputPoint复制dataSetSize次，成为一个矩阵，然后和原矩阵作差
    diffMat = tile(inputPoint,(dataSetSize,1))-dataSet  #样本与训练集的差值矩阵
    #这里的乘号不是我们线性代数里面学的矩阵乘法，而是对于每个元素乘方
    sqDiffMat = diffMat ** 2                    #差值矩阵平方
    sqDistances = sqDiffMat.sum(axis=1)         #计算每一行上元素的和
    distances = sqDistances ** 0.5              #开方得到欧拉距离矩阵
    #argsort返回的是数组从小到大的元素的索引
    sortedDistIndicies = distances.argsort()    #按distances中元素进行升序排序后得到的对应下标的列表
    #选择距离最小的k个点，统计每个类别的点的个数
    classCount = {}
    for i in range(k):
        voteIlabel = labels[ sortedDistIndicies[i] ]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #按classCount字典的第2个元素（即类别出现的次数）从大到小排序
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

#将图片矩阵转换为向量
def img2vector(filename):
    """
        filename:文件名字
        将这个文件的所有数据按照顺序写成一个一维向量并返回
    """
    returnVect = []
    fr = open(filename)			#打开文件
    for i in range(32):
        lineStr = fr.readline() #读取每一行
        for j in range(32):
            returnVect.append(int(lineStr[j]))
    return returnVect

#从文件名中解析分类数字
def classnumCut(fileName):
    """
    filename:文件名
    返回这个文件数据代表的实际数字
    """
    fileStr = fileName.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    return classNumStr

#构建训练集数据向量，及对应分类标签向量
def trainingDataSet():
    """
        从trainingDigits文件夹下面读取所有数据文件，返回：
        trainingMat：所有训练数据，每一行代表一个数据文件中的内容
        hwLabels：每一项表示traningMat中对应项的数据到底代表数字几
    """
    hwLabels = []
    #获取目录traningDigits内容(即数据集文件名)，并储存在一个list中
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)					   #当前目录文件数
    #初始化m维向量的训练集，每个向量1024维
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        hwLabels.append(classnumCut(fileNameStr))  #从文件名中解析分类数字，作为分类标签
        #将图片矩阵转换为向量并储存在新的矩阵中
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    return hwLabels,trainingMat

#测试函数
def handwritingTest(k):
    """
        主函数，直接执行算法测试
    """
    hwLabels,trainingMat = trainingDataSet()    #构建训练集
    #从testDigits里面拿到测试集
    testFileList = listdir('testDigits')
    errorCount = 0.0                            #错误数
    mTest = len(testFileList)                   #测试集总样本数
    t1 = time.time()							#获取程序运行到此处的时间（开始测试）
    for i in range(mTest):
        fileNameStr = testFileList[i]			#得到当前文件名
        classNumStr = classnumCut(fileNameStr)  #从文件名中解析分类数字
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)  #将图片矩阵转换为向量
        #调用knn算法进行测试
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, k)
        # print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        theErrorNumber = [0] * 10
        L = []
        if (classifierResult != classNumStr): 
            theErrorNumber[classNumStr] += 1
            with open('k.csv','a',newline='') as csvfile:
                spamwriter = csv.writer(csvfile,dialect='excel')
                spamwriter.writerow(str(classNumStr) + '->' + str(classifierResult))
            errorCount += 1.0      #若预测结果不一致，则错误数+1
    print( "\nthe total number of tests is: %d" % mTest)               #输出测试总样本数
    print ("the total number of errors is: %d" % errorCount )          #输出测试错误样本数
    print ("the total error rate is: %f" % (errorCount/float(mTest)))  #输出错误率
    t2 = time.time()							#获取程序运行到此处的时间（结束测试）
    print ("Cost time: %.2fmin, %.4fs."%((t2-t1)//60,(t2-t1)%60) )     #测试耗时
    return mTest , errorCount , errorCount/float(mTest) , (t2-t1)%60 , theErrorNumber , L

def file2matrix(folder):
    with open(folder,'r') as f:
        data = f.readlines()
        datingDataMat = list(i.replace('\n','').split('\t')[:3] for i in data)
        datingLabels = list(i.replace('\n','').split('\t')[3] for i in data)
        return np.array(datingDataMat) , datingLabels

if __name__ == "__main__":
    handwritingTest(3)
