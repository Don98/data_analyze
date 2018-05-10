#coding=utf-8
'''
项目名称：
作者
日期
'''

#导入必要库
from numpy import *

#创建实验样本
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec #postingList为词条切分后的文档集合，classVec为类别标签的集合

#创建词汇表：利用集合结构内元素的唯一性，创建一个包含所有词汇的词表。
def createVocabList(dataSet):
    vocabSet=set([])
    for docment in dataSet:
        vocabSet=vocabSet| set(docment) #union of tow sets
    return list(vocabSet) #convet if to list 

#vocablist为词汇表，inputSet为输入的邮件
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)    #他的大小与词向量一样
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1 #查找单词的索引
        else: print ("the word: %s is not in my vocabulary" %word) 
    return returnVec

#这里的trainMat是训练样本的词向量，其是一个矩阵，他的每一行为一个邮件的词向量
#trainGategory为与trainMat对应的类别，值为0，1表示正常，垃圾
def train(trainMat,trainCategory):
    numTrain=len(trainMat)
    numWords=len(trainMat[0])  #is vocabulary length
    pAbusive=sum(trainCategory)/float(numTrain)
    p0Num=ones(numWords);p1Num=ones(numWords)
    p0Denom=2.0;p1Denom=2.0
    for i in range(numTrain):
        if trainCategory[i] == 1:
            p1Num += trainMat[i] #统计类1中每个单词的个数
            p1Denom += sum(trainMat[i]) #类1的单词总数
        else:
            p0Num += trainMat[i]#统计类0中每个单词的个数
            p0Denom +=sum(trainMat[i])#类0的单词总数
    p1Vec=log(p1Num/p1Denom) #类1中每个单词的概率
    p0Vec=log(p0Num/p0Denom)#类0中每个单词的概率
    return p0Vec,p1Vec,pAbusive

# 分类函数
def classfy(vec2classfy,p0Vec,p1Vec,pClass1):
    p1=sum(vec2classfy*p1Vec)+log(pClass1)
    p0=sum(vec2classfy*p0Vec)+log(1-pClass1)
    if p1 > p0:
        return 1;
    else:
        return 0

#  对邮件的文本划分成词汇，长度小于2的默认为不是词汇，过滤掉即可。返回一串小写的拆分后的邮件信息。
def textParse(bigString):
    import re
    listOfTokens=re.split(r'\W+',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

		
#vocablist为词汇表，inputSet为输入的邮件
def bagOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)    #他的大小与词向量一样
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1 #查找单词的索引
        else: print ("the word is not in my vocabulary")
    return returnVec
	
'''
输入为25封正常邮件和25封垃圾邮件。50封邮件中随机选取10封作为测试样本，剩余40封作为训练样本。

　　　训练模型：40封训练样本，训练出先验概率和条件概率；

　　　测试模型：遍历10个测试样本，计算垃圾邮件分类的正确率。
'''
def spamTest():
    fullTest=[];docList=[];classList=[]
    for i in range(1,26): #it only 25 doc in every class
        wordList=textParse(open('email/spam/%d.txt' % i,encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt' % i,encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)   # create vocabulary
    trainSet=list(range(50));testSet=[]
#choose 10 sample to test ,it index of trainMat
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainSet)))#num in 0-49
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMat=[];trainClass=[]
    for docIndex in trainSet:
        trainMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0,p1,pSpam=train(array(trainMat),array(trainClass))
    errCount=0
    for docIndex in testSet:
        wordVec=bagOfWords2Vec(vocabList,docList[docIndex])
        if classfy(array(wordVec),p0,p1,pSpam) != classList[docIndex]:
            errCount +=1
            print (("classfication error"), docList[docIndex])
    print (("the error rate is ") , float(errCount)/len(testSet))

if __name__ == '__main__':
    spamTest()
