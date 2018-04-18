#此脚本运行需要有knn.py的脚本依赖
import knn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

def def_try():
    group , labels = knn.createDataSet()
    print(group , labels)
    result = knn.classify([0,0],group , labels , 3)
    print(result)

def file2matrix(folder):
    with open(folder,'r') as f:
        data = f.readlines()
        datingDataMat = list(i.replace('\n','').split('\t')[:3] for i in data)
        datingLabels = list(i.replace('\n','').split('\t')[3] for i in data)
        return np.array(datingDataMat) , datingLabels
    
def darw_picture(datingDataMat , datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
    plt.show()
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0 * np.array(datingLabels),15.0 * np.array(datingLabels))
    # plt.show()

def dis_digits():
    hwLabels,trainingMat = knn.trainingDataSet()
    with open('实验2.csv','w',newline='') as csvfile:
        spamwriter = csv.writer(csvfile,dialect='excel')
        L = ['mTest' , 'the number of error' , 'error rate','time','0','1','2','3','4','5','6','7','8','9']
        spamwriter.writerow(L)
        for k in range(1,30):
            getList = knn.handwritingTest(k)
            result = []
            for i in range(len(getList)):
                if i<= 3:
                    result.append(getList[i])
                else:
                    result += getList[i]
            spamwriter.writerow(result)
    
if __name__ == '__main__':
    folder = './lab6_knsp_code_and_data/code_and_data/datingTestSet.txt'
    datingDataMat , datingLabels = knn.file2matrix(folder)
    print(datingDataMat)
    print(datingLabels[0:20])
    darw_picture(datingDataMat , datingLabels)
    # dis_digits()
    
