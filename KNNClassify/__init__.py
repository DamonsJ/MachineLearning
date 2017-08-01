# -*- coding:utf8 -*-

from KNNClassify.KNNClassifyImpl import *
import sys
import os

#获取脚本的当前文件路径
def cur_file_dir():
    #获取脚本路径
    path = sys.path[0]
    #判断为脚本文件还是py2exe编译后的文件，如果是脚本文件，则返回的是脚本的目录，如果是py2exe编译后的文件，则返回的是编译后的文件路径
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)

def ReadTrainDataImpl(datapath):
    rows = 32  
    cols = 32  
    imgVector = []   
    with open(datapath) as fp: 
        for row in range(rows):  
            lineStr = fp.readline()  
            for col in range(cols):  
                imgVector.insert((row * 32 + col), int(lineStr[col]))
                
    return imgVector  
    
def ReadTraningTestData():
    print("reading train data ......")
    cur_path_d = cur_file_dir()
    cur_path = os.path.join(cur_path_d,"trainingDigits")
    cur_path_test = os.path.join(cur_path_d,"testDigits")
    file_names = os.listdir(cur_path)
    number_files = len(file_names)
    train_data = []
    train_labels=[]

    for idx in range(number_files):
        filename = file_names[idx]
        #get train data label
        label = int(filename.split('_')[0])
        train_labels.insert(idx,label)
        #get train data 
        fileNamePath = os.path.join(cur_path,filename)
        imgVector = ReadTrainDataImpl(fileNamePath)
        train_data.insert(idx,imgVector)
    
    print("train data size : ",len(train_data))
    print("train data label size : ",len(train_labels))
    
    print("reading test data ......")
    file_names_test = os.listdir(cur_path_test)
    number_files_test = len(file_names_test)
    test_data = []
    test_labels=[]
    for idx in range(number_files_test):
        filename = file_names_test[idx]
        #get train data label
        label = int(filename.split('_')[0])
        test_labels.insert(idx,label)
        #get train data 
        fileNamePath = os.path.join(cur_path_test,filename)
        imgVector = ReadTrainDataImpl(fileNamePath)
        test_data.insert(idx,imgVector)
        
    return train_data,train_labels,test_data,test_labels

def KNNClassify(test_data,train_data,train_labels,k):
    len_data = len(train_data[0])
    
    diff_result = [[(test_data[i] - subdata[i]) for i in range(len_data)]  for subdata in train_data]
    squar_result = [[t**2 for t in subdata] for subdata in diff_result]
    sum_result =  [sum(subdata) for subdata in squar_result]
    
#     dict_res = dict(zip(range(len_data),sum_result))
    dict_res = dict(enumerate(sum_result))
    dict_res = dict(sorted(dict_res.items(),key = lambda item:item[1]))
    list_res = [ train_labels[item] for item in dict_res.keys()]
    list_res = list_res[0 : k]
    
    dict_res = dict(Counter(list_res))
    maxCount = 0  
    for key, value in dict_res.items():  
        if value > maxCount:  
            maxCount = value  
            maxIndex = key  
  
    return maxIndex   

if __name__ == "__main__":
    train_data,train_labels,test_data,test_labels = ReadTraningTestData()
    
    numTestSamples = len(test_data)  
    matchCount = 0  
    for i in range(numTestSamples):  
        predict = KNNClassify(test_data[i], train_data, train_labels, 3)  
        if predict == test_labels[i]:
            matchCount += 1  
    accuracy = float(matchCount) / numTestSamples 
    
    print('The classify accuracy is: %.2f%%' % (accuracy * 100))