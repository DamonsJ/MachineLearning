# -*- coding: utf-8 -*-
import math
from collections import Counter
from DecisionTree.DrawDecisionTree import createPlot
 
def CalculateEntropy(dataset):   
    '''
        计算信息熵
    '''  
    #字典统计 label的数据个数
    feature_vector = [f[-1] for f in dataset]
    labelDict = dict(Counter(feature_vector)) 
    #数据集的长度
    numData = len(feature_vector)
    #计算熵
#     shannoEntropy = 0.0
#     for lvalue in labelDict.values():
#         v = float(lvalue)/numData
#         shannoEntropy -= (v*math.log(v,2))   
    probs = [float(lvalue)/numData for lvalue in labelDict.values()]
    shannoEntropy = sum([-prob*math.log(prob,2) for prob in probs])  
    return shannoEntropy   

def CalculateConditionalEntropy(dataset,dim):
    '''
        输入数据集 以及 要计算条件熵的数据维度
        返回条件熵
    '''
    #抽取第dim维的数据
    dimth_feature = [f[dim] for f in dataset]
    length_feature = len(dimth_feature)
    #定义条件熵
    conditionalEntropy = 0.0
    #统计特征个数并转化为字典
    featureDict = dict(Counter(dimth_feature))
    #抽取数据集计算条件熵
    for vk,vv in featureDict.items():
        dimth_dataset  = [[f[dim],f[-1]] for f in dataset if f[dim] == vk]
        prob = float(vv)/length_feature
        conditionalEntropy += prob * CalculateEntropy(dimth_dataset)
     
    return conditionalEntropy        

def CalculateInformationGain(dataset,dim):
    '''
    Args:
        dataset :需要计算信息增益的数据集
        dim:     需要计算的维度
        
    return:
        information gain
    '''
    #这里为了不传入参数，重新计算了熵
    baseEntropy = CalculateEntropy(dataset)
    infoGain = baseEntropy - CalculateConditionalEntropy(dataset, dim)

    return infoGain

def CalculateInformationGainRatio(dataset,dim):
    '''
        计算信息增益比率
    Args:
        dataset :需要计算信息增益比率的数据集
        dim:     需要计算的维度
        
    return:
        information gain
    '''
    infoGain    = CalculateInformationGain(dataset,dim)

    dimth_feature = [f[dim] for f in dataset]
    length_feature = len(dimth_feature)
    #定义条件熵
    spliteEntropy = 0.0
    #统计特征个数并转化为字典
    featureDict = dict(Counter(dimth_feature))
    #抽取数据集计算条件熵
    for vv in featureDict.values():
        prob = float(vv)/length_feature
        spliteEntropy -= (prob*math.log(prob,2))    
    
    return infoGain/spliteEntropy

def GetBestFeature(dataset,method = "ID3"):
    '''
        获取当前数据集中最有的特征并返回维度
        使用ID3算法
    Args:
        dataset :数据集
    return:
        dimension:最优维度
    '''
    #数据的最后一列不是特征
    features_number = len(dataset[0]) -1
    bestInfoGain = 0.0
    bestFeatureDim = 0
    
    for index in range(features_number):
        if method == "C45":
            infoGain = CalculateInformationGainRatio(dataset, index)
        else:
            infoGain = CalculateInformationGain(dataset, index)
            
        if bestInfoGain < infoGain:
            bestInfoGain = infoGain
            bestFeatureDim = index
    
    return bestFeatureDim

def GetMostApperanceFeature(featureVector):
    featureDict = dict(Counter(featureVector)) 
    # 排序
    sortedfeatureDict = sorted(featureDict.items(), key= lambda item : item[1],reverse = True)
    return sortedfeatureDict[0][0]

def CreateDecisionTree(dataset ,labels,method = "ID3"):
    '''
        创建决策树
    Args:
        dataset :数据集
        labels ： 标签集
        method : ID3 or C45
    return
        decision tree
    '''
    feature_vector = [f[-1] for f in dataset]
    #结束条件：所有类的标签相同
    if feature_vector.count(feature_vector[0]) == len(feature_vector):
        return feature_vector[0]
    #结束条件 ：所有特征用完 找到出现次数最多的标签
    if len(dataset[0]) == 1:
        return GetMostApperanceFeature(feature_vector)
    
    #最优特征
    bestFeatureDim   = GetBestFeature(dataset,method)
    bestFeatureLabel = labels[bestFeatureDim]
    # 使用字典类型储存树的信息
    decisionTree = {bestFeatureLabel:{}}
    #当前标签已使用        
    del(labels[bestFeatureDim])
    
    featValues = [example[bestFeatureDim] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 复制所有类标签，保证每次递归调用时不改变原始列表的内容
        subLabels = labels[:]
        datasubset = [[f[fv] for fv in range(0,len(f)) if fv != bestFeatureDim and value == f[bestFeatureDim]] for f in dataset]
        not_empty = lambda tlist: len(tlist)
        datasubset = list(filter(not_empty,datasubset))
#         if not_empty(datasubset):
        decisionTree[bestFeatureLabel][value] = CreateDecisionTree(datasubset,subLabels)
    return decisionTree

def ClassifyDecisionTree(decisionTree,featureLabels,testVector):
    '''
        利用决策树进行分类
    :param: inputTree:构造好的决策树模型
    :param: featLabels:所有的类标签
    :param: testVec:测试数据
    :return: 分类决策结果
    
    '''
    firstFeature = list(decisionTree.keys())[0]
    secondDict = decisionTree[firstFeature]
    featIndex = featureLabels.index(firstFeature)
    key = testVector[featIndex]
    NextTree = secondDict[key]
    if isinstance(NextTree, dict): 
        classLabel = ClassifyDecisionTree(NextTree, featureLabels, testVector)
    else: 
        classLabel = NextTree
        
    return classLabel

def createDataSet():
    '''
        导入数据
    '''
#     dataSet = [['youth', 'no', 'no', 1, 'refuse'],
#                ['youth', 'no', 'no', '2', 'refuse'],
#                ['youth', 'yes', 'no', '2', 'agree'],
#                ['youth', 'yes', 'yes', 1, 'agree'],
#                ['youth', 'no', 'no', 1, 'refuse'],
#                ['mid', 'no', 'no', 1, 'refuse'],
#                ['mid', 'no', 'no', '2', 'refuse'],
#                ['mid', 'yes', 'yes', '2', 'agree'],
#                ['mid', 'no', 'yes', '3', 'agree'],
#                ['mid', 'no', 'yes', '3', 'agree'],
#                ['elder', 'no', 'yes', '3', 'agree'],
#                ['elder', 'no', 'yes', '2', 'agree'],
#                ['elder', 'yes', 'no', '2', 'agree'],
#                ['elder', 'yes', 'no', '3', 'agree'],
#                ['elder', 'no', 'no', 1, 'refuse'],
#                ]
#     labels = ['age', 'working?', 'house?', 'credit_situation']
    
    dataSet = [
                  ['s','sd','no','no'],
                  ['s','ld','yes','yes'],
                  ['l','md','yes','yes'],
                  ['m','md','yes','yes'],
                  ['l','md','yes','yes'],
                  ['m','ld','no','yes'],
                  ['m','sd','no','no'],
                  ['l','md','no','yes'],
                  ['m','sd','no','yes'],
                  ['s','sd','yes','no']
                ]
    labels = ['L', 'F', 'H']   
     
    return dataSet, labels

def CreateDecisionTreeID3C45(isID3Alogrithm):
    dataSet, labels = createDataSet()

    if(isID3Alogrithm):
        decisionTree = CreateDecisionTree(dataSet, labels)
    else:
        decisionTree = CreateDecisionTree(dataSet, labels,"C45")
        
    print("decisionTree : ",decisionTree)
#     rlabels = labels[:]
#     testVector = ['s','md','yes']
#     result = ClassifyDecisionTree(decisionTree,rlabels,testVector)
#     print("result is : ",result)
    createPlot(decisionTree)