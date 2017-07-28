# -*- coding: utf-8 -*-
import math
from collections import Counter
from DecisionTree.DrawDecisionTree import createPlot
from itertools import *
from DecisionTree.DecisionTreeImpl import GetBestFeature

def FeatureSplit(features):
    '''
        对多于3个特征值的特征，进行分割，分割为两个特征
    '''  
    count = len(features)    
    featureind = list(range(count))
    featureind.pop(0) #get value 1~(count-1)  
    combiList = []   
    for i in featureind:      
        com = combinations(features, len(features[0:i])) 
        combiList.extend(com)   
    combiLen = len(combiList)
    featuresplitGroup = list(zip(combiList[0:combiLen//2],combiList[combiLen-1:combiLen//2-1:-1])) 
    
    return featuresplitGroup

def CalculateGini(dataset):
    '''
        计算整个数据集的基尼指数
    Args:
        dataset : 数据集
    return:
        gini指数
    '''
    #字典统计 label的数据个数
    feature_vector = [f[-1] for f in dataset]
    labelDict = dict(Counter(feature_vector)) 
    #数据集的长度
    numData = len(feature_vector)
    #计算熵
    Gini = 1.0
    probs = [float(lvalue)/numData for lvalue in labelDict.values()]  
    Gini = Gini - sum([prob*prob for prob in probs])  
    
    return Gini   

def CalConditionalGini(dataset,dim,featureValue):
    '''
        输入数据集 以及 要计算基尼指数的数据维度
        返回基尼指数
    '''
    #抽取第dim维的数据
    dimth_feature = [f[dim] for f in dataset]
    length_feature = len(dimth_feature)

    dimth_dataset1  = [[f[dim],f[-1]] for f in dataset if f[dim] in featureValue]
    dimth_dataset2  = [[f[dim],f[-1]] for f in dataset if f[dim] not in featureValue]
    
    conditionalGini = (len(dimth_dataset1)/length_feature)*CalculateGini(dimth_dataset1) + (len(dimth_dataset2)/length_feature)*CalculateGini(dimth_dataset2)
    return conditionalGini

def ChooseBestDimAndFeature(dataset):
    '''
        选取基尼指数最小点维度和分裂点
    '''
    #数据的最后一列不是特征
    features_number = len(dataset[0]) -1
    bestGiniInfo = 10000000.0
    bestFeatureDim = 0
    bestFeature = []
    
    for dim in range(features_number):
        #当前特征维度下的所有特征值
        dimth_feature = [f[dim] for f in dataset]
        set_feature = list(set(dimth_feature))
        #组合特征值，二叉分割
        featuresplitGroup = FeatureSplit(set_feature)
        for group in featuresplitGroup:
            current_feature_list = list(group[0])
            currentGini = CalConditionalGini(dataset, dim, current_feature_list)
            if bestGiniInfo > currentGini:
                bestGiniInfo = currentGini
                bestFeatureDim = dim
                bestFeature = current_feature_list
                    
    return bestFeatureDim,bestFeature

def GetMostApperanceCartFeature(featureVector):
    featureDict = dict(Counter(featureVector)) 
    # 排序
    sortedfeatureDict = sorted(featureDict.items(), key= lambda item : item[1],reverse = True)
    return sortedfeatureDict[0][0]

def CreateCartClassificationTreeImpl(dataset,labels):
    '''
        创建cart分类树
    Args:
        dataset:数据集
        labels：标签
    return:
        cart classification tree
    '''
    feature_list = [f[-1] for f in dataset]
    #停止条件：所有标签都归为一类
    if len(set(feature_list)) == 1:
        return feature_list[0]
    #停止条件：所有特征使用完毕
    if len(dataset[0]) == 1:
        return GetMostApperanceCartFeature(feature_list)
    #选择最优化分裂点
    bestDim,bestFeature = ChooseBestDimAndFeature(dataset)
    bestFeatLabel = labels[bestDim]
    
    current_feature_list = list(set([f[bestDim] for f in dataset]))
    len_list = len(current_feature_list)
    #如果当前最有维度内有多于2个特征则将特征分为最优特征和非最优特征两个部分
    if len_list > 2 :
        subLabels1 =  labels[:]
        subLabels2 =  labels[:]
        left_value = str(bestFeature)
        right_value = str("not ") + left_value
        if len(bestFeature) == 1:
            datasubset1 = [[f[fv] for fv in range(0,len(f)) if fv != bestDim and f[bestDim] in bestFeature] for f in dataset]
            del(subLabels1[bestDim])
        else:
            datasubset1 = [[f[fv] for fv in range(0,len(f)) if f[bestDim] in bestFeature] for f in dataset]
            
        if (len_list - len(bestFeature)) == 1:
            datasubset2 = [[f[fv] for fv in range(0,len(f)) if fv != bestDim and f[bestDim] not in bestFeature] for f in dataset] 
            del(subLabels2[bestDim])
        else:
            datasubset2 = [[f[fv] for fv in range(0,len(f)) if f[bestDim] not in bestFeature] for f in dataset]   
    #否则直接将特征分为两部分，并将特诊标签删除
    else:
        del(labels[bestDim])
        
        subLabels1 =  labels[:]
        subLabels2 =  labels[:]
        
        left_value = bestFeature[0]
        if current_feature_list[0] == left_value:
            right_value = current_feature_list[1]
        else:
            right_value = current_feature_list[0]
        
        datasubset1 = [[f[fv] for fv in range(0,len(f)) if fv != bestDim and f[bestDim] in bestFeature] for f in dataset]  
        datasubset2 = [[f[fv] for fv in range(0,len(f)) if fv != bestDim and f[bestDim] not in bestFeature] for f in dataset] 
    #去掉数据集中为空的部分
    not_empty = lambda tlist: len(tlist)
    datasubset1 = list(filter(not_empty,datasubset1))
    datasubset2 = list(filter(not_empty,datasubset2)) 
    #使用字典类型储存树的信息
    myCartTree = {bestFeatLabel:{}}
    #左右两部分迭代创建分类树
    myCartTree[bestFeatLabel][left_value]  = CreateCartClassificationTreeImpl(datasubset1, subLabels1)
    myCartTree[bestFeatLabel][right_value] = CreateCartClassificationTreeImpl(datasubset2,subLabels2)
    
    return myCartTree      

def createCartDataSet():
    '''
        导入数据
    '''
#     dataSet = [['youth', 'no', 'no', '1', 'refuse'],
#                ['youth', 'no', 'no', '2', 'refuse'],
#                ['youth', 'yes', 'no', '2', 'agree'],
#                ['youth', 'yes', 'yes', '1', 'agree'],
#                ['youth', 'no', 'no', '1', 'refuse'],
#                ['mid', 'no', 'no', '1', 'refuse'],
#                ['mid', 'no', 'no', '2', 'refuse'],
#                ['mid', 'yes', 'yes', '2', 'agree'],
#                ['mid', 'no', 'yes', '3', 'agree'],
#                ['mid', 'no', 'yes', '3', 'agree'],
#                ['elder', 'no', 'yes', '3', 'agree'],
#                ['elder', 'no', 'yes', '2', 'agree'],
#                ['elder', 'yes', 'no', '2', 'agree'],
#                ['elder', 'yes', 'no', '3', 'agree'],
#                ['elder', 'no', 'no', '1', 'refuse'],
#                ]
#     labels = ['age', 'working?', 'house?', 'credit_situation']

    dataSet = [
        ['young','myope','no','reduced','no lenses'],
        ['young' ,'myope','no','normal', 'soft' ],
        ['young' ,'myope','yes' , 'reduced','no lenses'],
        ['young' ,'hyper','no','normal'  , 'soft'],
        ['young' ,'hyper','yes' , 'reduced','no lenses'],
        ['young' ,'hyper','yes' , 'normal' ,  'hard'],
        ['pre' ,'myope','no','reduced','no lenses'],
        ['pre' ,'myope','no','normal',  'soft'],
        ['presbyopic','myope' , 'yes'  ,'reduced' ,'no lenses' ],
        ['presbyopic','hyper' , 'no'  ,'normal', 'soft'        ],
        ['presbyopic','hyper' , 'yes' , 'reduced', 'no lenses' ],
        ['presbyopic','hyper' , 'yes' , 'normal',  'no lenses'],
        ['pre'  , 'hyper',  'yes','normal', 'no lenses'],
        ['presbyopic', 'myope', 'no','reduced', 'no lenses'],
        ['pre','hyper' ,'yes','reduced'  ,  'no lenses'],
        ['presbyopic' ,'myope','no' ,'normal', 'no lenses'],
        ['pre' ,'myope','yes','normal' ,'hard'],
        ['pre' ,'hyper','no','reduced' ,'no lenses'],
        ]
    
    labels = ['age', 'prescript', 'asti', 'tearrate']
    return dataSet, labels

def CreateCartClassificationTree():
    dataSet, labels = createCartDataSet()
    myCartDecisionTree = CreateCartClassificationTreeImpl(dataSet, labels)
    print(myCartDecisionTree)
    createPlot(myCartDecisionTree)
 