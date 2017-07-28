from DecisionTree.DecisionTreeImpl import CreateDecisionTreeID3C45
from DecisionTree.CartDecisionTree import CreateCartClassificationTree
  
if __name__ == "__main__":
#     CreateDecisionTreeID3C45(True)
    CreateCartClassificationTree()


# from itertools import *
#  
# def featuresplit(features):   
#     count = len(features)    
#     featureind = list(range(count))
#     featureind.pop(0) #get value 1~(count-1)  
#     combiList = []   
#     for i in featureind:      
#         com = combinations(features, len(features[0:i])) 
#         combiList.extend(com)   
#     combiLen = len(combiList)
#     featuresplitGroup = list(zip(combiList[0:combiLen//2],combiList[combiLen-1:combiLen//2-1:-1])) 
#      
#     return featuresplitGroup
#  
# if __name__ == "__main__":   
#     test= list(range(3))    
#     splitGroup = featuresplit(test)   
#     print ("splitGroup", len(splitGroup), splitGroup)   
#    
#     test= list(range(4))    
#     splitGroup = featuresplit(test)  
#     print ("splitGroup2", len(splitGroup), splitGroup)
#    
#     test= list(range(5))  
#     splitGroup = featuresplit(test)    
#     print ("splitGroup3", len(splitGroup), splitGroup)  
 
#     test= ["young","middle","old","come"]   
#     splitGroup = featuresplit(test)  
#     print ("splitGroup4", len(splitGroup), splitGroup)
#     for group in splitGroup:
#         current_feature_list = list(group[0])
#         print(current_feature_list)
                
'''
splitGroup 3 [((0,), (1, 2)), ((1,), (0, 2)), ((2,), (0, 1))]
splitGroup2 7 [((0,), (1, 2, 3)), ((1,), (0, 2, 3)), ((2,), (0, 1, 3)), ((3,), (0, 1, 2)), ((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
splitGroup3 15 [((0,), (1, 2, 3, 4)), ((1,), (0, 2, 3, 4)), ((2,), (0, 1, 3, 4)), ((3,), (0, 1, 2, 4)), ((4,), (0, 1, 2, 3)), ((0, 1), (2, 3, 4)), ((0, 2), (1, 3, 4)), ((0, 3), (1, 2, 4)), ((0, 4), (1, 2, 3)), ((1, 2), (0, 3, 4)), ((1, 3), (0, 2, 4)), ((1, 4), (0, 2, 3)), ((2, 3), (0, 1, 4)), ((2, 4), (0, 1, 3)), ((3, 4), (0, 1, 2))]
splitGroup4 3 [(('young',), ('middle', 'old')), (('middle',), ('young', 'old')), (('old',), ('young', 'middle'))]
'''
