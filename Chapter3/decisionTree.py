import numpy as np
from math import log
import operator

def createDataAarray():
    dataArray = [[1, 1, 'yes'],
                 [1, 1, 'yes'],
                 [1, 0, 'no'],
                 [0, 1, 'no'],
                 [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataArray, labels

# 计算熵
def calcShannonEnt(dataArray):

    labelsCount = {}
    for iter in dataArray:
        currentLabels = iter[-1]
        # list不可hash，不能放入dict，要转换为tuple
        currentTuple = tuple(currentLabels)
        if currentTuple not in labelsCount.keys():
            labelsCount[currentTuple] = 0
        labelsCount[currentTuple] += 1
    shannoEnt = 0.0
    for key in labelsCount:
        prob = labelsCount[key]/len(dataArray)
        shannoEnt -= prob * log(prob, 2)
    return shannoEnt

# 划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 1.将原始的数据按照特征分成不同的组
# 2.计算每一个组下的熵
# 3.取出最优熵的特征
# page:38
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    # print(numFeatures)
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        # print(i)
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        # print('featList:{}'.format(featList))
        uniqueVals = set(featList)       #get a set of unique values
        # print('uniqueVals:{}'.format(uniqueVals))
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 递归创建决策树
# page：41
def createDecisionTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]

    # 当类别已经完全一样，就不需要再划分下去
    if classList.count(classList[0]) == len(classList):
        print('== classList: {}'.format(classList)  )
        return classList[0]

    # 所有类别都已划分完成
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        subLabels = labels[:]
        # 嵌套的字典取值：
        # s = {'a': {0: 'no', 1: {'flippers': {0: 'no', 1: 'maybe'}}}, 'b': {}}  # 构造字典
        # >> > s['a'][0]  # 取值
        # 'no'
        myTree[bestFeatLabel][value] = createDecisionTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree


if __name__ == '__main__':
    dataArray, dataLabels = createDataAarray()
    # print(calcShannonEnt(dataArray))
    # dataSet = splitDataSet(dataArray, 0, 1)
    # print(dataSet)
    # print(chooseBestFeatureToSplit(dataArray))
    print(createDecisionTree(dataArray, dataLabels))

