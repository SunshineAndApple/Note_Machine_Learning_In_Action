import numpy as np

# page：58
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                  ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                  ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                  ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    #1 is abusive, 0 not
    return postingList, classVec

# 利用集合创建不重复的词汇表
def createVocabList(dataSet):
    vocabSet = set([]) #利用结合不能重复特性来过滤重复词汇
    for doucument in dataSet:
        vocabSet = vocabSet | set(doucument)
    return list(vocabSet)

# 根据输入的词汇得到文档向量
# page:59
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1 #以0表示未出现，1表示出现
        else:
            print('The word {} is not in my Vocabulary.'.format(word))
    return returnVec


def trainNB(trainList, trainCategory):
    numTrainDoc = len(trainList)
    numWords = len(trainList[0])
    pAbusive = sum(trainCategory) / numTrainDoc

    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDoc):
        # 循环遍历输入list中所有单词向量，如果当前值在
        if trainCategory[i] == 1:
            p1Num += trainList[i]
            p1Denom += sum(trainList[i])
        else:
            p0Num += trainList[i]
            p1Denom += sum(trainList[i])

    p1Vec = np.log(p1Num / p1Denom)#下溢出，很多很小的数乘积会近似0，取对数
    p0Vec = np.log(p0Num / p0Denom)
    return p0Vec, p1Vec, pAbusive



if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    print('---\nInit list: {}'.format(postingList))
    myVocabList = createVocabList(postingList)
    print('---\nMy vocabulary list: {}'.format(myVocabList))
    testVec = setOfWords2Vec(myVocabList, postingList[0])
    print('---\nsetOfWords2Vec of 0: {}'.format(testVec))

    trainMat = []
    for p in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, p))
    print('---\ntrainMat: {}  '.format(trainMat))

    p0V, p1V, pAb = trainNB(trainMat, classVec)
    print('---\nTest naive bayes: {}'.format(pAb))
    print('---\np0V: {}'.format(p0V))
    print('---\np1V: {}'.format(p1V))

