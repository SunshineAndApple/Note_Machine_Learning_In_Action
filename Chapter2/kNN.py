# Page:19

import numpy as np
import operator


# create demo dataset
# page19
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# knn
# inX：目标点
# dataSet: 样本
# labels: 特征
# k：k值
# page19
def classify0(inX, dataSet, labels, k):
    # 取样本空间第一维的数值，这里是行
    dataSetSize = dataSet.shape[0]
    # tile将目标扩充为和样本空间一样的行，便于后续的计算，要维度相等。
    # 扩充后与样本空间求差
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 平方
    sqDiffMat = diffMat ** 2
    # 取上述结果第二维的和
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方求得距离
    distances = sqDistances ** 0.5
    # 获取数组从小到大的索引
    sortedDisIndicies = distances.argsort()

    # 定义字典开始按照k查找
    # print(sortedDisIndicies)
    # 定义一个空字典，字典的key：特征 value：权重
    classCount = {}
    '''
    print('loop: \n')
    '''
    for i in range(k):
        '''
        print('------------')
        print(labels)
        print('i:{}'.format(i))
        '''
        # 取出对应的特征值，索引项是距离，距离的取值按照循环量
        voteIlabel = labels[sortedDisIndicies[i]]
        '''
        print('sortedDisIndicies[i]: {}'.format(sortedDisIndicies[i]))
        print('labels[sortedDisIndicies[i]]: {}'.format(voteIlabel))
        print('classCount.get(voteIlabel, 0): {}'.format(classCount.get(voteIlabel, 0)))
        '''
        # 判断离哪个特征值更近：从字典中查找，如果有则权重+1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        '''
        print('classCount[voteIlabel]: {}'.format(classCount[voteIlabel]))
        print('this loop classCount:{}'.format(classCount))

    print(classCount)
    '''
    # 按照降序排列结果，将权重最大的排到最前面
    sortdClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    ''' 
    print('sortdClassCount: {}, sortdClassCount[0][0]: {}'.format(sortdClassCount, sortdClassCount[0][0]))
    '''
    # 取权重最大值作为结果
    return sortdClassCount[0][0]



if __name__ == '__main__':
    import kNN
    #
    # group, labels = kNN.createDataSet()
    # print(classify0((0, 2), group, labels, 3))

    a = np.array([[1, 2, 3], [4, 5, 4]])
    a=a**2
    print(a)
