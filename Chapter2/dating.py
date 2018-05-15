# page 21

import numpy as np
import matplotlib.pyplot as plt

# read file
def file2matrix(filename):
    dataReadlist = []
    dataLebels = []
    with open(filename) as f:
        for line in f:
            # 截取回车符
            line = line.strip()
            # 使用tab字符将数据转化为一行一行的
            lineSplit = line.split('\t')
            # 这里有两个操作，每一行的前三列是数据，所以先将前三列取出
            # 取出后每一行数据是字符串类型的，再使用list(map(type, data)转换为数值类型
            dataReadlist.append(list(map(float, lineSplit[0:3])))
            # 最后一列是特征，使用负索引的方式可以取出
            dataLebels.append(int(lineSplit[-1]))
    # 将list转换为矩阵
    dataMatrix = np.matrix(dataReadlist)
    # print('maxtix: {}, type: {}, shape: {}'.format(dataMatrix[0:2], dataMatrix.dtype, dataMatrix.shape))
    return dataMatrix, dataLebels

# page23
def showData(dataMatrix, dataLebels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 这里需要将矩阵的列转换为list
    ax.scatter(dataMatrix[:, 1].tolist(), dataMatrix[:, 2].tolist())

    plt.show()

# 数值归一化
# 公式：newValue = (oldValue - min)/(max-min)
# page:25
def autoNorm(dataArray):
    minValues = dataArray.min(0)
    maxValues = dataArray.max(0)
    ranges = maxValues - minValues
    # minValues、maxValues、ranges都是[1*3]的数组，因此需要将其扩展为dataArray相同维数
    m = dataArray.shape[0]
    normDataArray = dataArray - np.tile(minValues, (m, 1))
    normDataArray = normDataArray/(np.tile(ranges, (m, 1)))
    # print('normDataArray: {}, minValues: {}, maxValues: {}'.format(normDataArray[0:2], minValues, maxValues))
    return normDataArray

if __name__ == '__main__':
    dataMatrix, dataLebels = file2matrix('datingTestSet2.txt')
    # showData(dataMatrix, dataLebels)
    autoNorm(dataMatrix)