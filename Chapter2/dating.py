# page 21

import numpy as np


# read file
def file2matrix(filename):
    dataReadlist = []
    datalebels = []
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
            datalebels.append(int(lineSplit[-1]))
    # 将list转换为矩阵
    dataMatrix = np.matrix(dataReadlist)
    # print('maxtix: {}, type: {}, shape: {}'.format(dataMatrix[0:2], dataMatrix.dtype, dataMatrix.shape))

if __name__ == '__main__':
    file2matrix('datingTestSet2.txt')