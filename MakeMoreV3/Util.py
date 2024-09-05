import torch
import torch.nn.functional as F

def splitIntoTest(dataX, dataY, train=.8):
    indexes = torch.randperm(len(dataX))
    trainLen = int(len(indexes) * train)
    testLen = len(indexes) - trainLen
    splitIndexes = indexes.split([trainLen, testLen])

    trainX = dataX[splitIndexes[0]]
    trainY = dataY[splitIndexes[0]]
    testX = dataX[splitIndexes[1]]
    testY = dataY[splitIndexes[1]]
    return [[trainX, trainY], [testX, testY]]
