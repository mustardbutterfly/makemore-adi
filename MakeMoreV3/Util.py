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

def getLoss(mask, trainX, trainY, layers, maskedDimension):
    maskedTrainX = mask[trainX]
    input = maskedTrainX.view([-1, maskedDimension])

    for layer in layers:
        input = layer(input)

    return F.cross_entropy(input, trainY)