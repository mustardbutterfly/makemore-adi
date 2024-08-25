import matplotlib.pyplot as plt
import torch

from MakeMoreV3.Linear import LinearLayer
from MakeMoreV3.NamesGram import NamesGram
from MakeMoreV3.Util import splitIntoTest, getLoss

contextCount = 3
maskCount = 10
layerParamsCount = 300
biggerStep = .1
smallerStep = .01
stepCutoff = 100000
trainCount = 200000
sampleCount = 32

namesGram = NamesGram(contextCount)
[[trainX, trainY], [testX, testY]] = splitIntoTest(namesGram.getXs(), namesGram.getYs())

mask = torch.randn([27, maskCount], requires_grad=True).float()

linearLayers = [
    LinearLayer(maskCount * contextCount, layerParamsCount), torch.nn.Tanh(),
    LinearLayer(layerParamsCount, layerParamsCount), torch.nn.Tanh(),
    LinearLayer(layerParamsCount, layerParamsCount), torch.nn.Tanh(),
    LinearLayer(layerParamsCount, layerParamsCount), torch.nn.Tanh(),
    LinearLayer(layerParamsCount, layerParamsCount), torch.nn.Tanh(),
    LinearLayer(layerParamsCount, 27)
]

parameters = [mask]

for layer in linearLayers:
    if not isinstance(layer, torch.nn.Tanh):
        parameters += [layer.parameters()]


for i in range(trainCount):
    batchIndex = torch.randint(0, trainX.shape[0], (sampleCount,))
    loss = getLoss(mask, trainX[batchIndex], trainY[batchIndex], linearLayers, maskCount * contextCount)
    for p in parameters:
        p.grad = None
    loss.backward()

    step = smallerStep if i < stepCutoff else biggerStep

    for p in parameters:
        p.data = p.grad * step

loss = getLoss(mask, testX, testY, linearLayers, maskCount * contextCount)
print(loss.item())
