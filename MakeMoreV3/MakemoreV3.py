import matplotlib.pyplot as plt
import torch

from MakeMoreV3.BatchNormLayer import BatchNormLayer
from MakeMoreV3.LinearLayer import LinearLayer, Tanh
from MakeMoreV3.Model import Model
from MakeMoreV3.NamesGram import NamesGram
from MakeMoreV3.SerializeLayer import EmbeddingLayer, ConsecutiveFlatteningLayer
from MakeMoreV3.Util import splitIntoTest

contextCount = 8
maskCount = 10
layerParamsCount = 300
trainCount = 20000
sampleCount = 32

namesGram = NamesGram(contextCount)
[[trainX, trainY], [testX, testY]] = splitIntoTest(namesGram.getXs(), namesGram.getYs())

layers = [
    EmbeddingLayer(maskCount),
    ConsecutiveFlatteningLayer(2), LinearLayer(maskCount * 2, layerParamsCount), BatchNormLayer(layerParamsCount), Tanh(), LinearLayer(layerParamsCount, maskCount),
    ConsecutiveFlatteningLayer(2), LinearLayer(maskCount * 2, layerParamsCount), BatchNormLayer(layerParamsCount), Tanh(), LinearLayer(layerParamsCount, maskCount),
    ConsecutiveFlatteningLayer(2), LinearLayer(maskCount * 2, layerParamsCount), BatchNormLayer(layerParamsCount), Tanh(),
    LinearLayer(layerParamsCount, 27)
]

model = Model(layers)

for i in range(trainCount):
    batchIndex = torch.randint(0, trainX.shape[0], (sampleCount,))
    loss = model.train(trainX[batchIndex], trainY[batchIndex])
    if i % 1000 == 0:  # print every once in a while
        print(f'{i:7d} {loss.item():.4f}')

legends = []
for i, layer in enumerate(layers):
    if isinstance(layer, Tanh):
        hy, hx = torch.histogram(layer.out(), density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'tan layer activations {i} ({layer.__class__.__name__}')
plt.legend(legends)
plt.show()

legends = []
for i, layer in enumerate(layers):
    if isinstance(layer, Tanh):
        hy, hx = torch.histogram(layer.out().grad, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer grad {i} ({layer.__class__.__name__}')
plt.legend(legends)
plt.show()

legends = []
for i, p in enumerate(model.parameters()):
    hy, hx = torch.histogram(p.grad, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'parameters grad {i}')
plt.legend(legends)
plt.show()

legends = []
ud = model.getLoggingVars()['ud']
for i, p in enumerate(model.parameters()):
    if p.ndim == 2:
        plt.plot([ud[j][i] for j in range(len(ud))])
        legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k')
plt.legend(legends)
plt.show()

print('test loss %d' % model.getTestLoss(testX, testY).item())

