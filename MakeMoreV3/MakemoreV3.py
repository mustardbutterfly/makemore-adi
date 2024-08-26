import matplotlib.pyplot as plt
import torch

from MakeMoreV3.BatchNorm import BatchNorm
from MakeMoreV3.Linear import LinearLayer, Tanh
from MakeMoreV3.NamesGram import NamesGram
from MakeMoreV3.Util import splitIntoTest, getLoss

contextCount = 3
maskCount = 10
layerParamsCount = 300
biggerStep = .1
smallerStep = .01
stepCutoff = 10000
trainCount = 20000
sampleCount = 32

namesGram = NamesGram(contextCount)
[[trainX, trainY], [testX, testY]] = splitIntoTest(namesGram.getXs(), namesGram.getYs())

mask = torch.randn([27, maskCount], requires_grad=True).float()

linearLayers = [
    LinearLayer(maskCount * contextCount, layerParamsCount), BatchNorm(layerParamsCount), Tanh(),
    LinearLayer(layerParamsCount, layerParamsCount), BatchNorm(layerParamsCount), Tanh(),
    LinearLayer(layerParamsCount, layerParamsCount), BatchNorm(layerParamsCount), Tanh(),
    LinearLayer(layerParamsCount, layerParamsCount), BatchNorm(layerParamsCount), Tanh(),
    LinearLayer(layerParamsCount, layerParamsCount), BatchNorm(layerParamsCount), Tanh(),
    LinearLayer(layerParamsCount, 27)
]

parameters = [mask]

for layer in linearLayers:
    parameters += layer.parameters()

for p in parameters:
    p.requires_grad = True

ud = []

for i in range(trainCount):
    batchIndex = torch.randint(0, trainX.shape[0], (sampleCount,))
    loss = getLoss(mask, trainX[batchIndex], trainY[batchIndex], linearLayers, maskCount * contextCount)

    step = smallerStep if i < stepCutoff else biggerStep

    for layer in linearLayers:
        layer.out().retain_grad()
    for p in parameters:
        p.grad = None

    loss.backward()

    for p in parameters:
        p.data += -p.grad * step

    if i % 1000 == 0:  # print every once in a while
        print(f'{i:7d} {loss.item():.4f}')
    with torch.no_grad():
        newud = []
        for p in parameters:
            val = ((step * p.grad).std() / p.data.std()).log10().item()
            newud.append(val)
        ud.append(newud)

legends = []
for i, layer in enumerate(linearLayers):
    if isinstance(layer, Tanh):
        hy, hx = torch.histogram(layer.out(), density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'tan layer activations {i} ({layer.__class__.__name__}')
plt.legend(legends)
plt.show()

legends = []
for i, layer in enumerate(linearLayers):
    if isinstance(layer, Tanh):
        hy, hx = torch.histogram(layer.out().grad, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer grad {i} ({layer.__class__.__name__}')
plt.legend(legends)
plt.show()

legends = []
for i, p in enumerate(parameters):
    hy, hx = torch.histogram(p.grad, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'parameters grad {i}')
plt.legend(legends)
plt.show()

legends = []
for i, p in enumerate(parameters):
    if p.ndim == 2:
        plt.plot([ud[j][i] for j in range(len(ud))])
        legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k')
plt.legend(legends)
plt.show()

for layer in linearLayers:
    if isinstance(layer, BatchNorm):
        layer.training = False
loss = getLoss(mask, testX, testY, linearLayers, maskCount * contextCount)
print(loss.item())
