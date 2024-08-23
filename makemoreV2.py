import torch
import torch.nn.functional as F

namesText = open('names.txt', 'r').read()
chars = set(namesText)
chars.remove('\n')
chars = sorted(chars)

mapOfChars = {c: index + 1 for index, c in enumerate(chars)}
mapOfIdxToChars = {index + 1: c for index, c in enumerate(chars)}
mapOfChars['.'] = 0
mapOfIdxToChars[0] = '.'

names = namesText.splitlines()

contextCount = 3

xs = []
ys = []

for word in names:
    context = [0] * contextCount
    xs.append(context)
    for cw in word:
        c = mapOfChars[cw]
        ys.append(c)
        context = context[1:] + [c]
        xs.append(context)
    ys.append(0)

maskCount = 10
layerParamsCount = 200

xsTensor = torch.tensor(xs)
ys = torch.tensor(ys)

indexes = torch.randperm(len(xs))
i1 = int(len(indexes) * .8)
i2 = int(len(indexes) * .1)
splitIndexes = indexes.split([i1, i2, len(indexes)-i1-i2])

train = xsTensor[splitIndexes[0]]
trainY = ys[splitIndexes[0]]
verify = xsTensor[splitIndexes[1]]
verifyY = ys[splitIndexes[1]]
test = xsTensor[splitIndexes[2]]
testY = ys[splitIndexes[2]]

mask = torch.randn([27, maskCount], requires_grad=True).float()

W1 = torch.randn([maskCount * contextCount, layerParamsCount], requires_grad=True).float()
B1 = torch.randn(layerParamsCount, requires_grad=True).float()
W2 = torch.randn([layerParamsCount, 27], requires_grad=True).float()
B2 = torch.randn(27, requires_grad=True).float()

allWeights = [mask, W1, B1, W2, B2]


def getLoss(currentBatch, currentBatchY):
    EmbeddingLayerlookup  = mask[currentBatch]
    maskLayer = EmbeddingLayerlookup.view([-1, maskCount * contextCount])

    firstForwardPass = torch.tanh(maskLayer @ W1 + B1)

    logits = firstForwardPass @ W2 + B2  # len X 27
    return F.cross_entropy(logits, currentBatchY)


batchSize = 32
for i in range(200000):
    indexBatch = torch.randint(0, train.shape[0], (32,))
    trainBatch = train[indexBatch]
    trainBatchY = trainY[indexBatch]

    loss = getLoss(trainBatch, trainBatchY)

    for weights in allWeights:
        weights.grad = None
    print(loss.item())
    loss.backward()
    for weights in allWeights:
        if i > 100000:
            weights.data += -.01 * weights.grad
        else:
            weights.data += -.1 * weights.grad



testLoss = getLoss(test, testY)
print(testLoss.item())