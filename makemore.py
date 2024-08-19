import torch

namesText = open('names.txt', 'r').read()

chars = set(namesText)
chars.remove('\n')
chars = sorted(chars)

mapOfChars = {c: index+1 for index, c in enumerate(chars)}
mapOfIdxToChars = {index+1: c for index, c in enumerate(chars)}
mapOfChars['.'] = 0
mapOfIdxToChars[0] = '.'

names = namesText.splitlines()
firstNeighbor = []
nextNeighbor = []
for name in names:
    nameInInt = [mapOfChars[n] for n in name]
    firstNeighbor.append(0)
    nextNeighbor.append(nameInInt[0])

    for n1,n2 in zip(nameInInt, nameInInt[1:]):
        firstNeighbor.append(n1)
        nextNeighbor.append(n2)
    firstNeighbor.append(nameInInt[-1])
    nextNeighbor.append(0)


generator = torch.Generator().manual_seed(58395729474)
layer = torch.randn([27, 27], requires_grad=True, generator=generator)

for i in range(1000):
    firstNeighborEncoding = torch.nn.functional.one_hot(torch.tensor(firstNeighbor), 27).float()
    counts = firstNeighborEncoding @ layer
    counts = counts.exp()
    probs = counts / counts.sum(1, keepdim=True)
    temp = probs[torch.arange(len(firstNeighbor)), nextNeighbor]
    loss = -temp.log().mean() + .01 * (layer**2).mean()

    layer.grad = None
    loss.backward()
    layer.data += -5 * layer.grad
    print(loss.item())

