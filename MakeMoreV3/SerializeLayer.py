import torch


class EmbeddingLayer:
    def __init__(self, sizeOfEmbedding):
        self._embedding = torch.randn(27, sizeOfEmbedding)

    def __call__(self, x):
        self._out = self._embedding[x]
        return self._out

    def out(self):
        return self._out

    def parameters(self):
        return [self._embedding]


class ConsecutiveFlatteningLayer:
    def __init__(self, size):
        self._size = size

    def __call__(self, input):
        (data, reserve, out) = input.shape
        leftReserve = reserve // self._size
        self._out = input.view([data, leftReserve, self._size * out])
        if leftReserve == 1:
            self._out = self._out.squeeze(1)
        return self._out

    def parameters(self):
        return []

    def out(self):
        return self._out
