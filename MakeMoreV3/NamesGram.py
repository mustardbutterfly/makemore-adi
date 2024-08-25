import torch


class NamesGram:
    def __init__(self, contextCount = 3):
        namesText = open('../names.txt', 'r').read()
        chars = set(namesText)
        chars.remove('\n')
        chars = sorted(chars)

        self._mapOfChars = {c: index + 1 for index, c in enumerate(chars)}
        self._mapOfIdxToChars = {index + 1: c for index, c in enumerate(chars)}
        self._mapOfChars['.'] = 0
        self._mapOfIdxToChars[0] = '.'

        names = namesText.splitlines()

        self._xs = []
        self._ys = []

        for word in names:
            context = [0] * contextCount
            self._xs.append(context)
            for cw in word:
                c = self._mapOfChars[cw]
                self._ys.append(c)
                context = context[1:] + [c]
                self._xs.append(context)
            self._ys.append(0)
        self._ys = torch.tensor(self._ys)
        self._xs = torch.tensor(self._xs)

    def getIndex(self, char):
        return self._mapOfChars[char]
    def getChar(self, index):
        return self._mapOfIdxToChars[index]
    def getXs(self, indexes=None):
        if not indexes:
            return self._xs
        return self._xs[indexes]
    def getYs(self, indexes=None):
        if not indexes:
            return self._ys
        return self._ys[indexes]