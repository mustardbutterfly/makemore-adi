import torch


class LinearLayer:
    def __init__(self, inParams, outParams):
        self._variables = torch.randn([inParams, outParams]).float() / inParams ** .5

    def __call__(self, input):
        self._out = input @ self._variables
        return self._out

    def parameters(self):
        return [self._variables]

    def out(self):
        return self._out


class Tanh:

    def __init__(self):
        self.tanh = torch.nn.Tanh()

    def __call__(self, input):
        self._out = self.tanh(input)
        return self._out

    def out(self):
        return self._out

    def parameters(self):
        return []
