import torch


class LinearLayer():
    def __init__(self, inParams, outParams):
        self._variables = torch.randn([inParams, outParams], requires_grad=True).float()

    def __call__(self, input):
        return input @ self._variables

    def parameters(self):
        return self._variables