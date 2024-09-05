import torch
import torch.nn.functional as F

from MakeMoreV3.BatchNormLayer import BatchNormLayer


class Model:
    def __init__(self, layers):
        self._parameters = []
        self._layers = layers
        for layer in layers:
            parameters = layer.parameters()
            for p in parameters:
                p.requires_grad = True
            self._parameters += layer.parameters()

        self._ud = []

    def train(self, input, Y):

        for layer in self._layers:
            input = layer(input)

        loss = F.cross_entropy(input, Y)

        for layer in self._layers:
            layer.out().retain_grad()
        for p in self._parameters:
            p.grad = None

        loss.backward()

        for p in self._parameters:
            p.data += p.grad * -.1

        with torch.no_grad():
            newud = []
            for p in self._parameters:
                val = ((.1 * p.grad).std() / p.data.std()).log10().item()
                newud.append(val)
            self._ud.append(newud)

        return loss

    def getTestLoss(self, input, testY):
        for layer in self._layers:
            if isinstance(layer, BatchNormLayer):
                layer.training = False

        for layer in self._layers:
            input = layer(input)
        loss = F.cross_entropy(input, testY)
        print(loss.item())
        return loss

    def getLoggingVars(self):
        return {'ud': self._ud}

    def parameters(self):
        return self._parameters
