import torch


class BatchNormLayer:
    def __init__(self, dim):
        self.strech = torch.ones(dim)
        self.offset = torch.zeros(dim)

        self.running_strech = torch.ones(dim)
        self.running_offset = torch.zeros(dim)

        self.training = True

    def __call__(self, input: torch.Tensor):
        mean = input.mean(dim=(0,1), keepdim=True)
        variance = input.var(dim=(0,1), keepdim=True)
        input = (input - mean) / torch.sqrt(variance + .0001)
        self.running_strech = .01 * self.strech + .99 * self.running_strech
        self.running_offset = .01 * self.offset + .99 * self.running_offset

        if self.training:
            self.scaledAndShifted = self.strech * input + self.offset
        else:
            self.scaledAndShifted = self.running_strech * input + self.running_offset
        return self.scaledAndShifted

    def parameters(self):
        return [self.offset] + [self.strech]

    def out(self):
        return self.scaledAndShifted
