import torch.nn as nn
import torchinfo


class MyLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super(MyLinear, self).__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.linear.to('cuda')
    def forward(self, x):
        return self.linear(x)
    def summary(self, input_size=None):
        return torchinfo.summary(self, input_size, verbose=0)
