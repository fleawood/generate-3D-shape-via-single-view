import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    # make syntax inspection happy
    def __call__(self, *args, **kwargs):
        return super(Identity, self).__call__(*args, **kwargs)


class View(nn.Module):
    def __init__(self, *size):
        super(View, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(self.size)

    # make syntax inspection happy
    def __call__(self, *args, **kwargs):
        return super(View, self).__call__(*args, **kwargs)
