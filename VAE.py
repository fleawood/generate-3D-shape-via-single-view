import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# class FactorizedConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
#         super(FactorizedConv, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=stride, padding=padding
# , dilation=dilation)
#         self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=1, padding=0
# , dilation=dilation)
#
#     def forward(self):
#         pass

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class View(nn.Module):
    def __init__(self, *size):
        super(View, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(self.size)

    # make syntax inspection happy
    def __call__(self, *args, **kwargs):
        return super(View, self).__call__(*args, **kwargs)


class Sampler:
    def __init__(self, z_dim):
        self.z_dim = z_dim

    def __call__(self, mean, var):
        rand = Variable(torch.randn(mean.size())).cuda()
        z = torch.addcmul(mean, 1, rand, var)
        return z


class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, conv_trans=False):
        super(ShortCut, self).__init__()
        if conv_trans:
            self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                               padding=padding, dilation=dilation)
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.shortcut(x))
        return x

    # make syntax inspection happy
    def __call__(self, *args, **kwargs):
        return super(ShortCut, self).__call__(*args, **kwargs)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, conv_trans=False):
        super(ResidualBlock, self).__init__()
        if conv_trans:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   dilation=dilation)
        self.conv1_bn = nn.BatchNorm2d(out_channels)

        if conv_trans:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_channels)
        self.shortcut = ShortCut(in_channels, out_channels, kernel_size, stride, padding, dilation, conv_trans)

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.conv2_bn(self.conv2(x))
        x += residual
        return x

    # make syntax inspection happy
    def __call__(self, *args, **kwargs):
        return super(ResidualBlock, self).__call__(*args, **kwargs)


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        # conv1 224 -> 110
        self.conv1 = nn.Conv2d(1, 240, kernel_size=4, stride=2, padding=1, dilation=2)
        self.conv1_bn = nn.BatchNorm2d(num_features=240)

        # res blocks
        # 110 -> 53
        self.res1 = ResidualBlock(240, 320, kernel_size=4, stride=2, padding=1, dilation=2)
        # 53 -> 25
        self.res2 = ResidualBlock(320, 400, kernel_size=4, stride=2, padding=1, dilation=2)
        # 25 -> 11
        self.res3 = ResidualBlock(400, 400, kernel_size=4, stride=2, padding=1, dilation=2)
        # 11 -> 4
        self.res4 = ResidualBlock(400, 320, kernel_size=4, stride=2, padding=1, dilation=2)

        self.view1 = View(-1, 320 * 4 * 4)

        # mean, var
        self.linear1 = nn.Linear(320 * 4 * 4, z_dim)
        self.linear2 = nn.Linear(320 * 4 * 4, z_dim)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))

        x = F.relu(self.res1(x))
        x = F.relu(self.res2(x))
        x = F.relu(self.res3(x))
        x = F.relu(self.res4(x))

        x = self.view1(x)

        mean = self.linear1(x)
        var = self.linear2(x)
        return mean, var

    # make syntax inspection happy
    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        # from z and c
        self.linear1 = nn.Linear(z_dim, 120 * 4 * 4)
        self.view1 = View(-1, 120, 4, 4)
        self.bn1 = nn.BatchNorm2d(120)

        # res block
        # 4 -> 7
        self.res1 = ResidualBlock(120, 320, kernel_size=4, conv_trans=True)
        # 7 -> 14
        self.res2 = ResidualBlock(320, 400, kernel_size=4, stride=2, padding=1, conv_trans=True)
        # 14 -> 28
        self.res3 = ResidualBlock(400, 480, kernel_size=4, stride=2, padding=1, conv_trans=True)
        # 28 -> 56
        self.res4 = ResidualBlock(480, 320, kernel_size=4, stride=2, padding=1, conv_trans=True)
        # 56 -> 112
        self.res5 = ResidualBlock(320, 240, kernel_size=4, stride=2, padding=1, conv_trans=True)

        # # transpose conv1
        # self.conv1 = nn.ConvTranspose2d(out_channels * 7, out_channels * 6, kernel_size=4, stride=2, padding=1)
        # self.bn3 = nn.BatchNorm2d(out_channels * 6)
        #
        # # transpose conv2
        # self.conv2 = nn.ConvTranspose2d(out_channels * 6, out_channels * 4, kernel_size=4, stride=2, padding=1)
        # self.bn4 = nn.BatchNorm2d(out_channels * 4)

        # depth map
        # 112 -> 224
        self.conv1 = nn.ConvTranspose2d(240, 20, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.view1(self.linear1(x))))
        x = F.relu(self.res1(x))
        x = F.relu(self.res2(x))
        x = F.relu(self.res3(x))
        x = F.relu(self.res4(x))
        x = F.relu(self.res5(x))

        # x = F.relu(self.bn3(self.conv1(x)))
        # x = F.relu(self.bn4(self.conv2(x)))

        x = F.sigmoid(self.conv1(x))
        return x

    # make syntax inspection happy
    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        in_channels = 20
        out_channels = 60
        z_dim = 100

        self.encoder = Encoder(z_dim)
        self.sampler = Sampler(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        mean, var = self.encoder(x)
        z = self.sampler(mean, var)
        reconstruction = self.decoder(z)
        return reconstruction, mean, var

    # make syntax inspection happy
    def __call__(self, *args, **kwargs):
        return super(VAE, self).__call__(*args, **kwargs)
