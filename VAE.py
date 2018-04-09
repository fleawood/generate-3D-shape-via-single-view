import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.legacy.nn import Identity
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
        if in_channels == out_channels:
            self.shortcut = Identity()
        else:
            if conv_trans:
                self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                                   padding=padding, dilation=dilation)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                          padding=padding, dilation=dilation)

    def forward(self, x):
        x = self.shortcut(x)
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
        self.bn1 = nn.BatchNorm2d(out_channels)

        if conv_trans:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = ShortCut(in_channels, out_channels, kernel_size, stride, padding, dilation, conv_trans)

    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(self.bn1(y))
        y = self.conv2(y)
        z = self.shortcut(x)
        return y + z

    # make syntax inspection happy
    def __call__(self, *args, **kwargs):
        return super(ResidualBlock, self).__call__(*args, **kwargs)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim, num_cats, is_single=False):
        super(Encoder, self).__init__()
        # for single view point
        in_channels = 1
        # conv1
        self.conv1 = nn.Conv2d(in_channels, out_channels * 4, kernel_size=4, stride=2, padding=1, dilation=2)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels * 4)

        # res block
        self.res1 = ResidualBlock(out_channels * 4, out_channels * 6, kernel_size=4, stride=2, padding=1, dilation=2)
        self.res2 = ResidualBlock(out_channels * 6, out_channels * 8, kernel_size=4, stride=2, padding=1, dilation=2)
        self.res3 = ResidualBlock(out_channels * 8, out_channels * 6, kernel_size=4, stride=2, padding=1, dilation=2)
        self.res4 = ResidualBlock(out_channels * 6, out_channels, kernel_size=4, stride=2, padding=1, dilation=2)

        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.view1 = View(-1, out_channels * 4 * 4)

        # mean, var
        self.linear1 = nn.Linear(out_channels * 4 * 4, z_dim)
        self.linear2 = nn.Linear(out_channels * 4 * 4, z_dim)

        # category
        self.linear3 = nn.Linear(out_channels * 4 * 4, out_channels * 4 * 2)
        self.linear4 = nn.Linear(out_channels * 4 * 2, num_cats)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.bn2(x)
        x = self.view1(x)

        mean = self.linear1(x)
        var = self.linear2(x)
        category = self.linear4(F.relu(self.linear3(x)))
        return mean, var, category

    # make syntax inspection happy
    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim, num_cats):
        super(Decoder, self).__init__()
        # from z and c
        self.linear1 = nn.Linear(z_dim + num_cats, out_channels * 2 * 4 * 4)
        self.view1 = View(-1, out_channels * 2, 4, 4)
        self.bn1 = nn.BatchNorm2d(out_channels * 2)

        # res block
        self.res1 = ResidualBlock(out_channels * 2, out_channels * 6, kernel_size=4, conv_trans=True)
        self.res2 = ResidualBlock(out_channels * 6, out_channels * 8, kernel_size=4, stride=2, padding=1,
                                  conv_trans=True)
        self.res3 = ResidualBlock(out_channels * 8, out_channels * 7, kernel_size=4, stride=2, padding=1,
                                  conv_trans=True)
        self.bn2 = nn.BatchNorm2d(out_channels * 7)

        # transpose conv1
        self.conv1 = nn.ConvTranspose2d(out_channels * 7, out_channels * 6, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels * 6)

        # transpose conv2
        self.conv2 = nn.ConvTranspose2d(out_channels * 6, out_channels * 4, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels * 4)

        # depth map
        self.conv3 = nn.ConvTranspose2d(out_channels * 4, in_channels, kernel_size=4, stride=2, padding=1)

        # silhouette
        # self.conv4 = nn.ConvTranspose2d(out_channels * 4, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.view1(self.linear1(x))))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.bn2(x)

        x = F.relu(self.bn3(self.conv1(x)))
        x = F.relu(self.bn4(self.conv2(x)))

        depth_map = F.sigmoid(self.conv3(x))
        # silhouette = F.sigmoid(self.conv4(x))
        return depth_map

    # make syntax inspection happy
    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)


class VAE(nn.Module):
    def __init__(self, num_cats):
        super(VAE, self).__init__()
        in_channels = 20
        out_channels = 60
        z_dim = 200

        self.encoder = Encoder(in_channels, out_channels, z_dim, num_cats)
        self.sampler = Sampler(z_dim)
        self.decoder = Decoder(in_channels, out_channels, z_dim, num_cats)

    def forward(self, x, c):
        mean, var, category = self.encoder(x)
        z = self.sampler(mean, var)
        depth_maps = self.decoder(torch.cat((z, c), dim=1))
        return depth_maps, mean, var, category

    # make syntax inspection happy
    def __call__(self, *args, **kwargs):
        return super(VAE, self).__call__(*args, **kwargs)
