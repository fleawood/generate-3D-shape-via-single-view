import torch.nn as nn
import torch.nn.functional as F
from utils import View


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


def sigmoid(x):
    return F.sigmoid(x)


def relu(x):
    return F.relu(x)


def lrelu(x, rate=0.2):
    return F.leaky_relu(x, 0.2)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        channels = [1, 60, 120, 240, 360, 480, 360]
        z_dim = 400
        # 224 -> 112
        self.res1 = ResidualBlock(channels[0], channels[1], kernel_size=3, padding=1)
        # self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(channels[1])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 112 -> 56
        self.res2 = ResidualBlock(channels[1], channels[2], kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels[2])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 56 -> 28
        self.res3 = ResidualBlock(channels[2], channels[3], kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(channels[3])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 28 -> 14
        self.res4 = ResidualBlock(channels[3], channels[4], kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm2d(channels[4])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 14 -> 7
        self.res5 = ResidualBlock(channels[4], channels[5], kernel_size=3, padding=1)
        # self.conv5 = nn.Conv2d(channels[4], channels[5], kernel_size=3, padding=1)
        # self.bn5 = nn.BatchNorm2d(channels[5])
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 7 -> 4
        self.res6 = ResidualBlock(channels[5], channels[6], kernel_size=3, padding=1)
        # self.conv6 = nn.Conv2d(channels[5], channels[6], kernel_size=3, padding=1)
        # self.bn6 = nn.BatchNorm2d(channels[6])
        self.maxpool6 = nn.MaxPool2d(kernel_size=2, padding=1, stride=2)

        self.view1 = View(-1, 4 * 4 * channels[6])
        self.linear1 = nn.Linear(4 * 4 * channels[6], z_dim)

    def forward(self, x):
        x = self.res1(x)
        # x = self.conv1(x)
        # x = self.bn1(x)
        x = lrelu(x)
        x = self.maxpool1(x)

        x = self.res2(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        x = lrelu(x)
        x = self.maxpool2(x)

        x = self.res3(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        x = lrelu(x)
        x = self.maxpool3(x)

        x = self.res4(x)
        # x = self.conv4(x)
        # x = self.bn4(x)
        x = lrelu(x)
        x = self.maxpool4(x)

        x = self.res5(x)
        # x = self.conv5(x)
        # x = self.bn5(x)
        x = lrelu(x)
        x = self.maxpool5(x)

        x = self.res6(x)
        # x = self.conv6(x)
        # x = self.bn6(x)
        x = lrelu(x)
        x = self.maxpool6(x)

        x = self.view1(x)
        x = self.linear1(x)
        x = relu(x)
        return x

    # make syntax inspection happy
    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)


# Decoder/Generator
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # channels = [360, 480, 360, 240, 120, 60, 20]
        channels = [360, 480, 360, 240, 120, 60, 20]
        z_dim = 400
        self.linear1 = nn.Linear(z_dim, 4 * 4 * channels[0])
        self.view1 = View(-1, channels[0], 4, 4)
        # 4 -> 7
        self.res1 = ResidualBlock(channels[0], channels[1], kernel_size=3, stride=2, padding=1, conv_trans=True)
        # self.conv1 = nn.ConvTranspose2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(channels[1])
        # 7 -> 14
        self.res2 = ResidualBlock(channels[1], channels[2], kernel_size=4, stride=2, padding=1, conv_trans=True)
        # self.conv2 = nn.ConvTranspose2d(channels[1], channels[2], kernel_size=4, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels[2])
        # 14 -> 28
        self.res3 = ResidualBlock(channels[2], channels[3], kernel_size=4, stride=2, padding=1, conv_trans=True)
        # self.conv3 = nn.ConvTranspose2d(channels[2], channels[3], kernel_size=4, stride=2, padding=1)
        # self.bn3 = nn.BatchNorm2d(channels[3])
        # 28 -> 56
        self.res4 = ResidualBlock(channels[3], channels[4], kernel_size=4, stride=2, padding=1, conv_trans=True)
        # self.conv4 = nn.ConvTranspose2d(channels[3], channels[4], kernel_size=4, stride=2, padding=1)
        # self.bn4 = nn.BatchNorm2d(channels[4])
        # 56 -> 112
        self.res5 = ResidualBlock(channels[4], channels[5], kernel_size=4, stride=2, padding=1, conv_trans=True)
        # self.conv5 = nn.ConvTranspose2d(channels[4], channels[5], kernel_size=4, stride=2, padding=1)
        # self.bn5 = nn.BatchNorm2d(channels[5])
        # 112 -> 224
        # self.res6 = ResidualBlock(channels[5], channels[6], kernel_size=4, stride=2, padding=1, conv_trans=True)
        self.conv6 = nn.ConvTranspose2d(channels[5], channels[6], kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(channels[6])

    def forward(self, x):
        x = self.linear1(x)
        x = relu(x)
        x = self.view1(x)

        x = self.res1(x)
        # x = self.conv1(x)
        # x = self.bn1(x)
        x = relu(x)

        x = self.res2(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        x = relu(x)

        x = self.res3(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        x = relu(x)

        x = self.res4(x)
        # x = self.conv4(x)
        # x = self.bn4(x)
        x = relu(x)

        x = self.res5(x)
        # x = self.conv5(x)
        # x = self.bn5(x)
        x = relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = sigmoid(x)
        return x

    # make syntax inspection happy
    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


    # make syntax inspection happy
    def __call__(self, *args, **kwargs):
        return super(Autoencoder, self).__call__(*args, **kwargs)