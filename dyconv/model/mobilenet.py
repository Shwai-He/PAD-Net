'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer.dyconv import dynamic_convolution_generator, Conv2dWrapper, DynamicConvolution
from .layer.common import TempModule, CustomSequential
import math
class Block(TempModule):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1, conv_sparse='mask'):
        super(Block, self).__init__()

        ConvLayer = dynamic_convolution_generator(nof_kernels=4, conv_sparse=conv_sparse)
        self.conv1 = ConvLayer(in_planes, in_planes, kernel_size=3, stride=stride, padding=1,
                               groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = ConvLayer(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x, temperature):
        out = F.relu(self.bn1(self.conv1(x, temperature)))
        out = F.relu(self.bn2(self.conv2(out, temperature)))
        return out


class MobileNet(nn.Module):
    def __init__(self, sparsity=0., conv_sparse='mask'):
        super(MobileNet, self).__init__()
        self.temperature = 1
        self.sparsity = 1
        if '1' in conv_sparse:
            self.spa_delta = 1 - sparsity
        else:
            self.spa_delta = math.pow(1 - sparsity, 0.1)
        self.tem_delta = 3 if 'expert' not in conv_sparse else 0
        self.conv_sparse = conv_sparse
        conv = Conv2dWrapper if 'expert' in self.conv_sparse else nn.Conv2d
        ConvLayer = dynamic_convolution_generator(nof_kernels=4, conv_sparse=conv_sparse)

        def conv_bn(inp, oup, stride):
            return CustomSequential(
                conv(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return CustomSequential(
                ConvLayer(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                ConvLayer(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = CustomSequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.linear = conv(1024, 1000, kernel_size=1, stride=1)


    def forward(self, x):
        x = self.model(x, self.temperature)
        x = self.linear(x)
        x = x.view(-1, 1000)
        return x

    def update_sparsity(self):
        self.sparsity *= self.spa_delta

    def update_temperature(self):
        self.temperature -= self.tem_delta
        self.temperature = round(self.temperature, 2)

if __name__ == '__main__':

    kwargs = {
        'sparsity': 0.9,
        'conv_sparse': 'mask',
    }

    net = MobileNet(**kwargs)
    net.update_temperature()
    net.update_sparsity()
    print(net.sparsity, net.temperature)
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

