import torch
import torch.nn as nn
from .layer.odconv import ODConv2d_PAD, Attention
from .layer.dyconv import PAD_DynamicConvolution, Conv2dWrapper
from .layer.common import CustomSequential, TempModule
__all__ = ['OD_ResNet_PAD', 'od_resnet18_pad', 'od_resnet34_pad', 'od_resnet50_pad', 'od_resnet101_pad']


def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1, Lambda=None):
    return ODConv2d_PAD(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num, Lambda=Lambda)


def odconv1x1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1, Lambda=None):
    return ODConv2d_PAD(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                    reduction=reduction, kernel_num=kernel_num, Lambda=Lambda)


class BasicBlock(TempModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1, Lambda=None):
        super(BasicBlock, self).__init__()
        self.conv1 = odconv3x3(inplanes, planes, stride, reduction=reduction, kernel_num=kernel_num, Lambda=Lambda)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = odconv3x3(planes, planes, reduction=reduction, kernel_num=kernel_num, Lambda=Lambda)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, temperature=1.):
        identity = x

        out = self.conv1(x, temperature)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, temperature)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x, temperature)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(TempModule):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1, Lambda=None):
        super(Bottleneck, self).__init__()
        self.conv1 = odconv1x1(inplanes, planes, reduction=reduction, kernel_num=kernel_num, Lambda=Lambda)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = odconv3x3(planes, planes, stride, reduction=reduction, kernel_num=kernel_num, Lambda=Lambda)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = odconv1x1(planes, planes * self.expansion, reduction=reduction, kernel_num=kernel_num, Lambda=Lambda)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, temperature=1.):
        identity = x

        out = self.conv1(x, temperature)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, temperature)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, temperature)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x, temperature)

        out += identity
        out = self.relu(out)
        return out


class OD_ResNet_PAD(nn.Module):
    def __init__(self, block, layers, num_classes=1000, dropout=0.1, reduction=0.0625, kernel_num=4, Lambda=None):
        super(OD_ResNet_PAD, self).__init__()
        self.inplanes = 64
        self.temperature = 31
        self.temp_epoch = 10
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], reduction=reduction, kernel_num=kernel_num)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction=reduction, kernel_num=kernel_num)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, reduction=reduction, kernel_num=kernel_num)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, reduction=reduction, kernel_num=kernel_num)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1, reduction=0.625, kernel_num=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = CustomSequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, reduction=reduction, kernel_num=kernel_num))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, reduction=reduction, kernel_num=kernel_num))

        return CustomSequential(*layers)

    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, self.temperature)
        x = self.layer2(x, self.temperature)
        x = self.layer3(x, self.temperature)
        x = self.layer4(x, self.temperature)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    def update_temperature(self, temperature):
        self.temperature = temperature

    def reset_buffers(self):
        name_list, buf_list = [], []
        for name, module in self.named_modules():
            if isinstance(module, (PAD_DynamicConvolution, ODConv2d_PAD, Attention)):
                for name, buf in module.named_buffers():
                    if 'mask' in name:
                        print(buf.mean())
                        name_list.append(name.split('.')[-1])
                        buf_list.append(buf)
            for i in range(len(name_list)):
                module.register_buffer(name_list[i], buf_list[i])



def od_resnet10_pad(**kwargs):
    model = OD_ResNet_PAD(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def od_resnet18_pad(**kwargs):
    model = OD_ResNet_PAD(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def od_resnet34_pad(**kwargs):
    model = OD_ResNet_PAD(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def od_resnet50_pad(**kwargs):
    model = OD_ResNet_PAD(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def od_resnet101_pad(**kwargs):
    model = OD_ResNet_PAD(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model
