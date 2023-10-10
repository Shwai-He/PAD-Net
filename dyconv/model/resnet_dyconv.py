import torch.nn as nn
import torch.nn.functional as F
from .layer.dyconv import dynamic_convolution_generator, Conv2d, DynamicConvolution
from .layer.common import CustomSequential, TempModule
import math

ConvLayer = DynamicConvolution
class BasicBlock(TempModule):
    expansion = 1

    def __init__(self, ConvLayer, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvLayer(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = ConvLayer(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = CustomSequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = CustomSequential(
                ConvLayer(in_planes, self.expansion*planes, kernel_size=1,
                               stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, temperature):
        out = F.relu(self.bn1(self.conv1(x, temperature)), inplace=True)
        out = self.bn2(self.conv2(out, temperature))
        out += self.downsample(x, temperature)
        out = F.relu(out, inplace=True)
        return out


class Bottleneck(TempModule):
    expansion = 4

    def __init__(self, ConvLayer, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvLayer(
            in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = ConvLayer(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = ConvLayer(
            planes, self.expansion*planes, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = CustomSequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = CustomSequential(
                ConvLayer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, temperature):
        out = F.relu(self.bn1(self.conv1(x, temperature)))
        out = F.relu(self.bn2(self.conv2(out, temperature)))
        out = self.bn3(self.conv3(out, temperature))
        out += self.downsample(x, temperature)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ConvLayer, block, num_blocks, num_classes=1000, in_planes=64, scale=1.0):
        super(ResNet, self).__init__()
        self.temperature = 31

        self.ConvLayer = ConvLayer
        self.in_planes = in_planes
        self.conv1 = Conv2d(3, in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64 * scale), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128 * scale), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * scale), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * scale), num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Conv2d(int(512 * scale) * block.expansion, num_classes, kernel_size=1, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()

        # for name, param in self.named_parameters():
        #      # if 'bn' in name:
        #      #     param.requires_grad = True
        #      # else:
        #      #     param.requires_grad = False
        #     if 'attention' in name:
        #         param.requires_grad = True
        #     elif len(param.size()) == 5:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.ConvLayer, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return CustomSequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out, self.temperature)
        out = self.layer2(out, self.temperature)
        out = self.layer3(out, self.temperature)
        out = self.layer4(out, self.temperature)
        out = self.avgpool(out)
        out = self.fc(out)
        out = out.view(out.size(0), -1)
        return out

    def update_temperature(self, temperature):
        self.temperature = temperature

    def reset_buffers(self):
        name_list, buf_list = [], []
        for name, module in self.named_modules():
            if isinstance(module, (DynamicConvolution)):
                for name, buf in module.named_buffers():
                    if 'mask' in name:
                        print(buf.mean())
                        name_list.append(name.split('.')[-1])
                        buf_list.append(buf)
            for i in range(len(name_list)):
                module.register_buffer(name_list[i], buf_list[i])


def DyResNet18(**kwargs):
    # ConvLayer = dynamic_convolution_generator(nof_kernels=4)
    return ResNet(ConvLayer, BasicBlock, [2, 2, 2, 2], **kwargs)


def DyResNet10(**kwargs):
    # ConvLayer = dynamic_convolution_generator(nof_kernels=4)
    return ResNet(ConvLayer, BasicBlock, [1, 1, 1, 1], **kwargs)


def DyResNet34(**kwargs):
    # ConvLayer = dynamic_convolution_generator(nof_kernels=4)
    return ResNet(ConvLayer, BasicBlock, [3, 4, 6, 3], **kwargs)


def DyResNet50(**kwargs):
    # ConvLayer = dynamic_convolution_generator(nof_kernels=4)
    return ResNet(ConvLayer, Bottleneck, [3, 4, 6, 3], **kwargs)


def DyResNet101(**kwargs):
    ConvLayer = dynamic_convolution_generator(nof_kernels=4)
    return ResNet(ConvLayer, Bottleneck, [3, 4, 23, 3], **kwargs)


def DyResNet152(**kwargs):
    ConvLayer = dynamic_convolution_generator(nof_kernels=4)
    return ResNet(ConvLayer, Bottleneck, [3, 8, 36, 3], **kwargs)


def dy_resnet18(pretrained: bool = False):
    ConvLayer = dynamic_convolution_generator(nof_kernels=4)
    return ResNet(ConvLayer, BasicBlock, [2, 2, 2, 2])


def dy_resnet10(pretrained: bool = False):
    ConvLayer = dynamic_convolution_generator(nof_kernels=4)
    return ResNet(ConvLayer, BasicBlock, [1, 1, 1, 1])


def dy_resnet34(pretrained: bool = False):
    ConvLayer = dynamic_convolution_generator(nof_kernels=4)
    return ResNet(ConvLayer, BasicBlock, [3, 4, 6, 3])


def dy_resnet50(pretrained: bool = False):
    ConvLayer = dynamic_convolution_generator(nof_kernels=4)
    return ResNet(ConvLayer, Bottleneck, [3, 4, 6, 3])


def dy_resnet101(pretrained: bool = False):
    ConvLayer = dynamic_convolution_generator(nof_kernels=4)
    return ResNet(ConvLayer, Bottleneck, [3, 4, 23, 3])


def dy_resnet152(pretrained: bool = False):
    ConvLayer = dynamic_convolution_generator(nof_kernels=4)
    return ResNet(ConvLayer, Bottleneck, [3, 8, 36, 3])


