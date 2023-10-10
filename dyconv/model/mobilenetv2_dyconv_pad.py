import torch.nn as nn
import torch
from .layer.dyconv import fuse_convolution_generator, Conv2d, PAD_DynamicConvolution
from .layer.common import CustomSequential, TempModule
import math

ConvLayer = PAD_DynamicConvolution

__all__ = ['Dymobilenetv2_PAD', 'DyMobileNetV2_PAD']
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride, conv=nn.Conv2d):
    return CustomSequential(
        conv(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv=nn.Conv2d):
    return CustomSequential(
        conv(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(TempModule):
    def __init__(self, inp, oup, stride, expand_ratio, conv=nn.Conv2d, output_hidden_states=True):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.output_hidden_states = output_hidden_states
        self.hidden_states = () if output_hidden_states else None
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = CustomSequential(
                # dw
                conv(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = CustomSequential(
                conv(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                conv(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x, temperature):
        if self.identity:
            output = x + self.conv(x, temperature)
            self.hidden_states = output if self.output_hidden_states else None
            return output
        else:
            output = self.conv(x, temperature)
            self.hidden_states = output if self.output_hidden_states else None
            return output


class DyMobileNetV2_PAD(nn.Module):
    def __init__(self, conv_layer, num_classes=1000, width_multiplier=0.35, dropout=None, output_hidden_states=False):
        super(DyMobileNetV2_PAD, self).__init__()

        self.output_hidden_states = output_hidden_states
        self.hidden_states = () if output_hidden_states else None
        self.temperature = 31
        self.t_delta = 1
        self.ConvLayer = conv_layer
        self.cfgs = [(1, 16, 1, 1),
               (6, 24, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10
               (6, 32, 3, 2),
               (6, 64, 4, 2),
               (6, 96, 3, 1),
               (6, 160, 3, 2),
               (6, 320, 1, 1)]

        # building first layer
        input_channel = _make_divisible(32 * width_multiplier, 4 if width_multiplier == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2, conv=nn.Conv2d)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_multiplier, 4 if width_multiplier == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, conv=self.ConvLayer))
                input_channel = output_channel
        self.features = CustomSequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_multiplier, 4 if width_multiplier == 0.1 else 8) if width_multiplier > 1.0 else 1280
        self.output_channel = output_channel
        self.conv = conv_1x1_bn(input_channel, output_channel, conv=self.ConvLayer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        classifier = []
        if dropout is not None:
            classifier.append(nn.Dropout(p=dropout))

        classifier.append(Conv2d(output_channel, num_classes, kernel_size=1, stride=1))
        self.classifier = nn.Sequential(*classifier)

        self._initialize_weights()

    def forward(self, x):

        x = self.features(x, self.temperature)
        x = self.conv(x, self.temperature)
        self.hidden_states = x if self.output_hidden_states else None
        x = self.avgpool(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (PAD_DynamicConvolution, PAD_DynamicConvolution)):
                for i_kernel in range(m.nof_kernels):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.output_channel
                    m.kernels_weights[i_kernel].data.normal_(0, math.sqrt(2. / n))
                if m.kernels_bias is not None:
                    m.kernels_bias.data.zeros_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def update_temperature(self, temperature):
        self.temperature = temperature

    def reset_buffers(self):
        name_list, buf_list = [], []
        for name, module in self.named_modules():
            if isinstance(module, (PAD_DynamicConvolution, PAD_DynamicConvolution)):
                for name, buf in module.named_buffers():
                    if 'mask' in name:
                        print(buf.mean())
                        name_list.append(name.split('.')[-1])
                        buf_list.append(buf)
            for i in range(len(name_list)):
                module.register_buffer(name_list[i], buf_list[i])

def Dymobilenetv2_PAD(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return DyMobileNetV2_PAD(conv_layer=ConvLayer, **kwargs)

if __name__ == '__main__':

    x = torch.rand(1, 3, 64, 64)
    kwargs = {  'num_classes': 1000,
                'width_multiplier': 0.35,
                'sparsity': 0.9
                }
    model = Dymobilenetv2_PAD(**kwargs)
    x = model(x)
    print(x.size())