from collections import Iterable
import itertools
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch import nn
from .common import TempModule
import math

class AttentionLayer(nn.Module):
    def __init__(self, c_dim, hidden_dim, nof_kernels):
        super(AttentionLayer, self).__init__()

        self.fc1 = nn.Conv2d(c_dim, hidden_dim, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_dim, nof_kernels, 1, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        n = hidden_dim
        self.fc1.weight.data.normal_(0, math.sqrt(2. / n))
        n = nof_kernels
        self.fc2.weight.data.normal_(0, math.sqrt(2. / n))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, temperature=1):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / temperature, 1)


class DynamicConvolution(TempModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, nof_kernels=4,
                 dilation=1, groups=1, bias=True, reduce=16, output_hidden_states=True):
        super().__init__()
        self.stride = _pair(stride)
        self.groups = groups
        self.output_hidden_states = output_hidden_states
        self.hidden_states = () if output_hidden_states else None
        self.conv_args = {'stride': stride, 'padding': padding, 'dilation': dilation}
        self.attention = AttentionLayer(c_dim=in_channels, hidden_dim=max(1, in_channels // reduce),
                                        nof_kernels=nof_kernels)
        self.kernel_size = _pair(kernel_size)
        self.weight = nn.Parameter(torch.Tensor(
            nof_kernels, out_channels, in_channels // self.groups, *self.kernel_size), requires_grad=True)
        n = self.kernel_size[0] * self.kernel_size[1] * out_channels
        self.weight.data.normal_(0, math.sqrt(2. / n))
        if bias:
            self.bias = nn.Parameter(torch.zeros(nof_kernels, out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, temperature=31):
        batch_size = x.shape[0]
        weight = self.weight
        alphas = self.attention(x, temperature)
        agg_weights = torch.sum(torch.mul(weight, alphas.view(batch_size, -1, 1, 1, 1, 1)), dim=1)
        agg_weights = agg_weights.view(-1, *agg_weights.shape[-3:])
        if self.bias is not None:
            agg_bias = torch.sum(torch.mul(self.bias.unsqueeze(0), alphas.view(batch_size, -1, 1)), dim=1)
            agg_bias = agg_bias.view(-1)
        else:
            agg_bias = None
        x_grouped = x.view(1, -1, *x.shape[-2:])
        out = F.conv2d(x_grouped, agg_weights, agg_bias, groups=self.groups * batch_size,
                       **self.conv_args)
        out = out.view(batch_size, -1, *out.shape[-2:])
        self.hidden_states = out if self.output_hidden_states else None
        return out

class PAD_DynamicConvolution(TempModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, nof_kernels=4,
                 dilation=1, groups=1, bias=True, reduce=16, Lambda=None, output_hidden_states=False):
        super().__init__()
        self.Lambda = Lambda
        self.groups = groups
        self.nof_kernels = nof_kernels
        self.output_hidden_states = output_hidden_states
        self.hidden_states = () if output_hidden_states else None
        self.conv_args = {'stride': stride, 'padding': padding, 'dilation': dilation}
        self.attention = AttentionLayer(c_dim=in_channels, hidden_dim=max(1, in_channels // reduce),
                                        nof_kernels=nof_kernels)
        self.kernel_size = _pair(kernel_size)
        self.bias = None
        self.output_channel = out_channels
        self.kernels_weights = nn.Parameter(torch.Tensor(
            nof_kernels, out_channels, in_channels // self.groups, *self.kernel_size), requires_grad=True)
        n = self.kernel_size[0] * self.kernel_size[1] * out_channels
        self.weight = nn.Parameter(torch.Tensor(
            1, out_channels, in_channels // self.groups, *self.kernel_size), requires_grad=True)
        self.register_buffer('kernels_weights_mask', torch.ones(self.weight.shape))
        self.phi_d = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.kernels_weights.data.normal_(0, math.sqrt(2. / n))
        self.weight.data.normal_(0, math.sqrt(2. / n))
        if bias:
            self.kernels_bias = nn.Parameter(torch.zeros(nof_kernels, out_channels), requires_grad=True)
        else:
            self.register_parameter('kernels_bias', None)

    def forward(self, x, temperature=31):
        batch_size = x.shape[0]
        alphas = self.attention(x, temperature)
        # todo phi
        if self.Lambda == 'both':
            phi = 2 * torch.sigmoid(self.phi_d)
            agg_weights = self.kernels_weights_mask * torch.sum(
                torch.mul(self.kernels_weights, alphas.view(batch_size, -1, 1, 1, 1, 1)), dim=1) * phi.view(1, 1, 1, 1, 1) \
                          + self.weight * (1 - self.kernels_weights_mask) * (2 - phi).view(1, 1, 1, 1, 1)

        # todo lambda_s
        elif self.Lambda == 's':
            phi = 2 * torch.sigmoid(self.phi)
            agg_weights = self.kernels_weights_mask * torch.sum(
                torch.mul(self.kernels_weights, alphas.view(batch_size, -1, 1, 1, 1, 1)), dim=1) \
                          + self.weight * (1 - self.kernels_weights_mask) * (2 - phi).view(1, 1, 1, 1, 1)
        # todo lambda_d
        elif self.Lambda == 'd':
            phi = 2 * torch.sigmoid(self.phi)
            agg_weights = self.kernels_weights_mask * torch.sum(
                torch.mul(self.kernels_weights, alphas.view(batch_size, -1, 1, 1, 1, 1)), dim=1) * phi.view(1, 1, 1, 1, 1) \
                          + self.weight * (1 - self.kernels_weights_mask)

        # todo no phi
        else:
            agg_weights = self.kernels_weights_mask * torch.sum(
                torch.mul(self.kernels_weights, alphas.view(batch_size, -1, 1, 1, 1, 1)), dim=1) \
                          + self.weight * (1 - self.kernels_weights_mask)

        agg_weights = agg_weights.view(-1, *agg_weights.shape[-3:])
        if self.kernels_bias is not None:
            agg_bias = torch.sum(torch.mul(self.kernels_bias.unsqueeze(0), alphas.view(batch_size, -1, 1)), dim=1)
            agg_bias = agg_bias.view(-1)
        else:
            agg_bias = None
        x_grouped = x.view(1, -1, *x.shape[-2:])
        out = F.conv2d(x_grouped, agg_weights, agg_bias, groups=self.groups * batch_size,
                       **self.conv_args)
        out = out.view(batch_size, -1, *out.shape[-2:])
        self.hidden_states = out if self.output_hidden_states else None
        return out


class RouteLayer(nn.Module):
    def __init__(self, c_dim, hidden_dim, nof_kernels):
        super().__init__()

        self.fc1 = nn.Conv2d(c_dim, hidden_dim, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_dim, nof_kernels, 1, bias=True)
        self.fc_scale = nn.Conv2d(hidden_dim, 1, 1, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        n = hidden_dim
        self.fc1.weight.data.normal_(0, math.sqrt(2. / n))
        n = nof_kernels
        self.fc2.weight.data.normal_(0, math.sqrt(2. / n))
        nn.init.zeros_(self.fc2.bias)
        self.fc_scale.weight.data.normal_(0, 0.02)
        nn.init.zeros_(self.fc_scale.bias)

    def forward(self, x, temperature=1):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        phi = torch.sigmoid(self.fc_scale(x) / temperature)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / temperature, 1), phi


class FlexibleKernelsDynamicConvolution:
    def __init__(self, Base, nof_kernels):
        if isinstance(nof_kernels, Iterable):
            self.nof_kernels_it = iter(nof_kernels)
        else:
            self.nof_kernels_it = itertools.cycle([nof_kernels])
        self.Base = Base

    def __call__(self, *args, **kwargs):
        return self.Base(next(self.nof_kernels_it), *args, **kwargs)


def dynamic_convolution_generator(nof_kernels):
    return FlexibleKernelsDynamicConvolution(DynamicConvolution, nof_kernels)

def fuse_convolution_generator(nof_kernels):
    return FlexibleKernelsDynamicConvolution(PAD_DynamicConvolution, nof_kernels)

class Conv2dWrapper(TempModule):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.groups = groups
        self.out_channels = out_channels
        self.stride = _pair(stride)
        self.conv_args = {'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups}
        self.kernel_size = _pair(kernel_size)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size),
                                   requires_grad=True)
        n = self.kernel_size[0] * self.kernel_size[1] * out_channels
        self.weight.data.normal_(0, math.sqrt(2. / n))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, temp=1.):
        out = F.conv2d(x, self.weight, self.bias, **self.conv_args)
        return out


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode)

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        W = self.weight_mask * self.weight
        if self.bias is not None:
            b = self.bias_mask * self.bias
        else:
            b = self.bias
        return self._conv_forward(input, W, b)



if __name__ == '__main__':
    torch.manual_seed(41)
    t = torch.randn(1, 3, 16, 16)
    conv = DynamicConvolution(3, 1, in_channels=3, out_channels=8, kernel_size=3, padding=1, bias=True)
    print(conv(t, 10).sum())
