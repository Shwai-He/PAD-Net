import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from .common import TempModule

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16, Lambda='both'):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)
        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)

        if kernel_size != 1:  # point-wise convolution
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)

        if kernel_num != 1:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # @staticmethod
    def skip(self, x, temp=1.):
        return 1.0

    def forward(self, x, temperature=1.):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        #todo four types of attention
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / temperature)
        if self.in_planes == self.groups and self.in_planes == self.out_planes:  # depth-wise convolution
            filter_attention = self.skip(x)
        else:
            filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / temperature)

        #todo skip when kernel_size == 1
        if self.kernel_size == 1:
            spatial_attention = 1.
        else:
            spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
            spatial_attention = torch.sigmoid(spatial_attention / temperature)
        #todo skip when kernel_num == 1
        if self.kernel_num == 1:
            kernel_attention = 1.
        else:
            kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
            kernel_attention = F.softmax(kernel_attention / temperature, dim=1)
        return channel_attention, filter_attention, spatial_attention, kernel_attention


class ODConv2d(TempModule):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4, Lambda='none'):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.kernels_weights = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.kernels_weights[i], mode='fan_out', nonlinearity='relu')

    def forward(self, x, temperature=1.):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x, temperature)
        x = x * channel_attention
        if self.kernel_size == 1 and self.kernel_num == 1:
            output = F.conv2d(x, weight=self.kernels_weights.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)
        else:
            # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
            # while we observe that when using the latter method the models will run faster with less gpu memory cost.
            batch_size, in_planes, height, width = x.size()
            x = x.reshape(1, -1, height, width)
            aggregate_weight = spatial_attention * kernel_attention * self.kernels_weights.unsqueeze(dim=0)
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            aggregate_weight = aggregate_weight.view([-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output


class Attention_PAD(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16, Lambda='both'):
        super(Attention_PAD, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_phi = nn.Parameter(torch.Tensor([0]), requires_grad=True) if Lambda != 'none' \
            else nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.filter_phi = nn.Parameter(torch.Tensor([0]), requires_grad=True) if Lambda != 'none' \
            else nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.spatial_phi = nn.Parameter(torch.Tensor([0]), requires_grad=True) if Lambda != 'none' \
            else nn.Parameter(torch.Tensor([0]), requires_grad=False)

        self.channel_fc_weight = nn.Parameter(torch.Tensor(in_planes, attention_channel, 1, 1), requires_grad=True)
        self.channel_fc_bias = nn.Parameter(torch.Tensor(in_planes), requires_grad=True)
        self.channel_att = nn.Parameter(torch.Tensor(1, in_planes, 1, 1), requires_grad=True)
        self.register_buffer('channel_fc_mask', torch.ones(self.channel_fc_weight.shape))
        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc_weight = nn.Parameter(torch.Tensor(out_planes, attention_channel, 1, 1), requires_grad=True)
            self.filter_fc_bias = nn.Parameter(torch.Tensor(out_planes), requires_grad=True)
            self.filter_att = nn.Parameter(torch.Tensor(1, out_planes, 1, 1), requires_grad=True)
            self.register_buffer('filter_fc_mask', torch.ones(out_planes, 1, 1, 1))

        if kernel_size != 1:  # point-wise convolution
            self.spatial_fc_weight = nn.Parameter(torch.Tensor(kernel_size * kernel_size, attention_channel, 1, 1), requires_grad=True)
            self.spatial_fc_bias = nn.Parameter(torch.Tensor(kernel_size * kernel_size), requires_grad=True)
            self.spatial_att = nn.Parameter(torch.Tensor(1, kernel_size * kernel_size, 1, 1), requires_grad=True)
            self.register_buffer('spatial_fc_mask', torch.ones(kernel_size * kernel_size, 1, 1, 1))
        if kernel_num != 1:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def skip(self, x, temp=1.):
        return 1.0

    def forward(self, x, temperature=1.):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        channel_phi = 2 * torch.sigmoid(self.channel_phi)
        filter_phi = 2 * torch.sigmoid(self.filter_phi)
        spatial_phi = 2 * torch.sigmoid(self.spatial_phi)
        #todo four types of attention
        if self.in_planes == self.groups and self.in_planes == self.out_planes:  # depth-wise convolution
            filter_attention = self.skip(x)
        else:
            filter_attention = F.conv2d(x, self.filter_fc_mask * self.filter_fc_weight, self.filter_fc_bias)
            filter_attention = (2 - filter_phi) * self.filter_att + filter_phi * torch.sigmoid(
                filter_attention.view(x.size(0), -1, 1, 1) / temperature)

        channel_attention = F.conv2d(x, self.channel_fc_mask * self.channel_fc_weight, self.channel_fc_bias)
        channel_attention = (2 - channel_phi) * self.channel_att + channel_phi * torch.sigmoid(
                channel_attention.view(x.size(0), -1, 1, 1) / temperature)
        #todo skip when kernel_size == 1
        if self.kernel_size == 1:
            spatial_attention = 1.
        else:
            spatial_attention = F.conv2d(x, self.spatial_fc_mask * self.spatial_fc_weight, self.spatial_fc_bias)
            spatial_attention = (2 - spatial_phi) * self.spatial_att + spatial_phi * torch.sigmoid(
                spatial_attention.view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size) / temperature)
        #todo skip when kernel_num == 1
        if self.kernel_num == 1:
            kernel_attention = 1.
        else:
            kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
            kernel_attention = F.softmax(kernel_attention / temperature, dim=1)
        return channel_attention, filter_attention, spatial_attention, kernel_attention


class ODConv2d_PAD(TempModule):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4, Lambda='both'):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.phi_d = nn.Parameter(torch.Tensor([0]), requires_grad=True) if Lambda != 'none' \
            else nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.attention = Attention_PAD(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.kernels_weights = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self.weight = nn.Parameter(torch.randn(1, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self.register_buffer('kernels_weights_mask', torch.ones(self.weight.shape))
        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.kernels_weights[i], mode='fan_out', nonlinearity='relu')

    def forward(self, x, temperature=1.):
        phi = 2 * torch.sigmoid(self.phi_d)
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x, temperature)
        x = x * channel_attention
        if self.kernel_size == 1 and self.kernel_num == 1:
            output = F.conv2d(x, weight=self.kernels_weights.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)
        else:
            # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
            # while we observe that when using the latter method the models will run faster with less gpu memory cost.
            batch_size, in_planes, height, width = x.size()
            x = x.reshape(1, -1, height, width)
            aggregate_weight = spatial_attention * kernel_attention * self.kernels_weights.unsqueeze(dim=0)
            aggregate_weight = self.kernels_weights_mask * torch.sum(aggregate_weight, dim=1) * phi.view(1, 1, 1, 1, 1) \
                               + self.weight * (1 - self.kernels_weights_mask) * (2 - phi).view(1, 1, 1, 1, 1)
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            aggregate_weight = aggregate_weight.view([-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output
