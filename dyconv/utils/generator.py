from model.layer.dynamic_convolutions import FusionConvolution
from model.layer.odconv import ODConv2d, Attention
from model.resnet_dcd_cmd import conv_dy, conv_basic_dy
from model.mobilenetv2_dcd_cmd import DYCls, DYModule
import torch
from torch import nn
from torch.nn import Parameter, init

def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf

def trainable(module):
    r"""Returns boolean whether a module is trainable.
    """
    return not isinstance(module, (Identity1d, Identity2d))

def prunable(module):
    r"""Returns boolean whether a module is prunable.
    """
    isprunable = isinstance(module, (FusionConvolution, ODConv2d, Attention,
                                     conv_dy, conv_basic_dy, DYCls, DYModule))
    return isprunable

def parameters(model):
    r"""Returns an iterator over models trainable parameters, yielding just the
    parameter tensor.
    """
    for module in filter(lambda p: trainable(p), model.modules()):
        for name, param in module.named_parameters(recurse=False):
            if name.endswith('weight'):
                yield param


def mAp(module):
    for mask_name, buf in module.named_buffers():
        if "mask" in mask_name:
            for param_name, param in module.named_parameters(recurse=False):
                if param_name.replace('weight', 'mask') == mask_name:
                    yield buf, param


def masked_parameters(model, bias=False):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in filter(lambda p: prunable(p), model.module.modules()):
        for mask, param in mAp(module):
            label = (param is not module.bias) if hasattr(module, 'bias') else True
            if bias or label:
                yield mask, param


class Identity1d(nn.Module):
    def __init__(self, num_features):
        super(Identity1d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W


class Identity2d(nn.Module):
    def __init__(self, num_features):
        super(Identity2d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features, 1, 1))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W
