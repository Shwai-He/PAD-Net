import torch
from torch import nn
from torch.nn import *
from typing import TypeVar

T = TypeVar('T', bound=Module)


class Conv2dWrapper(nn.Conv2d):
    """
    Wrapper for pytorch Conv2d class which can take additional parameters(like temperature) and ignores them.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(x)


class TempModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, temperature) -> torch.Tensor:
        return x

class CustomSequential(TempModule):
    """Sequential container that supports passing temperature to TempModule"""

    def __init__(self, *args):
        super().__init__()
        self.layers = nn.ModuleList(args)

    def forward(self, x, temperature):
        for layer in self.layers:
            if isinstance(layer, TempModule):
                x = layer(x, temperature)
            else:
                x = layer(x)
        return x

    def __getitem__(self, idx):
        return CustomSequential(*list(self.layers[idx]))
        # if isinstance(idx, slice):
        #     return self.__class__(OrderedDict(list(self.layers.items())[idx]))
        # else:
        #     return self._get_item_by_idx(self.layers.values(), idx)


# Implementation inspired from
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py#L38 and
# https://github.com/pytorch/pytorch/issues/7455
