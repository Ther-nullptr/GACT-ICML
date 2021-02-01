from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor, device, dtype

from quantize.layers import QConv2d, QBatchNorm2d, QReLU, QMaxPool2d, QLinear
from quantize.conf import config


class QModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        QModule.convert_layers(model)

    @staticmethod
    def convert_layers(module):
        for name, child in module.named_children():
            # Do not convert layers that are already quantized
            if isinstance(child, (QConv2d, QBatchNorm2d, QReLU, QMaxPool2d, QLinear)):
                continue

            if isinstance(child, nn.Conv2d):
                setattr(module, name, QConv2d(child.in_channels, child.out_channels,
                    child.kernel_size, child.stride, child.padding, child.dilation,
                    child.groups, child.bias))
            elif isinstance(child, nn.BatchNorm2d):
                setattr(module, name, QBatchNorm2d(child.num_features, child.eps, child.momentum,
                    child.affine, child.track_running_stats))
            elif isinstance(child, nn.Linear):
                setattr(module, name, QLinear(child.in_features, child.out_features,
                    child.bias is not None))
            elif isinstance(child, nn.ReLU):
                setattr(module, name, QReLU())
            elif isinstance(child, nn.MaxPool2d):
                setattr(module, name, QMaxPool2d(child.kernel_size, child.stride,
                    child.padding, child.dilation, child.return_indices, child.ceil_mode))
            else:
                QModule.convert_layers(child)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, mode: bool = True):
        print('QModule.train')
        config.training = mode
        return super().train(mode)

    def eval(self):
        print('QModule.eval')
        config.training = False
        return super().eval(self)

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
                        strict: bool = True):
        # remove the prefix "model." added by this wrapper
        new_state_dict = OrderedDict([("model." + k,  v) for k, v in state_dict.items()])
        return super().load_state_dict(new_state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super().state_dict(destination, prefix, keep_vars)

        # remove the prefix "model." added by this wrapper
        ret = OrderedDict([(k[6:], v) for k, v in ret.items()])
        return ret

