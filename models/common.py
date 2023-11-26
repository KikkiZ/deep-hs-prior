from typing import Dict

import torch
import torch.nn as nn
import numpy as np
from models.downsampler import Downsampler


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module


class Concat(nn.Module):
    def __init__(self, dim, modules: Dict[str, nn.Module]):
        super(Concat, self).__init__()
        self.dim = dim

        for key, value in modules.items():
            self.add_module(key, value)

    def forward(self, input_data):
        inputs = []
        # _modules.values(): contains all operations in the current layer (layer)
        temp = input_data
        for module in self._modules.values():
            temp = module(temp)
            print(temp.shape)
            inputs.append(temp)
        print(len(inputs))
        print(inputs[0].shape)
        print(inputs[1].shape)

        inputs_shapes_2 = [x.shape[2] for x in inputs]
        inputs_shapes_3 = [x.shape[3] for x in inputs]
        print(inputs_shapes_2)
        print(inputs_shapes_3)

        if (np.all(np.array(inputs_shapes_2) == min(inputs_shapes_2)) and
                np.all(np.array(inputs_shapes_3) == min(inputs_shapes_3))):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes_2)
            target_shape3 = min(inputs_shapes_3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, inputs):
        a = list(inputs.size())
        a[1] = self.dim2
        # print (input.data.type())

        b = torch.zeros(a).type_as(inputs.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(activate='LeakyReLU'):
    """
        Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(activate, str):
        if activate == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activate == 'Swish':
            return Swish()
        elif activate == 'ELU':
            return nn.ELU()
        elif activate == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return activate


def bn(num_features):
    """Batch normalized layers

    Argument:
        num_features: the number of channels of the eigenvector
    """
    return nn.BatchNorm2d(num_features)


def conv(input_channel, output_channel, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    """Convolution layer
    Returns a convolution layer containing downsampling and reflection padding.

    Argument:
        in_f: the number of input channels
        out_f: the number of output channels
        pad: setting padding mode
        downsample_mode: setting downsample mode
    """
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=output_channel, factor=stride, kernel_type=downsample_mode, phase=0.5,
                                      preserve_size=True)
        else:
            assert False

        stride = 1

    padding = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padding = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padding, convolver, downsampler])
    return nn.Sequential(*layers)
