import torch.nn as nn

from models.downsampler import Downsampler


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
            downsampler = Downsampler(n_planes=output_channel,
                                      factor=stride,
                                      kernel_type=downsample_mode,
                                      phase=0.5,
                                      preserve_size=True)
        else:
            assert False

        stride = 1

    padding = None
    to_pad = kernel_size // 2
    if pad == 'reflection':
        padding = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padding, convolver, downsampler])
    return nn.Sequential(*layers)
