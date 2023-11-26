import torch
from torch import nn

from models.common import conv, bn, act


class UNet(nn.Module):
    def __init__(self,
                 input_channel=3,
                 output_channel=3,
                 num_channels_down=None,
                 num_channels_up=None,
                 num_channels_skip=4,
                 kernel_size_down=3,
                 kernel_size_up=3,
                 kernel_size_skip=3,
                 need_sigmoid=True,
                 downsample_mode='stride',
                 activate='LeakyReLU',
                 need_bias=True,
                 need1x1_up=True,
                 pad='zero',
                 upsample_mode='nearest'):

        super(UNet, self).__init__()

        if num_channels_up is None:
            num_channels_up = [16, 32, 64, 128, 128]
        if num_channels_down is None:
            num_channels_down = [16, 32, 64, 128, 128]

        self.encoder_1 = self._encoder(input_channel=input_channel,
                                       output_channel=num_channels_down[0],
                                       downsample_mode=downsample_mode,
                                       kernel_size=kernel_size_down,
                                       activate=activate,
                                       bias=need_bias,
                                       pad=pad)

        self.encoder_2 = self._encoder(input_channel=num_channels_down[0],
                                       output_channel=num_channels_down[1],
                                       downsample_mode=downsample_mode,
                                       kernel_size=kernel_size_down,
                                       activate=activate,
                                       bias=need_bias,
                                       pad=pad)

        self.encoder_3 = self._encoder(input_channel=num_channels_down[1],
                                       output_channel=num_channels_down[2],
                                       downsample_mode=downsample_mode,
                                       kernel_size=kernel_size_down,
                                       activate=activate,
                                       bias=need_bias,
                                       pad=pad)

        self.encoder_4 = self._encoder(input_channel=num_channels_down[2],
                                       output_channel=num_channels_down[3],
                                       downsample_mode=downsample_mode,
                                       kernel_size=kernel_size_down,
                                       activate=activate,
                                       bias=need_bias,
                                       pad=pad)

        self.encoder_5 = self._encoder(input_channel=num_channels_down[3],
                                       output_channel=num_channels_down[4],
                                       downsample_mode=downsample_mode,
                                       kernel_size=kernel_size_down,
                                       activate=activate,
                                       bias=need_bias,
                                       pad=pad)

        self.skip_1 = self._skip(input_channel=num_channels_down[0],
                                 output_channel=num_channels_skip,
                                 kernel_size=kernel_size_skip,
                                 activate=activate,
                                 bias=need_bias,
                                 pad=pad)

        self.skip_2 = self._skip(input_channel=num_channels_down[1],
                                 output_channel=num_channels_skip,
                                 kernel_size=kernel_size_skip,
                                 activate=activate,
                                 bias=need_bias,
                                 pad=pad)

        self.skip_3 = self._skip(input_channel=num_channels_down[2],
                                 output_channel=num_channels_skip,
                                 kernel_size=kernel_size_skip,
                                 activate=activate,
                                 bias=need_bias,
                                 pad=pad)

        self.skip_4 = self._skip(input_channel=num_channels_down[3],
                                 output_channel=num_channels_skip,
                                 kernel_size=kernel_size_skip,
                                 activate=activate,
                                 bias=need_bias,
                                 pad=pad)

        self.skip_5 = self._skip(input_channel=num_channels_down[4],
                                 output_channel=num_channels_skip,
                                 kernel_size=kernel_size_skip,
                                 activate=activate,
                                 bias=need_bias,
                                 pad=pad)

        self.upsample = self._upsample(upsample_mode)

        self.decoder_1 = self._decoder(input_channel=num_channels_up[1] + num_channels_skip,
                                       output_channel=num_channels_up[0],
                                       kernel_size=kernel_size_up,
                                       need1x1_up=need1x1_up,
                                       activate=activate,
                                       bias=need_bias,
                                       pad=pad)

        self.decoder_2 = self._decoder(input_channel=num_channels_up[2] + num_channels_skip,
                                       output_channel=num_channels_up[1],
                                       kernel_size=kernel_size_up,
                                       need1x1_up=need1x1_up,
                                       activate=activate,
                                       bias=need_bias,
                                       pad=pad)

        self.decoder_3 = self._decoder(input_channel=num_channels_up[3] + num_channels_skip,
                                       output_channel=num_channels_up[2],
                                       kernel_size=kernel_size_up,
                                       need1x1_up=need1x1_up,
                                       activate=activate,
                                       bias=need_bias,
                                       pad=pad)

        self.decoder_4 = self._decoder(input_channel=num_channels_up[4] + num_channels_skip,
                                       output_channel=num_channels_up[3],
                                       kernel_size=kernel_size_up,
                                       need1x1_up=need1x1_up,
                                       activate=activate,
                                       bias=need_bias,
                                       pad=pad)

        self.decoder_5 = self._decoder(input_channel=num_channels_down[4] + num_channels_skip,
                                       output_channel=num_channels_up[4],
                                       kernel_size=kernel_size_up,
                                       need1x1_up=need1x1_up,
                                       activate=activate,
                                       bias=need_bias,
                                       pad=pad)

        self.output = self._output(input_channel=num_channels_up[0],
                                   output_channel=output_channel,
                                   need_sigmoid=need_sigmoid,
                                   bias=need_bias,
                                   pad=pad)

    def _encoder(self, input_channel, output_channel, kernel_size, bias, pad, activate, downsample_mode):
        encoder = nn.Sequential()

        encoder.add_module('0:downsample', conv(input_channel=input_channel,
                                                output_channel=output_channel,
                                                downsample_mode=downsample_mode,
                                                kernel_size=kernel_size,
                                                stride=2,
                                                bias=bias,
                                                pad=pad))
        encoder.add_module('1:bn', bn(output_channel))
        encoder.add_module('2:act', act(activate=activate))
        encoder.add_module('3:conv', conv(input_channel=output_channel,
                                          output_channel=output_channel,
                                          kernel_size=kernel_size,
                                          bias=bias,
                                          pad=pad))
        encoder.add_module('4:bn', bn(output_channel))
        encoder.add_module('5:act', act(activate=activate))

        return encoder

    def _decoder(self, input_channel, output_channel, kernel_size, bias, pad, activate, need1x1_up):
        decoder = nn.Sequential()

        decoder.add_module('0:bn', bn(num_features=input_channel))
        decoder.add_module('1:conv', conv(input_channel=input_channel,
                                          output_channel=output_channel,
                                          kernel_size=kernel_size,
                                          bias=bias,
                                          pad=pad))
        decoder.add_module('2:bn', bn(num_features=output_channel))
        decoder.add_module('3:act', act(activate=activate))
        decoder.add_module('4:conv', conv(input_channel=output_channel,
                                          output_channel=output_channel,
                                          kernel_size=kernel_size,
                                          bias=bias,
                                          pad=pad))
        decoder.add_module('5:bn', bn(num_features=output_channel))
        decoder.add_module('6:act', act(activate=activate))

        if need1x1_up:
            decoder.add_module('7:conv', conv(input_channel=output_channel,
                                              output_channel=output_channel,
                                              kernel_size=1,
                                              bias=bias,
                                              pad=pad))
            decoder.add_module('8:bn', bn(num_features=output_channel))
            decoder.add_module('9:act', act(activate=activate))

        return decoder

    def _upsample(self, upsample_mode):
        return nn.Upsample(scale_factor=2, mode=upsample_mode)

    def _skip(self, input_channel, output_channel, kernel_size, bias, pad, activate):
        skip_connect = nn.Sequential()

        skip_connect.add_module('0:conv', conv(input_channel=input_channel,
                                               output_channel=output_channel,
                                               kernel_size=kernel_size,
                                               bias=bias,
                                               pad=pad))
        skip_connect.add_module('1:bn', bn(num_features=output_channel))
        skip_connect.add_module('2:act', act(activate=activate))

        return skip_connect

    def _output(self, input_channel, output_channel, bias, pad, need_sigmoid):
        layer = nn.Sequential()

        layer.add_module('0:conv', conv(input_channel=input_channel,
                                        output_channel=output_channel,
                                        kernel_size=1,
                                        bias=bias,
                                        pad=pad))

        if need_sigmoid:
            layer.add_module('1:act', nn.Sigmoid())

        return layer

    def forward(self, data):

        enc_1 = self.encoder_1(data)
        enc_2 = self.encoder_2(enc_1)
        enc_3 = self.encoder_3(enc_2)
        enc_4 = self.encoder_4(enc_3)
        enc_5 = self.encoder_5(enc_4)

        dec_5 = enc_5
        dec_5 = torch.cat([dec_5, self.skip_5(enc_5)], dim=1)
        up_5 = self.upsample(dec_5)

        dec_4 = self.decoder_5(up_5)
        dec_4 = torch.cat([dec_4, self.skip_4(enc_4)], dim=1)
        up_4 = self.upsample(dec_4)

        dec_3 = self.decoder_4(up_4)
        dec_3 = torch.cat([dec_3, self.skip_3(enc_3)], dim=1)
        up_3 = self.upsample(dec_3)

        dec_2 = self.decoder_3(up_3)
        dec_2 = torch.cat([dec_2, self.skip_2(enc_2)], dim=1)
        up_2 = self.upsample(dec_2)

        dec_1 = self.decoder_2(up_2)
        dec_1 = torch.cat([dec_1, self.skip_1(enc_1)], dim=1)
        up_1 = self.upsample(dec_1)

        out = self.decoder_1(up_1)
        return self.output(out)
