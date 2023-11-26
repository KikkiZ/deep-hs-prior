from torch import nn

from models.common import conv, bn, act, Concat


def type_check(obj):
    return isinstance(obj, list) or isinstance(obj, tuple)


def skip(input_channel=2,
         output_channel=3,
         num_channels_up=None,
         num_channels_down=None,
         num_channels_skip=None,
         filter_size_up=3,
         filter_size_down=3,
         filter_size_skip=1,
         need_sigmoid=True,
         need_bias=True,
         pad='zero',  # padding mode
         upsample_mode='nearest',
         downsample_mode='stride',
         act_fun='LeakyReLU',  # activate function
         need1x1_up=True):
    """Assembles encoder-decoder with skip connections.
    Construct a ten-layer encoder-decoder network with skip connections.

    :param input_channel: number of input channels
    :param output_channel: number of output channels
    :param num_channels_up: number of upsample channels
    :param num_channels_down: number of downsample channels
    :param num_channels_skip: number of skip connect channels
    :param filter_size_up: upsample convolution kernel size
    :param filter_size_down: downsample convolution kernel size
    :param filter_size_skip: skip connect convolution kernel size
    :param need_sigmoid: whether to activate the function
    :param need_bias: whether bias is required
    :param pad: zero|reflection (default: 'zero')
    :param upsample_mode: 'nearest|bilinear' (default: 'nearest')
    :param downsample_mode: 'stride|avg|max|lanczos2' (default: 'stride')
    :param act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
    :param need1x1_up: whether 1*1 convolution is required
    """

    if num_channels_up is None:
        num_channels_up = [16, 32, 64, 128, 128]
    if num_channels_down is None:
        num_channels_down = [16, 32, 64, 128, 128]
    if num_channels_skip is None:
        num_channels_skip = [4, 4, 4, 4, 4]

    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    # number of encoder-decoder groups
    encoder_decoder_pair = len(num_channels_down)

    # the isinstance() function determines whether an object is of a known type, similar to type()
    if not type_check(upsample_mode):
        # expands the list to a specified length
        upsample_mode = [upsample_mode] * encoder_decoder_pair

    if not type_check(downsample_mode):
        downsample_mode = [downsample_mode] * encoder_decoder_pair

    if not type_check(filter_size_up):
        filter_size_up = [filter_size_up] * encoder_decoder_pair

    if not type_check(filter_size_down):
        filter_size_down = [filter_size_down] * encoder_decoder_pair

    model = nn.Sequential()
    model_tmp = model

    encoder_channel = input_channel
    # structural network
    for i in range(encoder_decoder_pair):
        encoder = nn.Sequential()
        skip_connect = nn.Sequential()

        # downsample layer
        encoder.add_module('0:downsample', conv(encoder_channel, num_channels_down[i], filter_size_down[i],
                                                stride=2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        encoder.add_module('1:bn', bn(num_channels_down[i]))
        encoder.add_module('2:act', act(act_fun))
        encoder.add_module('3:conv', conv(num_channels_down[i], num_channels_down[i], filter_size_down[i],
                                          bias=need_bias, pad=pad))
        encoder.add_module('4:bn', bn(num_channels_down[i]))
        encoder.add_module('5:act', act(act_fun))

        if num_channels_skip[i] != 0 and i != 4:
            # 当跳跃连接通道数不为零时，即存在跳跃连接时添加module
            skip_connect.add_module('skip:conv', conv(num_channels_down[i], num_channels_skip[i], filter_size_skip,
                                                      bias=need_bias, pad=pad))
            skip_connect.add_module('skip:bn', bn(num_channels_skip[i]))
            skip_connect.add_module('skip:activate', act(act_fun))

            model_tmp.add(Concat(1, {'encoder:' + str(i + 1): encoder, 'skip connect': skip_connect}))
        else:
            model_tmp.add_module('encoder:' + str(i + 1), encoder)

        next_layer = nn.Sequential()
        if i != encoder_decoder_pair - 1:
            model_tmp.add(next_layer)

        # upsample layer
        # scale_factor is the multiple of the upsampling
        model_tmp.add_module('upsample layer', nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        decoder_channel = num_channels_up[i] if i == 4 else num_channels_up[i] + num_channels_skip[i]

        decoder = nn.Sequential()
        # Adding a normalization layer, the number of eigenvector channels is
        # the sum of the jump connections and the number of channels in the next layer
        decoder.add_module('0:bn', bn(decoder_channel))
        decoder.add_module('1:conv', conv(decoder_channel, num_channels_up[i - 1],
                                          filter_size_up[i], stride=1, bias=need_bias, pad=pad))
        decoder.add_module('2:bn', bn(num_channels_up[i - 1]))
        decoder.add_module('3:act', act(act_fun))
        decoder.add_module('4:conv', conv(num_channels_up[i - 1], num_channels_up[i - 1], filter_size_up[i],
                                          stride=1, bias=need_bias, pad=pad))
        decoder.add_module('5:bn', bn(num_channels_up[i - 1]))
        decoder.add_module('6:act', act(act_fun))

        if need1x1_up:
            # 1x1 convolution, but did not change the number of channels,
            # presumably cross-channel information fusion
            decoder.add_module('7:conv', conv(num_channels_up[i - 1], num_channels_up[i - 1],
                                              kernel_size=1, bias=need_bias, pad=pad))
            decoder.add_module('8:bn', bn(num_channels_up[i - 1]))
            decoder.add('9:act', act(act_fun))

        model_tmp.add_module('decoder:' + str(i + 1), decoder)
        encoder_channel = num_channels_down[i]
        model_tmp = next_layer  # recursive initialization network model

    # add a 1x1 convolution layer to the last layer of the model
    model.add_module('output', conv(num_channels_up[0], output_channel, kernel_size=1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    return model
