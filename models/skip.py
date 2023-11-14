from .common import *


def type_check(obj):
    return isinstance(obj, list) or isinstance(obj, tuple)


def skip(input_channels=2,
         output_channels=3,
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

    Arguments:
        input_channels: number of input channels
        output_channels: number of output channels
        num_channels_up: number of upsample channels
        num_channels_down: number of downsample channels
        num_channels_skip: number of skip connect channels
        filter_size_up: upsample convolution kernel size
        filter_size_down: downsample convolution kernel size
        filter_size_skip: skip connect convolution kernel size
        need_sigmoid: whether to activate the function
        need_bias: whether bias is required
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        need1x1_up: whether 1*1 convolution is required
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

    last_scale = encoder_decoder_pair - 1

    model = nn.Sequential()
    model_tmp = model

    input_depth = input_channels
    # structural network
    for i in range(encoder_decoder_pair):
        deep_conv = nn.Sequential()
        skip_connect = nn.Sequential()

        # don't know what that means
        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, {'skip connect': skip_connect, 'conv layer': deep_conv}))
        else:
            model_tmp.add(deep_conv)

        if num_channels_skip[i] != 0:
            # The skip connection consists of a convolution layer,
            # a batch normalization layer, and a nonlinear activation layer
            skip_connect.add_module('skip conv', conv(input_depth, num_channels_skip[i], filter_size_skip,
                                                      bias=need_bias, pad=pad))
            skip_connect.add_module('skip bn', bn(num_channels_skip[i]))
            skip_connect.add_module('skip activate', act(act_fun))

        # downsample layer
        encoder = nn.Sequential()
        encoder.add_module('1', conv(input_depth, num_channels_down[i], filter_size_down[i],
                                     stride=2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        encoder.add_module('2', bn(num_channels_down[i]))
        encoder.add_module('3', act(act_fun))
        encoder.add_module('4', conv(num_channels_down[i], num_channels_down[i], filter_size_down[i],
                                     bias=need_bias, pad=pad))
        encoder.add_module('5', bn(num_channels_down[i]))
        encoder.add_module('6', act(act_fun))
        deep_conv.add_module('encoder:' + str(i + 1), encoder)

        next_layer = nn.Sequential()

        if i == encoder_decoder_pair - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deep_conv.add(next_layer)
            k = num_channels_up[i + 1]

        # upsample layer
        # scale_factor is the multiple of the upsampling
        deep_conv.add_module('upsample layer', nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        decoder = nn.Sequential()
        # Adding a normalization layer, the number of eigenvector channels is
        # the sum of the jump connections and the number of channels in the next layer
        decoder.add_module('1', bn(num_channels_skip[i] +
                                   (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))
        decoder.add_module('2', conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i],
                                     stride=1, bias=need_bias, pad=pad))
        decoder.add_module('3', bn(num_channels_up[i]))
        decoder.add_module('4', act(act_fun))
        model_tmp.add_module('decoder:' + str(i + 1), decoder)

        if need1x1_up:
            # 1x1 convolution, but did not change the number of channels,
            # presumably cross-channel information fusion
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], kernel_size=1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = next_layer  # recursive initialization network model

    # add a 1x1 convolution layer to the last layer of the model
    model.add_module('1x1 conv', conv(num_channels_up[0], output_channels, kernel_size=1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
