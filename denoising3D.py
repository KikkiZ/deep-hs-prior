import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from models.unet3D import UNet
from utils.common_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

i = 0                # 模型的迭代次数
out_avg = None       # 上次迭代的输出
last_net = None      # 上次迭代的参数
net_input = None     # 网络输入
psnr_noisy_last = 0  # 上次迭代的psnr


def func(args):
    file_name = './data/denoising.mat'
    image, decrease_image = read_data(file_name)
    print_images(image, decrease_image, 'origin image')

    reg_noise_std = args.reg_noise_std
    learning_rate = args.learning_rate
    exp_weight = args.exp_weight
    show_every = args.show_every
    num_iter = args.num_iter  # number of network iterations

    data_type = torch.cuda.FloatTensor
    net = UNet(input_channel=image.shape[0],
               output_channel=image.shape[0],
               num_channels_down=[args.down_channel] * 5,
               num_channels_up=[args.up_channel] * 5,
               num_channel_skip=args.skip_channel,
               kernel_size_up=3,
               kernel_size_down=3,
               kernel_size_skip=3,
               upsample_mode=args.upsample_mode,  # downsample_mode='avg',
               need1x1_up=False,
               need_sigmoid=False,
               need_bias=True,
               pad='reflection',
               activate='LeakyReLU').type(data_type)
    device = torch.device('cuda')
    net.to(device)

    loss_func = torch.nn.MSELoss().type(data_type)

    decrease_image = decrease_image[None, :].cuda()
    net_input = get_noise(image.shape[0], '3D', (image.shape[1], image.shape[2])).type(data_type).detach()
    net_input_saved = net_input.detach().clone()  # clone the noise tensor without grad
    noise = net_input.detach().clone()

    date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
    writer = SummaryWriter('./logs/denoising3D/' + date)  # the location where the data record is saved

    def closure():
        global i, out_avg, psnr_noisy_last, last_net, net_input

        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)  # the result of a network iteration

        # smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        total_loss = loss_func(out, decrease_image).to(device)  # calculate the loss value of the loss function
        total_loss.backward()                                   # back propagation gradient calculation

        psnr_noisy = psnr_gpu(decrease_image.squeeze(), out.squeeze())
        psnr_gt = psnr_gpu(image, out.squeeze())
        psnr_gt_sm = psnr_gpu(image, out_avg.squeeze())

        writer.add_scalar('compare with de', psnr_noisy, i)
        writer.add_scalar('compare with gt', psnr_gt, i)
        writer.add_scalar('compare with gt_sm', psnr_gt_sm, i)

        # backtracking
        if i % show_every == 0:
            msg = 'iteration times: [' + str(i) + '/' + str(num_iter) + ']'
            print(msg)
            out = torch.clamp(out, 0, 1)
            out_avg = torch.clamp(out_avg, 0, 1)

            out_normalize = max_min_normalize(out.squeeze().detach())
            out_avg_normalize = max_min_normalize(out_avg.squeeze().detach())
            print_images(out_normalize, out_avg_normalize, msg)

        i += 1

        return total_loss

    params = get_params('net', net, net_input)
    optimize('adam', params, closure, learning_rate, num_iter)

    writer.close()
