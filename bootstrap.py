import argparse

import denoising2D
import denoising_red


def parse_args():
    parser = argparse.ArgumentParser(description='deep_hs_prior')

    parser.add_argument('--net', dest='net', default='default', type=str)  # 网络选择
    parser.add_argument('--num_iter', dest='num_iter', default=3000, type=int)  # 网络迭代次数
    parser.add_argument('--optimizer', dest='optimizer', default='adam', type=str)  # 优化器
    parser.add_argument('--reg_noise_std', dest='reg_noise_std', default=0.03, type=float)
    parser.add_argument('--show_every', dest='show_every', default=50, type=int)
    parser.add_argument('--exp_weight', dest='exp_weight', default=0.99, type=float)
    parser.add_argument('--lr', dest='learning_rate', default=0.01, type=float)
    parser.add_argument('--skip_channel', dest='skip_channel', default=4, type=int)
    parser.add_argument('--up_channel', dest='up_channel', default=128, type=int)
    parser.add_argument('--down_channel', dest='down_channel', default=128, type=int)
    parser.add_argument('--upsample_mode', dest='upsample_mode', default='bilinear', type=str)
    parser.add_argument('--downsample_mode', dest='downsample_mode', default='stride', type=str)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.net == '2d':
        denoising2D.func(args)
        exit()
    elif args.net == 'red':
        denoising_red.func(args)
        exit()
    else:
        assert False
