import argparse

import denoising2D_gpu


def parse_args():
    parser = argparse.ArgumentParser(description='nas-dhsip')

    parser.add_argument('--net', dest='net', default='default', type=str)  # 网络选择
    parser.add_argument('--num_iter', dest='num_iter', default=3000, type=int)  # 网络迭代次数
    parser.add_argument('--optimizer', dest='optimizer', default='adam', type=str)  # 优化器
    parser.add_argument('--reg_noise_std', dest='reg_noise_std', default=0.03, type=float)
    parser.add_argument('--show_every', dest='show_every', default=50, type=int)
    parser.add_argument('--exp_weight', dest='exp_weight', default=0.99, type=float)
    parser.add_argument('--lr', dest='lr', default=0.01, type=float)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.net == '2d':
        denoising2D_gpu.func(args)
        exit()
    elif args.net == '3d':
        exit()
    else:
        assert False
