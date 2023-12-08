import datetime
import time

from torch.autograd import profiler
from torch.utils.tensorboard import SummaryWriter

from models.non_local_means import fast_non_local_means, non_local_means
from models.unet2D import UNet
from utils.common_utils import *
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
data_type = torch.cuda.FloatTensor


def func(args):
    file_name = './data/denoising.mat'
    image, decrease_image = read_data(file_name)
    print_image([image, decrease_image], 'origin image')

    reg_noise_std = args.reg_noise_std  # 扰动噪声张量的常量
    learning_rate = args.learning_rate  # 学习率
    exp_weight = args.exp_weight        # 平滑参数, 将网络输出与过往输出值
    show_every = args.show_every
    num_iter = args.num_iter            # 模型迭代次数
    mu = 0.5                            # ADMM参数，希腊字母μ
    beta = 0.5                          # 正则化参数

    net = UNet(image.shape[0],
               image.shape[0],
               num_channels_up=[args.up_channel] * 5,
               num_channels_down=[args.down_channel] * 5,
               num_channel_skip=args.skip_channel,
               kernel_size_up=3,
               kernel_size_down=3,
               kernel_size_skip=3,
               upsample_mode=args.upsample_mode,
               need1x1_up=False,
               need_sigmoid=False,
               need_bias=True,
               pad='reflection',
               activate='LeakyReLU').type(data_type)
    device = torch.device('cuda')
    net.to(device)
    print('module running in: ', device)

    s = sum([np.prod(list(p.size())) for p in net.parameters()])  # 计算模型参数
    print('number of params: ', s)
    print('memory occupied by parameter: ', s * 4 / 1024 / 1024, 'MB')

    # 该值代表损失函数约束的拉格朗日乘子张量
    lagrange_multiplier = torch.zeros(image.shape).type(data_type)[None, :].cuda()
    # 该值初始以退化图像作为基准值, 后续会不断更新该值
    benchmark_image = decrease_image.clone().type(data_type)[None, :].cuda()
    temp_benchmark = benchmark_image.clone()  # 基准值的副本, 用于中间计算
    out_avg = None                            # 用于记录平滑输出的累计值

    # 拓展数据的维度
    decrease_image = decrease_image[None, :].cuda()
    # 自动生成一个指定维度的噪声图像
    net_input = noise_generator('2D', list(image.shape)).type(data_type).detach()
    net_input_saved = net_input.clone().detach_()  # clone the noise tensor without grad
    noise = net_input.clone().detach_()            # clone twice

    date = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
    writer = SummaryWriter('./logs/denoising_red/' + date)  # the location where the data record is saved

    print('start iteration...')
    start_time = time.time()

    criterion = torch.nn.MSELoss().type(data_type)                    # 损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 优化器

    # 运行基准函数, 更新基准值
    benchmark_image = non_local_means(benchmark_image.clone().squeeze(), 3)

    for i in range(num_iter + 1):
        optimizer.zero_grad()  # 清空梯度

        inputs = net_input_saved + (noise.normal_() * reg_noise_std)  # 扰动每一轮的网络输入噪声
        out = net(inputs)

        temp = benchmark_image - lagrange_multiplier
        loss_net = criterion(out, decrease_image)  # 网络的loss
        loss_red = criterion(out, temp.detach_())  # RED's loss

        total_loss = loss_net + mu * loss_red

        total_loss.backward()
        optimizer.step()  # update the parameters of the model

        # 模型每迭代一定次数更新一次基准值
        if i % 25 == 0:
            temp_benchmark = non_local_means(benchmark_image.clone().squeeze(), 3)

            msg = 'benchmark: [' + str(i) + '/' + str(num_iter) + ']'
            benchmark_image_normalize = max_min_normalize(benchmark_image.squeeze().detach())
            lagrange_multiplier_normalize = max_min_normalize(lagrange_multiplier.squeeze().detach())
            print_image([benchmark_image_normalize, lagrange_multiplier_normalize], msg)

        benchmark_image = 1 / (beta + mu) * (beta * temp_benchmark + mu * (out + lagrange_multiplier))
        lagrange_multiplier = lagrange_multiplier + out - benchmark_image

        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        psnr_noisy = psnr_gpu(decrease_image.squeeze(), out.squeeze())
        psnr_gt = psnr_gpu(image, out.squeeze())
        psnr_gt_sm = psnr_gpu(image, out_avg.squeeze())
        psnr_temp = psnr_gpu(image, benchmark_image.squeeze())
        psnr_temp_u = psnr_gpu(image, (benchmark_image - lagrange_multiplier).squeeze())

        writer.add_scalar('compare with de', psnr_noisy, i)
        writer.add_scalar('compare with gt', psnr_gt, i)
        writer.add_scalar('compare with gt_sm', psnr_gt_sm, i)
        writer.add_scalar('compare with temp', psnr_temp, i)
        writer.add_scalar('compare with temp_u', psnr_temp_u, i)

        if i % show_every == 0:
            msg = 'iteration times: [' + str(i) + '/' + str(num_iter) + ']'
            print(msg)

            out_normalize = max_min_normalize(out.squeeze().detach())
            out_avg_normalize = max_min_normalize(out_avg.squeeze().detach())
            print_image([out_normalize, out_avg_normalize], msg)

    writer.close()
    end_time = time.time()
    print('cost time', end_time - start_time, 's')
