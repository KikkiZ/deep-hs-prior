import time

import scipy.io as sio
from torch.utils.tensorboard import SummaryWriter

from denoising2D import print_images
from models import *
from utils.denoising_utils import *
from utils.max_min_normalize import max_min_normalize
from utils.psnr import psnr_gpu

print('import success...')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
data_type = torch.cuda.FloatTensor

# load data
file_name = 'data/denoising/denoising.mat'
mat = sio.loadmat(file_name)
image = mat['image']
decrease_image = mat['image_noisy']

image = torch.from_numpy(image).type(data_type)
decrease_image = torch.from_numpy(decrease_image).type(data_type)
print_images(image, decrease_image)

copy_image = image.clone().detach()
copy_decrease_image = decrease_image.clone().detach()

print('load data success...')

reg_noise_std = 0.03  # 0 0.01 0.05 0.08
learning_rate = 0.01
exp_weight = 0.99
show_every = 200
save_every = 200
num_iter = 2000       # number of network iterations

# build the network
net = skip(image.shape[0],
           image.shape[0],
           num_channels_up=[128] * 5,
           num_channels_down=[128] * 5,
           num_channels_skip=[4] * 5,
           filter_size_up=3,
           filter_size_down=3,
           filter_size_skip=1,
           upsample_mode='bilinear',  # downsample_mode='avg',
           need1x1_up=False,
           need_sigmoid=False,
           need_bias=True,
           pad='reflection',
           act_fun='LeakyReLU').type(data_type)
device = torch.device('cuda')
net.to(device)
print('module running in: ', device)

# compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()])
print('number of params: ', s)
print('initialize model success...')

# loss function
loss_func = torch.nn.MSELoss().type(data_type)

# extend the dimension of the tensor
# from [191, 200, 200] to [1, 191, 200, 200]
decrease_image = decrease_image[None, :].cuda()
# generates a noise tensor of a specified size
net_input = get_noise(image.shape[0], '2D', (image.shape[1], image.shape[2])).type(data_type).detach()
net_input_saved = net_input.detach().clone()  # clone the noise tensor without grad
noise = net_input.detach().clone()            # clone twice

i = 0                                         # the number of iterations of the model
out_avg = None                                # the output from the previous iteration
last_net = None                               # the parameters during model iteration
psnr_noisy_last = 0                           # psnr of the previous iteration
writer = SummaryWriter('./logs/denoising')    # the location where the data record is saved


def closure():
    # declare the following variables as global variables
    global i, out_avg, psnr_noisy_last, last_net, net_input

    # normal_() is used to generate a tensor of a specified size that follows a normal distribution
    # this step adds some random noise to the noise tensor, preventing the model from overfitting
    net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)  # the result of a network iteration

    # smoothing
    # combine the results of the previous run with this one
    # which can make the results more stable
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    total_loss = loss_func(out, decrease_image)  # calculate the loss value of the loss function
    total_loss.backward()                        # back propagation gradient calculation

    psnr_noisy = psnr_gpu(copy_decrease_image, out.squeeze())
    psnr_gt = psnr_gpu(copy_image, out.squeeze())
    psnr_gt_sm = psnr_gpu(copy_image, out_avg.squeeze())

    writer.add_scalar('compare with de', psnr_noisy, i)
    writer.add_scalar('compare with gt', psnr_gt, i)
    writer.add_scalar('compare with gt_sm', psnr_gt_sm, i)

    if i % save_every == 0:  # save model output during iteration
        sio.savemat('results/result_denoising_2D_it%05d.mat' % i, {'pred': out.transpose(2, 0).detach().cpu(),
                                                                   'pred_avg': out_avg.transpose(2, 0).detach().cpu()})

    # backtracking
    if i % show_every == 0:
        out = torch.clamp(out, 0, 1)
        out_avg = torch.clamp(out_avg, 0, 1)

        out_normalize = max_min_normalize(out.squeeze().detach())
        out_avg_normalize = max_min_normalize(out_avg.squeeze().detach())
        print_images(out_normalize, out_avg_normalize)

        if psnr_noisy - psnr_noisy_last < -5:                           # model produced an overfit
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.detach().copy_(new_param.cuda())              # copy the paras from saved to model

            return total_loss * 0                                       # clean the loss
        else:                                                           # psnr volatility is still within expectations
            last_net = [x.detach().cpu() for x in net.parameters()]     # save the parameters in the model
            psnr_noisy_last = psnr_noisy                                # renew psnr_noisy_last

    i += 1

    return total_loss


params = get_params('net', net, net_input)

print('start iteration...')
start_time = time.time()

optimize('adam', params, closure, learning_rate, num_iter)

writer.close()
end_time = time.time()
print('cost time', end_time - start_time, 's')
