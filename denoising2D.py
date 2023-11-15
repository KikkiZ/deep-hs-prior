import time

import scipy.io as sio
from skimage.metrics import peak_signal_noise_ratio

from models import *
from utils.common_utils import *
from utils.denoising_utils import *

print('import success...')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
data_type = torch.cuda.FloatTensor

sigma = 100
sigma_ = sigma / 255.

# ensure dimensions [0][1] are divisible by 32 (or 2^depth)!
file_name = 'data/denoising/denoising.mat'
mat = sio.loadmat(file_name)
image = mat['image']
decrease_image = mat['image_noisy']

image = torch.from_numpy(image).type(data_type)
# img_noisy_np = get_noisy_image(image, sigma_)
decrease_image = torch.from_numpy(decrease_image).type(data_type)

print('origin image shape:', image.shape)
print('decrease image shape:', decrease_image.shape)
print_images(image, decrease_image)
print('load data success...')

method = '2D'
pad = 'reflection'
OPT_OVER = 'net'  # 'net,input'

reg_noise_std = 0.03  # 0 0.01 0.05 0.08
learning_rate = 0.01

OPTIMIZER = 'adam'  # 'LBFGS'
exp_weight = 0.99

show_every = 200
save_every = 200

# number of network iterations
num_iter = 2000
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
           pad=pad,
           act_fun='LeakyReLU').type(data_type)
# print network structure
# print(net)
# generates a noise tensor of a specified size
net_input = get_noise(image.shape[0], method, (image.shape[1], image.shape[2])).type(data_type).detach()

# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()])
print('number of params: ', s)

# Loss function
loss_func = torch.nn.MSELoss().type(data_type)

# Extend the dimension of the tensor
# from [191, 200, 200] to [1, 191, 200, 200]
decrease_image = decrease_image[None, :].cuda()
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

i = 0
out_avg = None
last_net = None
psnr_noisy_last = 0

copy_image = image.cpu().numpy().astype(np.float32)
copy_image = copy_image.squeeze()
copy_decrease_image = np.array(decrease_image.cpu().numpy().astype(np.float32), copy=True)
copy_decrease_image = copy_decrease_image.squeeze()
print(copy_image.shape, copy_decrease_image.shape)


def closure():
    # declare the following variables as global variables
    global i, out_avg, psnr_noisy_last, last_net, net_input

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)  # the result of a network iteration

    # Smoothing
    # don't know why smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    total_loss = loss_func(out, decrease_image)  # calculate the loss value of the loss function
    total_loss.backward()                        # back propagation gradient calculation

    out_np = out.detach().cpu().squeeze().numpy()
    out_avg_np = out_avg.detach().cpu().squeeze().numpy()
    # print(type(out_np), type(out_avg_np))

    psnr_noisy = peak_signal_noise_ratio(copy_decrease_image, np.clip(out_np, 0, 1))
    psnr_gt = peak_signal_noise_ratio(copy_image, np.clip(out_np, 0, 1))
    psnr_gt_sm = peak_signal_noise_ratio(copy_image, np.clip(out_avg_np, 0, 1))

    if i % 10 == 0:
        print('iteration', i, ' loss ', total_loss.item(), ' PSNR_gt: ', psnr_gt, ' PSNR_gt_sm: ', psnr_gt_sm)

    if i % show_every == 0:
        out_np = np.clip(out_np, 0, 1)
        out_avg_np = np.clip(out_avg_np, 0, 1)

        print_images(torch.Tensor(out_np), torch.Tensor(out_avg_np))

    if i % save_every == 0:
        sio.savemat("results/result_denoising_2D_it%05d.mat" % i, {'pred': out_np.transpose(1, 2, 0),
                                                                   'pred_avg': out_avg_np.transpose(1, 2, 0)})

    # Backtracking
    if i % show_every:
        if psnr_noisy - psnr_noisy_last < -5:
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.detach().copy_(new_param.cuda())

            return total_loss * 0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psnr_noisy_last = psnr_noisy

    i += 1

    return total_loss


# input: 'net', skip net, __
p = get_params(OPT_OVER, net, net_input)

start_time = time.time()
optimize(OPTIMIZER, p, closure, learning_rate, num_iter)
end_time = time.time()
print('cost time', end_time - start_time)
