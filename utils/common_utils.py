import warnings
from typing import Iterable

import scipy.io as sio
import torch
from matplotlib import pyplot as plt


def print_images(var_1, var_2, title=None):
    warnings.warn('该方法已弃用，生成了一个可以打印任意长度图片的方法', DeprecationWarning)
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 8))
    # 设置输出图片的标题
    if title is not None:
        f.suptitle(title)
    ax1.imshow(torch.stack((var_1[56, :, :], var_1[26, :, :], var_1[16, :, :]), 2).cpu())
    ax2.imshow(torch.stack((var_2[56, :, :], var_2[26, :, :], var_2[16, :, :]), 2).cpu())
    plt.show()


def print_image(images: list[torch.Tensor], title=None):
    """ 输出一组任意长度的(高光谱)图像

    :param images: list of images
    :param title: title of the image
    """

    f, axs = plt.subplots(nrows=1,
                          ncols=len(images),
                          sharey=True,
                          figsize=(5 if len(images) == 1 else 4 * len(images), 5))

    if title is not None:
        f.suptitle(title)

    if isinstance(axs, Iterable):
        for index, ax in enumerate(axs):
            var = images[index]
            ax.imshow(torch.stack((var[56, :, :], var[26, :, :], var[16, :, :]), 2).cpu())
    else:
        var = images[0]
        axs.imshow(torch.stack((var[56, :, :], var[26, :, :], var[16, :, :]), 2).cpu())
    plt.show()


def crop_image(img, d=32):
    """Make dimensions divisible by `d`"""

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def get_params(opt_over, net, net_input, downsampler=None):
    """Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
        downsampler:
    """
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:
        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    warnings.warn('该方法已弃用', DeprecationWarning)
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a torch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for filling tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicand by. Basically it is standard deviation scale.
    """
    warnings.warn('该方法已弃用，生成了一个更简单的', DeprecationWarning)
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    if method == '2D':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
    elif method == '3D':
        shape = [1, 1, input_depth, spatial_size[0], spatial_size[1]]
    else:
        assert False

    # get the size of the target noise tensor
    net_input = torch.zeros(shape)

    fill_noise(net_input, noise_type)
    net_input *= var

    return net_input


def noise_generator(method, shape: list):
    if method == '2D':
        shape.insert(0, 1)
    elif method == '3D':
        shape.insert(0, 1)
        shape.insert(0, 1)
    else:
        assert False

    noise_tensor = torch.zeros(shape)
    noise_tensor.uniform_()
    noise_tensor *= 0.1

    return noise_tensor


def optimize(optimizer_type, parameters, closure, learning_rate, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        learning_rate: learning rate
        num_iter: number of iterations 
    """

    # LBFGS should have been one of the options,
    # but it was not used in the literature
    if optimizer_type == 'LBFGS':
        # do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('starting optimization with LBFGS')

        def closure2():
            optimizer.zero_grad()
            return closure()

        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=learning_rate, tolerance_grad=-1,
                                      tolerance_change=-1)
        optimizer.step(closure2)
    elif optimizer_type == 'adam':
        # print('starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

        # iterative execution network
        for j in range(num_iter + 1):
            optimizer.zero_grad()  # clean gradient
            closure()  # execution model, which includes backwards
            optimizer.step()  # update the parameters of the model
    else:
        assert False


def _tensor_repeat(inputs, x, y):
    inputs = inputs.repeat(x, y, 1)
    inputs = torch.transpose(inputs, 0, 2)
    inputs = torch.transpose(inputs, 1, 2)
    return inputs


def max_min_normalize(inputs: torch.Tensor):
    max_tensor = torch.max(inputs, dim=1).values
    max_tensor = torch.max(max_tensor, dim=1).values
    max_tensor = _tensor_repeat(max_tensor, x=inputs.shape[1], y=inputs.shape[2])

    min_tensor = torch.min(inputs, dim=1).values
    min_tensor = torch.min(min_tensor, dim=1).values
    min_tensor = _tensor_repeat(min_tensor, x=inputs.shape[1], y=inputs.shape[2])

    return (inputs - min_tensor) / (max_tensor - min_tensor)


def _mean_squared_error(image0, image1):
    return torch.mean((image0 - image1) ** 2)


def psnr_gpu(image_true: torch.Tensor, image_test: torch.Tensor):
    if not image_true.shape == image_test.shape:
        print(image_true.shape)
        print(image_test.shape)
        raise ValueError('Input must have the same dimensions.')

    if image_true.dtype != image_test.dtype:
        raise TypeError("Inputs have mismatched dtype. Set both tensors to be of the same type.")

    true_max = torch.max(image_true)
    true_min = torch.min(image_true)
    if true_max > 1 or true_min < -1:
        raise ValueError("image_true has intensity values outside the range expected "
                         "for its data type. Please manually specify the data_range.")
    if true_min >= 0:
        # most common case (255 for uint8, 1 for float)
        data_range = 1
    else:
        data_range = 2

    err = _mean_squared_error(image_true, image_test)
    return (10 * torch.log10((data_range ** 2) / err)).item()


def read_data(file_name):
    data_type = torch.cuda.FloatTensor

    mat = sio.loadmat(file_name)
    image = mat['image']
    decrease_image = mat['image_noisy']

    image = torch.from_numpy(image).type(data_type)
    decrease_image = torch.from_numpy(decrease_image).type(data_type)

    return image, decrease_image
