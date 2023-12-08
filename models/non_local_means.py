import numpy as np
import torch
from skimage.restoration import denoise_nl_means

from models.apply_2d_ops import nlm2d

has_cuda = torch.cuda.is_available()
print('has cuda: ', has_cuda)


def non_local_means(image: torch.Tensor, sigma, fast_mode=True) -> torch.Tensor:
    if has_cuda and image.is_cuda:
        image = image.cpu()
    image = image.detach_().numpy()

    sigma = sigma / 255.
    h = 0.6 * sigma if fast_mode else 0.8 * sigma
    channels = image.shape[0]
    denoise_img = []
    for num_channel in range(channels):
        temp = denoise_nl_means(image[num_channel, :, :], h=h, sigma=sigma,
                                fast_mode=fast_mode, patch_size=5, patch_distance=6)
        denoise_img += [temp]

    image = np.array(denoise_img, dtype=np.float32)

    return torch.from_numpy(image).cuda() if has_cuda else torch.from_numpy(image)


# 比原来的方法快了一倍多
def fast_non_local_means(image: torch.Tensor,
                         kernel_size: int = 3,
                         kernel_size_mean=3,
                         std: int = 1,
                         sub_filter_size: int = 256) -> torch.Tensor:
    channels = image.shape[0]
    denoise_img = []

    for num_channel in range(channels):
        temp = nlm2d(image[num_channel, :, :], kernel_size, kernel_size_mean, std, sub_filter_size)
        denoise_img += [temp]

    return torch.stack(denoise_img)
