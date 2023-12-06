import numpy as np
from skimage.restoration import denoise_nl_means


def non_local_means(image, sigma, fast_mode=True):
    sigma = sigma / 255.
    h = 0.6 * sigma if fast_mode else 0.8 * sigma
    channels = image.shape[0]
    # TODO 这里可以直接输入3D数据, 或许重写速度会更快？
    denoise_img = []
    for num_channel in range(channels):
        temp = denoise_nl_means(image[num_channel, :, :], h=h, sigma=sigma,
                                fast_mode=fast_mode, patch_size=5, patch_distance=6)
        denoise_img += [temp]

    return np.array(denoise_img, dtype=np.float32)
