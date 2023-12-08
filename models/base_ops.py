"""
Generic operations for torch_nlm.
"""

__author__ = "José Guilherme de Almeida"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "José Guilherme de Almeida"
__email__ = "jose.almeida@research.fchampalimaud.org"

import numpy as np
import torch
import torch.nn.functional as func
from itertools import product
from tqdm import tqdm
from typing import Tuple, List


def get_gaussian_kernel(kernel_size: int = 5, sigma: float = 1., dim: int = 2) -> np.ndarray:
    """Creates gaussian kernel with side length kernel_size and a standard 
    deviation of sigma.

    Based on: https://stackoverflow.com/a/43346070

    Args:
        kernel_size (int, optional): size of kernel. Defaults to 5.
        sigma (float, optional): sigma for the normal distribution. Defaults 
            to 1.
        dim (int, optional): number of dimensions in output kernel.

    Returns:
        np.ndarray: Gaussian filter kernel.
    """
    ax = torch.linspace(-(kernel_size - 1) / 2.,
                        (kernel_size - 1) / 2.,
                        kernel_size)
    gauss = torch.exp(-0.5 * np.square(ax) / np.square(sigma))
    if dim == 1:
        kernel = gauss
    elif dim == 2:
        kernel = torch.outer(gauss, gauss)
    elif dim == 3:
        kernel = gauss[None, None, :] * gauss[None, :, None] * gauss[:, None, None]
    else:
        assert False

    return kernel / torch.sum(kernel)


def unsqueeze_tensor_at_dim(image: torch.Tensor, n_dim: int, dim: int = 0) -> torch.Tensor:
    """
    Adds dimensions as necessary.

    Args:
        image (torch.Tensor): tensor.
        n_dim (int): number of output dimensions.
        dim (int): dimension which will be unsqueezed. Defaults to 0.

    Returns:
        torch.Tensor: unqueezed tensor.
    """
    sh = len(image.shape)
    diff = n_dim - sh
    if diff > 0:
        for _ in range(diff):
            image = image.unsqueeze(0)
    return image


def make_neighbours_kernel(kernel_size: int = 3, dim: int = 2) -> torch.Tensor:
    """
    Make convolutional kernel that extracts neighbours within a kernel_size
        neighbourhood (each filter is 1 for the corresponding neighbour and
        0 otherwise). 

    Args:
        kernel_size (int, optional): size of the neighbourhood. Defaults to 3.
        dim (int, optional): number of dimensions. Only 2 or 3 possible.
            Defaults to 2.

    Returns:
        torch.Tensor: convolutional kernel for neighbourhood extraction.
    """
    k = kernel_size ** dim
    filters = torch.zeros([k, 1, *[kernel_size for _ in range(dim)]])
    generators = [range(kernel_size) for _ in range(dim)]
    for i, coord in enumerate(product(*generators)):
        if dim == 2:
            filters[i, :, coord[0], coord[1]] = 1
        elif dim == 3:
            filters[i, :, coord[0], coord[1], coord[2]] = 1
    return filters


def get_neighbours(image: torch.Tensor,
                   kernel_size: int = 3,
                   dim: int = 2) -> torch.Tensor:
    """
    Retrieves neighbours in an image and stores them in the channel dimension.
    Expects the input to be padded.

    Args:
        image (torch.Tensor): 4-D or 5-D (batched) tensor.
        kernel_size (int, optional): size of the neighbourhood. Defaults to 3.
        dim (int, optional): number of dimensions. Only 2 or 3 possible.
            Defaults to 2.

    Returns:
        torch.Tensor: Gaussian filter-normalised X.
    """
    filters = make_neighbours_kernel(kernel_size, dim).to(image)
    if dim == 2:
        image = func.conv2d(image, filters, padding=0)
    elif dim == 3:
        image = func.conv3d(image, filters, padding=0)
    else:
        raise NotImplementedError("dim must be 2 or 3.")
    return image


def apply_gaussian_filter(image: torch.Tensor,
                          kernel_size: int = 3,
                          dim: int = 2,
                          sigma: float = 1.0) -> torch.Tensor:
    """
    Simple function to apply Gaussian filter.

    Args:
        image (torch.Tensor): input tensor.
        kernel_size (int, optional): size of the neighbourhood. Defaults to 3.
        dim (int, optional): number of dimensions. Only 2 or 3 possible.
            Defaults to 2.
        sigma (float, optional): standard deviation for the filter. Defaults to
            1.0.

    Returns:
        torch.Tensor: mean filter-normalised X.
    """
    image = unsqueeze_tensor_at_dim(image, n_dim=dim + 2)
    n_channels = image.shape[1]
    gaussian_kernel = get_gaussian_kernel(kernel_size, sigma, dim)
    gaussian_kernel = torch.from_numpy(gaussian_kernel)
    gaussian_kernel = unsqueeze_tensor_at_dim(gaussian_kernel, n_dim=dim)
    gaussian_kernel = torch.cat([gaussian_kernel for _ in range(n_channels)], 0)
    gaussian_kernel = torch.cat([gaussian_kernel for _ in range(n_channels)], 1)
    if dim == 2:
        image = func.conv2d(image, gaussian_kernel, bias=0, padding=kernel_size // 2)
    elif dim == 3:
        image = func.conv3d(image, gaussian_kernel, bias=0, padding=kernel_size // 2)
    else:
        raise NotImplementedError("dim must be 2 or 3.")
    return image


def apply_mean_filter(image: torch.Tensor,
                      kernel_size: int = 3,
                      dim: int = 2):
    """
    Simple function to apply mean filter.

    Args:
        image (torch.Tensor): input tensor.
        kernel_size (int, optional): size of the neighbourhood. Defaults to 3.
        dim (int, optional): number of dimensions. Only 2 or 3 possible. 
            Defaults to 2.

    Returns:
        torch.Tensor: _description_
    """
    filters = torch.ones([kernel_size for _ in range(dim)]).to(image)
    filters = filters / filters.sum()
    filters = unsqueeze_tensor_at_dim(filters, dim + 2)
    pad = kernel_size // 2
    padding = tuple([pad for _ in range(dim * 2)])
    image = func.pad(image, padding, mode="reflect")
    if dim == 2:
        image = func.conv2d(image, filters, padding=0)
    if dim == 3:
        image = func.conv3d(image, filters, padding=0)
    return image


def array_chunk(arr: np.ndarray, chunk_size: int):
    for i in range(0, arr.shape[0], chunk_size):
        yield arr[i:i + chunk_size]


def non_local_means_loop_index(image: torch.Tensor,
                               kernel_size: int = 3,
                               dim: int = 2,
                               sub_filter_size: int = 1,
                               std: float = 1.0) -> torch.Tensor:
    """
    Calculates non-local means using a for loop to select at each iteration a
    different set of neighbours. This avoids storing at any given stage a large 
    number of neighbours, making the calculation of larger neighbourhoods 
    possible without running the risk of OOM. Here, neighbours are selected 
    using simple indexing.

    Args:
        image (torch.Tensor): input tensor.
        kernel_size (int, optional): size of neighbourhood. Defaults to 3.
        dim (int, optional): number of dimensions. Must be 2 or 3. Defaults to
            2.
        sub_filter_size (int, optional): approximate size of neighbourhood set
            at each iteration. Defaults to 1 (regular non-local means).
        std (float, optional): standard deviation for weights. Defaults to 1.0.

    Returns:
        torch.Tensor: non-local mean-normalised input.
    """

    def cat_index(index: List[torch.Tensor]):
        for i in range(len(index)):
            index[i] = torch.cat(index[i])
        return index

    def preprocess_index(index: List[torch.Tensor],
                         original_index: List[torch.Tensor],
                         sizes: List[int]):
        for i in range(len(index)):
            d = torch.abs(original_index[i] - index[i])
            index[i] = torch.where(index[i] < 0, original_index[i] + d, index[i])
            index[i] = torch.where(index[i] > sizes[i] - 1, original_index[i] - d, index[i])
        return index

    def calculate_weights_2d(img: torch.Tensor,
                             index: List[torch.Tensor],
                             std2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = len(index[0])
        index = cat_index(index)
        neighbour = img[:, :, index[0], index[1]].reshape(
            1, n, img.shape[2], img.shape[3])
        weight = torch.square(img - neighbour).negative().divide(std2).exp()
        return weight, neighbour

    def calculate_weights_3d(img: torch.Tensor,
                             index: List[torch.Tensor],
                             std2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = len(index[0])
        index = cat_index(index)
        neighbour = img[:, :, index[0], index[1], index[2]].reshape(
            1, n, img.shape[2], img.shape[3], img.shape[4])
        weight = torch.square(
            img - neighbour).negative().divide(std2).exp()
        return weight, neighbour

    weights_sum = torch.zeros_like(image)
    output = torch.zeros_like(image)
    k2 = kernel_size // 2
    std_2 = torch.as_tensor(std ** 2).to(image)
    if dim == 2:
        _, _, upper_h, upper_w = image.shape
        h, w = torch.where(image[0, 0] == image[0, 0])
        all_index = [[], []]
        range_h = torch.arange(-k2, k2 + 1, dtype=torch.long)
        range_w = torch.arange(-k2, k2 + 1, dtype=torch.long)
        for i in range_h:
            for j in range_w:
                new_index = preprocess_index(
                    [h + i, w + j], [h, w], [upper_h, upper_w])
                all_index[0].append(new_index[0])
                all_index[1].append(new_index[1])
                if len(all_index[0]) >= sub_filter_size:
                    weights, neighbours = calculate_weights_2d(image, all_index, std_2)
                    weights_sum += torch.sum(weights, 1)
                    output += weights.multiply(neighbours).sum(1)
                    all_index = [[], []]
        if len(all_index[0]) >= 0:
            weights, neighbours = calculate_weights_2d(image, all_index, std_2)
            weights_sum += torch.sum(weights, 1)
            output += weights.multiply(neighbours).sum(1)

    if dim == 3:
        _, _, upper_h, upper_w, upper_d = image.shape
        h, w, d = torch.where(image[0, 0] == image[0, 0])
        all_index = [[], [], []]
        range_h = torch.arange(-k2, k2 + 1, dtype=torch.long)
        range_w = torch.arange(-k2, k2 + 1, dtype=torch.long)
        range_d = torch.arange(-k2, k2 + 1, dtype=torch.long)
        for i in range_h:
            for j in range_w:
                for k in range_d:
                    new_index = preprocess_index(
                        [h + i, w + j, d + k], [h, w, d], [upper_h, upper_w, upper_d])
                    all_index[0].append(new_index[0])
                    all_index[1].append(new_index[1])
                    all_index[2].append(new_index[2])
                    if len(all_index[0]) >= sub_filter_size:
                        weights, neighbours = calculate_weights_3d(image, all_index, std_2)
                        weights_sum += torch.sum(weights, 1)
                        output += weights.multiply(neighbours).sum(1)
                        all_index = [[], [], []]
        if len(all_index[0]) >= 0:
            weights, neighbours = calculate_weights_3d(image, all_index, std_2)
            weights_sum += torch.sum(weights, 1)
            output += weights.multiply(neighbours).sum(1)

    return image


def non_local_means_loop(image: torch.Tensor,
                         kernel_size: int = 3,
                         dim: int = 2,
                         sub_filter_size: int = 1,
                         std: float = 1.0) -> torch.Tensor:
    """
    Calculates non-local means using a for loop to select at each iteration a
    different set of neighbours. This avoids storing at any given stage a large 
    number of neighbours, making the calculation of larger neighbourhoods 
    possible without running the risk of OOM. Here, neighbours are selected 
    using convolutional filters.

    Args:
        image (torch.Tensor): input tensor.
        kernel_size (int, optional): size of neighbourhood. Defaults to 3.
        dim (int, optional): number of dimensions. Must be 2 or 3. Defaults to
            2.
        sub_filter_size (int, optional): approximate size of neighbourhood set
            at each iteration. Defaults to 1 (regular non-local means).
        std (float, optional): standard deviation for weights. Defaults to 1.0.

    Returns:
        torch.Tensor: non-local mean-normalised input.
    """
    filters = make_neighbours_kernel(kernel_size, dim).to(image)
    pad_size = kernel_size // 2
    padding = tuple([pad_size for _ in range(4)])
    weights_sum = torch.zeros_like(image)
    output = torch.zeros_like(image)
    n_filters = filters.shape[0]
    if sub_filter_size > n_filters:
        blocks = [np.arange(n_filters, dtype=int)]
    else:
        blocks = list(
            array_chunk(np.arange(n_filters, dtype=int), sub_filter_size))
    padded_x = func.pad(image, padding)
    std_2 = torch.as_tensor(std ** 2).to(image)
    with torch.no_grad():
        for block in tqdm(blocks):
            neighbours = func.conv2d(
                padded_x, filters[block], padding=0)
            weights = torch.square(image - neighbours).negative().divide(std_2).exp()
            weights_sum += torch.sum(weights, 1)
            output += weights.multiply(neighbours).sum(1)
    output = output / weights_sum
    return output
