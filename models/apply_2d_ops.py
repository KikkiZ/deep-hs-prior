"""
Functions for 2d NLM using torch.
"""

__author__ = "José Guilherme de Almeida"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "José Guilherme de Almeida"
__email__ = "jose.almeida@research.fchampalimaud.org"

import torch

import torch.nn.functional as func
from itertools import product

from functorch import einops
from tqdm import tqdm
from models.base_ops import unsqueeze_tensor_at_dim, apply_mean_filter, get_neighbours, non_local_means_loop_index


def apply_nonlocal_means_2d(image: torch.Tensor,
                            kernel_size: int = 3,
                            std: float = 1,
                            kernel_size_mean=3):
    """
    Calculates non-local means for an input X with 2 dimensions.

    Args:
        image (torch.Tensor): input tensor with shape [h,w].
        kernel_size (int, optional): size of neighbourhood. Defaults to 3.
        std (float, optional): standard deviation for weights. Defaults to 1.0.
        kernel_size_mean (int, optional): kernel size for the initial mean
        filtering.

    Returns:
        torch.Tensor: non-local mean-normalised input.
    """
    dim = 2
    # include batch and channel dimensions
    image = unsqueeze_tensor_at_dim(image, dim + 2)
    padding = (
        kernel_size // 2, kernel_size // 2,
        kernel_size // 2, kernel_size // 2)
    # apply mean filter
    image = apply_mean_filter(image, kernel_size_mean, dim)
    # retrieve neighbourhood
    neighbours_x = get_neighbours(func.pad(image, padding), kernel_size=kernel_size, dim=dim)
    distances = torch.sqrt(torch.square(image - neighbours_x))
    weights = torch.exp(- distances / std ** 2)
    weights = weights / weights.sum(1, keepdim=True)
    output = (neighbours_x * weights).sum(1).squeeze(0)
    return output


def apply_windowed_nonlocal_means_2d(image: torch.Tensor,
                                     kernel_size: int = 3,
                                     std: int = 1,
                                     kernel_size_mean=3,
                                     window_size=None,
                                     strides=None):
    """
    Calculates non-local means for an input X with 2 dimensions by calculating 
    a local (windowed) non-local means. Leads to artefacts because of this - 
    not perfect as an implementation, kept mostly as a curiosity.

    Args:
        image (torch.Tensor): input tensor with shape [h,w].
        kernel_size (int, optional): size of neighbourhood. Defaults to 3.
        std (float, optional): standard deviation for weights. Defaults to 1.0.
        kernel_size_mean (int, optional): kernel size for the initial mean
            filtering.
        window_size (Tuple[int,int], optional): size of window. Defaults to 
            [128,128].
        strides (Tuple[int,int], optional): size of window. Defaults to 
            [64,64].

    Returns:
        torch.Tensor: non-local mean-normalised input.
    """
    if strides is None:
        strides = [64, 64]
    if window_size is None:
        window_size = [128, 128]

    dim = 2
    output = torch.zeros_like(image)
    denominator = torch.zeros_like(image)
    # include batch and channel dimensions
    image = unsqueeze_tensor_at_dim(image, dim + 2)
    sh = image.shape
    b, c, h, w = sh[0], sh[1], sh[2], sh[3]
    padding = (
        kernel_size // 2, kernel_size // 2,
        kernel_size // 2, kernel_size // 2)
    image = apply_mean_filter(image, kernel_size_mean, dim)
    image = func.pad(image, padding)
    neighbours_x = get_neighbours(image, kernel_size=kernel_size, dim=dim)
    image = image[:, :, padding[0]:(image.shape[2] - padding[0]), padding[1]:(image.shape[3] - padding[1])]

    all_ij = list(product(range(0, h, strides[0]), range(0, h, strides[1])))

    for i, j in tqdm(all_ij):
        i_1, i_2 = i, i + window_size[0]
        j_1, j_2 = j, j + window_size[1]
        if i_2 > h:
            i_1, i_2 = h - window_size[0], h
        if j_2 > w:
            j_1, j_2 = w - window_size[1], w
        # reshape X to calculate distances
        sub_x = image[:, :, i_1:i_2, j_1:j_2]
        sub_neighbours = neighbours_x[:, :, i_1:i_2, j_1:j_2]
        reshaped_neighbours_x = einops.rearrange(
            sub_neighbours, "b c h w -> b (h w) c")
        # calculate distances
        neighbour_dists = torch.cdist(
            reshaped_neighbours_x, reshaped_neighbours_x)
        # calculate weights from distances
        weights = torch.exp(-neighbour_dists / (std ** 2))
        # calculate the new values 
        flat_x = einops.rearrange(sub_x, "b c h w -> b (h w) c")
        weighted_x = (weights @ flat_x).squeeze(-1)
        output[i_1:i_2, j_1:j_2] += einops.rearrange(
            weighted_x, "b (h w) -> b h w",
            b=b, h=window_size[0], w=window_size[1]).squeeze(0)
        denominator[i_1:i_2, j_1:j_2] += einops.rearrange(
            weights.sum(-1), "b (h w) -> b h w",
            b=b, h=window_size[0], w=window_size[1]).squeeze(0)
    output = output / denominator
    return output


def apply_nonlocal_means_2d_mem_efficient(image: torch.Tensor,
                                          kernel_size: int = 3,
                                          std: int = 1,
                                          kernel_size_mean=3,
                                          sub_filter_size: int = 256):
    """
    Calculates non-local means using a for loop to select at each iteration a
    different set of neighbours. Most of the heavy lifting is performed by
    non_local_means_loop.

    Args:
        image (torch.Tensor): input tensor with shape [h,w].
        kernel_size (int, optional): size of neighbourhood. Defaults to 3.
        std (float, optional): standard deviation for weights. Defaults to 1.0.
        kernel_size_mean (int, optional): kernel size for the initial mean
            filtering.
        sub_filter_size (int, optional): approximate size of neighbourhood set
            at each iteration. Defaults to 1 (regular non-local means).

    Returns:
        torch.Tensor: non-local mean-normalised input.
    """
    dim = 2
    # include batch and channel dimensions
    image = unsqueeze_tensor_at_dim(image, dim + 2)
    # apply mean filter
    image = apply_mean_filter(image, kernel_size_mean, dim)
    # retrieve neighbourhood
    output = non_local_means_loop_index(
        image, kernel_size=kernel_size, std=std,
        dim=dim, sub_filter_size=sub_filter_size).squeeze(0).squeeze(0)
    return output


nlm2d = apply_nonlocal_means_2d_mem_efficient
