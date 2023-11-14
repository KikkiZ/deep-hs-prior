import torch


def mean_squared_error(image0, image1):
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

    err = mean_squared_error(image_true, image_test)
    return (10 * torch.log10((data_range ** 2) / err)).item()
