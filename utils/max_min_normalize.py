import torch


def tensor_repeat(inputs):
    inputs = inputs.repeat(200, 200, 1)
    return torch.transpose(inputs, 0, 2)


def max_min_normalize(inputs: torch.Tensor):
    max_tensor = torch.max(inputs, dim=1).values
    max_tensor = torch.max(max_tensor, dim=1).values
    max_tensor = tensor_repeat(max_tensor)

    min_tensor = torch.max(inputs, dim=1).values
    min_tensor = torch.max(min_tensor, dim=1).values
    min_tensor = tensor_repeat(min_tensor)

    return (inputs - min_tensor) / (max_tensor - min_tensor)
