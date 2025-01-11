import einops as ei
import torch


def expect(tensor, shape, **axes_lengths):
    return ei.rearrange(tensor, f'{shape} -> {shape}', **axes_lengths)


def to_tensor(*args):
    if len(args) == 1:
        return torch.tensor(args[0])
    return tuple(torch.tensor(arg) for arg in args)


def to_device(*args: torch.Tensor, device: torch.device):
    if len(args) == 1:
        return args[0].to(device)
    else:
        return tuple(arg.to(device) for arg in args)


def get_device(module_or_tensor: torch.nn.Module | torch.Tensor | None = None):
    if module_or_tensor is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(module_or_tensor, torch.Tensor):
        return module_or_tensor.device
    else:
        return next(module_or_tensor.parameters()).device
