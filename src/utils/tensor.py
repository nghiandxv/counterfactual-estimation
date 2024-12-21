import einops as ei


def expect_shape(tensor, shape, **axes_lengths):
    return ei.rearrange(tensor, f'{shape} -> {shape}', **axes_lengths)
