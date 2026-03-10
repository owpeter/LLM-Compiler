import torch

import ntops
from ntops.torch.utils import _cached_make


def causal_softmax(input, dtype=None):
    tensor_dtype = dtype if dtype is not None else input.dtype
    output = torch.empty_like(input, dtype=tensor_dtype)

    kernel = _cached_make(ntops.kernels.causal_softmax.premake, input.ndim)

    kernel(input, output)

    return output
