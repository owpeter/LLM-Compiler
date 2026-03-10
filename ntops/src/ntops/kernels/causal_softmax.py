import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def _exp(x, dtype):
    exp_dtype = (
        ntl.float32
        if dtype == ntl.float16
        else (ntl.float32 if dtype == ntl.bfloat16 else dtype)
    )
    return ntl.cast(ntl.exp(ntl.cast(x, exp_dtype)), dtype)


def application(input, output):
    dtype = output.dtype.dtype
    prev_max = ntl.cast(float("-inf"), dtype)
    denominator = ntl.cast(0, dtype)
    offset = input.source.shape[-1] - input.source.shape[-2]

    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], dtype)
        mask = input[i].offsets(-1) <= input[i].offsets(-2) + offset
        masked_input_i = ntl.where(mask, input_i, float("-inf"))
        curr_max = ntl.cast(ntl.maximum(prev_max, ntl.max(masked_input_i)), dtype)
        input_max_diff_exp = _exp(masked_input_i - curr_max, dtype)
        prev_curr_max_diff_exp = _exp(prev_max - curr_max, dtype)
        denominator = denominator * prev_curr_max_diff_exp + ntl.sum(input_max_diff_exp)
        prev_max = curr_max

    for i in range(input.shape[0]):
        mask = input[i].offsets(-1) <= input[i].offsets(-2) + offset
        numerator = _exp(input[i] - prev_max, dtype)
        output[i] = ntl.where(mask, numerator / denominator, 0)


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=-1, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype, other=float("-inf")),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
