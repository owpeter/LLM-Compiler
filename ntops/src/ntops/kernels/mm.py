import enum
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()
BLOCK_SIZE_K = ninetoothed.block_size()


class InputPrecisionVariant(enum.IntEnum):
    TF32 = enum.auto()

    IEEE = enum.auto()


def arrangement(
    input,
    other,
    output,
    input_precision,
    unroll,
    beta,
    has_bias,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N

    if block_size_k is None:
        block_size_k = BLOCK_SIZE_K

    output_arranged = output.tile((block_size_m, block_size_n))

    input_arranged = input.tile((block_size_m, block_size_k))
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    other_arranged = other.tile((block_size_k, block_size_n))
    other_arranged = other_arranged.tile((-1, 1))
    other_arranged = other_arranged.expand((output_arranged.shape[0], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze(1)

    input_precision_arranged = input_precision
    unroll_arranged = unroll
    beta_arranged = beta
    has_bias_arranged = has_bias

    return (
        input_arranged,
        other_arranged,
        output_arranged,
        input_precision_arranged,
        unroll_arranged,
        beta_arranged,
        has_bias_arranged,
    )


def application(input, other, output, input_precision, unroll, beta, has_bias):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    if has_bias == 1:
        accumulator += ntl.cast(beta, ntl.float32) * ntl.cast(output, ntl.float32)

    if input_precision == 2:  # InputPrecisionVariant.IEEE:
        input_precision_: ntl.constexpr = "ieee"
    else:
        input_precision_: ntl.constexpr = "tf32"

    for k in range(0, input.shape[0], unroll):
        for ku in ntl.static_range(unroll):
            k_index = k + ku
            if k_index < input.shape[0]:
                accumulator += ntl.dot(
                    input[k_index],
                    other[k_index],
                    input_precision=input_precision_,
                )

    output = accumulator


def premake(
    input_precision=None,
    dtype=None,
    unroll=1,
    has_bias=None,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    tensors = (
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
        Tensor(0, constexpr=True, value=input_precision),
        Tensor(0, constexpr=True, value=unroll),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(0, constexpr=True, value=has_bias),
    )

    return arrangement_, application, tensors
