import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def _exp(x):
    # Always compute exp in float32 for stability
    return ntl.exp(ntl.cast(x, ntl.float32))


def application(probs, result, random_val, topp, topk, temperature):
    dtype = probs.dtype.dtype
    # Use float32 for all accumulations and probability computations
    # to avoid overflow/underflow issues common in fp16
    accum_dtype = ntl.float32

    # Avoid division by zero
    temp_safe = ntl.maximum(temperature, 1e-6)

    # State for Pass 1
    prev_max = ntl.cast(float("-inf"), accum_dtype)
    denominator = ntl.cast(0, accum_dtype)
    max_idx = ntl.cast(0, ntl.int64)

    # Pass 1: Compute Max, Argmax, and Sum (Z)
    for i in range(probs.shape[0]):
        input_i = ntl.cast(probs[i], accum_dtype)
        block_offsets = probs[i].offsets()

        # Max/Argmax
        curr_max = ntl.max(input_i)
        curr_argmax = ntl.argmax(input_i, axis=0)  # local index within block
        # curr_argmax_global = block_offsets[curr_argmax]
        range_indices = ntl.arange(0, input_i.shape[0])
        curr_argmax_global = ntl.sum(ntl.where(range_indices == curr_argmax, block_offsets, ntl.cast(0, ntl.int64)))

        # Update global max/argmax
        update_max = curr_max > prev_max
        max_idx = ntl.where(update_max, curr_argmax_global, max_idx)

        # Update Sum (Z) for softmax
        # Note: We apply temperature here
        new_max = ntl.maximum(prev_max, curr_max)

        input_max_diff_exp = _exp((input_i - new_max) / temp_safe)
        prev_curr_max_diff_exp = _exp((prev_max - new_max) / temp_safe)

        denominator = denominator * prev_curr_max_diff_exp + ntl.sum(input_max_diff_exp)
        prev_max = new_max

    # Prepare for Pass 2
    # target = random_val * Z
    target = random_val * denominator
    cdf = ntl.cast(0, accum_dtype)
    sample_idx = ntl.cast(-1, ntl.int64)
    MAX_INT64 = ntl.cast(9223372036854775807, ntl.int64)

    # Pass 2: CDF and Sample
    for i in range(probs.shape[0]):
        input_i = ntl.cast(probs[i], accum_dtype)
        block_offsets = probs[i].offsets()

        # Recompute probs using global max
        prob = _exp((input_i - prev_max) / temp_safe)

        block_sum = ntl.sum(prob)
        block_cdf = ntl.cumsum(prob)  # cumsum within block

        current_cdf = cdf + block_cdf

        # Find first index where current_cdf >= target
        mask = current_cdf >= target

        # We want the smallest index where mask is True
        candidate_indices = ntl.where(mask, block_offsets, MAX_INT64)
        min_candidate = ntl.min(candidate_indices)

        # Update sample_idx if we found a better one (and haven't found one yet)
        found_now = min_candidate != MAX_INT64
        should_update = (sample_idx == -1) & found_now
        sample_idx = ntl.where(should_update, min_candidate, sample_idx)

        cdf = cdf + block_sum

    # Select result
    # If random_val == 0 or topp == 0 or topk == 1 or temperature == 0, return max_idx
    # Note: Currently full Top-P/Top-K sorting is not implemented in this kernel due to lack of sorting primitives.
    # We fall back to standard Multinomial Sampling unless argmax conditions are met.
    use_argmax = (random_val == 0) | (topp == 0) | (topk == 1) | (temperature == 0)

    # Fallback if sample not found (rounding errors)
    sample_idx = ntl.where(sample_idx == -1, max_idx, sample_idx)

    final_idx = ntl.where(use_argmax, max_idx, sample_idx)

    # Write result
    # result is reduced to size 1 in the reduction dim.
    # We write to the first element of the result block.
    result[0] = final_idx


def premake(ndim, dtype=None, block_size=None):
    # We reduce the last dimension (vocab size)
    arrangement_ = functools.partial(arrangement, dim=-1, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),  # probs
        Tensor(ndim, dtype=ninetoothed.int64),  # result
        Tensor(0, dtype=ninetoothed.float32),   # random_val
        Tensor(0, dtype=ninetoothed.float32),   # topp
        Tensor(0, dtype=ninetoothed.int32),     # topk
        Tensor(0, dtype=ninetoothed.float32),   # temperature
    )

    return arrangement_, application, tensors
