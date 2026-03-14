import functools

import ninetoothed
from ntops.kernels import rms_norm

import infiniop.ninetoothed.build


def build(
    dtype_values=None,
    ndim_values=(2, 3),
    num_normalized_dims_values=(1,),
    block_size_values=(128,),
    num_warps=4,
    num_stages=2,
    use_vectorized_application=False,
):
    if dtype_values is None:
        dtype_values = (
            ninetoothed.float16,
            # ninetoothed.bfloat16,
            # ninetoothed.float32,
        )

    constexpr_param_grid = {
        "ndim": ndim_values,
        "num_normalized_dims": num_normalized_dims_values,
        "input_dtype": dtype_values,
        "weight_dtype": dtype_values,
        "output_dtype": dtype_values,
        "block_size": block_size_values,
    }

    infiniop.ninetoothed.build.build(
        functools.partial(
            rms_norm.premake, use_vectorized_application=use_vectorized_application
        ),
        constexpr_param_grid,
        caller="cuda",
        op_name="rms_norm",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
        num_warps=num_warps,
        num_stages=num_stages,
    )
