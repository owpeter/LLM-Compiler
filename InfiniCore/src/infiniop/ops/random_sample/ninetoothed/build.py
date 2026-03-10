import ninetoothed
from ntops.kernels import random_sample

import infiniop.ninetoothed.build


def build(
    ndim_values=(1, 2),
    dtype_values=None,
    block_size_values=(128,),
    num_warps=4,
    num_stages=2,
):
    if dtype_values is None:
        dtype_values = (
            ninetoothed.float16,
            ninetoothed.bfloat16,
            ninetoothed.float32,
        )

    constexpr_param_grid = {
        "ndim": ndim_values,
        "dtype": dtype_values,
        "block_size": block_size_values,
    }

    infiniop.ninetoothed.build.build(
        random_sample.premake,
        constexpr_param_grid,
        caller="cuda",
        op_name="random_sample",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
        num_warps=num_warps,
        num_stages=num_stages,
    )
