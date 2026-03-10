import ninetoothed
from ntops.kernels import swiglu

import infiniop.ninetoothed.build


def build(
    dtype_values=None,
    ndim_values=(2,3,),
    block_size_values=(512,),
):
    if dtype_values is None:
        dtype_values = (
            ninetoothed.float16,
            ninetoothed.bfloat16,
            ninetoothed.float32,
            ninetoothed.float64,
        )

    constexpr_param_grid = {
        "ndim": ndim_values,
        "dtype": dtype_values,
        "block_size": block_size_values,
    }

    infiniop.ninetoothed.build.build(
        swiglu.premake,
        constexpr_param_grid,
        caller="cuda",
        op_name="swiglu",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
    )
