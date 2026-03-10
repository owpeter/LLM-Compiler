import ninetoothed
from ntops.kernels import causal_softmax

import infiniop.ninetoothed.build


def build(
    dtype_values=None,
    ndim_values=(2, 3),
    block_size_values=(128, 256, 512, 1024),
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
        causal_softmax.premake,
        constexpr_param_grid,
        caller="cuda",
        op_name="causal_softmax",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
    )
