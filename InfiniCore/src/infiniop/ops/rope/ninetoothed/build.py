import ninetoothed
from ntops.kernels import rotary_position_embedding

import infiniop.ninetoothed.build


def build(
    dtype_values=None,
    ndim_values=(4,),
    emb_dim_values=(64,),
    interleaved_values=(True, False),
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
        "emb_dim": emb_dim_values,
        "dtype": dtype_values,
        "interleaved": interleaved_values,
    }

    infiniop.ninetoothed.build.build(
        rotary_position_embedding.premake,
        constexpr_param_grid,
        caller="cuda",
        op_name="rope",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
