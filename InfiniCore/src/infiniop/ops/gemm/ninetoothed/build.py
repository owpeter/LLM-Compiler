import ninetoothed
from ntops.kernels import mm

import infiniop.ninetoothed.build


def build(
    dtype_values=None,
    input_precision_values=None,
    block_size_m_values=(128,),
    block_size_n_values=(128,),
    block_size_k_values=(64,),
    unroll_values=(4,),
    num_warps=4,
    num_stages=2,
):
    if dtype_values is None:
        dtype_values = (
            ninetoothed.float16,
            ninetoothed.bfloat16,
            ninetoothed.float32,
        )

    if input_precision_values is None:
        input_precision_values = (
            mm.InputPrecisionVariant.TF32,
            mm.InputPrecisionVariant.IEEE,
        )

    constexpr_param_grid = {
        "dtype": dtype_values,
        "input_precision": input_precision_values,
        "block_size_m": block_size_m_values,
        "block_size_n": block_size_n_values,
        "block_size_k": block_size_k_values,
        "unroll": unroll_values,
    }

    infiniop.ninetoothed.build.build(
        mm.premake,
        constexpr_param_grid,
        caller="cuda",
        op_name="gemm",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
        num_warps=num_warps,
        num_stages=num_stages,
    )
