import ninetoothed
from . import flash_attention
from .flash_attention import CausalVariant

import infiniop.ninetoothed.build

import torch

import os


def build(
    with_kv_cache_values=(0,),
    emb_dim_values=(64,),
    is_causal_values=(0, 1),
    with_attn_mask_values=(0,),
    causal_variant_values=(CausalVariant.UPPER_LEFT, CausalVariant.LOWER_RIGHT),
    dtype_values=(ninetoothed.float16,),
    block_size_m_values=(256,),
    block_size_n_values=(64,),
    num_warps=4,
    num_stages=2,
):

    env_vars_to_check = ["MACA_HOME", "MACA_PATH", "MACA_ROOT"]
    if any(var in os.environ for var in env_vars_to_check):
        return

    constexpr_param_grid = {
        "with_kv_cache": with_kv_cache_values,
        "emb_dim": emb_dim_values,
        "is_causal": is_causal_values,
        "with_attn_mask": with_attn_mask_values,
        "causal_variant": causal_variant_values,
        "dtype": dtype_values,
        "block_size_m": block_size_m_values,
        "block_size_n": block_size_n_values,
    }

    infiniop.ninetoothed.build.build(
        flash_attention.premake,
        constexpr_param_grid,
        caller="cuda",
        op_name="flash_attention",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
        num_warps=num_warps,
        num_stages=num_stages,
    )
