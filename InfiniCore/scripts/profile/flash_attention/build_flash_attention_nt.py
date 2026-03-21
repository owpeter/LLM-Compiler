import argparse
import os
import pathlib
import shutil
import sys

CURRENT_FILE_PATH = pathlib.Path(__file__).resolve()


def _find_project_root(start: pathlib.Path) -> pathlib.Path:
    for candidate in [start.parent, *start.parents]:
        if (candidate / "src" / "infiniop").is_dir():
            return candidate
    return start.parents[2]


PROJECT_ROOT_PATH = _find_project_root(CURRENT_FILE_PATH)
SRC_DIR_PATH = PROJECT_ROOT_PATH / "src"
sys.path.insert(0, str(SRC_DIR_PATH))


def _normalize_path_str(path_str: str | None):
    if not path_str:
        return path_str
    return path_str.replace("\\", "/")


def _maybe_add_module_path(candidate: pathlib.Path, module_name: str):
    if not candidate.exists():
        return

    python_path = candidate / "python"
    if (python_path / module_name).is_dir():
        sys.path.insert(0, str(python_path))
        return

    src_path = candidate / "src"
    if (src_path / module_name).is_dir():
        sys.path.insert(0, str(src_path))
        return

    if (candidate / module_name).is_dir():
        sys.path.insert(0, str(candidate))


def _ensure_ninetoothed_available(explicit_path: str | None):
    candidates = []
    if explicit_path:
        candidates.append(pathlib.Path(_normalize_path_str(explicit_path)).expanduser())

    env_path = os.getenv("NINETOOTHED_PATH")
    if env_path:
        candidates.append(pathlib.Path(_normalize_path_str(env_path)).expanduser())

    candidates.extend(
        [
            PROJECT_ROOT_PATH / "ninetoothed",
            PROJECT_ROOT_PATH / "third_party" / "ninetoothed",
            PROJECT_ROOT_PATH.parent / "ninetoothed",
        ]
    )

    for candidate in candidates:
        _maybe_add_module_path(candidate, "ninetoothed")

    try:
        import ninetoothed
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "未找到 ninetoothed 模块。请先安装/克隆 ninetoothed，并使用 --ninetoothed-path 或 NINETOOTHED_PATH 指定路径。"
        ) from exc


def _parse_int_values(value: str):
    return tuple(int(v.strip()) for v in value.split(","))


def build_flash_attention(
    ninetoothed_path: str = "",
    with_kv_cache: str = "0",
    emb_dim: str = "64,128",
    is_causal: str = "0,1",
    with_attn_mask: str = "0",
    causal_variant: str = "UPPER_LEFT,LOWER_RIGHT",
    dtype: str = "float16",
    block_size_m: str = "32",
    block_size_n: str = "128",
    num_warps: int = 4,
    num_stages: int = 2,
    skip_cleanup: bool = False,
):
    _ensure_ninetoothed_available(ninetoothed_path or None)

    import ninetoothed
    from infiniop.ninetoothed.build import BUILD_DIRECTORY_PATH
    from infiniop.ops.flash_attention.ninetoothed.build import build
    from infiniop.ops.flash_attention.ninetoothed.flash_attention import CausalVariant

    if not skip_cleanup and BUILD_DIRECTORY_PATH.exists():
        print(f"清理构建目录: {BUILD_DIRECTORY_PATH}")
        for item in BUILD_DIRECTORY_PATH.glob("flash_attention*"):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            print(f"  已删除: {item.name}")

    dtype_map = {
        "float16": ninetoothed.float16,
        "bfloat16": ninetoothed.bfloat16,
        "float32": ninetoothed.float32,
    }
    causal_variant_map = {
        "UPPER_LEFT": CausalVariant.UPPER_LEFT,
        "LOWER_RIGHT": CausalVariant.LOWER_RIGHT,
        "1": CausalVariant.UPPER_LEFT,
        "2": CausalVariant.LOWER_RIGHT,
    }

    dtype_values = tuple(dtype_map[d.strip()] for d in dtype.split(","))
    with_kv_cache_values = _parse_int_values(with_kv_cache)
    emb_dim_values = _parse_int_values(emb_dim)
    is_causal_values = _parse_int_values(is_causal)
    with_attn_mask_values = _parse_int_values(with_attn_mask)
    block_size_m_values = _parse_int_values(block_size_m)
    block_size_n_values = _parse_int_values(block_size_n)
    causal_variant_values = tuple(
        causal_variant_map[v.strip().upper()] for v in causal_variant.split(",")
    )

    print("开始构建 FlashAttention...")
    print("参数:")
    print(f"  with_kv_cache_values: {with_kv_cache_values}")
    print(f"  emb_dim_values: {emb_dim_values}")
    print(f"  is_causal_values: {is_causal_values}")
    print(f"  with_attn_mask_values: {with_attn_mask_values}")
    print(f"  causal_variant_values: {causal_variant_values}")
    print(f"  dtype_values: {dtype_values}")
    print(f"  block_size_m_values: {block_size_m_values}")
    print(f"  block_size_n_values: {block_size_n_values}")
    print(f"  num_warps: {num_warps}")
    print(f"  num_stages: {num_stages}")

    build(
        with_kv_cache_values=with_kv_cache_values,
        emb_dim_values=emb_dim_values,
        is_causal_values=is_causal_values,
        with_attn_mask_values=with_attn_mask_values,
        causal_variant_values=causal_variant_values,
        dtype_values=dtype_values,
        block_size_m_values=block_size_m_values,
        block_size_n_values=block_size_n_values,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    print("构建完成。")


def main():
    parser = argparse.ArgumentParser(
        description="仅构建 FlashAttention (Ninetoothed) 的脚本"
    )
    parser.add_argument(
        "--ninetoothed-path", default="", help="ninetoothed 仓库或包根目录的路径"
    )
    parser.add_argument(
        "--with-kv-cache", default="0", help="with_kv_cache_values (逗号分隔)"
    )
    parser.add_argument("--emb-dim", default="64,128", help="emb_dim_values (逗号分隔)")
    parser.add_argument(
        "--is-causal", default="0,1", help="is_causal_values (逗号分隔)"
    )
    parser.add_argument(
        "--with-attn-mask", default="0", help="with_attn_mask_values (逗号分隔)"
    )
    parser.add_argument(
        "--causal-variant",
        default="UPPER_LEFT,LOWER_RIGHT",
        help="causal_variant_values (逗号分隔，可选 UPPER_LEFT/LOWER_RIGHT 或 1/2)",
    )
    parser.add_argument(
        "--dtype", default="float32", help="数据类型 (逗号分隔，例如 float16,bfloat16)"
    )
    parser.add_argument(
        "--block-size-m", default="32", help="block_size_m_values (逗号分隔)"
    )
    parser.add_argument(
        "--block-size-n", default="128", help="block_size_n_values (逗号分隔)"
    )
    parser.add_argument("--num-warps", type=int, default=4, help="num_warps")
    parser.add_argument("--num-stages", type=int, default=2, help="num_stages")
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="跳过清理旧的 flash_attention* 文件",
    )

    args = parser.parse_args()

    build_flash_attention(
        ninetoothed_path=args.ninetoothed_path,
        with_kv_cache=args.with_kv_cache,
        emb_dim=args.emb_dim,
        is_causal=args.is_causal,
        with_attn_mask=args.with_attn_mask,
        causal_variant=args.causal_variant,
        dtype=args.dtype,
        block_size_m=args.block_size_m,
        block_size_n=args.block_size_n,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        skip_cleanup=args.skip_cleanup,
    )


if __name__ == "__main__":
    main()
