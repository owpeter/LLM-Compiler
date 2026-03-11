import argparse
import os
import pathlib
import sys
import shutil

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


def _ensure_ntops_available(explicit_path: str | None):
    candidates = []
    if explicit_path:
        candidates.append(pathlib.Path(_normalize_path_str(explicit_path)).expanduser())

    env_path = os.getenv("NTOPS_PATH")
    if env_path:
        candidates.append(pathlib.Path(_normalize_path_str(env_path)).expanduser())

    candidates.extend(
        [
            PROJECT_ROOT_PATH / "ntops",
            PROJECT_ROOT_PATH / "third_party" / "ntops",
            PROJECT_ROOT_PATH.parent / "ntops",
        ]
    )

    for candidate in candidates:
        _maybe_add_module_path(candidate, "ntops")

    try:
        import ntops  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "未找到 ntops 模块。请先安装/克隆 ntops，并使用 --ntops-path 或 NTOPS_PATH 指定路径。"
        ) from exc


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
        import ninetoothed  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "未找到 ninetoothed 模块。请先安装/克隆 ninetoothed，并使用 --ninetoothed-path 或 NINETOOTHED_PATH 指定路径。"
        ) from exc


def build_gemm(
    ntops_path: str = "",
    ninetoothed_path: str = "",
    dtype: str | None = None,
    input_precision: str | None = None,
    block_m: str = "256",
    block_n: str = "128",
    block_k: str = "64",
    unroll: str = "4",
    num_warps: int = 4,
    num_stages: int = 2,
    skip_cleanup: bool = False,
):
    """
    构建 GEMM (Ninetoothed) 算子的函数封装。

    Args:
        ntops_path: ntops 仓库或包根目录的路径
        ninetoothed_path: ninetoothed 仓库或包根目录的路径
        dtype: 数据类型 (逗号分隔，例如 "float16,bfloat16")
        input_precision: 输入精度 (逗号分隔，例如 "TF32,IEEE")
        block_m: block_size_m_values (逗号分隔，例如 "128,256")
        block_n: block_size_n_values (逗号分隔)
        block_k: block_size_k_values (逗号分隔)
        unroll: unroll_values (逗号分隔)
        num_warps: num_warps
        num_stages: num_stages
        skip_cleanup: 是否跳过清理 build 目录下的 gemm* 文件
    """
    # 确保依赖项可用
    _ensure_ninetoothed_available(ninetoothed_path or None)
    _ensure_ntops_available(ntops_path or None)

    import ninetoothed
    from ntops.kernels import mm
    from infiniop.ops.gemm.ninetoothed.build import build
    from infiniop.ninetoothed.build import BUILD_DIRECTORY_PATH

    # 清理 build 目录下所有 gemm* 的文件和文件夹
    if not skip_cleanup:
        if BUILD_DIRECTORY_PATH.exists():
            print(f"清理构建目录: {BUILD_DIRECTORY_PATH}")
            for item in BUILD_DIRECTORY_PATH.glob("gemm*"):
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                print(f"  已删除: {item.name}")

    # 处理参数映射
    dtype_map = {
        "float16": ninetoothed.float16,
        "bfloat16": ninetoothed.bfloat16,
        "float32": ninetoothed.float32,
    }

    precision_map = {
        "TF32": mm.InputPrecisionVariant.TF32,
        "IEEE": mm.InputPrecisionVariant.IEEE,
    }

    # 解析逗号分隔的字符串
    dtype_values = None
    if dtype:
        dtype_values = tuple(dtype_map[d.strip()] for d in dtype.split(","))

    input_precision_values = None
    if input_precision:
        input_precision_values = tuple(
            precision_map[p.strip()] for p in input_precision.split(",")
        )

    block_size_m_values = tuple(int(v.strip()) for v in block_m.split(","))
    block_size_n_values = tuple(int(v.strip()) for v in block_n.split(","))
    block_size_k_values = tuple(int(v.strip()) for v in block_k.split(","))
    unroll_values = tuple(int(v.strip()) for v in unroll.split(","))

    print(f"开始构建 GEMM...")
    print(f"参数: ")
    print(f"  dtype_values: {dtype_values}")
    print(f"  input_precision_values: {input_precision_values}")
    print(f"  block_size_m_values: {block_size_m_values}")
    print(f"  block_size_n_values: {block_size_n_values}")
    print(f"  block_size_k_values: {block_size_k_values}")
    print(f"  unroll_values: {unroll_values}")
    print(f"  num_warps: {num_warps}")
    print(f"  num_stages: {num_stages}")

    build(
        dtype_values=dtype_values,
        input_precision_values=input_precision_values,
        block_size_m_values=block_size_m_values,
        block_size_n_values=block_size_n_values,
        block_size_k_values=block_size_k_values,
        unroll_values=unroll_values,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    print("构建完成。")


def main():
    parser = argparse.ArgumentParser(description="仅构建 GEMM (Ninetoothed) 的脚本")
    parser.add_argument("--ntops-path", default="", help="ntops 仓库或包根目录的路径")
    parser.add_argument(
        "--ninetoothed-path", default="", help="ninetoothed 仓库或包根目录的路径"
    )

    # gemm build 函数的参数
    parser.add_argument(
        "--dtype", default=None, help="数据类型 (逗号分隔，例如 float16,bfloat16)"
    )
    parser.add_argument(
        "--input-precision", default=None, help="输入精度 (逗号分隔，例如 TF32,IEEE)"
    )
    parser.add_argument("--block-m", default="128", help="block_size_m_values (逗号分隔)")
    parser.add_argument("--block-n", default="1024", help="block_size_n_values (逗号分隔)")
    parser.add_argument("--block-k", default="32", help="block_size_k_values (逗号分隔)")
    parser.add_argument("--unroll", default="4", help="unroll_values (逗号分隔)")
    parser.add_argument("--num-warps", type=int, default=4, help="num_warps")
    parser.add_argument("--num-stages", type=int, default=2, help="num_stages")
    parser.add_argument("--skip-cleanup", action="store_true", help="跳过清理旧的 gemm* 文件")

    args = parser.parse_args()

    # 调用封装好的函数
    build_gemm(
        ntops_path=args.ntops_path,
        ninetoothed_path=args.ninetoothed_path,
        dtype=args.dtype,
        input_precision=args.input_precision,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        unroll=args.unroll,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        skip_cleanup=args.skip_cleanup,
    )


if __name__ == "__main__":
    main()
