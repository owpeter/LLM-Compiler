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
        import ntops
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
        import ninetoothed
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "未找到 ninetoothed 模块。请先安装/克隆 ninetoothed，并使用 --ninetoothed-path 或 NINETOOTHED_PATH 指定路径。"
        ) from exc


def build_causal_softmax(
    ntops_path: str = "",
    ninetoothed_path: str = "",
    dtype: str | None = None,
    ndim: str = "2,3",
    block_size: str = "128",
    skip_cleanup: bool = False,
):
    _ensure_ninetoothed_available(ninetoothed_path or None)
    _ensure_ntops_available(ntops_path or None)

    import ninetoothed
    from infiniop.ninetoothed.build import BUILD_DIRECTORY_PATH
    from infiniop.ops.causal_softmax.ninetoothed.build import build

    if not skip_cleanup and BUILD_DIRECTORY_PATH.exists():
        print(f"清理构建目录: {BUILD_DIRECTORY_PATH}")
        for item in BUILD_DIRECTORY_PATH.glob("causal_softmax*"):
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

    dtype_values = None
    if dtype:
        dtype_values = tuple(dtype_map[d.strip()] for d in dtype.split(","))

    ndim_values = tuple(int(v.strip()) for v in ndim.split(","))
    block_size_values = tuple(int(v.strip()) for v in block_size.split(","))

    print("开始构建 CausalSoftmax...")
    print("参数:")
    print(f"  dtype_values: {dtype_values}")
    print(f"  ndim_values: {ndim_values}")
    print(f"  block_size_values: {block_size_values}")

    build(
        dtype_values=dtype_values,
        ndim_values=ndim_values,
        block_size_values=block_size_values,
    )
    print("构建完成。")


def main():
    parser = argparse.ArgumentParser(
        description="仅构建 CausalSoftmax (Ninetoothed) 的脚本"
    )
    parser.add_argument("--ntops-path", default="", help="ntops 仓库或包根目录的路径")
    parser.add_argument(
        "--ninetoothed-path", default="", help="ninetoothed 仓库或包根目录的路径"
    )
    parser.add_argument(
        "--dtype", default=None, help="数据类型 (逗号分隔，例如 float16,bfloat16)"
    )
    parser.add_argument("--ndim", default="2,3", help="ndim_values (逗号分隔)")
    parser.add_argument(
        "--block-size", default="128", help="block_size_values (逗号分隔)"
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="跳过清理旧的 causal_softmax* 文件",
    )

    args = parser.parse_args()

    build_causal_softmax(
        ntops_path=args.ntops_path,
        ninetoothed_path=args.ninetoothed_path,
        dtype=args.dtype,
        ndim=args.ndim,
        block_size=args.block_size,
        skip_cleanup=args.skip_cleanup,
    )


if __name__ == "__main__":
    main()
