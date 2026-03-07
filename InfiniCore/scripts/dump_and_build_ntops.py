import argparse
import importlib
import os
import pathlib
import shutil
import sys
import time

from set_env import set_env

CURRENT_FILE_PATH = pathlib.Path(__file__)
PROJECT_ROOT_PATH = CURRENT_FILE_PATH.parent.parent
SRC_DIR_PATH = PROJECT_ROOT_PATH / "src"
BUILD_DIRECTORY_PATH = PROJECT_ROOT_PATH / "build" / "ninetoothed"
DEFAULT_EXPORT_DIR = PROJECT_ROOT_PATH / "build" / "ninetoothed_ptx"

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


def _find_ops(only_ops: set[str] | None):
    ops_path = SRC_DIR_PATH / "infiniop" / "ops"

    for op_dir in sorted(ops_path.iterdir(), key=lambda p: p.name):
        if only_ops and op_dir.name not in only_ops:
            continue

        ninetoothed_path = op_dir / "ninetoothed"
        if not ninetoothed_path.is_dir():
            continue

        build_file = ninetoothed_path / "build.py"
        if not build_file.exists():
            continue

        yield ninetoothed_path


def _build(ninetoothed_path):
    module_path = ninetoothed_path / "build"
    relative_path = module_path.relative_to(SRC_DIR_PATH)
    import_name = ".".join(relative_path.parts)
    module = importlib.import_module(import_name)
    module.build()


def _artifact_files(cache_dir: pathlib.Path):
    for suffix in (".ptx", ".cubin"):
        for p in cache_dir.rglob(f"*{suffix}"):
            if p.is_file():
                yield p


def _safe_copy(src: pathlib.Path, dst_dir: pathlib.Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        dst = dst_dir / f"{src.stem}_{int(time.time() * 1e6)}{src.suffix}"
    shutil.copy2(src, dst)
    return dst


def _collect_new_artifacts(cache_dir: pathlib.Path, seen_sources: set[str]):
    new_files = []
    for artifact in _artifact_files(cache_dir):
        artifact_key = str(artifact.resolve())
        if artifact_key in seen_sources:
            continue
        seen_sources.add(artifact_key)
        new_files.append(artifact)
    return new_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default="", help="Comma-separated op names to build")
    parser.add_argument("--jobs", type=int, default=0, help="Max processes per op build")
    parser.add_argument("--dry-run", action="store_true", help="List selected ops and exit")
    parser.add_argument("--ntops-path", default="", help="Path to ntops repo or package root")
    parser.add_argument(
        "--ninetoothed-path", default="", help="Path to ninetoothed repo or package root"
    )
    parser.add_argument(
        "--export-dir",
        default=str(DEFAULT_EXPORT_DIR),
        help="Directory to export ptx/cubin artifacts",
    )
    args = parser.parse_args()

    only_ops = {name for name in args.ops.split(",") if name} if args.ops else None
    selected = list(_find_ops(only_ops))
    if args.dry_run:
        for p in selected:
            print(p.parent.name)
        return

    set_env()
    BUILD_DIRECTORY_PATH.mkdir(parents=True, exist_ok=True)
    export_dir = pathlib.Path(_normalize_path_str(args.export_dir)).expanduser()
    cache_dir = export_dir / f"triton_cache_{int(time.time())}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)

    if args.jobs and args.jobs > 0:
        os.environ["NINETOOTHED_MAX_WORKERS"] = str(args.jobs)

    _ensure_ninetoothed_available(args.ninetoothed_path or None)
    _ensure_ntops_available(args.ntops_path or None)

    import ninetoothed

    original_make = ninetoothed.make
    current_op = {"name": "unknown"}
    exported_records = []
    copied_sources = set()

    def hooked_make(*make_args, **make_kwargs):
        result = original_make(*make_args, **make_kwargs)
        kernel_name = str(make_kwargs.get("kernel_name", "unknown_kernel"))
        op_name = current_op["name"]
        target_dir = export_dir / op_name / kernel_name
        newly_copied = []
        for artifact in _artifact_files(cache_dir):
            artifact_key = str(artifact.resolve())
            if artifact_key in copied_sources:
                continue
            copied_sources.add(artifact_key)
            copied = _safe_copy(artifact, target_dir)
            newly_copied.append(str(copied))
        if newly_copied:
            exported_records.append((op_name, kernel_name, newly_copied))
        return result

    ninetoothed.make = hooked_make
    try:
        for op_path in selected:
            current_op["name"] = op_path.parent.name
            _build(op_path)
            collected = _collect_new_artifacts(cache_dir, copied_sources)
            if collected:
                target_dir = export_dir / current_op["name"] / "__auto_collected__"
                copied = [str(_safe_copy(artifact, target_dir)) for artifact in collected]
                exported_records.append(
                    (current_op["name"], "__auto_collected__", copied)
                )
    finally:
        ninetoothed.make = original_make

    if not exported_records:
        raise SystemExit(
            f"未采集到 PTX/CUBIN 产物。请检查 ninetoothed 编译链是否在 {cache_dir} 输出中间产物。"
        )

    for op_name, kernel_name, files in exported_records:
        print(f"[{op_name}] {kernel_name}")
        for file in files:
            print(file)


if __name__ == "__main__":
    main()
