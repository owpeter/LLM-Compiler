import argparse
import importlib
import os
import pathlib
import sys

CURRENT_FILE_PATH = pathlib.Path(__file__)
PROJECT_ROOT_PATH = CURRENT_FILE_PATH.parent.parent
SRC_DIR_PATH = PROJECT_ROOT_PATH / "src"
sys.path.insert(0, str(SRC_DIR_PATH))
BUILD_DIRECTORY_PATH = PROJECT_ROOT_PATH / "build" / "ninetoothed"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default="", help="Comma-separated op names to build")
    parser.add_argument("--jobs", type=int, default=0, help="Max processes per op build")
    parser.add_argument("--dry-run", action="store_true", help="List selected ops and exit")
    parser.add_argument("--ntops-path", default="", help="Path to ntops repo or package root")
    parser.add_argument(
        "--ninetoothed-path", default="", help="Path to ninetoothed repo or package root"
    )
    args = parser.parse_args()

    only_ops = {name for name in args.ops.split(",") if name} if args.ops else None

    BUILD_DIRECTORY_PATH.mkdir(parents=True, exist_ok=True)

    if args.jobs and args.jobs > 0:
        os.environ["NINETOOTHED_MAX_WORKERS"] = str(args.jobs)

    selected = list(_find_ops(only_ops))
    if args.dry_run:
        for p in selected:
            print(p.parent.name)
        return

    _ensure_ninetoothed_available(args.ninetoothed_path or None)
    _ensure_ntops_available(args.ntops_path or None)

    for p in selected:
        _build(p)


if __name__ == "__main__":
    main()
