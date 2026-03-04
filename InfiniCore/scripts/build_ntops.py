import argparse
import importlib
import os
import pathlib
import sys

CURRENT_FILE_PATH = pathlib.Path(__file__)

SRC_DIR_PATH = CURRENT_FILE_PATH.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR_PATH))

from infiniop.ninetoothed.build import BUILD_DIRECTORY_PATH


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

    for p in selected:
        _build(p)


if __name__ == "__main__":
    main()
