import os
import subprocess
import platform
import sys
import argparse
import glob
from typing import Iterable
from set_env import set_env

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_DIR)

def run_cmd(cmd):
    subprocess.run(cmd, text=True, encoding="utf-8", check=True, shell=True)


def install(xmake_config_flags=""):
    run_cmd(f"xmake f {xmake_config_flags} -cv")
    run_cmd("xmake")
    run_cmd("xmake install")
    run_cmd("xmake build infiniop-test")
    run_cmd("xmake install infiniop-test")

def _is_truthy_xmake_value(value: str) -> bool:
    return value.strip().lower() in {"y", "yes", "true", "1", "on"}


def _get_xmake_option_value(args: Iterable[str], name: str) -> str | None:
    prefix = f"--{name}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _get_available_ninetoothed_ops() -> set[str]:
    matches = glob.glob(os.path.join(PROJECT_DIR, "src", "infiniop", "ops", "*", "ninetoothed", "build.py"))
    return {os.path.basename(os.path.dirname(os.path.dirname(match))) for match in matches}


def _parse_ops(ops_value: str | None) -> list[str]:
    if ops_value is None:
        return []
    parts = []
    for chunk in ops_value.replace(",", " ").split():
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    seen = set()
    result = []
    for op in parts:
        if op not in seen:
            seen.add(op)
            result.append(op)
    return result


if __name__ == "__main__":
    try:
        set_env()

        parser = argparse.ArgumentParser(add_help=True)
        parser.add_argument(
            "--ops",
            type=str,
            default=None,
            help="Comma/space-separated ninetoothed ops to compile (e.g. --ops gemm,swiglu). Requires --ninetoothed=y.",
        )
        parsed, xmake_args = parser.parse_known_args(sys.argv[1:])

        ops = _parse_ops(parsed.ops)
        if ops:
            ninetoothed_value = _get_xmake_option_value(xmake_args, "ninetoothed")
            if ninetoothed_value is None or not _is_truthy_xmake_value(ninetoothed_value):
                raise RuntimeError("--ops requires --ninetoothed=y")

            available = _get_available_ninetoothed_ops()
            unknown = [op for op in ops if op not in available]
            if unknown:
                raise RuntimeError(
                    f"Unknown ninetoothed ops: {', '.join(unknown)}. Available: {', '.join(sorted(available))}"
                )

            xmake_args = [arg for arg in xmake_args if not arg.startswith("--ninetoothed_ops=")]
            xmake_args.append(f"--ninetoothed_ops={','.join(ops)}")
            print(f"[install.py] ninetoothed: ops={','.join(ops)}")

        install(" ".join(xmake_args))
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)
