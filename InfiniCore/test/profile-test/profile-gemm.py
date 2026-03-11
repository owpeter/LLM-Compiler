import argparse
import csv
import ctypes
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any



PROJECT_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = PROJECT_ROOT / "scripts" / "profile" / "gemm.yaml"
BUILD_GEMM_NT_PATH = PROJECT_ROOT / "scripts" / "profile" / "build_gemm_nt.py"
INFINIOP_TEST_ROOT = PROJECT_ROOT / "test" / "infiniop"


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _status(msg: str):
    print(f"[{_ts()}] {msg}", flush=True)


def _strip_comment(line: str) -> str:
    if "#" not in line:
        return line
    before, _hash, _after = line.partition("#")
    return before


def _parse_scalar(text: str) -> Any:
    s = text.strip()
    if s == "":
        return ""
    try:
        if s.startswith("0") and len(s) > 1 and s.isdigit():
            return s
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _yaml_lines(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in raw:
        line = _strip_comment(line).rstrip("\n")
        if line.strip() == "":
            continue
        out.append(line.rstrip())
    return out


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parse_block(lines: list[str], start: int, indent: int) -> tuple[Any, int]:
    i = start
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i >= len(lines):
        return {}, i
    if _indent_of(lines[i]) < indent:
        return {}, i

    is_list = lines[i].lstrip(" ").startswith("- ") and _indent_of(lines[i]) == indent
    if is_list:
        out: list[Any] = []
        while i < len(lines):
            line = lines[i]
            if _indent_of(line) < indent:
                break
            if _indent_of(line) > indent:
                break
            stripped = line.lstrip(" ")
            if not stripped.startswith("- "):
                break
            item_text = stripped[2:].strip()
            if item_text == "":
                child, i = _parse_block(lines, i + 1, indent + 2)
                out.append(child)
            else:
                out.append(_parse_scalar(item_text))
                i += 1
        return out, i

    out_dict: dict[str, Any] = {}
    while i < len(lines):
        line = lines[i]
        cur_indent = _indent_of(line)
        if cur_indent < indent:
            break
        if cur_indent > indent:
            break
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line: {line}")
        key, rest = stripped.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        if rest == "":
            child, i = _parse_block(lines, i + 1, indent + 2)
            out_dict[key] = child
        else:
            out_dict[key] = _parse_scalar(rest)
            i += 1
    return out_dict, i


def load_simple_yaml(path: Path) -> dict[str, Any]:
    lines = _yaml_lines(path)
    obj, _ = _parse_block(lines, 0, 0)
    if not isinstance(obj, dict):
        raise ValueError("Root YAML must be a mapping")
    return obj


def lhs_sample_discrete(
    space: dict[str, list[Any]],
    n: int,
    seed: int,
    max_unique_attempts: int = 20,
) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    keys = list(space.keys())
    for k in keys:
        if not isinstance(space[k], list) or len(space[k]) == 0:
            raise ValueError(f"Invalid discrete space for key={k}")

    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    attempts = 0
    while len(samples) < n and attempts < max_unique_attempts:
        perms = {k: rng.sample(range(n), n) for k in keys}
        for i in range(n):
            combo: dict[str, Any] = {}
            signature: list[Any] = []
            for k in keys:
                u = (perms[k][i] + rng.random()) / n
                idx = int(u * len(space[k]))
                if idx >= len(space[k]):
                    idx = len(space[k]) - 1
                v = space[k][idx]
                combo[k] = v
                signature.append(v)
            sig_t = tuple(signature)
            if sig_t in seen:
                continue
            seen.add(sig_t)
            samples.append(combo)
            if len(samples) >= n:
                break
        attempts += 1

    if len(samples) < n:
        raise RuntimeError(
            f"Failed to sample {n} unique combos, got {len(samples)}. Reduce n or adjust space."
        )
    return samples


def _load_build_gemm_func(path: Path):
    spec = importlib.util.spec_from_file_location("build_gemm_nt", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_gemm_nt"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_gemm"):
        raise RuntimeError("build_gemm not found in build_gemm_nt.py")
    return getattr(mod, "build_gemm")


def _ensure_libinfiniop_importable():
    sys.path.insert(0, str(INFINIOP_TEST_ROOT))


def rebuild_infiniop_gemm():
    cmd = [
        sys.executable,
        "scripts/install.py",
        "--nv-gpu=y",
        "--ninetoothed=y",
        "--ops",
        "gemm",
    ]
    env = os.environ.copy()
    start = time.time()
    _status("开始重新编译 Infinicore 算子库 (ops=gemm)")
    p = os.spawnve(os.P_WAIT, sys.executable, cmd, env)
    if p != 0:
        raise RuntimeError(f"Rebuild failed with exit code {p}")
    elapsed = time.time() - start
    _status(f"重新编译完成，用时 {elapsed:.2f}s")


@dataclass(frozen=True)
class ScheduleParams:
    dtype: str
    block_m: int
    block_n: int
    block_k: int
    unroll: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class Workload:
    m: int
    n: int
    k: int


def schedule_from_dict(d: dict[str, Any]) -> ScheduleParams:
    return ScheduleParams(
        dtype=str(d["dtype"]),
        block_m=int(d["block_m"]),
        block_n=int(d["block_n"]),
        block_k=int(d["block_k"]),
        unroll=int(d["unroll"]),
        num_warps=int(d["num_warps"]),
        num_stages=int(d["num_stages"]),
    )


def sample_workloads(
    workload_cfg: dict[str, Any],
    prefill_count: int,
    decode_count: int,
    seed: int,
) -> list[Workload]:
    out: list[Workload] = []
    prefill_space = {
        "m": list(workload_cfg["prefill"]["m"]),
        "n": list(workload_cfg["prefill"]["n"]),
        "k": list(workload_cfg["prefill"]["k"]),
    }
    decode_space = {
        "m": list(workload_cfg["decode"]["m"]),
        "n": list(workload_cfg["decode"]["n"]),
        "k": list(workload_cfg["decode"]["k"]),
    }

    prefill = lhs_sample_discrete(prefill_space, prefill_count, seed=seed + 101)
    decode = lhs_sample_discrete(decode_space, decode_count, seed=seed + 202)

    for d in prefill + decode:
        out.append(Workload(m=int(d["m"]), n=int(d["n"]), k=int(d["k"])))
    return out


def _dtype_to_infini(dtype: str):
    from libinfiniop import InfiniDtype

    if dtype == "float16":
        return InfiniDtype.F16
    if dtype == "bfloat16":
        return InfiniDtype.BF16
    if dtype == "float32":
        return InfiniDtype.F32
    raise ValueError(f"Unsupported dtype: {dtype}")


def profile_gemm_ms(
    device,
    dtype: str,
    workload: Workload,
    num_prerun: int,
    num_iterations: int,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> float:

    _ensure_libinfiniop_importable()
    _status(f"开始配置 GEMM 算子 (dtype={dtype}, workload={workload})")
    import torch
    from ctypes import c_size_t
    from libinfiniop import (
        LIBINFINIOP,
        TestTensor,
        TestWorkspace,
        check_error,
        create_handle,
        destroy_handle,
        timed_op,
        torch_device_map,
        infiniopOperatorDescriptor_t,
    )

    if torch_device_map[device] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but NVIDIA device selected")

    device_id = 0
    if torch_device_map[device] == "cuda":
        for k in (
            "LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "SLURM_LOCALID",
        ):
            v = os.environ.get(k)
            if v is None:
                continue
            v = str(v).strip()
            if v == "":
                continue
            try:
                device_id = int(v)
                break
            except ValueError:
                continue
        torch.cuda.set_device(device_id)

    check_error(LIBINFINIOP.infinirtSetDevice(device, ctypes.c_int(device_id)))

    handle = create_handle()
    try:
        a = TestTensor((workload.m, workload.k), None, _dtype_to_infini(dtype), device)
        b = TestTensor((workload.k, workload.n), None, _dtype_to_infini(dtype), device)
        c = TestTensor(
            (workload.m, workload.n),
            None,
            _dtype_to_infini(dtype),
            device,
            mode="zeros",
        )

        descriptor = infiniopOperatorDescriptor_t()
        check_error(
            LIBINFINIOP.infiniopCreateGemmDescriptor(
                handle,
                ctypes.byref(descriptor),
                c.descriptor,
                a.descriptor,
                b.descriptor,
            )
        )

        for t in (a, b, c):
            t.destroy_desc()

        workspace_size = c_size_t(0)
        check_error(
            LIBINFINIOP.infiniopGetGemmWorkspaceSize(
                descriptor, ctypes.byref(workspace_size)
            )
        )
        workspace = TestWorkspace(workspace_size.value, device)

        def lib_gemm():
            alpha_ = ctypes.c_float(alpha)
            beta_ = ctypes.c_float(beta)
            check_error(
                LIBINFINIOP.infiniopGemm(
                    descriptor,
                    ctypes.c_void_p(workspace.data() or 0),
                    ctypes.c_size_t(workspace_size.value),
                    ctypes.c_void_p(c.data()),
                    ctypes.c_void_p(a.data()),
                    ctypes.c_void_p(b.data()),
                    alpha_,
                    beta_,
                    None,
                )
            )

        for _ in range(num_prerun):
            lib_gemm()

        elapsed_sec = timed_op(lib_gemm, num_iterations, torch_device_map[device])
        check_error(LIBINFINIOP.infiniopDestroyGemmDescriptor(descriptor))
        return float(elapsed_sec * 1000.0)
    finally:
        destroy_handle(handle)


def _profile_workloads_in_worker(
    device: str,
    dtype: str,
    workloads: list[Workload],
    num_prerun: int,
    num_iterations: int,
) -> list[float]:
    payload = {
        "device": str(device),
        "dtype": str(dtype),
        "num_prerun": int(num_prerun),
        "num_iterations": int(num_iterations),
        "workloads": [{"m": w.m, "n": w.n, "k": w.k} for w in workloads],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as f:
        json.dump(payload, f)
        in_path = f.name
    out_path = f"{in_path}.out.json"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-in",
        in_path,
        "--worker-out",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    data = json.loads(Path(out_path).read_text(encoding="utf-8"))
    times = data.get("times_ms")
    if not isinstance(times, list) or len(times) != len(workloads):
        raise RuntimeError("Worker returned invalid profile results")
    return [float(x) for x in times]


def _worker_main(worker_in: str, worker_out: str):
    payload = json.loads(Path(worker_in).read_text(encoding="utf-8"))
    device_str = str(payload["device"])
    dtype = str(payload["dtype"])
    num_prerun = int(payload["num_prerun"])
    num_iterations = int(payload["num_iterations"])
    ws = payload["workloads"]
    if not isinstance(ws, list):
        raise RuntimeError("Invalid worker input")

    _ensure_libinfiniop_importable()
    from libinfiniop import InfiniDeviceEnum

    if device_str == "nvidia":
        device = InfiniDeviceEnum.NVIDIA
    elif device_str == "cpu":
        device = InfiniDeviceEnum.CPU
    else:
        raise RuntimeError(f"Unsupported device: {device_str}")

    out: list[float] = []
    for idx, w in enumerate(ws, start=1):
        workload = Workload(m=int(w["m"]), n=int(w["n"]), k=int(w["k"]))
        _status(f"Profile {idx}/{len(ws)}: m={workload.m}, n={workload.n}, k={workload.k}")
        out.append(
            profile_gemm_ms(
                device=device,
                dtype=dtype,
                workload=workload,
                num_prerun=num_prerun,
                num_iterations=num_iterations,
            )
        )
        _status(f"Profile 完成: run_time={out[-1]:.6f} ms")

    Path(worker_out).write_text(json.dumps({"times_ms": out}), encoding="utf-8")


def _write_csv_row(path: Path, row: dict[str, Any]):
    header = [
        "dtype",
        "block_m",
        "block_n",
        "block_k",
        "unroll",
        "num_warps",
        "num_stages",
        "m",
        "n",
        "k",
        "run_time",
    ]
    new_file = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", default=str(YAML_PATH))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "test" / "profile-test" / "gemm_profile.csv"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--schedule-samples", type=int, default=1)
    parser.add_argument("--max-total-schedule", type=int, default=1000)
    parser.add_argument("--prefill-samples", type=int, default=20)
    parser.add_argument("--decode-samples", type=int, default=20)
    parser.add_argument("--num-prerun", type=int, default=1000)
    parser.add_argument("--num-iterations", type=int, default=10000)
    parser.add_argument("--device", choices=["nvidia", "cpu"], default="nvidia")
    parser.add_argument("--ntops-path", default="")
    parser.add_argument("--ninetoothed-path", default="")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-in", default="")
    parser.add_argument("--worker-out", default="")
    args = parser.parse_args()

    if args.worker:
        if args.worker_in == "" or args.worker_out == "":
            raise RuntimeError("--worker requires --worker-in and --worker-out")
        _worker_main(args.worker_in, args.worker_out)
        return

    _status("GEMM profile 脚本启动")
    _status(f"project_root={PROJECT_ROOT}")
    _status(f"yaml={args.yaml}")
    _status(f"output={args.output}")
    _status(
        "params="
        + f"seed={args.seed}, rounds={args.rounds}, schedule_samples={args.schedule_samples}, "
        + f"max_total_schedule={args.max_total_schedule}, prefill_samples={args.prefill_samples}, "
        + f"decode_samples={args.decode_samples}, num_prerun={args.num_prerun}, "
        + f"num_iterations={args.num_iterations}, device={args.device}"
    )
    _status(
        "flags="
        + f"dry_run={args.dry_run}, skip_build={args.skip_build}, skip_install={args.skip_install}, "
        + f"skip_cleanup={args.skip_cleanup}"
    )

    cfg = load_simple_yaml(Path(args.yaml))
    schedule_cfg = cfg["schedule"]
    workload_cfg = cfg["workload"]

    schedule_space = {
        "dtype": list(schedule_cfg["dtype"]),
        "block_m": list(schedule_cfg["block_m"]),
        "block_n": list(schedule_cfg["block_n"]),
        "block_k": list(schedule_cfg["block_k"]),
        "unroll": list(schedule_cfg["unroll"]),
        "num_warps": list(schedule_cfg["num_warps"]),
        "num_stages": list(schedule_cfg["num_stages"]),
    }
    device = args.device

    build_gemm = _load_build_gemm_func(BUILD_GEMM_NT_PATH)

    total_schedule = 0
    out_path = Path(args.output)
    _status(f"CSV 输出路径: {out_path}")

    for r in range(args.rounds):
        if total_schedule >= args.max_total_schedule:
            _status("已达到 max_total_schedule，停止")
            break
        remaining = args.max_total_schedule - total_schedule
        n_schedule = min(args.schedule_samples, remaining)
        _status(f"Round {r + 1}/{args.rounds}: 采样 schedule 数量={n_schedule} (remaining={remaining})")
        schedule_dicts = lhs_sample_discrete(
            schedule_space, n_schedule, seed=args.seed + 1000 * r
        )
        schedules = [schedule_from_dict(d) for d in schedule_dicts]

        for s in schedules:
            total_schedule += 1
            _status(
                f"Schedule {total_schedule}/{args.max_total_schedule}: "
                + f"dtype={s.dtype}, block_m={s.block_m}, block_n={s.block_n}, block_k={s.block_k}, "
                + f"unroll={s.unroll}, num_warps={s.num_warps}, num_stages={s.num_stages}"
            )
            workloads = sample_workloads(
                workload_cfg,
                prefill_count=args.prefill_samples,
                decode_count=args.decode_samples,
                seed=args.seed + 1000 * r + total_schedule,
            )
            _status(f"生成 workloads 完成，总数={len(workloads)} (prefill={args.prefill_samples}, decode={args.decode_samples})")

            if args.dry_run:
                _status("dry-run: 跳过构建/编译/profile，仅写入 CSV 占位行")
                for w in workloads:
                    _write_csv_row(
                        out_path,
                        {
                            "dtype": s.dtype,
                            "block_m": s.block_m,
                            "block_n": s.block_n,
                            "block_k": s.block_k,
                            "unroll": s.unroll,
                            "num_warps": s.num_warps,
                            "num_stages": s.num_stages,
                            "m": w.m,
                            "n": w.n,
                            "k": w.k,
                            "run_time": "",
                        },
                    )
                continue

            if not args.skip_build:
                _status("开始构建算子 (build_gemm)")
                build_gemm(
                    ntops_path=args.ntops_path,
                    ninetoothed_path=args.ninetoothed_path,
                    dtype=s.dtype,
                    block_m=str(s.block_m),
                    block_n=str(s.block_n),
                    block_k=str(s.block_k),
                    unroll=str(s.unroll),
                    num_warps=int(s.num_warps),
                    num_stages=int(s.num_stages),
                    skip_cleanup=bool(args.skip_cleanup),
                )
                _status("算子构建完成")
            else:
                _status("skip-build: 跳过算子构建")

            if not args.skip_install:
                os.chdir(PROJECT_ROOT)
                rebuild_infiniop_gemm()
            else:
                _status("skip-install: 跳过重新编译算子库")

            _status("开始 profile workloads")
            times_ms = _profile_workloads_in_worker(
                device=device,
                dtype=s.dtype,
                workloads=workloads,
                num_prerun=args.num_prerun,
                num_iterations=args.num_iterations,
            )
            for w, run_time_ms in zip(workloads, times_ms):
                _write_csv_row(
                    out_path,
                    {
                        "dtype": s.dtype,
                        "block_m": s.block_m,
                        "block_n": s.block_n,
                        "block_k": s.block_k,
                        "unroll": s.unroll,
                        "num_warps": s.num_warps,
                        "num_stages": s.num_stages,
                        "m": w.m,
                        "n": w.n,
                        "k": w.k,
                        "run_time": f"{run_time_ms:.6f}",
                    },
                )
            _status("当前 schedule 的 workloads 全部完成")
    _status("脚本运行结束")


if __name__ == "__main__":
    main()
