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
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


CURRENT_FILE_PATH = Path(__file__).resolve()


def _find_project_root(start: Path) -> Path:
    for candidate in [start.parent, *start.parents]:
        if (candidate / "src" / "infiniop").is_dir():
            return candidate
    return start.parents[3]


PROJECT_ROOT = _find_project_root(CURRENT_FILE_PATH)
YAML_PATH = PROJECT_ROOT / "scripts" / "profile" / "rms_norm" / "rms_norm.yaml"
BUILD_RMS_NORM_NT_PATH = (
    PROJECT_ROOT / "scripts" / "profile" / "rms_norm" / "build_rms_norm_nt.py"
)
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
    if s == "True":
        return True
    if s == "False":
        return False
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
                continue
            if ":" in item_text:
                key, rest = item_text.split(":", 1)
                key = key.strip()
                rest = rest.strip()
                if rest == "":
                    child, i = _parse_block(lines, i + 1, indent + 4)
                    out.append({key: child})
                else:
                    out.append({key: _parse_scalar(rest)})
                continue
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
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Root YAML must be a mapping")
    return obj


def _flatten_single_key_dict_list(items: list[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"Invalid config item: {item}")
        k = next(iter(item.keys()))
        out[str(k)] = item[k]
    return out


def parse_rms_norm_config(path: Path) -> tuple[dict[str, list[Any]], dict[str, dict[str, list[Any]]]]:
    cfg = load_simple_yaml(path)
    schedule_raw = cfg.get("schedule")
    workload_raw = cfg.get("workload")
    if not isinstance(schedule_raw, list) or not isinstance(workload_raw, list):
        raise ValueError("Invalid rms_norm.yaml format")

    schedule_map = _flatten_single_key_dict_list(schedule_raw)
    schedule_space = {
        "block_size_value": list(schedule_map["block_size_value"]),
        "num_warps": list(schedule_map["num_warps"]),
        "num_stages": list(schedule_map["num_stages"]),
        "use_vectorized_application": list(schedule_map["use_vectorized_application"]),
    }

    workload_map = _flatten_single_key_dict_list(workload_raw)
    prefill_map = _flatten_single_key_dict_list(list(workload_map["prefill"]))
    decode_map = _flatten_single_key_dict_list(list(workload_map["decode"]))
    workload_space = {
        "prefill": {
            "batch_size": list(prefill_map["batch_size"]),
            "sequence_length": list(prefill_map["sequence_length"]),
            "hidden_size": list(prefill_map["hidden_size"]),
        },
        "decode": {
            "batch_size": list(decode_map["batch_size"]),
            "sequence_length": list(decode_map["sequence_length"]),
            "hidden_size": list(decode_map["hidden_size"]),
        },
    }
    return schedule_space, workload_space


def lhs_sample_discrete(
    space: dict[str, list[Any]],
    n: int,
    seed: int,
    max_unique_attempts: int = 20,
    allow_repeat: bool = False,
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
    max_unique = 1
    for k in keys:
        max_unique *= len(space[k])
    target_unique = min(n, max_unique)
    lattice_size = max(1, min(n, max_unique))
    attempts = 0

    while len(samples) < target_unique and attempts < max_unique_attempts:
        perms = {k: rng.sample(range(lattice_size), lattice_size) for k in keys}
        for i in range(lattice_size):
            combo: dict[str, Any] = {}
            signature: list[Any] = []
            for k in keys:
                u = (perms[k][i] + rng.random()) / lattice_size
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
            if len(samples) >= target_unique:
                break
        attempts += 1

    if len(samples) < target_unique:
        raise RuntimeError(
            f"Failed to sample {target_unique} unique combos, got {len(samples)}. Reduce n or adjust space."
        )
    if len(samples) < n:
        if not allow_repeat:
            raise RuntimeError(
                f"Failed to sample {n} unique combos, max available is {max_unique}. "
                + "Reduce n or enable repeated sampling."
            )
        while len(samples) < n:
            samples.append(dict(rng.choice(samples)))
    return samples


@dataclass(frozen=True)
class ScheduleParams:
    block_size_value: int
    num_warps: int
    num_stages: int
    use_vectorized_application: bool


@dataclass(frozen=True)
class Workload:
    task: str
    batch_size: int
    sequence_length: int
    hidden_size: int


def schedule_from_dict(d: dict[str, Any]) -> ScheduleParams:
    return ScheduleParams(
        block_size_value=int(d["block_size_value"]),
        num_warps=int(d["num_warps"]),
        num_stages=int(d["num_stages"]),
        use_vectorized_application=bool(d["use_vectorized_application"]),
    )


def sample_workloads(
    workload_cfg: dict[str, dict[str, list[Any]]],
    prefill_count: int,
    decode_count: int,
    seed: int,
) -> list[Workload]:
    prefill_space = {
        "batch_size": list(workload_cfg["prefill"]["batch_size"]),
        "sequence_length": list(workload_cfg["prefill"]["sequence_length"]),
        "hidden_size": list(workload_cfg["prefill"]["hidden_size"]),
    }
    decode_space = {
        "batch_size": list(workload_cfg["decode"]["batch_size"]),
        "sequence_length": list(workload_cfg["decode"]["sequence_length"]),
        "hidden_size": list(workload_cfg["decode"]["hidden_size"]),
    }
    prefill = lhs_sample_discrete(
        prefill_space, prefill_count, seed=seed + 101, allow_repeat=True
    )
    decode = lhs_sample_discrete(
        decode_space, decode_count, seed=seed + 202, allow_repeat=True
    )

    out: list[Workload] = []
    for d in prefill:
        out.append(
            Workload(
                task="prefill",
                batch_size=int(d["batch_size"]),
                sequence_length=int(d["sequence_length"]),
                hidden_size=int(d["hidden_size"]),
            )
        )
    for d in decode:
        out.append(
            Workload(
                task="decode",
                batch_size=int(d["batch_size"]),
                sequence_length=int(d["sequence_length"]),
                hidden_size=int(d["hidden_size"]),
            )
        )
    return out


def _load_build_rms_norm_func(path: Path):
    spec = importlib.util.spec_from_file_location("build_rms_norm_nt", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_rms_norm_nt"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_rms_norm"):
        raise RuntimeError("build_rms_norm not found in build_rms_norm_nt.py")
    return getattr(mod, "build_rms_norm")


def _ensure_libinfiniop_importable():
    p = str(INFINIOP_TEST_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def rebuild_infiniop_rms_norm():
    installer = "scripts/install.py"
    if not (PROJECT_ROOT / installer).exists():
        installer = "scripts/install"
    cmd = [
        sys.executable,
        installer,
        "--nv-gpu=y",
        "--ninetoothed=y",
        "--ops",
        "rms_norm",
    ]
    env = os.environ.copy()
    start = time.time()
    _status("开始重新编译 Infinicore 算子库 (ops=rms_norm)")
    p = os.spawnve(os.P_WAIT, sys.executable, cmd, env)
    if p != 0:
        raise RuntimeError(f"Rebuild failed with exit code {p}")
    elapsed = time.time() - start
    _status(f"重新编译完成，用时 {elapsed:.2f}s")


def profile_rms_norm_ms(
    device,
    dtype: str,
    workload: Workload,
    num_prerun: int,
    num_iterations: int,
) -> float:
    _ensure_libinfiniop_importable()
    import torch
    from ctypes import c_uint64
    from libinfiniop import (
        LIBINFINIOP,
        TestTensor,
        TestWorkspace,
        check_error,
        create_handle,
        destroy_handle,
        get_sync_func,
        profile_operation,
        torch_device_map,
        InfiniDtype,
        infiniopOperatorDescriptor_t,
    )

    if torch_device_map[device] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but NVIDIA device selected")

    dtype_map = {
        "float16": InfiniDtype.F16,
        "bfloat16": InfiniDtype.BF16,
        "float32": InfiniDtype.F32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    dt = dtype_map[dtype]

    y_shape = (workload.batch_size, workload.sequence_length, workload.hidden_size)
    x_shape = y_shape
    w_shape = (workload.hidden_size,)
    eps = 1e-6

    check_error(LIBINFINIOP.infinirtSetDevice(device, ctypes.c_int(0)))
    handle = create_handle()
    descriptor = infiniopOperatorDescriptor_t()
    try:
        y = TestTensor(y_shape, None, dt, device, mode="ones")
        x = TestTensor(x_shape, None, dt, device, scale=0.01)
        w = TestTensor(w_shape, None, dt, device)

        sync = get_sync_func(device)
        sync()
        check_error(
            LIBINFINIOP.infiniopCreateRMSNormDescriptor(
                handle,
                ctypes.byref(descriptor),
                y.descriptor,
                x.descriptor,
                w.descriptor,
                eps,
            )
        )

        for tensor in [x, y, w]:
            tensor.destroy_desc()

        workspace_size = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetRMSNormWorkspaceSize(
                descriptor, ctypes.byref(workspace_size)
            )
        )
        workspace = TestWorkspace(workspace_size.value, y.device)

        def lib_rms_norm():
            check_error(
                LIBINFINIOP.infiniopRMSNorm(
                    descriptor,
                    workspace.data(),
                    workspace_size.value,
                    y.data(),
                    x.data(),
                    w.data(),
                    None,
                )
            )

        lib_rms_norm()
        lib_sec = float(
            profile_operation(
                "    lib",
                lambda: lib_rms_norm(),
                device,
                int(num_prerun),
                int(num_iterations),
            )
        )
        return lib_sec * 1000.0
    finally:
        check_error(LIBINFINIOP.infiniopDestroyRMSNormDescriptor(descriptor))
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
        "workloads": [
            {
                "task": w.task,
                "batch_size": w.batch_size,
                "sequence_length": w.sequence_length,
                "hidden_size": w.hidden_size,
            }
            for w in workloads
        ],
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
        workload = Workload(
            task=str(w["task"]),
            batch_size=int(w["batch_size"]),
            sequence_length=int(w["sequence_length"]),
            hidden_size=int(w["hidden_size"]),
        )
        _status(
            f"Profile {idx}/{len(ws)}: task={workload.task}, "
            + f"batch_size={workload.batch_size}, sequence_length={workload.sequence_length}, "
            + f"hidden_size={workload.hidden_size}"
        )
        out.append(
            profile_rms_norm_ms(
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
        "batch_size",
        "sequence_length",
        "hidden_size",
        "block_size_value",
        "num_warps",
        "num_stages",
        "use_vectorized_application",
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
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "test" / "profile-test" / "rms_norm" / "rms_norm_profile.csv"),
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--schedule-samples", type=int, default=1)
    parser.add_argument("--max-total-schedule", type=int, default=100)
    parser.add_argument("--prefill-samples", type=int, default=20)
    parser.add_argument("--decode-samples", type=int, default=20)
    parser.add_argument("--num-prerun", type=int, default=1000)
    parser.add_argument("--num-iterations", type=int, default=10000)
    parser.add_argument("--device", choices=["nvidia", "cpu"], default="nvidia")
    parser.add_argument("--ntops-path", default="")
    parser.add_argument("--ninetoothed-path", default="")
    parser.add_argument("--ndim", default="3")
    parser.add_argument("--num-normalized-dims", default="1")
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

    _status("RMSNorm profile 脚本启动")
    _status(f"project_root={PROJECT_ROOT}")
    _status(f"yaml={args.yaml}")
    _status(f"output={args.output}")
    _status(
        "params="
        + f"dtype={args.dtype}, seed={args.seed}, rounds={args.rounds}, "
        + f"schedule_samples={args.schedule_samples}, max_total_schedule={args.max_total_schedule}, "
        + f"prefill_samples={args.prefill_samples}, decode_samples={args.decode_samples}, "
        + f"num_prerun={args.num_prerun}, num_iterations={args.num_iterations}, device={args.device}"
    )
    _status(
        "flags="
        + f"dry_run={args.dry_run}, skip_build={args.skip_build}, skip_install={args.skip_install}, "
        + f"skip_cleanup={args.skip_cleanup}"
    )

    schedule_space, workload_cfg = parse_rms_norm_config(Path(args.yaml))
    build_rms_norm = None
    if not args.skip_build and not args.dry_run:
        build_rms_norm = _load_build_rms_norm_func(BUILD_RMS_NORM_NT_PATH)

    total_schedule = 0
    success_schedule = 0
    failed_schedule = 0
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
                + f"block_size_value={s.block_size_value}, num_warps={s.num_warps}, "
                + f"num_stages={s.num_stages}, use_vectorized_application={s.use_vectorized_application}"
            )
            workloads = sample_workloads(
                workload_cfg,
                prefill_count=args.prefill_samples,
                decode_count=args.decode_samples,
                seed=args.seed + 1000 * r + total_schedule,
            )
            _status(
                f"生成 workloads 完成，总数={len(workloads)} "
                + f"(prefill={args.prefill_samples}, decode={args.decode_samples})"
            )

            if args.dry_run:
                _status("dry-run: 跳过构建/编译/profile，仅写入 CSV 占位行")
                for w in workloads:
                    _write_csv_row(
                        out_path,
                        {
                            "dtype": args.dtype,
                            "batch_size": w.batch_size,
                            "sequence_length": w.sequence_length,
                            "hidden_size": w.hidden_size,
                            "block_size_value": s.block_size_value,
                            "num_warps": s.num_warps,
                            "num_stages": s.num_stages,
                            "use_vectorized_application": s.use_vectorized_application,
                            "run_time": "",
                        },
                    )
                continue

            try:
                if not args.skip_build:
                    if build_rms_norm is None:
                        raise RuntimeError("build_rms_norm loader is not initialized")
                    _status("开始构建算子 (build_rms_norm)")
                    build_rms_norm(
                        ntops_path=args.ntops_path,
                        ninetoothed_path=args.ninetoothed_path,
                        dtype=args.dtype,
                        ndim=args.ndim,
                        num_normalized_dims=args.num_normalized_dims,
                        block_size=str(s.block_size_value),
                        num_warps=int(s.num_warps),
                        num_stages=int(s.num_stages),
                        use_vectorized_application=bool(s.use_vectorized_application),
                        skip_cleanup=bool(args.skip_cleanup),
                    )
                    _status("算子构建完成")
                else:
                    _status("skip-build: 跳过算子构建")

                if not args.skip_install:
                    os.chdir(PROJECT_ROOT)
                    rebuild_infiniop_rms_norm()
                else:
                    _status("skip-install: 跳过重新编译算子库")

                _status("开始 profile workloads")
                times_ms = _profile_workloads_in_worker(
                    device=args.device,
                    dtype=args.dtype,
                    workloads=workloads,
                    num_prerun=args.num_prerun,
                    num_iterations=args.num_iterations,
                )
                for w, run_time_ms in zip(workloads, times_ms):
                    _write_csv_row(
                        out_path,
                        {
                            "dtype": args.dtype,
                            "batch_size": w.batch_size,
                            "sequence_length": w.sequence_length,
                            "hidden_size": w.hidden_size,
                            "block_size_value": s.block_size_value,
                            "num_warps": s.num_warps,
                            "num_stages": s.num_stages,
                            "use_vectorized_application": s.use_vectorized_application,
                            "run_time": f"{run_time_ms:.6f}",
                        },
                    )
                _status("当前 schedule 的 workloads 全部完成")
                success_schedule += 1
            except Exception as exc:
                failed_schedule += 1
                _status(
                    "当前 schedule 失败并跳过: "
                    + f"{type(exc).__name__}: {exc}"
                )
                _status(traceback.format_exc())
                continue
    if (not args.dry_run) and success_schedule == 0:
        raise RuntimeError(
            "所有 schedule 均执行失败，未产生有效 profile 数据。"
            + "请先单独排查 build_rms_norm 可用性，或在已有可用算子时使用 --skip-build --skip-install 进行 profile。"
        )
    _status(
        f"脚本运行结束，成功 schedule={success_schedule}，失败 schedule={failed_schedule}"
    )


if __name__ == "__main__":
    main()
