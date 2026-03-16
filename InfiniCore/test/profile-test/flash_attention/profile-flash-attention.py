import argparse
import csv
import ctypes
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback
from ctypes import c_int8, c_uint64
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CURRENT_FILE_PATH = Path(__file__).resolve()


def _find_project_root(start: Path) -> Path:
    for candidate in [start.parent, *start.parents]:
        if (candidate / "src" / "infiniop").is_dir():
            return candidate
    return start.parents[3]


PROJECT_ROOT = _find_project_root(CURRENT_FILE_PATH)
YAML_PATH = PROJECT_ROOT / "scripts" / "profile" / "flash_attention" / "flash_attention.yaml"
BUILD_FLASH_ATTENTION_NT_PATH = (
    PROJECT_ROOT / "scripts" / "profile" / "flash_attention" / "build_flash_attention_nt.py"
)
INFINIOP_TEST_ROOT = PROJECT_ROOT / "test" / "infiniop"
FLASH_ATTENTION_DESCRIPTOR_TMP_PATH = (
    PROJECT_ROOT / "src" / "infiniop" / "ops" / "flash_attention" / "ninetoothed" / "descriptor.h.tmp"
)
FLASH_ATTENTION_DESCRIPTOR_OUT_PATH = FLASH_ATTENTION_DESCRIPTOR_TMP_PATH.with_suffix("")


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
    if s.lower() == "true":
        return True
    if s.lower() == "false":
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
            f"Failed to sample {target_unique} unique combos, got {len(samples)}. "
            + "Reduce n or adjust space."
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


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool value: {v}")


@dataclass(frozen=True)
class ScheduleParams:
    dtype: str
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class Workload:
    task: str
    batch: int
    num_heads: int
    q_len: int
    kv_len_buffer: int
    total_kv_len: int
    head_dim: int
    is_causal: bool


def schedule_from_dict(d: dict[str, Any]) -> ScheduleParams:
    return ScheduleParams(
        dtype=str(d["dtype"]),
        block_m=int(d["block_m"]),
        block_n=int(d["block_n"]),
        num_warps=int(d["num_warps"]),
        num_stages=int(d["num_stages"]),
    )


def sample_workloads(
    workload_cfg: dict[str, Any],
    prefill_count: int,
    decode_count: int,
    seed: int,
) -> list[Workload]:
    prefill_space = {
        "batch": list(workload_cfg["prefill"]["batch"]),
        "num_heads": list(workload_cfg["prefill"]["num_heads"]),
        "q_len": list(workload_cfg["prefill"]["q_len"]),
        "kv_len_buffer": list(workload_cfg["prefill"]["kv_len_buffer"]),
        "total_kv_len": list(workload_cfg["prefill"]["total_kv_len"]),
        "head_dim": list(workload_cfg["prefill"]["head_dim"]),
        "is_causal": list(workload_cfg["prefill"]["is_causal"]),
    }
    decode_space = {
        "batch": list(workload_cfg["decode"]["batch"]),
        "num_heads": list(workload_cfg["decode"]["num_heads"]),
        "q_len": list(workload_cfg["decode"]["q_len"]),
        "kv_len_buffer": list(workload_cfg["decode"]["kv_len_buffer"]),
        "total_kv_len": list(workload_cfg["decode"]["total_kv_len"]),
        "head_dim": list(workload_cfg["decode"]["head_dim"]),
        "is_causal": list(workload_cfg["decode"]["is_causal"]),
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
                batch=int(d["batch"]),
                num_heads=int(d["num_heads"]),
                q_len=int(d["q_len"]),
                kv_len_buffer=int(d["kv_len_buffer"]),
                total_kv_len=int(d["total_kv_len"]),
                head_dim=int(d["head_dim"]),
                is_causal=_to_bool(d["is_causal"]),
            )
        )
    for d in decode:
        out.append(
            Workload(
                task="decode",
                batch=int(d["batch"]),
                num_heads=int(d["num_heads"]),
                q_len=int(d["q_len"]),
                kv_len_buffer=int(d["kv_len_buffer"]),
                total_kv_len=int(d["total_kv_len"]),
                head_dim=int(d["head_dim"]),
                is_causal=_to_bool(d["is_causal"]),
            )
        )
    return out


def _load_build_flash_attention_func(path: Path):
    spec = importlib.util.spec_from_file_location("build_flash_attention_nt", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_flash_attention_nt"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_flash_attention"):
        raise RuntimeError("build_flash_attention not found in build_flash_attention_nt.py")
    return getattr(mod, "build_flash_attention")


def _ensure_libinfiniop_importable():
    p = str(INFINIOP_TEST_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def rebuild_infiniop_flash_attention():
    installer = "scripts/install"
    if not (PROJECT_ROOT / installer).exists():
        installer = "scripts/install.py"
    cmd = [
        sys.executable,
        installer,
        "--nv-gpu=y",
        "--ninetoothed=y",
        "--ops",
        "flash_attention",
    ]
    env = os.environ.copy()
    start = time.time()
    _status("开始重新编译 Infinicore 算子库 (ops=flash_attention)")
    p = os.spawnve(os.P_WAIT, sys.executable, cmd, env)
    if p != 0:
        raise RuntimeError(f"Rebuild failed with exit code {p}")
    elapsed = time.time() - start
    _status(f"重新编译完成，用时 {elapsed:.2f}s")


def _render_flash_attention_descriptor(template_text: str, s: ScheduleParams) -> str:
    block_m_pattern = re.compile(
        r"(constexpr\s+auto\s+block_size_m_\s*\{)\s*[^}]+\s*(\};)"
    )
    block_n_pattern = re.compile(
        r"(constexpr\s+auto\s+block_size_n_\s*\{)\s*[^}]+\s*(\};)"
    )
    rendered, count_m = block_m_pattern.subn(rf"\g<1>{s.block_m}\g<2>", template_text, count=1)
    rendered, count_n = block_n_pattern.subn(rf"\g<1>{s.block_n}\g<2>", rendered, count=1)
    if count_m != 1:
        raise RuntimeError("Failed to patch block_size_m_ in descriptor.h.tmp")
    if count_n != 1:
        raise RuntimeError("Failed to patch block_size_n_ in descriptor.h.tmp")
    return rendered


def _materialize_flash_attention_descriptor(s: ScheduleParams):
    template_text = FLASH_ATTENTION_DESCRIPTOR_TMP_PATH.read_text(encoding="utf-8")
    rendered = _render_flash_attention_descriptor(template_text, s)
    FLASH_ATTENTION_DESCRIPTOR_OUT_PATH.write_text(rendered, encoding="utf-8")
    _status(
        "已更新 descriptor.h: "
        + f"block_size_m_={s.block_m}, block_size_n_={s.block_n}"
    )


def profile_flash_attention_ms(
    device,
    dtype: str,
    workload: Workload,
    num_prerun: int,
    num_iterations: int,
) -> float:
    if workload.total_kv_len > workload.kv_len_buffer:
        raise ValueError(
            "Invalid workload: total_kv_len exceeds kv_len_buffer, "
            + f"{workload.total_kv_len} > {workload.kv_len_buffer}"
        )

    _ensure_libinfiniop_importable()
    import torch
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

    scale = 1.0 / (workload.head_dim**0.5)
    not_implemented_status = 2

    check_error(LIBINFINIOP.infinirtSetDevice(device, ctypes.c_int(0)))
    handle = create_handle()
    descriptor = infiniopOperatorDescriptor_t()
    descriptor_created = False
    try:
        q = TestTensor(
            (workload.batch, workload.num_heads, workload.q_len, workload.head_dim),
            None,
            dt,
            device,
            scale=0.1,
        )
        k = TestTensor(
            (workload.batch, workload.num_heads, workload.kv_len_buffer, workload.head_dim),
            None,
            dt,
            device,
            scale=0.1,
        )
        v = TestTensor(
            (workload.batch, workload.num_heads, workload.kv_len_buffer, workload.head_dim),
            None,
            dt,
            device,
            scale=0.1,
        )
        total_kv_len = TestTensor(
            (workload.batch,),
            None,
            InfiniDtype.I32,
            device,
            mode="randint",
            randint_low=workload.total_kv_len,
            randint_high=workload.total_kv_len + 1,
        )
        out = TestTensor(
            (workload.batch, workload.num_heads, workload.q_len, workload.head_dim),
            None,
            dt,
            device,
            mode="zeros",
        )

        sync = get_sync_func(device)
        if sync is not None:
            sync()

        check_error(
            LIBINFINIOP.infiniopCreateFlashAttentionDescriptor(
                handle,
                ctypes.byref(descriptor),
                out.descriptor,
                q.descriptor,
                k.descriptor,
                v.descriptor,
                total_kv_len.descriptor,
                scale,
                c_int8(1 if workload.is_causal else 0),
            )
        )
        descriptor_created = True

        for tensor in [out, q, k, v, total_kv_len]:
            tensor.destroy_desc()

        workspace_size = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetFlashAttentionWorkspaceSize(
                descriptor, ctypes.byref(workspace_size)
            )
        )
        workspace = TestWorkspace(workspace_size.value, out.device)

        def _run_once() -> bool:
            status = LIBINFINIOP.infiniopFlashAttention(
                descriptor,
                workspace.data(),
                workspace_size.value,
                out.data(),
                q.data(),
                k.data(),
                v.data(),
                total_kv_len.data(),
                None,
            )
            if status == not_implemented_status:
                return False
            check_error(status)
            return True

        def _run_once_checked():
            if not _run_once():
                raise RuntimeError(
                    "Runtime unsupported flash_attention config: "
                    + f"q_len={workload.q_len}, kv_len_buffer={workload.kv_len_buffer}, "
                    + f"head_dim={workload.head_dim}, is_causal={workload.is_causal}"
                )

        _run_once_checked()
        elapsed = float(
            profile_operation(
                "    lib",
                lambda: _run_once_checked(),
                device,
                int(num_prerun),
                int(num_iterations),
            )
        )
        return elapsed * 1000.0
    finally:
        if descriptor_created:
            check_error(LIBINFINIOP.infiniopDestroyFlashAttentionDescriptor(descriptor))
        destroy_handle(handle)


def _profile_workloads_in_worker(
    device: str,
    dtype: str,
    workloads: list[Workload],
    num_prerun: int,
    num_iterations: int,
) -> list[float | None]:
    payload = {
        "device": str(device),
        "dtype": str(dtype),
        "num_prerun": int(num_prerun),
        "num_iterations": int(num_iterations),
        "workloads": [
            {
                "task": w.task,
                "batch": w.batch,
                "num_heads": w.num_heads,
                "q_len": w.q_len,
                "kv_len_buffer": w.kv_len_buffer,
                "total_kv_len": w.total_kv_len,
                "head_dim": w.head_dim,
                "is_causal": w.is_causal,
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
    return [None if x is None else float(x) for x in times]


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

    out: list[float | None] = []
    for idx, w in enumerate(ws, start=1):
        workload = Workload(
            task=str(w["task"]),
            batch=int(w["batch"]),
            num_heads=int(w["num_heads"]),
            q_len=int(w["q_len"]),
            kv_len_buffer=int(w["kv_len_buffer"]),
            total_kv_len=int(w["total_kv_len"]),
            head_dim=int(w["head_dim"]),
            is_causal=_to_bool(w["is_causal"]),
        )
        _status(
            f"Profile {idx}/{len(ws)}: task={workload.task}, batch={workload.batch}, "
            + f"num_heads={workload.num_heads}, q_len={workload.q_len}, "
            + f"kv_len_buffer={workload.kv_len_buffer}, total_kv_len={workload.total_kv_len}, "
            + f"head_dim={workload.head_dim}, is_causal={workload.is_causal}"
        )
        try:
            run_time = profile_flash_attention_ms(
                device=device,
                dtype=dtype,
                workload=workload,
                num_prerun=num_prerun,
                num_iterations=num_iterations,
            )
            _status(f"Profile 完成: run_time={run_time:.6f} ms")
            out.append(run_time)
        except Exception as exc:
            _status(f"Profile 失败并跳过: {type(exc).__name__}: {exc}")
            out.append(None)

    Path(worker_out).write_text(json.dumps({"times_ms": out}), encoding="utf-8")


def _write_csv_row(path: Path, row: dict[str, Any]):
    header = [
        "dtype",
        "batch",
        "num_heads",
        "q_len",
        "kv_len_buffer",
        "total_kv_len",
        "head_dim",
        "is_causal",
        "block_m",
        "block_n",
        "num_warps",
        "num_stages",
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
        default=str(
            PROJECT_ROOT
            / "test"
            / "profile-test"
            / "flash_attention"
            / "flash_attention_profile.csv"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--schedule-samples", type=int, default=1)
    parser.add_argument("--max-total-schedule", type=int, default=120)
    parser.add_argument("--prefill-samples", type=int, default=20)
    parser.add_argument("--decode-samples", type=int, default=20)
    parser.add_argument("--num-prerun", type=int, default=10)
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--device", choices=["nvidia", "cpu"], default="nvidia")
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

    _status("FlashAttention profile 脚本启动")
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
        "num_warps": list(schedule_cfg["num_warps"]),
        "num_stages": list(schedule_cfg["num_stages"]),
    }
    build_flash_attention = None
    if not args.skip_build and not args.dry_run:
        build_flash_attention = _load_build_flash_attention_func(BUILD_FLASH_ATTENTION_NT_PATH)

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
        _status(
            f"Round {r + 1}/{args.rounds}: 采样 schedule 数量={n_schedule} (remaining={remaining})"
        )
        schedule_dicts = lhs_sample_discrete(
            schedule_space, n_schedule, seed=args.seed + 1000 * r
        )
        schedules = [schedule_from_dict(d) for d in schedule_dicts]

        for s in schedules:
            total_schedule += 1
            _status(
                f"Schedule {total_schedule}/{args.max_total_schedule}: "
                + f"dtype={s.dtype}, block_m={s.block_m}, block_n={s.block_n}, "
                + f"num_warps={s.num_warps}, num_stages={s.num_stages}"
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
                            "dtype": s.dtype,
                            "batch": w.batch,
                            "num_heads": w.num_heads,
                            "q_len": w.q_len,
                            "kv_len_buffer": w.kv_len_buffer,
                            "total_kv_len": w.total_kv_len,
                            "head_dim": w.head_dim,
                            "is_causal": w.is_causal,
                            "block_m": s.block_m,
                            "block_n": s.block_n,
                            "num_warps": s.num_warps,
                            "num_stages": s.num_stages,
                            "run_time": "",
                        },
                    )
                continue

            try:
                if not args.skip_build:
                    if build_flash_attention is None:
                        raise RuntimeError("build_flash_attention loader is not initialized")
                    _status("开始构建算子 (build_flash_attention)")
                    build_flash_attention(
                        ninetoothed_path=args.ninetoothed_path,
                        dtype=s.dtype,
                        block_size_m=str(s.block_m),
                        block_size_n=str(s.block_n),
                        num_warps=int(s.num_warps),
                        num_stages=int(s.num_stages),
                        skip_cleanup=bool(args.skip_cleanup),
                    )
                    _status("算子构建完成")
                else:
                    _status("skip-build: 跳过算子构建")

                if not args.skip_install:
                    _materialize_flash_attention_descriptor(s)
                    os.chdir(PROJECT_ROOT)
                    rebuild_infiniop_flash_attention()
                else:
                    _status("skip-install: 跳过重新编译算子库")

                _status("开始 profile workloads")
                times_ms = _profile_workloads_in_worker(
                    device=args.device,
                    dtype=s.dtype,
                    workloads=workloads,
                    num_prerun=args.num_prerun,
                    num_iterations=args.num_iterations,
                )

                valid_rows = 0
                for w, run_time_ms in zip(workloads, times_ms):
                    if run_time_ms is not None:
                        valid_rows += 1
                    _write_csv_row(
                        out_path,
                        {
                            "dtype": s.dtype,
                            "batch": w.batch,
                            "num_heads": w.num_heads,
                            "q_len": w.q_len,
                            "kv_len_buffer": w.kv_len_buffer,
                            "total_kv_len": w.total_kv_len,
                            "head_dim": w.head_dim,
                            "is_causal": w.is_causal,
                            "block_m": s.block_m,
                            "block_n": s.block_n,
                            "num_warps": s.num_warps,
                            "num_stages": s.num_stages,
                            "run_time": (
                                f"{run_time_ms:.6f}" if run_time_ms is not None else ""
                            ),
                        },
                    )
                _status(f"当前 schedule 的 workloads 全部完成，成功 {valid_rows}/{len(workloads)}")
                if valid_rows == 0:
                    raise RuntimeError("No valid profile rows in current schedule")
                success_schedule += 1
            except Exception as exc:
                failed_schedule += 1
                _status(f"当前 schedule 失败并跳过: {type(exc).__name__}: {exc}")
                _status(traceback.format_exc())
                continue

    if (not args.dry_run) and success_schedule == 0:
        raise RuntimeError(
            "所有 schedule 均执行失败，未产生有效 profile 数据。"
            + "请先单独排查 build_flash_attention 可用性，或在已有可用算子时使用 "
            + "--skip-build --skip-install 进行 profile。"
        )
    _status(f"脚本运行结束，成功 schedule={success_schedule}，失败 schedule={failed_schedule}")


if __name__ == "__main__":
    main()
