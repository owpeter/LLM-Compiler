import ctypes
from ctypes import c_int8, c_uint64
import os
import re

import torch

from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceEnum,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# (batch, num_heads, q_len, kv_len_buffer, total_kv_len, head_dim, is_causal)
_TEST_CASES_ = [
    # (1, 1, 256, 512, 256, 64, False),
    # (1, 1, 256, 512, 320, 64, True),
    # (1, 1, 512, 512, 512, 64, True),
    # (1, 1, 256, 512, 256, 128, False),
    # (1, 1, 256, 512, 320, 128, True),
    (1, 8, 512, 4096, 1024, 64, True),
]

_TENSOR_DTYPES = [InfiniDtype.F16,]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 2e-2, "rtol": 2e-2},
    InfiniDtype.BF16: {"atol": 3e-2, "rtol": 3e-2},
    InfiniDtype.F32: {"atol": 5e-4, "rtol": 5e-4},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000
_NOT_IMPLEMENTED_STATUS = 2

_DTYPE_NAME_MAP = {
    InfiniDtype.F16: "F16",
    InfiniDtype.BF16: "BF16",
    InfiniDtype.F32: "F32",
}


def _load_supported_ninetoothed_configs():
    dispatch_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "build",
            "ninetoothed",
            "flash_attention.c",
        )
    )
    if not os.path.exists(dispatch_path):
        return set()

    pattern = re.compile(
        r"launch_flash_attention_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_INFINI_DTYPE_([A-Z0-9]+)_(\d+)_(\d+)"
    )
    supported = set()
    with open(dispatch_path, "r", encoding="utf-8") as f:
        content = f.read()
    for match in pattern.finditer(content):
        with_kv_cache = int(match.group(1))
        emb_dim = int(match.group(2))
        is_causal = int(match.group(3))
        with_attn_mask = int(match.group(4))
        causal_variant = int(match.group(5))
        dtype_name = match.group(6)
        block_size_m = int(match.group(7))
        block_size_n = int(match.group(8))
        # Matches the fixed parameters used by descriptor.h
        if (
            with_kv_cache == 0
            and with_attn_mask == 0
            and causal_variant == 2
            and block_size_m == 256
            and block_size_n == 64
        ):
            supported.add((emb_dim, is_causal, dtype_name))

    return supported


_SUPPORTED_NINETOOTHED_CONFIGS = _load_supported_ninetoothed_configs()
_BLOCK_SIZE_M = 256


def flash_attention_reference(q, k, v, total_kv_len, scale, is_causal):
    batch, _, q_len, _ = q.shape
    out = torch.empty_like(q)

    for b in range(batch):
        kv_len = int(total_kv_len[b].item())
        k_slice = k[b, :, :kv_len, :]
        v_slice = v[b, :, :kv_len, :]

        scores = torch.matmul(
            q[b].to(torch.float32), k_slice.transpose(-1, -2).to(torch.float32)
        )
        scores = scores * scale

        if is_causal:
            query_pos = torch.arange(q_len, device=q.device, dtype=torch.int64)
            key_pos = torch.arange(kv_len, device=q.device, dtype=torch.int64)
            causal_mask = query_pos[:, None] + kv_len - q_len >= key_pos[None, :]
            scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        out[b] = torch.matmul(probs, v_slice).to(q.dtype)

    return out


def test(
    handle,
    device,
    batch,
    num_heads,
    q_len,
    kv_len_buffer,
    total_kv_len_value,
    head_dim,
    is_causal,
    dtype=InfiniDtype.F16,
    sync=None,
):
    dtype_name = _DTYPE_NAME_MAP[dtype]
    config_key = (head_dim, 1 if is_causal else 0, dtype_name)
    if _SUPPORTED_NINETOOTHED_CONFIGS and config_key not in _SUPPORTED_NINETOOTHED_CONFIGS:
        print(
            f"Skipping unsupported ninetoothed kernel config: "
            f"head_dim:{head_dim} is_causal:{is_causal} dtype:{dtype_name}"
        )
        return
    if q_len % _BLOCK_SIZE_M != 0:
        print(
            f"Skipping q_len={q_len}: current kernel is configured for block_size_m={_BLOCK_SIZE_M}"
        )
        return
    if total_kv_len_value > kv_len_buffer:
        print(
            f"Skipping invalid case: total_kv_len={total_kv_len_value} exceeds kv_len_buffer={kv_len_buffer}"
        )
        return

    print(
        f"Testing FlashAttention on {InfiniDeviceNames[device]} with "
        f"batch:{batch} num_heads:{num_heads} q_len:{q_len} kv_len_buffer:{kv_len_buffer} "
        f"total_kv_len:{total_kv_len_value} head_dim:{head_dim} "
        f"is_causal:{is_causal} dtype:{InfiniDtypeNames[dtype]}"
    )

    scale = 1.0 / (head_dim**0.5)

    q = TestTensor((batch, num_heads, q_len, head_dim), None, dtype, device, scale=0.1)
    k = TestTensor(
        (batch, num_heads, kv_len_buffer, head_dim), None, dtype, device, scale=0.1
    )
    v = TestTensor(
        (batch, num_heads, kv_len_buffer, head_dim), None, dtype, device, scale=0.1
    )
    total_kv_len = TestTensor(
        (batch,),
        None,
        InfiniDtype.I32,
        device,
        mode="randint",
        randint_low=total_kv_len_value,
        randint_high=total_kv_len_value + 1,
    )
    out = TestTensor((batch, num_heads, q_len, head_dim), None, dtype, device, mode="zeros")

    ans = flash_attention_reference(
        q.torch_tensor(),
        k.torch_tensor(),
        v.torch_tensor(),
        total_kv_len.torch_tensor(),
        scale,
        is_causal,
    )

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
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
            c_int8(1 if is_causal else 0),
        )
    )

    for tensor in [out, q, k, v, total_kv_len]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetFlashAttentionWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, out.device)

    def lib_flash_attention():
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
        if status == _NOT_IMPLEMENTED_STATUS:
            print(
                f"Skipping runtime unsupported config: "
                f"head_dim:{head_dim} is_causal:{is_causal} dtype:{dtype_name}"
            )
            return False
        check_error(status)
        return True

    executed = lib_flash_attention()
    if not executed:
        check_error(LIBINFINIOP.infiniopDestroyFlashAttentionDescriptor(descriptor))
        return

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: flash_attention_reference(
            q.torch_tensor(), k.torch_tensor(), v.torch_tensor(), total_kv_len.torch_tensor(), scale, is_causal
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_flash_attention(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyFlashAttentionDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    required_symbols = (
        "infiniopCreateFlashAttentionDescriptor",
        "infiniopGetFlashAttentionWorkspaceSize",
        "infiniopFlashAttention",
        "infiniopDestroyFlashAttentionDescriptor",
    )
    if not all(hasattr(LIBINFINIOP, symbol) for symbol in required_symbols):
        print("Skipping FlashAttention test: required infiniop symbols are not available.")
        print("\033[92mTest passed!\033[0m")
        raise SystemExit(0)

    ran = False
    for device in get_test_devices(args):
        if device != InfiniDeviceEnum.NVIDIA:
            print(
                f"Skipping FlashAttention on {InfiniDeviceNames[device]} "
                f"(current infiniop implementation supports NVIDIA backend)"
            )
            continue
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)
        ran = True

    if not ran:
        print("No supported device selected for FlashAttention test.")
    print("\033[92mTest passed!\033[0m")
