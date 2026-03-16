import torch
import ctypes
from ctypes import c_uint64
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
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # Llama2-like linear GEMM cases (flattened N = bs * seq_len)
    # alpha, beta, a_shape, b_shape, c_shape, a_stride, b_stride, c_stride

    # ---- Prefill phase (e.g., bs=1, seq=128 => N=128) ----
    # Q/K/V projection: [N, hidden] x [hidden, hidden] -> [N, hidden]
    (1.0, 0.0, (128, 4096), (4096, 4096), (128, 4096), None, None, None),
    # Q/K/V projection with bias already materialized in C (beta=1 path)
    (1.0, 1.0, (128, 4096), (4096, 4096), (128, 4096), None, None, None),

    # MLP up/gate projection: [N, hidden] x [hidden, intermediate] -> [N, intermediate]
    (1.0, 0.0, (128, 4096), (4096, 11008), (128, 11008), None, None, None),
    # MLP down projection: [N, intermediate] x [intermediate, hidden] -> [N, hidden]
    (1.0, 0.0, (128, 11008), (11008, 4096), (128, 4096), None, None, None),

    # ---- Decode phase (e.g., bs=1, seq=1 => N=1) ----
    # Attention/MLP projections at single-token step
    (1.0, 0.0, (1, 4096), (4096, 4096), (1, 4096), None, None, None),
    # LM head: [N, hidden] x [hidden, vocab] -> [N, vocab]
    (1.0, 0.0, (1, 4096), (4096, 32000), (1, 32000), None, None, None),
]

# Data types used for testing
_TENSOR_DTYPES = [
    InfiniDtype.F16,
    # InfiniDtype.BF16,
    # InfiniDtype.F32,
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 1e-2},
    # InfiniDtype.F32: {"atol": 0, "rtol": 1e-3},
    # InfiniDtype.BF16: {"atol": 0, "rtol": 5e-2},
}

DEBUG = True
PROFILE = False
NUM_PRERUN = 100
NUM_ITERATIONS = 10000


# PyTorch implementation for matrix multiplication
def gemm(d, _c, beta, _a, _b, alpha):
    try:
        if _c.ndim == 2:
            torch.addmm(_c, _a, _b, beta=beta, alpha=alpha, out=d)
        elif _c.ndim == 3:
            torch.baddbmm(_c, _a, _b, beta=beta, alpha=alpha, out=d)
        else:
            raise
    except Exception:
        torch.matmul(_a, _b, out=d)
        d.mul_(alpha).add_(_c, alpha=beta)


# The argument list should be (lib, handle, torch_device, <param list>, dtype)
# The <param list> should keep the same order as the one specified in _TEST_CASES
def test(
    handle,
    device,
    alpha,
    beta,
    a_shape,
    b_shape,
    c_shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing Gemm on {InfiniDeviceNames[device]} with alpha:{alpha}, beta:{beta},"
        f" a_shape:{a_shape}, b_shape:{b_shape}, c_shape:{c_shape},"
        f" a_stride:{a_stride}, b_stride:{b_stride}, c_stride:{c_stride}, dtype:{InfiniDtypeNames[dtype]}"
    )

    # Initialize tensors
    a = TestTensor(a_shape, a_stride, dtype, device)
    b = TestTensor(b_shape, b_stride, dtype, device)
    c = TestTensor(c_shape, c_stride, dtype, device, mode="ones")
    ans = TestTensor(c_shape, c_stride, dtype, device, mode="zeros")

    # Compute the PyTorch reference result
    def torch_gemm():
        gemm(
            ans.torch_tensor(),
            c.torch_tensor(),
            beta,
            a.torch_tensor(),
            b.torch_tensor(),
            alpha,
        )

    torch_gemm()

    if sync is not None:
        sync()

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

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [a, b, c]:
        tensor.destroy_desc()

    # Get workspace size and create workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetGemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Execute infiniop gemm operator
    def lib_gemm():
        check_error(
            LIBINFINIOP.infiniopGemm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                c.data(),
                a.data(),
                b.data(),
                alpha,
                beta,
                None,
            )
        )

    lib_gemm()

    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    if DEBUG:
        debug(c.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(c.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    lib_ms = None
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
        lib_ms = float(profile_operation("    lib", lambda: lib_gemm(), device, NUM_PRERUN, NUM_ITERATIONS) * 1000.0)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyGemmDescriptor(descriptor))
    return lib_ms


# ==============================================================================
#  Main Execution
# ==============================================================================
if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
