#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/gemm.h"
#include <cstdio>
#include <cstdlib>

#ifdef ENABLE_CPU_API
#include "cpu/gemm_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/gemm_nvidia.cuh"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/gemm_bang.h"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/gemm_ascend.h"
#endif
#if defined(ENABLE_NINETOOTHED) && (defined(ENABLE_NINETOOTHED_ALL_OPS) || defined(ENABLE_NINETOOTHED_OP_GEMM))
#include "ninetoothed/gemm.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/gemm_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/gemm_moore.h"
#endif
#ifdef ENABLE_KUNLUN_API
#include "kunlun/gemm_kunlun.h"
#endif

static const char *gemmDeviceName(infiniDevice_t device) {
    switch (device) {
    case INFINI_DEVICE_CPU:
        return "CPU";
    case INFINI_DEVICE_NVIDIA:
        return "NVIDIA";
    case INFINI_DEVICE_CAMBRICON:
        return "CAMBRICON";
    case INFINI_DEVICE_ASCEND:
        return "ASCEND";
    case INFINI_DEVICE_METAX:
        return "METAX";
    case INFINI_DEVICE_MOORE:
        return "MOORE";
    case INFINI_DEVICE_ILUVATAR:
        return "ILUVATAR";
    case INFINI_DEVICE_KUNLUN:
        return "KUNLUN";
    case INFINI_DEVICE_HYGON:
        return "HYGON";
    case INFINI_DEVICE_QY:
        return "QY";
    case INFINI_DEVICE_ALI:
        return "ALI";
    default:
        return "UNKNOWN";
    }
}

__C infiniStatus_t infiniopCreateGemmDescriptor(
    infiniopHandle_t handle,
    infiniopGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

#define CREATE(CASE, NAMESPACE, IMPL)                                      \
    case CASE:                                                             \
        return op::gemm::NAMESPACE::Descriptor::create(                    \
            handle,                                                         \
            reinterpret_cast<op::gemm::NAMESPACE::Descriptor **>(desc_ptr), \
            c_desc,                                                         \
            a_desc,                                                         \
            b_desc)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu, "cpu");
#endif
#ifdef ENABLE_NVIDIA_API
#if defined(ENABLE_NINETOOTHED) && (defined(ENABLE_NINETOOTHED_ALL_OPS) || defined(ENABLE_NINETOOTHED_OP_GEMM))
        CREATE(INFINI_DEVICE_NVIDIA, ninetoothed, "ninetoothed");
#else
        CREATE(INFINI_DEVICE_NVIDIA, nvidia, "nvidia");
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
#if defined(ENABLE_NINETOOTHED) && (defined(ENABLE_NINETOOTHED_ALL_OPS) || defined(ENABLE_NINETOOTHED_OP_GEMM))
        CREATE(INFINI_DEVICE_ILUVATAR, ninetoothed, "ninetoothed");
#else
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia, "nvidia");
#endif
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia, "nvidia");
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia, "nvidia");
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia, "nvidia");
#endif
#ifdef ENABLE_CAMBRICON_API
        CREATE(INFINI_DEVICE_CAMBRICON, bang, "bang");
#endif
#ifdef ENABLE_ASCEND_API
        CREATE(INFINI_DEVICE_ASCEND, ascend, "ascend");
#endif
#ifdef ENABLE_METAX_API
#if defined(ENABLE_NINETOOTHED) && (defined(ENABLE_NINETOOTHED_ALL_OPS) || defined(ENABLE_NINETOOTHED_OP_GEMM))
        CREATE(INFINI_DEVICE_METAX, ninetoothed, "ninetoothed");
#else
        CREATE(INFINI_DEVICE_METAX, metax, "metax");
#endif
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore, "moore");
#endif

#ifdef ENABLE_KUNLUN_API
        CREATE(INFINI_DEVICE_KUNLUN, kunlun, "kunlun");
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t
infiniopGetGemmWorkspaceSize(
    infiniopGemmDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE, IMPL)                                                               \
    case CASE:                                                                                   \
        *size = reinterpret_cast<const op::gemm::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu, "cpu");
#endif
#ifdef ENABLE_NVIDIA_API
#if defined(ENABLE_NINETOOTHED) && (defined(ENABLE_NINETOOTHED_ALL_OPS) || defined(ENABLE_NINETOOTHED_OP_GEMM))
        GET(INFINI_DEVICE_NVIDIA, ninetoothed, "ninetoothed");
#else
        GET(INFINI_DEVICE_NVIDIA, nvidia, "nvidia");
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
#if defined(ENABLE_NINETOOTHED) && (defined(ENABLE_NINETOOTHED_ALL_OPS) || defined(ENABLE_NINETOOTHED_OP_GEMM))
        GET(INFINI_DEVICE_ILUVATAR, ninetoothed, "ninetoothed");
#else
        GET(INFINI_DEVICE_ILUVATAR, nvidia, "nvidia");
#endif
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia, "nvidia");
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia, "nvidia");
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia, "nvidia");
#endif
#ifdef ENABLE_CAMBRICON_API
        GET(INFINI_DEVICE_CAMBRICON, bang, "bang");
#endif
#ifdef ENABLE_ASCEND_API
        GET(INFINI_DEVICE_ASCEND, ascend, "ascend");
#endif
#ifdef ENABLE_METAX_API
#if defined(ENABLE_NINETOOTHED) && (defined(ENABLE_NINETOOTHED_ALL_OPS) || defined(ENABLE_NINETOOTHED_OP_GEMM))
        GET(INFINI_DEVICE_METAX, ninetoothed, "ninetoothed");
#else
        GET(INFINI_DEVICE_METAX, metax, "metax");
#endif
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore, "moore");
#endif
#ifdef ENABLE_KUNLUN_API
        GET(INFINI_DEVICE_KUNLUN, kunlun, "kunlun");
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__C infiniStatus_t infiniopGemm(
    infiniopGemmDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    float alpha,
    float beta,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE, IMPL)                                       \
    case CASE:                                                                 \
        return reinterpret_cast<const op::gemm::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size,                             \
                        c, beta,                                               \
                        a, b, alpha,                                           \
                        stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu, "cpu");
#endif
#ifdef ENABLE_NVIDIA_API
#if defined(ENABLE_NINETOOTHED) && (defined(ENABLE_NINETOOTHED_ALL_OPS) || defined(ENABLE_NINETOOTHED_OP_GEMM))
        CALCULATE(INFINI_DEVICE_NVIDIA, ninetoothed, "ninetoothed");
#else
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia, "nvidia");
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
#if defined(ENABLE_NINETOOTHED) && (defined(ENABLE_NINETOOTHED_ALL_OPS) || defined(ENABLE_NINETOOTHED_OP_GEMM))
        CALCULATE(INFINI_DEVICE_ILUVATAR, ninetoothed, "ninetoothed");
#else
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia, "nvidia");
#endif
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia, "nvidia");
#endif
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia, "nvidia");
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia, "nvidia");
#endif
#ifdef ENABLE_CAMBRICON_API
        CALCULATE(INFINI_DEVICE_CAMBRICON, bang, "bang");
#endif
#ifdef ENABLE_ASCEND_API
        CALCULATE(INFINI_DEVICE_ASCEND, ascend, "ascend");
#endif
#ifdef ENABLE_METAX_API
#if defined(ENABLE_NINETOOTHED) && (defined(ENABLE_NINETOOTHED_ALL_OPS) || defined(ENABLE_NINETOOTHED_OP_GEMM))
        CALCULATE(INFINI_DEVICE_METAX, ninetoothed, "ninetoothed");
#else
        CALCULATE(INFINI_DEVICE_METAX, metax, "metax");
#endif
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore, "moore");
#endif
#ifdef ENABLE_KUNLUN_API
        CALCULATE(INFINI_DEVICE_KUNLUN, kunlun, "kunlun");
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyGemmDescriptor(infiniopGemmDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE, IMPL)                                           \
    case CASE:                                                                  \
        delete reinterpret_cast<const op::gemm::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu, "cpu");
#endif
#ifdef ENABLE_NVIDIA_API
#if defined(ENABLE_NINETOOTHED) && (defined(ENABLE_NINETOOTHED_ALL_OPS) || defined(ENABLE_NINETOOTHED_OP_GEMM))
        DELETE(INFINI_DEVICE_NVIDIA, ninetoothed, "ninetoothed");
#else
        DELETE(INFINI_DEVICE_NVIDIA, nvidia, "nvidia");
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
#if defined(ENABLE_NINETOOTHED) && (defined(ENABLE_NINETOOTHED_ALL_OPS) || defined(ENABLE_NINETOOTHED_OP_GEMM))
        DELETE(INFINI_DEVICE_ILUVATAR, ninetoothed, "ninetoothed");
#else
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia, "nvidia");
#endif
#endif
#ifdef ENABLE_ALI_API
        DELETE(INFINI_DEVICE_ALI, nvidia, "nvidia");
#endif
#ifdef ENABLE_QY_API
        DELETE(INFINI_DEVICE_QY, nvidia, "nvidia");
#endif
#ifdef ENABLE_HYGON_API
        DELETE(INFINI_DEVICE_HYGON, nvidia, "nvidia");
#endif
#ifdef ENABLE_CAMBRICON_API
        DELETE(INFINI_DEVICE_CAMBRICON, bang, "bang");
#endif
#ifdef ENABLE_ASCEND_API
        DELETE(INFINI_DEVICE_ASCEND, ascend, "ascend");
#endif
#ifdef ENABLE_METAX_API
#if defined(ENABLE_NINETOOTHED) && (defined(ENABLE_NINETOOTHED_ALL_OPS) || defined(ENABLE_NINETOOTHED_OP_GEMM))
        DELETE(INFINI_DEVICE_METAX, ninetoothed, "ninetoothed");
#else
        DELETE(INFINI_DEVICE_METAX, metax, "metax");
#endif
#endif
#ifdef ENABLE_MOORE_API
        DELETE(INFINI_DEVICE_MOORE, moore, "moore");
#endif
#ifdef ENABLE_KUNLUN_API
        DELETE(INFINI_DEVICE_KUNLUN, kunlun, "kunlun");
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
