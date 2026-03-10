#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/random_sample.h"

#include <cstdio>
#include <cstdlib>

#ifdef ENABLE_CPU_API
#include "cpu/random_sample_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/random_sample_nvidia.cuh"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/random_sample_bang.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/random_sample_metax.h"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/random_sample_aclnn.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/random_sample_moore.h"
#endif
#ifdef ENABLE_KUNLUN_API
#include "kunlun/random_sample_kunlun.h"
#endif
#ifdef ENABLE_NINETOOTHED
#include "ninetoothed/random_sample.h"
#endif

namespace {
inline bool random_sample_ninetoothed_debug_enabled() {
    const char *value = std::getenv("INFINI_RANDOM_SAMPLE_DISPATCH_DEBUG");
    return value != nullptr && value[0] != '0';
}
inline void random_sample_ninetoothed_log_create(infiniDevice_t device, int device_id) {
    if (!random_sample_ninetoothed_debug_enabled()) {
        return;
    }
    std::fprintf(stderr, "[infiniop][random_sample] create backend=ninetoothed device=%d device_id=%d\n",
                 static_cast<int>(device),
                 device_id);
}
inline void random_sample_ninetoothed_log_call(float random_val, float topp, int topk, float temperature, size_t workspace_size) {
    if (!random_sample_ninetoothed_debug_enabled()) {
        return;
    }
    std::fprintf(stderr,
                 "[infiniop][random_sample] call backend=ninetoothed random_val=%g topp=%g topk=%d temperature=%g workspace_size=%zu\n",
                 random_val,
                 topp,
                 topk,
                 temperature,
                 workspace_size);
}
}

__C infiniStatus_t
infiniopCreateRandomSampleDescriptor(
    infiniopHandle_t handle,
    infiniopRandomSampleDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t result,
    infiniopTensorDescriptor_t probs) {

#define CREATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        return op::random_sample::NAMESPACE::Descriptor::create(                     \
            handle,                                                                  \
            reinterpret_cast<op::random_sample::NAMESPACE::Descriptor **>(desc_ptr), \
            result,                                                                  \
            probs)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
#ifdef ENABLE_NINETOOTHED
    case INFINI_DEVICE_NVIDIA:
        random_sample_ninetoothed_log_create(handle->device, handle->device_id);
        return op::random_sample::ninetoothed::Descriptor::create(
            handle,
            reinterpret_cast<op::random_sample::ninetoothed::Descriptor **>(desc_ptr),
            result,
            probs);
#else
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_CAMBRICON_API
        CREATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_ASCEND_API
        CREATE(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
#endif
#ifdef ENABLE_KUNLUN_API
        CREATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
};

__C infiniStatus_t infiniopGetRandomSampleWorkspaceSize(
    infiniopRandomSampleDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                          \
    case CASE: {                                                      \
        using Ptr = const op::random_sample::NAMESPACE::Descriptor *; \
        *size = reinterpret_cast<Ptr>(desc)->minWorkspaceSize();      \
    }                                                                 \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
#ifdef ENABLE_NINETOOTHED
    case INFINI_DEVICE_NVIDIA: {
        using Ptr = const op::random_sample::ninetoothed::Descriptor *;
        *size = reinterpret_cast<Ptr>(desc)->minWorkspaceSize();
    }
        return INFINI_STATUS_SUCCESS;
#else
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_CAMBRICON_API
        GET(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_ASCEND_API
        GET(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore);
#endif
#ifdef ENABLE_KUNLUN_API
        GET(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__C infiniStatus_t infiniopRandomSample(
    infiniopRandomSampleDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    float random_val,
    float topp,
    int topk,
    float temperature,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                          \
        return reinterpret_cast<const op::random_sample::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size,                                      \
                        result, probs,                                                  \
                        random_val,                                                     \
                        topp, topk, temperature,                                        \
                        stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
#ifdef ENABLE_NINETOOTHED
    case INFINI_DEVICE_NVIDIA:
        random_sample_ninetoothed_log_call(random_val, topp, topk, temperature, workspace_size);
        return reinterpret_cast<const op::random_sample::ninetoothed::Descriptor *>(desc)
            ->calculate(workspace, workspace_size,
                        result, probs,
                        random_val,
                        topp, topk, temperature,
                        stream);
#else
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_CAMBRICON_API
        CALCULATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_ASCEND_API
        CALCULATE(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
#endif
#ifdef ENABLE_KUNLUN_API
        CALCULATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyRandomSampleDescriptor(
    infiniopRandomSampleDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                           \
        delete reinterpret_cast<const op::random_sample::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
#ifdef ENABLE_NINETOOTHED
    case INFINI_DEVICE_NVIDIA:
        delete reinterpret_cast<const op::random_sample::ninetoothed::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
#else
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        DELETE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_QY_API
        DELETE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DELETE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_CAMBRICON_API
        DELETE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_ASCEND_API
        DELETE(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_MOORE_API
        DELETE(INFINI_DEVICE_MOORE, moore);
#endif
#ifdef ENABLE_KUNLUN_API
        DELETE(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
