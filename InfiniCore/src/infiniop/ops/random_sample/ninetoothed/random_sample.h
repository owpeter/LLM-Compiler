#ifndef __RANDOM_SAMPLE_NINETOOTHED_H__
#define __RANDOM_SAMPLE_NINETOOTHED_H__

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"
#include <cstdio>
#include <cstdlib>

#include "../../../../../build/ninetoothed/random_sample.h"
#include "../../../ninetoothed/utils.h"

#ifdef ENABLE_NVIDIA_API
#include "../nvidia/random_sample_nvidia.cuh"
#endif

namespace op::random_sample::ninetoothed {

class Descriptor final : public InfiniopDescriptor {
public:
    Descriptor(
        infiniopHandle_t handle,
        infiniopTensorDescriptor_t result_desc,
        infiniopTensorDescriptor_t probs_desc,
#ifdef ENABLE_NVIDIA_API
        op::random_sample::nvidia::Descriptor *fallback_desc
#else
        void *fallback_desc
#endif
    )
        : InfiniopDescriptor{handle->device, handle->device_id},
          result_shape_{result_desc->shape()},
          result_strides_{result_desc->strides()},
          probs_shape_{probs_desc->shape()},
          probs_strides_{probs_desc->strides()},
          dtype_{probs_desc->dtype()},
          fallback_{fallback_desc} {}

    ~Descriptor() {
#ifdef ENABLE_NVIDIA_API
        delete fallback_;
#endif
    }

    size_t minWorkspaceSize() const {
#ifdef ENABLE_NVIDIA_API
        if (fallback_) return fallback_->minWorkspaceSize();
#endif
        return 0;
    }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t result_desc,
        infiniopTensorDescriptor_t probs_desc) {

#ifdef ENABLE_NVIDIA_API
        op::random_sample::nvidia::Descriptor *fallback_desc{nullptr};
        auto status = op::random_sample::nvidia::Descriptor::create(handle, &fallback_desc, result_desc, probs_desc);
        if (status != INFINI_STATUS_SUCCESS) {
            return status;
        }
        *desc_ptr = new Descriptor(handle, result_desc, probs_desc, fallback_desc);
#else
        *desc_ptr = new Descriptor(handle, result_desc, probs_desc, nullptr);
#endif
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *result,
        const void *probs,
        float random_val,
        float topp,
        int topk,
        float temperature,
        void *stream) const {
        
        int ndim = probs_shape_.size();
        
        bool supported = (ndim == 2 || ndim == 3);
        if (dtype_ != INFINI_DTYPE_F16 && dtype_ != INFINI_DTYPE_BF16 && dtype_ != INFINI_DTYPE_F32) {
            supported = false;
        }

        if (!supported) {
            dispatchLog("param_mismatch");
#ifdef ENABLE_NVIDIA_API
            return fallback_->calculate(workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream);
#else
            return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
        }

        auto probs_nt{::ninetoothed::Tensor(probs, probs_shape_, probs_strides_)};
        auto result_nt{::ninetoothed::Tensor(result, result_shape_, result_strides_)};
        
        ::ninetoothed::Tensor<float> random_val_nt{random_val};
        ::ninetoothed::Tensor<float> topp_nt{topp};
        ::ninetoothed::Tensor<int> topk_nt{topk};
        ::ninetoothed::Tensor<float> temperature_nt{temperature};

        static constexpr int block_size_values[] = {128};

        for (auto block_size : block_size_values) {
             if (launch_random_sample(stream,
                                      probs_nt,
                                      result_nt,
                                      random_val_nt,
                                      topp_nt,
                                      topk_nt,
                                      temperature_nt,
                                      ndim,
                                      dtype_,
                                      block_size) == 0) {
                 return INFINI_STATUS_SUCCESS;
             }
        }

        dispatchLog("kernel_not_found");

#ifdef ENABLE_NVIDIA_API
        return fallback_->calculate(workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream);
#else
        return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
    }

private:
    using Size = ::ninetoothed::Tensor<>::Size;
    using Stride = ::ninetoothed::Tensor<>::Stride;
    
    static bool dispatchDebugEnabled() {
        const char *value = std::getenv("INFINI_RANDOM_SAMPLE_DISPATCH_DEBUG");
        return value != nullptr && value[0] != '0';
    }
    static void dispatchLog(const char *reason) {
        if (!dispatchDebugEnabled()) {
            return;
        }
        std::fprintf(stderr, "[infiniop][random_sample] ninetoothed fallback=%s\n", reason);
    }

    std::vector<Size> result_shape_;
    std::vector<Stride> result_strides_;
    std::vector<Size> probs_shape_;
    std::vector<Stride> probs_strides_;
    infiniDtype_t dtype_;

#ifdef ENABLE_NVIDIA_API
    op::random_sample::nvidia::Descriptor *fallback_{nullptr};
#else
    void *fallback_{nullptr};
#endif
};

} // namespace op::random_sample::ninetoothed

#endif // __RANDOM_SAMPLE_NINETOOTHED_H__
