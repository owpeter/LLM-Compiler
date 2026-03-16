#ifndef __CAUSAL_SOFTMAX_NINETOOTHED_H__
#define __CAUSAL_SOFTMAX_NINETOOTHED_H__

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"
#include "../info.h"

#include "../../../../../build/ninetoothed/causal_softmax.h"
#include "../../../ninetoothed/utils.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "../nvidia/causal_softmax_nvidia.cuh"
#endif

namespace op::causal_softmax::ninetoothed {
class Descriptor final : public InfiniopDescriptor {
public:
    Descriptor(
        infiniopHandle_t handle,
        CausalSoftmaxInfo info,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
        op::causal_softmax::nvidia::Descriptor *fallback_desc
#else
        void *fallback_desc
#endif
        )
        : InfiniopDescriptor{handle->device, handle->device_id},
          info_{std::move(info)},
          y_shape_{y_desc->shape()},
          y_strides_{y_desc->strides()},
          x_shape_{x_desc->shape()},
          x_strides_{x_desc->strides()},
          fallback_{fallback_desc} {}

    ~Descriptor() {
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
        delete fallback_;
#endif
    }

    size_t workspaceSize() const {
        return 0;
    }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc) {
        auto info_result = CausalSoftmaxInfo::create(y_desc, x_desc);
        if (!info_result) {
            return info_result.status();
        }
        CausalSoftmaxInfo info = info_result.take();

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
        op::causal_softmax::nvidia::Descriptor *fallback_desc{nullptr};
        auto status = op::causal_softmax::nvidia::Descriptor::create(handle, &fallback_desc, y_desc, x_desc);
        if (status != INFINI_STATUS_SUCCESS) {
            return status;
        }
        *desc_ptr = new Descriptor(handle, std::move(info), y_desc, x_desc, fallback_desc);
#else
        *desc_ptr = new Descriptor(handle, std::move(info), y_desc, x_desc, nullptr);
#endif
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const {
        auto y_nt{::ninetoothed::Tensor(y, y_shape_, y_strides_)};
        auto x_nt{::ninetoothed::Tensor(x, x_shape_, x_strides_)};

        int ndim = y_shape_.size();
        static constexpr int block_sizes[] = {64, 128, 256, 512, 1024};
        for (auto block_size : block_sizes) {
            if (launch_causal_softmax(stream, x_nt, y_nt, ndim, info_.dtype, block_size) == 0) {
                return INFINI_STATUS_SUCCESS;
            }
        }

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
        return fallback_->calculate(workspace, workspace_size, y, x, stream);
#else
        return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
    }

private:
    using Size = ::ninetoothed::Tensor<>::Size;
    using Stride = ::ninetoothed::Tensor<>::Stride;

    CausalSoftmaxInfo info_;
    std::vector<Size> y_shape_;
    std::vector<Stride> y_strides_;
    std::vector<Size> x_shape_;
    std::vector<Stride> x_strides_;

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
    op::causal_softmax::nvidia::Descriptor *fallback_{nullptr};
#else
    void *fallback_{nullptr};
#endif
};
} // namespace op::causal_softmax::ninetoothed

#endif // __CAUSAL_SOFTMAX_NINETOOTHED_H__
