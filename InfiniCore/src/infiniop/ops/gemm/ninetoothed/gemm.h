#ifndef __GEMM_NINETOOTHED_H__
#define __GEMM_NINETOOTHED_H__

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include "../../../../../build/ninetoothed/gemm.h"
#include "../../../ninetoothed/utils.h"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/gemm_nvidia.cuh"
#endif

namespace op::gemm::ninetoothed {
class Descriptor final : public InfiniopDescriptor {
public:
    Descriptor(
        infiniopHandle_t handle,
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc,
#ifdef ENABLE_NVIDIA_API
        op::gemm::nvidia::Descriptor *fallback_desc
#else
        void *fallback_desc
#endif
        )
        : InfiniopDescriptor{handle->device, handle->device_id},
          c_shape_{c_desc->shape()},
          c_strides_{c_desc->strides()},
          a_shape_{a_desc->shape()},
          a_strides_{a_desc->strides()},
          b_shape_{b_desc->shape()},
          b_strides_{b_desc->strides()},
          dtype_{c_desc->dtype()},
          fallback_{fallback_desc} {}

    ~Descriptor() {
#ifdef ENABLE_NVIDIA_API
        delete fallback_;
#endif
    }

    size_t workspaceSize() const {
        return 0;
    }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc) {
#ifdef ENABLE_NVIDIA_API
        op::gemm::nvidia::Descriptor *fallback_desc{nullptr};
        auto status = op::gemm::nvidia::Descriptor::create(handle, &fallback_desc, c_desc, a_desc, b_desc);
        if (status != INFINI_STATUS_SUCCESS) {
            return status;
        }
        *desc_ptr = new Descriptor(handle, c_desc, a_desc, b_desc, fallback_desc);
#else
        *desc_ptr = new Descriptor(handle, c_desc, a_desc, b_desc, nullptr);
#endif
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *c,
        float beta,
        const void *a,
        const void *b,
        float alpha,
        void *stream) const {
        if (alpha != 1.0f || beta != 0.0f || a_shape_.size() != 2 || b_shape_.size() != 2 || c_shape_.size() != 2) {
#ifdef ENABLE_NVIDIA_API
            return fallback_->calculate(workspace, workspace_size, c, beta, a, b, alpha, stream);
#else
            return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
        }

        auto c_nt{::ninetoothed::Tensor(c, c_shape_, c_strides_)};
        auto a_nt{::ninetoothed::Tensor(a, a_shape_, a_strides_)};
        auto b_nt{::ninetoothed::Tensor(b, b_shape_, b_strides_)};

        static constexpr int input_precision_values[] = {1, 2};
        // todo: change values by reading config file
        static constexpr int block_size_values[][3] = {
            // {16, 16, 32},
            // {32, 128, 32},
            {128, 128, 32},
        };

        for (auto input_precision_value : input_precision_values) {
            ::ninetoothed::Tensor<int> input_precision_tensor{input_precision_value};
            for (const auto &block_size : block_size_values) {
                if (launch_gemm(stream,
                                a_nt,
                                b_nt,
                                c_nt,
                                input_precision_tensor,
                                dtype_,
                                input_precision_value,
                                block_size[0],
                                block_size[1],
                                block_size[2]) == 0) {
                    return INFINI_STATUS_SUCCESS;
                }
            }
        }

#ifdef ENABLE_NVIDIA_API
        return fallback_->calculate(workspace, workspace_size, c, beta, a, b, alpha, stream);
#else
        return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
    }

private:
    using Size = ::ninetoothed::Tensor<>::Size;
    using Stride = ::ninetoothed::Tensor<>::Stride;

    std::vector<Size> c_shape_;
    std::vector<Stride> c_strides_;

    std::vector<Size> a_shape_;
    std::vector<Stride> a_strides_;

    std::vector<Size> b_shape_;
    std::vector<Stride> b_strides_;

    infiniDtype_t dtype_;

#ifdef ENABLE_NVIDIA_API
    op::gemm::nvidia::Descriptor *fallback_{nullptr};
#else
    void *fallback_{nullptr};
#endif
};
} // namespace op::gemm::ninetoothed

#endif // __GEMM_NINETOOTHED_H__
