#ifndef __SWIGLU_NINETOOTHED_H__
#define __SWIGLU_NINETOOTHED_H__

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include "../../../../../build/ninetoothed/swiglu.h"
#include "../../../ninetoothed/utils.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
#include "../nvidia/swiglu_nvidia.cuh"
#endif

namespace op::swiglu::ninetoothed {
class Descriptor final : public InfiniopDescriptor {
public:
    Descriptor(
        infiniopHandle_t handle,
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t up_desc,
        infiniopTensorDescriptor_t gate_desc,
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
        op::swiglu::nvidia::Descriptor *fallback_desc
#else
        void *fallback_desc
#endif
        )
        : InfiniopDescriptor{handle->device, handle->device_id},
          out_shape_{out_desc->shape()},
          out_strides_{out_desc->strides()},
          up_shape_{up_desc->shape()},
          up_strides_{up_desc->strides()},
          gate_shape_{gate_desc->shape()},
          gate_strides_{gate_desc->strides()},
          dtype_{out_desc->dtype()},
          fallback_{fallback_desc} {}

    ~Descriptor() {
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
        delete fallback_;
#endif
    }

    size_t workspaceSize() const {
        return 0;
    }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec) {
        auto up_desc = input_desc_vec.at(0);
        auto gate_desc = input_desc_vec.at(1);
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
        op::swiglu::nvidia::Descriptor *fallback_desc{nullptr};
        auto status = op::swiglu::nvidia::Descriptor::create(handle, &fallback_desc, out_desc, input_desc_vec);
        if (status != INFINI_STATUS_SUCCESS) {
            return status;
        }
        *desc_ptr = new Descriptor(handle, out_desc, up_desc, gate_desc, fallback_desc);
#else
        *desc_ptr = new Descriptor(handle, out_desc, up_desc, gate_desc, nullptr);
#endif
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const {
        auto out_nt{::ninetoothed::Tensor(output, out_shape_, out_strides_)};
        auto up_nt{::ninetoothed::Tensor(inputs[0], up_shape_, up_strides_)};
        auto gate_nt{::ninetoothed::Tensor(inputs[1], gate_shape_, gate_strides_)};

        int ndim = out_shape_.size();
        static constexpr int block_sizes[] = {512};

        for (auto block_size : block_sizes) {
            if (launch_swiglu(stream, up_nt, gate_nt, out_nt, ndim, dtype_, block_size) == 0) {
                return INFINI_STATUS_SUCCESS;
            }
        }

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
        return fallback_->calculate(workspace, workspace_size, output, inputs, stream);
#else
        return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
    }

private:
    using Size = ::ninetoothed::Tensor<>::Size;
    using Stride = ::ninetoothed::Tensor<>::Stride;

    std::vector<Size> out_shape_;
    std::vector<Stride> out_strides_;

    std::vector<Size> up_shape_;
    std::vector<Stride> up_strides_;

    std::vector<Size> gate_shape_;
    std::vector<Stride> gate_strides_;

    infiniDtype_t dtype_;

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
    op::swiglu::nvidia::Descriptor *fallback_{nullptr};
#else
    void *fallback_{nullptr};
#endif
};
} // namespace op::swiglu::ninetoothed

#endif // __SWIGLU_NINETOOTHED_H__
