#ifndef __RMS_NORM_NINETOOTHED_H__
#define __RMS_NORM_NINETOOTHED_H__

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"
#include "../info.h"

#include "../../../../../build/ninetoothed/rms_norm.h"
#include "../../../ninetoothed/utils.h"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/rms_norm_nvidia.cuh"
#endif

namespace op::rms_norm::ninetoothed {
class Descriptor final : public InfiniopDescriptor {
public:
    Descriptor(
        infiniopHandle_t handle,
        RMSNormInfo info,
#ifdef ENABLE_NVIDIA_API
        op::rms_norm::nvidia::Descriptor *fallback_desc
#else
        void *fallback_desc
#endif
        )
        : InfiniopDescriptor{handle->device, handle->device_id},
          info_{std::move(info)},
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
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t w_desc,
        float epsilon) {

        auto info_result = RMSNormInfo::create(y_desc, x_desc, w_desc, epsilon);
        if (!info_result) {
             return info_result.status();
        }
        RMSNormInfo info = info_result.take();

#ifdef ENABLE_NVIDIA_API
        op::rms_norm::nvidia::Descriptor *fallback_desc{nullptr};
        auto status = op::rms_norm::nvidia::Descriptor::create(handle, &fallback_desc, y_desc, x_desc, w_desc, epsilon);
        if (status != INFINI_STATUS_SUCCESS) {
            return status;
        }
        *desc_ptr = new Descriptor(handle, std::move(info), fallback_desc);
#else
        *desc_ptr = new Descriptor(handle, std::move(info), nullptr);
#endif
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        const void *w,
        void *stream) const {
        
        int ndim = info_.shape.size();
        
        // Prepare tensors
        std::vector<::ninetoothed::Tensor<>::Size> shape(info_.shape.begin(), info_.shape.end());
        std::vector<::ninetoothed::Tensor<>::Stride> x_strides(info_.x_strides.begin(), info_.x_strides.end());
        std::vector<::ninetoothed::Tensor<>::Stride> y_strides(info_.y_strides.begin(), info_.y_strides.end());
        
        auto x_nt{::ninetoothed::Tensor(const_cast<void*>(x), shape, x_strides)};
        auto y_nt{::ninetoothed::Tensor(y, shape, y_strides)};
        
        size_t dim = info_.shape.back();
        std::vector<::ninetoothed::Tensor<>::Size> w_shape = shape;
        std::vector<::ninetoothed::Tensor<>::Stride> w_strides(ndim, 0);
        w_strides.back() = 1;
        auto w_nt{::ninetoothed::Tensor(const_cast<void*>(w), w_shape, w_strides)};

        ::ninetoothed::Tensor<double> eps_nt{static_cast<double>(info_.epsilon)};
        ::ninetoothed::Tensor<long> num_elems_nt{static_cast<long>(dim)};

        int num_normalized_dims = 1;
        
        static constexpr int block_sizes[] = {32, 64, 128, 256};
        
        for (auto block_size : block_sizes) {
             if (launch_rms_norm(stream,
                                 x_nt,
                                 w_nt,
                                 eps_nt,
                                 y_nt,
                                 num_elems_nt,
                                 ndim,
                                 num_normalized_dims,
                                 info_.atype,
                                 info_.wtype,
                                 info_.atype,
                                 block_size) == 0) {
                 return INFINI_STATUS_SUCCESS;
             }
        }

#ifdef ENABLE_NVIDIA_API
        return fallback_->calculate(workspace, workspace_size, y, x, w, stream);
#else
        return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
    }

private:
    RMSNormInfo info_;

#ifdef ENABLE_NVIDIA_API
    op::rms_norm::nvidia::Descriptor *fallback_{nullptr};
#else
    void *fallback_{nullptr};
#endif
};
} // namespace op::rms_norm::ninetoothed

#endif // __RMS_NORM_NINETOOTHED_H__
