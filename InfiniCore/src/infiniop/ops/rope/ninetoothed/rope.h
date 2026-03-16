#ifndef __ROPE_NINETOOTHED_H__
#define __ROPE_NINETOOTHED_H__

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"
#include <cstdio>
#include <cstdlib>

#include "../../../../../build/ninetoothed/rope.h"
#include "../../../ninetoothed/utils.h"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/rope_nvidia.cuh"
#endif

namespace op::rope::ninetoothed {
class Descriptor final : public InfiniopDescriptor {
public:
    Descriptor(
        infiniopHandle_t handle,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t pos_ids_desc,
        infiniopTensorDescriptor_t sin_table_desc,
        infiniopTensorDescriptor_t cos_table_desc,
        infiniopRoPEAlgo_t algo,
#ifdef ENABLE_NVIDIA_API
        op::rope::nvidia::Descriptor *fallback_desc
#else
        void *fallback_desc
#endif
        )
        : InfiniopDescriptor{handle->device, handle->device_id},
          y_shape_{y_desc->shape()},
          y_strides_{y_desc->strides()},
          x_shape_{x_desc->shape()},
          x_strides_{x_desc->strides()},
          pos_ids_shape_{pos_ids_desc->shape()},
          pos_ids_strides_{pos_ids_desc->strides()},
          sin_table_shape_{sin_table_desc->shape()},
          sin_table_strides_{sin_table_desc->strides()},
          cos_table_shape_{cos_table_desc->shape()},
          cos_table_strides_{cos_table_desc->strides()},
          dtype_{y_desc->dtype()},
          algo_{algo},
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
        infiniopTensorDescriptor_t pos_ids_desc,
        infiniopTensorDescriptor_t sin_table_desc,
        infiniopTensorDescriptor_t cos_table_desc,
        infiniopRoPEAlgo_t algo) {
#ifdef ENABLE_NVIDIA_API
        op::rope::nvidia::Descriptor *fallback_desc{nullptr};
        auto status = op::rope::nvidia::Descriptor::create(handle, &fallback_desc, y_desc, x_desc, pos_ids_desc, sin_table_desc, cos_table_desc, algo);
        if (status != INFINI_STATUS_SUCCESS) {
            return status;
        }
        *desc_ptr = new Descriptor(handle, y_desc, x_desc, pos_ids_desc, sin_table_desc, cos_table_desc, algo, fallback_desc);
#else
        *desc_ptr = new Descriptor(handle, y_desc, x_desc, pos_ids_desc, sin_table_desc, cos_table_desc, algo, nullptr);
#endif
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        const void *pos_ids,
        const void *sin_table,
        const void *cos_table,
        void *stream) const {
        if (x_shape_.size() != 4 || y_shape_.size() != 4 || sin_table_shape_.size() != 2 || cos_table_shape_.size() != 2) {
            dispatchLog("param_mismatch");
#ifdef ENABLE_NVIDIA_API
            return fallback_->calculate(workspace, workspace_size, y, x, pos_ids, sin_table, cos_table, stream);
#else
            return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
        }

        auto y_nt{::ninetoothed::Tensor(y, y_shape_, y_strides_)};
        auto x_nt{::ninetoothed::Tensor(x, x_shape_, x_strides_)};
        std::vector<Size> sin_table_shape_view = {
            1,
            1,
            sin_table_shape_[0],
            sin_table_shape_[1],
        };
        std::vector<Stride> sin_table_strides_view = {
            0,
            0,
            sin_table_strides_[0],
            sin_table_strides_[1],
        };
        std::vector<Size> cos_table_shape_view = {
            1,
            1,
            cos_table_shape_[0],
            cos_table_shape_[1],
        };
        std::vector<Stride> cos_table_strides_view = {
            0,
            0,
            cos_table_strides_[0],
            cos_table_strides_[1],
        };
        auto sin_table_nt{::ninetoothed::Tensor(
            sin_table, sin_table_shape_view, sin_table_strides_view)};
        auto cos_table_nt{::ninetoothed::Tensor(
            cos_table, cos_table_shape_view, cos_table_strides_view)};

        const bool preferred_interleaved =
            (algo_ == infiniopRoPEAlgo_t::INFINIOP_ROPE_ALGO_GPT_J);

        int ndim = y_shape_.size();
        int emb_dim = y_shape_.back();
        int dtype = static_cast<int>(dtype_);

        for (int attempt = 0; attempt < 2; ++attempt) {
            const bool interleaved_value =
                (attempt == 0) ? preferred_interleaved : !preferred_interleaved;

            if (launch_rope(stream,
                            x_nt,
                            sin_table_nt,
                            cos_table_nt,
                            y_nt,
                            ndim,
                            emb_dim,
                            dtype,
                            interleaved_value) == 0) {
                return INFINI_STATUS_SUCCESS;
            }
        }

        dispatchLog("kernel_launch_failed");
#ifdef ENABLE_NVIDIA_API
        return fallback_->calculate(workspace, workspace_size, y, x, pos_ids, sin_table, cos_table, stream);
#else
        return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
    }

private:
    static bool dispatchDebugEnabled() {
        const char *value = std::getenv("INFINI_ROPE_DISPATCH_DEBUG");
        return value != nullptr && value[0] != '0';
    }
    static void dispatchLog(const char *reason) {
        if (!dispatchDebugEnabled()) {
            return;
        }
        std::fprintf(stderr, "[infiniop][rope] ninetoothed fallback=%s\n", reason);
    }

    using Size = ::ninetoothed::Tensor<>::Size;
    using Stride = ::ninetoothed::Tensor<>::Stride;

    std::vector<Size> y_shape_;
    std::vector<Stride> y_strides_;
    std::vector<Size> x_shape_;
    std::vector<Stride> x_strides_;
    std::vector<Size> pos_ids_shape_;
    std::vector<Stride> pos_ids_strides_;
    std::vector<Size> sin_table_shape_;
    std::vector<Stride> sin_table_strides_;
    std::vector<Size> cos_table_shape_;
    std::vector<Stride> cos_table_strides_;
    infiniDtype_t dtype_;
    infiniopRoPEAlgo_t algo_;
#ifdef ENABLE_NVIDIA_API
    op::rope::nvidia::Descriptor *fallback_;
#else
    void *fallback_;
#endif
};

} // namespace op::rope::ninetoothed

#endif // __ROPE_NINETOOTHED_H__
