## 具体实现

### 替换已有算子

以“我想把某个已有 op 的 NVIDIA 实现替换为 ninetoothed 实现”为例。

#### 1）写 ninetoothed 实现（放到该算子的 ninetoothed 目录）

参考现有结构：

- Python 侧：`src/infiniop/ops/<op>/ninetoothed/<op>.py`
- 生成入口：`src/infiniop/ops/<op>/ninetoothed/build.py`

例子可以直接对照：

- ReLU 的 build：[relu/ninetoothed/build.py](file:///home/owpeter/InfiniCore/src/infiniop/ops/relu/ninetoothed/build.py#L7-L30)
- SwiGLU 的 “application/premake”：[swiglu.py](file:///home/owpeter/InfiniCore/src/infiniop/ops/swiglu/ninetoothed/swiglu.py#L9-L23)

你需要做的核心事：

- 定义 `application(...)`（算子数学表达式）
- 定义 `premake(...)` 返回 `(arrangement, application, tensors)`
- 在 `build.py` 枚举 constexpr 参数网格（至少 dtype、ndim，可能还要 block_size 等）

#### 2）运行生成脚本，产出 `build/ninetoothed/<op>.h/.c` 等文件

仓库已经提供“一键扫描所有 ops 并生成”的脚本：

- [scripts/build_ntops.py](file:///home/owpeter/InfiniCore/scripts/build_ntops.py#L12-L46)

它会遍历 `src/infiniop/ops/*/ninetoothed/build.py` 并调用 `module.build()`，生成目录是 `InfiniCore/build/ninetoothed/`（由 [ninetoothed/build.py](file:///home/owpeter/InfiniCore/src/infiniop/ninetoothed/build.py#L12-L15) 定义）。

#### 3）打开编译开关：ENABLE_NINETOOTHED

InfiniCore 用 xmake 管理，开关名叫 `ninetoothed`：

- 选项定义与宏： [xmake.lua](file:///home/owpeter/InfiniCore/xmake.lua#L213-L221)

开启后会：

- 定义 `ENABLE_NINETOOTHED`
- 并在各设备后端（例如 NVIDIA）把 `build/ninetoothed/*.c/*.cpp` 编译进来：[xmake/nvidia.lua](file:///home/owpeter/InfiniCore/xmake/nvidia.lua#L76-L79)

#### 4）让算子在运行时走 ninetoothed 实现（两种接法，选其一）

**接法 1（最常见）：在 operator.cc 层切换到 ninetoothed Descriptor**
SwiGLU 就是这样做的：`ENABLE_NINETOOTHED` 时 include ninetoothed 头并在 create/get/calc/destroy 分发到 `op::<op>::ninetoothed`：

- [swiglu/operator.cc](file:///home/owpeter/InfiniCore/src/infiniop/ops/swiglu/operator.cc#L8-L68)

你如果要替换某个 op，可以按这个模式：

- 增加 `src/infiniop/ops/<op>/ninetoothed/<op>.h`（C++ Descriptor，内部调用 `launch_<op>`）
- 在 `src/infiniop/ops/<op>/operator.cc` 里用 `#ifdef ENABLE_NINETOOTHED` 选择 `ninetoothed` namespace

**接法 2：在某个 device 实现文件里直接切换**
ReLU 的 NVIDIA 实现已经内置了这个开关：`ENABLE_NINETOOTHED` 时直接调用 `launch_relu(...)`，否则走原 CUDA elementwise 实现：

- [relu_nvidia.cu](file:///home/owpeter/InfiniCore/src/infiniop/ops/relu/nvidia/relu_nvidia.cu#L47-L80)

这条路的好处是你不需要新增 namespace/Descriptor，只是在现有后端里加一个分支。

#### 5）在 Descriptor::calculate 里如何“对接 tensor/shape/stride/stream”

C++ 侧对接方式已经统一封装好了：把 data + shape + strides 包装成 `ninetoothed::Tensor`，它会隐式转成 `NineToothedTensor`：

- 转换器定义：[ninetoothed/utils.h](file:///home/owpeter/InfiniCore/src/infiniop/ninetoothed/utils.h#L12-L31)
- SwiGLU 的调用样例：[swiglu.h](file:///home/owpeter/InfiniCore/src/infiniop/ops/swiglu/ninetoothed/swiglu.h#L42-L63)
- FlashAttention 的更复杂样例：[descriptor.h](file:///home/owpeter/InfiniCore/src/infiniop/ops/flash_attention/ninetoothed/descriptor.h#L51-L96)

---

### 新增算子

下面给你一套“在 InfiniCore 里借助 ninetoothed 新增一个**融合算子**（新 op 名称、新 C 接口、新实现），并能在推理时调用”的完整实施步骤。整体路径是：**写 Python（ninetoothed）→ AOT 生成 `build/ninetoothed/*.h/*.c` → C++ Descriptor 调用 `launch_<op>` → InfiniOP 对外暴露 C API**。

---

#### 1. 定义api

你需要先明确：

- 输入/输出张量个数、dtype 支持范围（fp16/bf16/fp32…）
- 允许的 shape/stride 约束（是否必须 contiguous、是否支持 broadcast、ndim 最大多少）
- 是否需要标量参数（epsilon、scale、mask flag 等）
- 是否需要 workspace（ninetoothed 一般可以做成 0 workspace）

建议：先做一版“约束更强”的（例如只支持 contiguous 或特定 layout），跑通后再逐步放宽。

---

#### 2. 添加 InfiniOP 的对外 C 头文件

参考现有格式（例如 [swiglu.h](file:///home/owpeter/InfiniCore/include/infiniop/ops/swiglu.h#L1-L25)），新增：

1) 新文件：`InfiniCore/include/infiniop/ops/<fused_op>.h`

- 定义 `typedef struct InfiniopDescriptor *infiniop<FusedOp>Descriptor_t;`
- 声明四段式接口：
  - `infiniopCreate<FusedOp>Descriptor(...)`
  - `infiniopGet<FusedOp>WorkspaceSize(...)`（可选但建议提供，返回 0）
  - `infiniop<FusedOp>(..., void* stream)`
  - `infiniopDestroy<FusedOp>Descriptor(...)`

2) 把它 include 到总入口 [infiniop.h](file:///home/owpeter/InfiniCore/include/infiniop.h#L4-L44)（新增一行 `#include "infiniop/ops/<fused_op>.h"`）。

---

#### 3. 添加算子分发入口（operator.cc）

新建目录：`InfiniCore/src/infiniop/ops/<fused_op>/`

新增 `operator.cc`（可以照抄 RMSNorm / SwiGLU 的结构），核心点：

- `infiniopCreate...` 根据 `handle->device` 分发到不同后端 `op::<fused_op>::<backend>::Descriptor::create`
- `infiniop...` 调用 `Descriptor::calculate`
- `infiniopDestroy...` delete 对应 Descriptor

你如果只想先支持 NVIDIA + ninetoothed，可以先只实现 `ENABLE_NVIDIA_API` 路径；其它设备先返回 `INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`。

xmake 会自动把 `src/infiniop/ops/*/operator.cc` 编进库里，见 [xmake.lua](file:///home/owpeter/InfiniCore/xmake.lua#L317-L368)。

---

#### 4. 写 ninetoothed 版本的 C++ Descriptor（运行时调用 launch_<op>）**

推荐仿照 SwiGLU 的做法：建一个 `ninetoothed` 后端命名空间。

目录结构建议：

- `src/infiniop/ops/<fused_op>/ninetoothed/descriptor.h`（或 `<fused_op>.h`）

Descriptor 里要做的事（参考 [swiglu.h](file:///home/owpeter/InfiniCore/src/infiniop/ops/swiglu/ninetoothed/swiglu.h#L42-L63)）：

- 在构造函数里把输入/输出的 `shape()` / `strides()` 保存成 `std::vector`
- `calculate()` 里把：
  - `output`、`inputs[i]` 包装成 `ninetoothed::Tensor(data, shape, strides)`（包装器在 [utils.h](file:///home/owpeter/InfiniCore/src/infiniop/ninetoothed/utils.h#L12-L31)）
  - 然后调用 `launch_<fused_op>(stream, tensor..., ndim, dtype, block_size, ...)`
- include 生成的 AOT 头：`#include "../../../../../build/ninetoothed/<fused_op>.h"`（参考 [swiglu.h](file:///home/owpeter/InfiniCore/src/infiniop/ops/swiglu/ninetoothed/swiglu.h#L8-L10)）

标量参数怎么传：

- 可以像 FlashAttention 那样构造 `NineToothedTensor scale{&value, nullptr, nullptr}`（见 [descriptor.h](file:///home/owpeter/InfiniCore/src/infiniop/ops/flash_attention/ninetoothed/descriptor.h#L59-L64)）
- 或用 `ninetoothed::Tensor<double>(value)` 这种 0-dim tensor（包装器支持，见 [utils.h](file:///home/owpeter/InfiniCore/src/infiniop/ninetoothed/utils.h#L27-L31)），再传给 `launch_<op>`

---

#### 5. 写 Python 侧 ninetoothed 实现（算子逻辑 + AOT 参数网格）**

在 `src/infiniop/ops/<fused_op>/ninetoothed/` 下新增：

- `<fused_op>.py`：定义 `application(...)` + `premake(...)`
- `build.py`：定义 `build()`，调用 `infiniop.ninetoothed.build.build(...)`

可以直接对照：

- `build.py` 的写法：[swiglu/ninetoothed/build.py](file:///home/owpeter/InfiniCore/src/infiniop/ops/swiglu/ninetoothed/build.py#L7-L29)
- `premake` 返回结构：[swiglu.py](file:///home/owpeter/InfiniCore/src/infiniop/ops/swiglu/ninetoothed/swiglu.py#L13-L23)

constexpr 参数网格怎么选（决定会生成多少 kernel）：

- 必选：`dtype`、`ndim`
- 常见：`block_size` / tile sizes
- 经验：先做“小网格”（比如 ndim=2~4、dtype=fp16/bf16、block_size 一个值），跑通后再扩展，否则生成量大、编译慢、资源容易爆。

---

#### 6. 生成 AOT 产物（build/ninetoothed）**

生成脚本是 [build_ntops.py](file:///home/owpeter/InfiniCore/scripts/build_ntops.py#L1-L75)。建议只生成你的新 op，避免全量生成：

```bash
cd /home/owpeter/InfiniCore
PYTHONPATH=$PWD/src python scripts/build_ntops.py --ops <fused_op> --jobs 1
```

生成后应该出现：

- `InfiniCore/build/ninetoothed/<fused_op>.h`
- `InfiniCore/build/ninetoothed/<fused_op>.c`
  以及一堆 `build/ninetoothed/<fused_op>_<suffix>.h`（专用 kernel 头）

---

#### 7. 编译 InfiniCore（启用 ninetoothed）**

xmake 里开关名是 `ninetoothed`，启用会定义 `ENABLE_NINETOOTHED`，见 [xmake.lua](file:///home/owpeter/InfiniCore/xmake.lua#L213-L221)。

并且 NVIDIA 后端会把 `build/ninetoothed/*.c/*.cpp` 编进 `infiniop-nvidia`，见 [nvidia.lua](file:///home/owpeter/InfiniCore/xmake/nvidia.lua#L76-L79)。

你需要确保编译时同时开启：

- 目标设备后端（例如 `--nv-gpu=y`）
- `--ninetoothed=y`

---

#### 8. 验证与集成到模型**

验证建议分三层：

- **单算子正确性**：写一个最小 C/C++ 或 Python（通过 infinicore python binding）调用新 op，与 PyTorch 参考实现对比。
- **形状覆盖**：覆盖模型会遇到的 batch/seq/hidden 等形状组合，以及 dtype。
- **回退机制**（可选但强烈建议）：在 operator.cc 里保留一个非 ninetoothed 的实现或直接返回 NOT_IMPLEMENTED，让上层可检测并切回旧路径。