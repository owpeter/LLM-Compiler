## 传参

1. 修改`block_size`

## 改写kernel

1. 指令替换：用乘法替换除法

当前的 `application` 实现逻辑比较偏向标量（Scalar）的逐元素计算，存在较大的指令和访存优化空间。

- 在当前代码的第二个循环中，`output[i] = input[i] / rms * weight[i]` 使用了除法操作。在大多数底层硬件指令集中，计算平方根的倒数并执行乘法，要比直接除以一个变量快得多。
- **改写建议：** 如果 `ntl` 提供了 `rsqrt` (Reciprocal Square Root) 方法，请优先使用它。

```python
# 原始逻辑
# rms = ntl.sqrt(ntl.sum(_rms) / num_normalized_elements + eps)
# output[i] = input[i] / rms * ...

# 优化逻辑
inv_rms = ntl.rsqrt(ntl.sum(_rms) / num_normalized_elements + eps)
for i in range(input.shape[0]):
    output[i] = input[i] * inv_rms * weight[i]

```

- 2. 向量化计算：尽量消除标量 For 循环
  - 在基于 Python 的算子 DSL 中，显式地编写 `for i in range(...)` 有时会被编译器解析为串行的标量循环或缺乏内存合并（Coalesced Memory Access）的操作。
  - **改写建议：** 如果 `ninetoothed.language` 支持张量/向量级别的直接运算，应直接表达整体计算，让编译器去处理底层的循环展开和向量化。

  ```python
  # 优化逻辑：如果 ntl 支持整体数组操作
  input_f32 = ntl.cast(input, ntl.float32)
  _rms = ntl.sum(input_f32 * input_f32) 
  # ... 计算 inv_rms ...
  output = input * inv_rms * weight

  ```

