# REASONING COMPILER

## 流程

1. PyTorch Model -> TVM MetaSchedule -> 特定算子程序$p_0$
2. $p_0$ -> 输入给 -> REASONING COMPILER (MCTS + LLM)
3. REASONING COMPILER -> 输出 -> Optimized Schedule (最佳调度)
4. Optimized Schedule -> 编译 -> GPU Machine Code (CUDA)

## 问题形式化

* **定义**：将优化问题定义为寻找一个转换序列 ，使得程序在目标硬件上的目标函数（如延迟、吞吐量）最大化。为了有效搜索，将问题建模为有限视界的马尔可夫决策过程。
* **状态**：经过转换后的程序代码。
* **动作**：应用特定的转换操作（如平铺、融合）。
* **奖励**：优化后的性能指标。

## 方法

该部分详细介绍了框架的两个核心组件：

### LLM部分

LLM 接收包含**当前代码、父节点代码、祖父节点代码**及其**对应的转换历史**和性能分数的 Prompt 。代码值TVM的TensorIR，节点指蒙特卡洛树搜索中的节点，代表某次优化状态后程序的代码状态。


prompt：要求 LLM 分析不同变体间的差异，识别性能变化的来源，并基于硬件成本模型生成新的转换建议。LLM 输出的转换建议会被解析和过滤，以确保其在有效转换集合中。

### MCTS部分

- **初始化**：pytorch能够非常方便的获得模型的计算图，并将计算图转换为TensorIR的TVM Script。将该TVM Script作为初始程序$p_0$

* **变换**：从根节点$p_0$开始遍历，使用UCT公式来选择下一个进行变换的节点。这一步决定了是继续深入挖掘一个当前性能很好的程序变体，还是去尝试一个很少被测试的变体。

* **扩展**：当 MCTS 决定扩展一个节点 $p_i$ 时，将对应的prompt发送给LLM，提出它认为最合理的下一个转换操作，生成新的子节点 $p_{i+1}$ 

* **模拟**：从新节点 $p_{i+1}$ 开始，**随机应用一系列合法转换**，得到一个模拟的最终程序 $p_{sim}$，并使用一个**硬件成本模型**来预估该模拟程序的性能（长期性能）


* **反向传播**：将成本模型预测出的分数（奖励 $W$）反向传回给该节点的所有祖先节点，同时更新路径上每个节点的访问次数 ($N$) 和累积奖励 ($W$) 

## 我们的思路

基于TensorIR可以对算子的实现进行很细粒度的调整，例如这篇论文的调整集合为$\{\text{TileSize, Parallel, ComputeLocation, Unroll}\}$

ninetoothed控制粒度：TileSize，流水线阶段数、每个 Threadblock 中的 Warp 数量

- **初始化**：直接将C语言形式的推理脚本作为$p_0$，并附带一个算子调优状态说明
- **变换**：从根节点$p_0$开始遍历，使用UCT公式来选择下一个进行变换的节点$p_i$。
- **扩展**：LLM对某个算子进行某种类型的调优，得到新的状态$p_{i+1}$
- **模拟**：从新节点 $p_{i+1}$ 开始，**随机应用一系列合法转换**，得到一个模拟的最终程序 $p_{sim}$，并使用一个**硬件成本模型**来预估该模拟程序的性能（长期性能）
- **反向传播**：将成本模型预测出的分数（奖励 $W$）反向传回给该节点的所有祖先节点，同时更新路径上每个节点的访问次数 ($N$) 和累积奖励 ($W$) 
---

# 九齿

## 现状

英伟达，沐曦，天数，支持直接使用九齿算子替换原有算子实现；需要在编写算子后重新编译infinicore

## 常见 LLM 算子里会用到的 kernel 特性参数（来自 ntops 内核）

不同的算子可以使用这些参数来调整生成的kernel

- `GEMM/BMM/AddMM`：block_size_m/n/k 决定矩阵 tiling 形态与内存访问块大小
- `Softmax/Reduction`：block_size 控制归约维度的 tile 大小
- `LayerNorm/RMSNorm`：num_normalized_dims 或 normalized_shape 决定归一化维度，block_size 决定归约 tile 
- `Rotary Position Embedding`：interleaved 与 emb_dim 控制旋转方式与最后维度大小 rotary_position_embedding.py
- `Scaled Dot-Product Attention`：block_size_m/n 控制 Q/K/V tile，with_kv_cache/with_attn_mask/is_causal/causal_variant 控制分支与掩码路径 
- `Conv2D`：stride/padding/dilation 与 block_size_m/n/k 共同决定展开方式与计算 tile
- `Element-wise` 激活（如 `silu/gelu/exp/neg` 等）：通常只用 block_size 控制扁平 tile 

`num_warps/num_stages` 这类编译配置参数对所有算子适用

---

## 算子与`ntops/kernel`对应关系

- rmsnorm → rms_norm.py:L1-L41
- linear（GEMM）→ mm.py:L1-L89 ；带 bias/alpha/beta 组合更接近 addmm.py:L1-L94
- rope → rotary_position_embedding.py:L1-L66
- causalSoftmax → 没有同名 kernel；现有 softmax.py:L1-L42 不含因果遮罩； scaled_dot_product_attention.py:L154-L199 内部带 is_causal 逻辑但属于整套 attention
- swiglu → 没有同名 kernel；可由 silu+ mul.py 组合得到
- randomSample：未找到对应实现
---

创新点：
- 国产异构卡智能感知
- agent自动算子调优

问题：

- 最终应该给评委讲一个什么样的故事？算子库？推理系统？调优系统？
- 计算任务解耦和任务性能偏好分析、异构硬件资源感知、互感知单元

- 论文结构：
    - 摘要
    - 绪论
    - 关键技术：
        1. 架构总览
        2. 3. 4. 三项关键技术
    - 实验结果与性能分析：
        1. 2. 3. 三项实验与分析
        4. 实验总结
    - 发展前景
    - 社会价值
    - 结论
    - 参考文献
---
## TODO

- 做个前端。酷炫的动画！
- 蒙特卡洛或许就是一个亮点，要论清楚到底为什么要用、为什么好用、与单纯用LLM搜索相比有什么优势
