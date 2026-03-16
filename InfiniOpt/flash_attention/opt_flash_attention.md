# 优化GEMM算子

基于MCTS算法与LLM对flash-attention算子进行调优

## 定义算子workload

首先需要定义flash-attention算子的workload，即算子的输入输出张量的大小和数据类型。

## 初始化

随机从`LLM-Compiler/InfiniCore/scripts/profile/flash_attention/flash_attention.yaml`中读取flash-attention算子的schedule参数作为初始算子，记为$p_0$，将$p_0$作为MCTS的初始状态。

## 变换

从根节点 $p_0$ 开始遍历，使用UCT公式来选择下一个进行变换的节点 $p_i$。

## 扩展

使用LLM对算子进行某种调优

需要将以下内容作为LLM的输入：
1. $p_i$，当前算子的代码/调优配置文件，以及性能分数
2. 父节点与祖父节点的算子代码/调优配置文件，以及性能分数
3. 当前硬件名称
4. 当前算子的workload
5. 当前算子schedule参数的可选择空间
6. 具体要求：
   ```
   性能分数代表对某个候选配置在对应问题上“更优”的相对评分，分数越大，排序越靠前。根据以上信息，分析$p_i$的性能分数，判断是否需要进行调优。如果需要，根据你的先验知识，分析$p_i$的性能分数与其他变体的差异，识别性能变化的来源。根据分析结果，生成新的调优建议。
   生成的调优建议必须是json格式的...
   ```
之后，解析LLM输出的json，根据json中的调优建议，生成新的状态 $p_{i+1}$

## 模拟

根据新的状态 $p_{i+1}$，调用`LLM-Compiler/InfiniOpt/flash_attention/xgboost/flash_attention_xgboost.py中的`predict_xgboost`函数对 $p_{i+1}$ 进行预测，得到新的算子的性能分数。

## 反向传播

将成本模型预测出的分数（奖励 $W$）反向传回给该节点的所有祖先节点，同时更新路径上每个节点的访问次数 ($N$) 和累积奖励 ($W$)。

## 停止条件

当MCTS搜索达到预设的迭代次数或满足其他停止条件时，停止搜索。