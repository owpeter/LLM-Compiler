# 训练XGBoost模型

所有数据的来源为gemm_profile.csv文件

## 预处理

根据`block_m, block_n, block_k`这三列，计算出`total_flops`这一列的值。

`total_flops` = 2 * `block_m` * `block_n` * `block_k`

计算TFLOP：
`TFLOP` = `total_flops` / `run_time` * 1e9

将`TFLOP`作为预测标签

## 训练

将`[m,n,k,block_m,block_n,block_k,unroll,num_warps,num_stages]`拼接成一个向量，作为模型的输入，模型的目标是预测`TFLOP`。模型使用`rank:pairwise`

## 模型参数

将模型参数保存为之后方便使用的格式