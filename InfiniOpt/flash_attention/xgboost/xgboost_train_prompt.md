# 训练XGBoost模型

所有数据的来源为flash_attention.csv文件

## 预处理

根据`batch, num_heads, q_len, total_kv_len, head_dim`中的数据，计算出`total_flops`这一列的值。

`total_flops` = $4 \times \text{batch} \times \text{num\_heads} \times \text{q\_len} \times \text{total\_kv\_len} \times \text{head\_dim}$

计算TFLOP：
`TFLOP` = `total_flops` / `run_time` * 1e9

将`TFLOP`作为预测标签

## 训练

将`[batch, num_heads, q_len, total_kv_len, head_dim,block_m,block_n,num_warps,num_stages]`拼接成一个向量，作为模型的输入，模型的目标是预测`TFLOP`。模型使用`rank:pairwise`

## 模型参数

将模型参数保存为之后方便使用的格式