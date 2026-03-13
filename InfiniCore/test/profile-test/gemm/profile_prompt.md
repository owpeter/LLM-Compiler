# 1. 生成算子

## 算子schedule参数

你需要编写一个基于拉丁超立方抽样 (LHS) 算法的 Python 函数，从`LLM-Compiler/InfiniCore/scripts/profile/gemm.yaml`中的`schedule`部分中分别读取参数，生成一个参数组合。参数组合要求：
1. 从`schedule`中随机选择一个`dtype`、`block_m`、`block_n`、`block_k`、`unroll`、`num_warps`、`num_stages`值。
2. 总的参数组合数量不能大于1000条

## 生成算子

对于每一组参数，调用`LLM-Compiler/InfiniCore/scripts/profile/build_gemm_nt.py`中的`build_gemm`函数并传入参数，等待生成指定算子。

## 重新编译算子库

当`build_gemm`函数执行完成后，使用`python scripts/install --nv-gpu=y --ninetoothed=y --ops gemm`命令重新编译Infinicore算子库

# 2. 生成workload

## workload参数

通过LHS算法从`LLM-Compiler/InfiniCore/scripts/profile/gemm.yaml`中的`workload`部分中分别读取参数，生成一个参数组合。参数组合要求：
1. 从`workload`中随机选择一个`prefill`或`decode`任务，并从该任务中随机选择一个`m`、`n`、`k`值。输入矩阵的维度为`m * n`, `n * k`，输出矩阵的维度为`m * k`。
2. `prefill`和`decode`任务数量分别20条

## 生成随机workload

根据矩阵大小生成随机矩阵

# 3. 进行profile

仿照`LLM-Compiler/InfiniCore/test/infiniop/gemm.py`脚本，调用gemm算子并输入随机生成的矩阵，记录算子执行时间。

# 4. 记录结果

将算子执行时间记录到一个csv文件中，文件格式为：
```
dtype,block_m,block_n,block_k,unroll,num_warps,num_stages,m,n,k,run_time
```

# 注意

**以上流程为一个完整工作流**，当一个流程完成后，重新使用新的schedule参数生成新的算子并进行下一轮测试