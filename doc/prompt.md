该算子的实现路径在 InfiniCore/src/infiniop/ops/${name}下，${name}为算子的名称。

## 增加流程

1. 在该算子`${name}`目录下构建ninetoothed目录，准备实现
2. 在`${name}/ninetoothed目录下实现算子的build.py`，其含义是构建多种模式的算子，具体可以参考`InfiniCore/src/infiniop/ops/gemm/ninebooted/build.py`。`build.py`的实现应该与`LLM-Compiler/ntops/src/ntops/kernels`下的核函数像对应
3. 在`${name}/ninetoothed目录下实现${name}.h`，其含义是C语言的调用接口，同样具体可以参考`InfiniCore/src/infiniop/ops/gemm/ninebooted/gemm.h`
4. 在`${name}`目录下修改`operator.cc`，让其支持ninetoothed的调用模式

## 验证流程

1. 使用`InfiniCore/scripts/build_ntops.py --ops ${name} --jobs 4`构建ninetoothed算子
2. 在`LLM-Compiler/InfiniCore/build/ninetoothed`下的`${name}.c`编写一定的调试信息，使得当调用该算子时，能够知道ninetoothed实现的算子被正确调用了，而非被回退了。
3. 构建完成后运行`InfiniCore/scripts/install.py --nv-gpu=y --ninetoothed=y`编译ninetoothed算子。
4. 运行`InfiniCore/test/infiniop/${name}.py --nvidia`验证算子正确性

