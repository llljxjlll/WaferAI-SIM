现在已经讨论了很多了，我想收敛一下前端设计的想法。现在我需要实现模型训练/推理负载在这个仿真器上的端到端运行，并在其中插入我设计的GEMM+collective融合算子，体现我的两点技术点：跨Die计算通信Swizzling编排和Die上数据流编排。
我先介绍一下我的技术点大致思路：
# Inter-Die编排：
思路：所有的负载（训练/推理）可以收敛为：Mesh阵列+GEMM运算+collective op（图1所示）
（晶圆级芯片的特殊表征方式，需要考虑Mesh阵列）
可配置参数：（按顺序依次搜索）
训练：
1. 并行切分：决定GEMM大小与collective op
2. 位置mapping：mesh阵列情况
3. 推导出最优Swizzling匹配模式

推理：
1. PD融合/分离：决定GEMM大小与跨Die流量（逐层KV传输）
2. instance内并行切分：决定GEMM大小与collective op
3. 位置mapping：mesh阵列情况
4. 推导出最优Swizzling匹配模式

输出：（给片上数据流优化的信息）
单Die计算通信图：（图2所示）
- 通信op包含单Die的跨Die流量大小与方向(N/S/E/W)
- 计算op包含单Die GEMM任务大小

（Exp）前端构建：
1. 支持可配置参数
2. 支持mesh表征且能计算等效带宽
3. 支持SPMD，切分自动推导：由并行切分得到GEMM大小、插入collective op
4. 支持推理的动态跨Die流量表征(e.g. KV Cache Transfer，需要在仿真器中补充能力)
5. 支持MoE训推
6. 尽量复用仿真器已有负载模式
7. op要表征非GEMM op，方便decode核融入更多相邻的计算op，更好掩盖通信/访存

# Intra-Die编排：
得到单die任务之后，考虑片上NoC通信能力、端口放置位置等优化单Die任务在片上的编排
（图3展示了架构）

现在需要设计一种前端来完成实验：
1. 设计一种inter-die负载表征的方式，不同于一般的计算通信图，不仅要表示计算、通信算子，还需要表征mesh阵列信息（同一Die可能在不同的算子中处在不同的mesh阵列，比如在TP group和EP group中会处于不同的mesh阵列）
2. 设计intra-die负载的表征方式，可以表征单die的计算通信图，作为片上数据流编排
3. 留下各层次优化的替换接口（方便将naive版本与优化版本比较，也为后续消融实验提供基础）：GEMM+collective算子逻辑替换（inter-die优化替换）、片上数据流编排；为了同时实现两种优化，可能需要考虑两种负载映射方式的兼容问题：JSON文件与ISA形式