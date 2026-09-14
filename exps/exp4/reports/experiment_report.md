> **已作废：** 本报告对应错误的 duration-ledger scaling v1，数值和结论不得使用。
> 请改读 [corrected_experiment_report.md](corrected_experiment_report.md) 以及
> `results/validation_summary.json`。


# Exp4 软硬件协同优化实验报告

## 结论

本次实验已遍历383个Pareto/哨兵硬件候选、36个exp-2 primitive负载及naive/sw_opt
两种软件状态，生成27,576条主结果和9,192条请求级派生结果。所有候选均按500 MHz、
`P_core=2*N_PE*f`、独立DTE channel、单core SRAM读写各自共享预算以及解析D2D/HBM3
语义闭合。

严格可行的训练—请求Pareto前沿为空。原因不是聚合失败，而是主请求指标要求B64和B512
同时可行；exp-2 B512驻留需求约4--178 TB，而候选的36-module单副本最大HBM容量约
2.3 TB。因此报告另外输出了明确标记为
`capacity-infeasible performance projection`的机理前沿，且没有把这些点混入可行Pareto。

在该投影前沿内，六模型平均的training、inference和balanced最优点在naive与sw_opt下均为
`cand-704ce24e9c9f1aa7`，未发生硬件argmax切换。这是规划中预先允许的负结果：在当前候选
空间和解析模型下，软件优化没有改变最优硬件选择。

该候选配置为：`n_ctrl=2, router=broadcast, B=512 GB/s, K=3 MiB/core,
B_s=1024 GB/s/core/direction, N_PE=12288, N=4, e_H=2, m=1,
DTE_channel=4, D2D_edge=512 GB/s`，物理扫描给出的wafer容量为96 modules。

## 结果规模和门禁

- 候选数：383；primitive数：36；软件状态数：2。
- 主结果：27,576；请求派生：9,192；压缩资源账本：220,608。
- 主结果中11,964条为解析可行，15,612条为明确标注的不可行性能投影。
- 非有限/非正周期：0；理论下界违规：0；`T_sw_opt > T_naive`：0。
- exp-2控制表精确连接48行，其中36行primitive和12行request；周期与
  `source_result_digest`原样继承，测试相对容差为`1e-12`。
- 10项标准库单元测试通过，包括383候选闭合、四个阻塞项、SRAM共享合同、exp-2连接、
  36-rank映射、Pareto和SVG生成。

## 三类优化收益

以`H_exp2 + naive`为共同分母，六模型范围如下：

|负载|sw_opt_only|hw_opt_only|sw_hw_opt|
|---|---:|---:|---:|
|training|1.039--1.071x|1.372--1.574x|1.399--1.630x|
|prefill|1.116--1.227x|0.552--0.951x|0.552--0.978x|
|decode|1.000--1.025x|0.635--1.018x|0.635--1.018x|

训练从更高候选算力获益；prefill/decode相对exp-2参考硬件未普遍加速，主要因为exp-2采用
候选集外的1 TB/s D2D参考语义，且当前请求容量约束很强。不能把后两行解释为候选硬件的
绝对性能退化，而应解释为相对该外部参考点的结果。

## 解析模型和敏感性

全量扫描没有用单一`old_time * old_peak/new_peak`缩放。每条exp-2账本分别恢复并重新服务
tensor/vector、SRAM read/write、NoC、独立DTE channel、共享D2D有向边、物理HBM stack及
control资源，再以理论下界约束结果。软件收益以exp-2精确speedup为参考锚点，并按候选的
compute/memory/fabric重叠机会调整。

exp-2发布的prefill表没有逐资源账本，因此prefill使用同模型、同序列训练forward账本作为
资源类别代理，再分别校准到exp-2的prefill naive/sw_opt精确周期；结果带有解析代理语义，
不属于直接周期证据。

一次一参数敏感性覆盖DTE效率、D2D PHY效率、HBM利用率、HBM first-byte和stripe不均衡
共10个变体。六模型平均投影前沿的2个点在所有变体中均保留；LLaMA3-8B与Mixtral的6点
前沿对扰动敏感。HBM first-byte因exp-2未发布可分离计数，使用已记录的2%固定服务归因。

## 周期精确证据状态

规划的168次短运行清单已生成（96次motif拟合、72次E2E holdout），但当前仓库没有这些
current-build直接双跑日志；已完成数为0。因此所有完整候选结果标记为
`resource_explicit_analytical_extrapolation`，发布状态为
`analytical_only_current_build_anchor_pending`。exp-2已有证据仅作结构证据，未冒充本次
直接周期精确校准。

可复现digest：

- run: `38894a3e961555a837c71d49fd7cf03cc1956a8d67c3fd150b89344b6d13af00`
- Pareto: `1bd968cdf8e08377e1e05ab62c1fcdef6e083ee67d96b9fc78da8acdb090aee0`

注：上述digest来自本次结果生成；若重新生成源文件或结果，应以`run_manifest.json`和
`pareto_summary.json`内的新值为准。
