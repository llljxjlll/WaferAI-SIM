# 实验 1-2 调试经验与问题记录

更新日期：2026-08-27

## 1. 坐标必须统一为 (x,y)

早期 placement 与 HBM attachment 混用了 row/column 和 (x,y)，会把 South/North stack
错误地放到左右边。最终统一使用：

- die_id = y×6+x；
- compact center group = (2,2)(3,2)(2,3)(3,3)；
- noncompact center group = (1,1)(4,1)(1,4)(4,4)；
- HBM = (1,0)(4,0)(1,5)(4,5)；
- route = X-first。

回归测试同时检查坐标、die id 和完整 route。

## 2. HBM 地址必须是全局唯一范围

不能让四个 stack 都从 address 0 开始。当前每个 stack 暴露独立的 16 GiB 全局范围，
起点为 0/16/32/48 GiB；gate/up/down root 全部按 256-byte alignment 分配并验证 owner
range。Mixtral 与 DeepSeek 的完整 expert weight 都通过容量检查。

## 3. 先保存 assignment，再逐 expert padding

S×top-k 必须先落到整数 A[source,expert]，再计算每个 expert 的 M_e 和
align_up(M_e,128)。DeepSeek 短序列不能使用 align_up(sum M_e,128)，否则会严重漏算
256 个 expert 各自的 underfill/padding。

## 4. Dispatch 是两个 GEMM

Dispatch 包含不同的 gate 和 up weights，logical/runtime FLOPs 的系数为 4；Combine 的
down GEMM 系数为 2。两者的 HBM root 和 replay bytes 也不能共用一个普通 GEMM 公式。
SwiGLU 与 weighted combine 必须留在 vector work 中。

## 5. local assignment 不进入 D2D

balanced routing 也有约 1/4 assignment 的 expert home 与 source rank 相同。这些 work
保留在本地 GEMM，但不能产生 D2D payload。回归按 A[source,expert] 和 expert_home_rank
重新计算 remote count，与 physical flow 守恒字段比较。

## 6. loaded contention 必须由流量推出

旧实现给 noncompact 固定写 contention=3，无法区分 isolated 与 loaded。当前 loaded
显式放置 9 个 group，再把每个 group 的 remote flow 经 X-first route 加到有向 edge。

实际观察到：

- compact loaded/isolated D2D max-link=1，因为九个 2×2 group 链路不重叠；
- noncompact loaded/isolated D2D max-link=3，因为九组长 route 在中心重叠。

如果未来 assignment 或 placement 改变，这个比值会自然变化。

## 7. local NoC 的 broadcast 必须乘 Tk

A/B broadcast 都发生在每一个 K tile。遗漏 Tk 会让高 K 模型的 NoC 流量低估数十倍。
现在公式逐 expert 使用 output_tiles_e×Tk，并把消息映射到 4×4 core mesh 的实际有向
route。测试从结果的 PE/PM/PN/PK 反推 bytes，避免只检查正值。

## 8. 允许 underfill 和负收益

Grouped-GEMM 的合法条件是 PE×PM×PN×PK≤16，不是必须等于 16。architecture 实际出现
2/8-core underfill，因为 HBM 主导时更多并行核不一定改变选择。四状态也不能设严格偏序；
本轮虽然全部 actual speedup>1，但测试只检查守恒和公式，并用人工负样本验证图不会隐藏
T11>T00。

## 9. HBM-free 不能删除 HBM 审计

消融只令 modeled_hbm_cycles=0。原始 HBM bytes、stack service、route 和 capacity 字段
仍保留，以确认 workload 未变。HBM-free 的高 speedup 尤其依赖 provisional tensor
efficiency/vector rate，只能作为计算通信潜力，不是部署预测。

## 10. loaded 吞吐必须区分 focus 与 scenario

loaded T11 包含 9 个 group 的资源竞争。如果仍用单 group FLOPs 作分子，得到的是 focus
group effective throughput；若用 9×FLOPs，则是 wafer scenario aggregate throughput。
结果分别保存 optimized_tflops 和 scenario_optimized_tflops，报告不得混用。

## 11. 周期精确 evidence 必须检查全部 digest

仓库中的旧 168-sample profile 与当前 npusim SHA 不同，即使 sample 完整也不能直接复用。
当前 preflight 正确拒绝该 profile。本轮新跑的 Flexible smoke 虽然双跑完全重复，但
workload 是 H16/I32/top-1、硬件是 tiny local-HBM fixture，只能证明入口、ProgramIO 和
资源释放闭合。

判断 evidence 能否用于定量，必须同时匹配 hardware、simulation、mapping、tool 和
workload digest；缺一项只能标 structural prior。

## 12. theory attainment 不是准确性

architecture 中 max-stack HBM service 同时进入 T11 与 theory resource floor，因此
attainment 接近 1。这只是同一模型内部“达到自身 HBM roof”的结果，不能代替
cycle-accurate replay error。正式发布仍需独立的 median/p95 error 与 CI。

## 13. 当前仍待解决

inter fused replay 的一个 wave 仍使用解析后的复合 compute resource，没有完全拆成
production 的 tensor core、vector、LSU/HBM port 和逐 local route action。目标
H128/H2000 的 unit closure、top-k=2/8 motif、skew/capacity 和直接 T00/T11 反事实也未跑。
因此当前结果保持 analytical 与 calibration_pending 状态。
