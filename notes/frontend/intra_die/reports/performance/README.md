# Intra-die 性能收益验收状态

本目录记录 resource_timeline_cost/v5 的正式证据。16 核同资源 naive/AUTO
对照满足 gain >= 5% 总验收；新增候选把树形归约与目标核直达 DMA 作为一项
可独立开关的组合优化。

## v5 树形归约 + 目标核直达 DMA

冻结 workload 为 `notes/frontend/examples/intra_die_perf_16core.yaml`，硬件为
2 dies、每 die 4x4 cores。naive 和 AUTO 均激活全部 32 个 core，重复三次且
签名稳定，timing/address_lifecycle/transport_control 全部通过。

| workload | naive cycles | v5 AUTO cycles | AUTO 选择 | speedup | gain | 预测误差 |
|---|---:|---:|---|---:|---:|---:|
| 16-core | 21432 | 14273 | 16-way tree reduce + target-core direct DMA | 1.5016x | 33.403% | 0.0701% |

相对旧的 16-way streaming-reduce AUTO（20293 cycles），新组合优化为
1.4218x，周期进一步降低 29.665%。每个目标核直接从 state DMA 读取自己的
连续 K 分片，消除源核到目标核的输入 handoff；16 个 partial 按 4 层二叉树
归约，共 15 条树边。实测 local NoC send/recv/wait 从各 90 次降到各 60 次。
正式报告见 `model_v5_tree_direct_16core/comparison.json`，新旧 AUTO 增量报告见
`model_v5_tree_direct_16core/incremental_comparison.json`。

## 旧 v4 16 核 streaming-reduce 结果

冻结 workload 为
`notes/frontend/examples/intra_die_perf_16core.yaml`，硬件为 2 dies、每 die
4x4 cores。naive 和 AUTO 的 artifact 都包含 32 个 core，ACK/DONE 都严格覆盖
core 0--31，因此两边均为每 die 16 核，不再以单核 identity 作为 baseline。

| workload | naive cycles | AUTO cycles | AUTO 选择 | speedup | gain | 预测误差 |
|---|---:|---:|---|---:|---:|---:|
| 16-core | 21432 | 20293 | 16-way streaming reduce | 1.0561x | 5.314% | naive 7.13%, AUTO 9.99% |

naive 使用 16-way split-K，等待全部 16 个 partial 后归约；AUTO 在同样的
16 个 compute groups 上比较 barrier/streaming，选择 handoff/reduce 流式交错。
两边各运行三次且签名稳定，搜索过程中 simulator 调用为零。正式报告见
`model_v4_16core/comparison.json`。

## 旧 OFF/AUTO（单核 identity）结果

| workload | OFF cycles | AUTO cycles | AUTO 选择 | gain | 预测误差 | 状态 |
|---|---:|---:|---|---:|---:|---|
| compute | 14760 | 14760 | identity | 0% | 0% | no-regression pass |
| sync | 8597 | 8597 | identity | 0% | 2.48% | identity fallback pass |

两组对照都满足同 workload、hardware、simulation、mapping、IR1、pre-refine projection、finalizer 和 simulator SHA；OFF/AUTO 各运行三次，artifact、makespan 和关键计数稳定；product search 的 simulator 调用为零。

显式 FORCE split-K(2) 在同一 compute workload 上为 18070 cycles，重复稳定，相比 OFF 退化 22.43%。v3 模型预测该候选 19556 cycles，误差 8.22%，并在 AUTO 中以 break_even_not_met 淘汰。parts=4 同样被淘汰。

## 资源结论

- OFF 每个 source core 的 compute busy 为 1374 cycles，DTE/LSU busy 约 10438 cycles。
- FORCE 虽把 terminal GEMM 的单核工作分到第二 compute group，但增加了逐行 pack、输入 handoff、partial handoff、wait 和 reduce；固定 transport/control 开销大于可节省的 compute cycles。
- 当前 64 KiB SRAM 加 uint16 relocated address 限制了更大的 K/V 正向 workload。扩大 JSON 中 SRAM 容量会被 finalizer 正确拒绝，不能作为收益证据。
- 旧对照把单核 identity 作为 baseline，不能回答同为 16 核时的编排收益；
  保留这些结果仅用于验证 identity fallback/no-regression。

## 2x2

2x2 实测为 A00=14760、A01=14760、A10=6391、A11=6391 cycles。inter-die swizzle 的组合 speedup 为 2.3095x，intra 的边际收益仍为 1.0x。naive 两格已校准；swizzle 两格的绝对 timing prediction 误差超过 20%，所以 matrix 明确记录 status=fail_uncalibrated，不能作为完整验收成功报告。

## 已闭环与剩余硬门槛

已闭环：OFF/AUTO/FORCE 接口、digest 绑定、候选上限、零 simulator 搜索、cycle 权重、split-K zero-copy/direct receive/streaming reduce/slot lifetime、容量溢出时的 liveness fallback、公平对照、资源证据、稳定性和 no-regression 回退。

已闭环：至少一个冻结 workload 的 5% intra-only 收益（16 核对照）。

仍未闭环：独立 overlap marker 提供的 DTE/compute overlap 证据，以及
swizzle 两格不超过 20% 的绝对预测误差。16 核结果的收益来自
handoff/reduce 编排交错，当前由任务顺序、per-core primitive timeline 和
makespan 共同证明。
