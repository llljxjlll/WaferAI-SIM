# exp1-1 开发方案：周期精确 Trace 校准的两层快速估算

## 1. 目标与结果定义

目标是在不实现完整模型 M/N/K tiled lowering、不为 176 个 case 分别编译完整
DAG 的前提下，得到 AG+GEMM、GEMM+RS 的全部实验数据，并分离：

- inter-die 编排带来的性能提升；
- intra-die 16 核编排带来的性能提升；
- 两类优化组合后的协同效应；
- 片上拥塞对稳态吞吐的影响。

只周期精确仿真固定大小的多 tile 窗口，完整模型时间由实测 trace 校准后回放。
结果定义为 cycle-accurate trace-calibrated estimate，不宣称是完整模型端到端
周期精确仿真。

## 2. 核心方法

只实现一种 Q=8 的最大重叠 tile motif，即完整优化版本 T11。程序包含：

- 全部实际 die 和实际 mesh 路由；
- 每 die 搜索 PM×PN×PK=16 的自适应核映射；
- HBM tile DMA；
- AG 或 RS；
- local NoC handoff；
- streaming reduce；
- SRAM lifecycle；
- 8 个连续 tile 的稳态重叠。

从周期精确 trace 提取事件的开始/结束周期、资源和依赖。Python 回放器复用同一
批事件，只调整依赖边，生成其余三个反事实调度结果。

| 结果 | Inter-die | Intra-die | 来源 |
|---|---|---|---|
| T00 | naive | naive | 周期精确 trace 回放 |
| T10 | optimized | naive | 周期精确 trace 回放 |
| T01 | naive | optimized | 周期精确 trace 回放 |
| T11 | optimized | optimized | 周期精确仿真实测 |

四种结果使用完全相同的 logical/runtime shape、padding、tile、FLOPs、HBM bytes
和 collective bytes，只允许依赖与资源占用顺序不同。

HBM 使用与 exp1-2 共用的 `fused_boundary_tile_replay_v1` 口径：

```text
AG+GEMM: HBM_bytes = Tm × K × rank_N × dtype_bytes
GEMM+RS: HBM_bytes = Tn × M × rank_K × dtype_bytes
                         + Tm × rank_K × N × dtype_bytes
```

AG activation 从网络到达，权重按 M tile 重放；GEMM+RS 按 output-stationary
顺序计入 A 跨 N tile、B 跨 M tile的重放。实际与理论使用同一 HBM bytes。

## 3. 固定 Tile 与 SRAM 约束

首版固定：

    Q = 8
    Mt = 128
    Nt = 512
    Kt = 256
    dtype = FP16
    candidates = {(PM, PN, PK) | PM*PN*PK=16, PM<=Tm, PN<=Tn,
                  PK<=ceil(rank_K/Kt)}
    spatial_util = (Tm*Tn) / (ceil(Tm/PM)*PM*ceil(Tn/PN)*PN)
    cost = max(T_HBM, T_compute/spatial_util + T_local_transport)

其中 PM/PN 是并发输出 tile 的核心网格，T_local_transport 同时计算 A 广播、
B 广播和 (PK-1) 份部分和归约。尾波不足16核时通过 spatial_util 显式计入。
按 cost 最小选择编排；同分时依次选择 transport 更小、PK 更小的候选。

程序生成前计算保守 live set：

    live_bytes =
        2 * FP16_bytes * (Mt*Kt + Kt*Nt)  # A/B 双缓冲
      + FP32_bytes * Mt*Nt                # accumulator
      + FP32_bytes * Mt*Nt                # reduce/NoC scratch 上界
      + fixed_runtime_reserve

要求 live_bytes 不超过 3 MiB。首版不开发 tile 自动搜索；固定 tile 不合法时，只
按固定候选表尝试更小的 Mt/Nt/Kt。

## 4. Unsupported Case 规范化

每个逻辑 case 先规范化，而不是直接交给完整模型 lowering：

    runtime_M = align_up(logical_M, Mt)

    # AG+GEMM：先按 TP 对 N 分片，再在每个 die 内做 tile 对齐
    rank_N    = align_up(ceil(logical_N / TP), Nt)
    runtime_N = TP * rank_N
    runtime_K = align_up(logical_K, Kt)

    # GEMM+RS：先按 TP 对 K 分片，再在每个 die 内做 tile 对齐
    rank_K    = align_up(ceil(logical_K / TP), Kt)
    runtime_K = TP * rank_K
    runtime_N = align_up(logical_N, Nt)

    Tm = runtime_M / Mt
    Tn = rank_N / Nt                     # AG；RS 使用 runtime_N / Nt
    Tk = ceil(rank_K / Kt)
    intra_K_waves = ceil(Tk / PK)
    Qfull = Tm * Tn

该过程覆盖当前主要失败：

- 分片维不能被 TP 整除：在 rank-local shard 上做最小 padding；
- rank-local tile 存在尾块：只对齐到一个 Mt/Nt/Kt，不为填满 16 核而 padding；
- 完整 weight、activation 或 partial 超过 3 MiB：tile streaming；
- 完整 DAG 太大或 linker 过慢：只编译 Q=8 motif；
- 完整 shape 没有 Wang-1D candidate：先用 tile shape 重新规划，仍失败时使用
  topology 合法的 canonical experiment fallback。

所有 padding 都保存 logical/runtime 两套 shape。性能分子使用 logical FLOPs，
padding 额外工作只体现在仿真周期中。

理论下界必须使用同一组 runtime_M/runtime_N/runtime_K，不得使用未 padding 的
logical shape 与 replay 周期相除。logical shape 只用于报告有效 FLOPs。
主图理论定义与 FlashOverlap 一致，并扩展到当前 16 核 lowering：

    theory_naive = T_comp_single_core + T_HBM + T_local_transport(PM,PN,PK) + T_comm
    ideal_intra = max(T_comp_16core_peak, T_HBM, T_local_transport(PM,PN,PK))
    theory_upper = max(ideal_intra, T_comm) + min(ideal_intra, T_comm) / Qfull
    theory_speedup = theory_naive / theory_upper

最后一项表示不可隐藏的一个边界 wave。理论项使用相同 padded runtime shape 和 tile
数量，但去除实际计算中的 90% 利用率折损。分析回放的实际 fused 时间另外建模：

    segmented_comm_efficiency = 0.72 + 0.12 * min(1, segment_bytes / 512KiB)
    topology_efficiency = 1 / sqrt(congestion_factor)
    wave_fill = Qfull / (Qfull + 256)
    split_k_penalty = 0.006 * log2(PK)
    fanout_penalty = 0.0015 * (PM + PN - 2)
    spatial_penalty = 0.04 * (1 - spatial_utilization)
    core_schedule_efficiency = clamp(
        0.765 + 0.065 * wave_fill
        - split_k_penalty - fanout_penalty - spatial_penalty,
        0.70, 0.83)

实际 fused overlap 还保留片上资源争用尾部，避免 Qfull 较大时退化成完全重叠：

    stage_balance = min(T_comm, T_intra) / max(T_comm, T_intra)
    mesh_span = min(1, (mesh_rows + mesh_columns - 2) / 10)
    contention_tail = 0.012 + 0.028 * stage_balance + 0.010 * mesh_span
    T_fused = max(T_comm, T_intra)
              + min(T_comm, T_intra) * (1 / Qfull + contention_tail)

这些项分别表示分段通信带宽、mesh 跳数/拥塞、wave 填充、Split-K reduction、
PM/PN 广播扇出、空间尾块和通信/核内阶段共享注入端口造成的争用。效率及各 penalty、
contention tail、actual/theory attainment rate 必须随结果输出。原纯
compute/collective algorithmic upper 作为辅助字段保留，不用于主图。

## 5. Q=8 周期精确 Trace

每个事件至少记录：

    tile_id
    stage
    start_cycle
    end_cycle
    die_id
    core_id
    resources
    dependency_ids

stage 包括：

    HBM_LOAD_A
    HBM_LOAD_B
    AG
    GEMM
    LOCAL_SEND_RECV
    LOCAL_REDUCE
    RS
    HBM_STORE

从 tile completion marker 得到：

    Tfill = completion(tile_0)
    II = median(completion(tile_i) - completion(tile_{i-1}), i=2..7)
    Tdrain = program_done - completion(tile_7)

完整模型的 T11：

    T11_estimated = Tfill + (Qfull - 1) * II + Tdrain

K 方向必须按 Tk 个 K-step 回放，不能把不同模型的 K 次数隐含进固定常数。

## 6. 反事实调度回放

回放器按事件依赖和资源可用时间执行：

    for event in topological_order:
        dependency_ready = max(finish[dep] for dep in event.dependencies)
        resource_ready = max(available[r] for r in event.resources)
        start[event] = max(dependency_ready, resource_ready)
        finish[event] = start[event] + duration[event]
        for resource in event.resources:
            available[resource] = finish[event]

四种依赖规则：

### T00：naive inter + naive intra

    collective 全部完成
    -> DMA
    -> 按选定 PM×PN×PK 完成16核工作
    -> barrier
    -> 完成必要的广播和 (PK-1) 路 reduce
    -> 下一 tile

### T10：optimized inter + naive intra

允许 AG/GEMM 或 GEMM/RS 跨 tile 流水，但保留选定 die 内映射的 barrier。

### T01：naive inter + optimized intra

collective 与 GEMM 大阶段不重叠，但允许 A/B 双缓冲、自适应 PM/PN/PK、local NoC handoff 和必要的 streaming reduce。

### T11：optimized inter + optimized intra

直接使用 Q=8 周期精确 trace 的 Tfill/II/Tdrain，不用回放结果替代实测。

## 7. 拥塞校准

正常 Q=8 motif 使用真实 staggered 调度，包含：

- 所有 die 同时通信；
- 每 die 16 核注入 local NoC；
- router/link queue；
- credit/backpressure；
- SRAM、DTE、HBM 竞争。

推荐每个 mesh/operator 再运行一个同步注入压力版本：

    4 meshes * 2 operators = 8 calls

计算：

    congestion_factor = II_synchronized / II_staggered
    normal_estimate = Tfill + (Qfull-1)*II_staggered + Tdrain
    congested_upper_bound =
        Tfill + (Qfull-1)*II_synchronized + Tdrain

回放 T00/T10/T01 时使用 trace 中实测的重叠事件时长和竞争膨胀，不能把所有重叠
简单写成 max(stage_time)。

## 8. 调度签名与缓存

仿真缓存 key：

    mesh_rows, mesh_columns,
    Px, Py,
    operator,
    inter_die_algorithm,
    Mt, Nt, Kt,
    collective_tile_bytes,
    injection_mode

模型名、完整 M/N/K 和 seq_len 不进入 key。多个模型只要使用相同 tile、mesh 和
调度，就复用同一次周期精确仿真。

planner 对 tile shape 仍无候选时允许固定 canonical schedule，但必须记录：

    schedule_source = canonical_experiment_fallback

拓扑确实不可行、缺少 opcode、最小 tile 仍超过 3 MiB 或不能 drain 时，不虚构
结果，记录 algorithm_unavailable。

## 9. 仿真调用数量

最快配置：

    4 meshes * 2 operators
    = 8 个 staggered T11 Q=8 calls

推荐配置：

    8 个 staggered T11 Q=8
    + 8 个 synchronized congestion calls
    + 2~4 个 T00/T10/T01 直接仿真校验点
    = 18~20 calls

若 planner 实际选择多个不同的 1D/2D/Px/Py 调度签名，只为新签名补跑，目标是
将总调用量控制在 20~30 次以内。

## 10. 回放验证

选择 2~4 个代表点直接仿真对应反事实版本：

    error = abs(Treplay - Tsim) / Tsim

验收标准：

- error 不超过 10%：直接采用；
- 10% 到 20%：采用并输出误差区间；
- 超过 20%：为该调度签名增加周期精确调用或使用分段 II；
- 所有周期精确调用必须 program drain，router/link residual 为 0。

## 11. 性能提升分解

    inter_speedup_without_intra = T00 / T10
    inter_speedup_with_intra    = T01 / T11
    intra_speedup_without_inter = T00 / T01
    intra_speedup_with_inter    = T10 / T11
    total_speedup               = T00 / T11
    synergy = (T10 * T01) / (T00 * T11)

synergy 大于 1 表示正协同，约等于 1 表示基本独立，小于 1 表示资源争用。

## 12. 输出

每个 case 输出：

    case_id, mesh, operator, model, layer, seq_len,
    logical_M, logical_N, logical_K,
    runtime_M, runtime_N, runtime_K,
    tile_M, tile_N, tile_K, Tm, Tn, Tk,
    logical_flops, runtime_flops,
    T00_cycles, T10_cycles, T01_cycles, T11_cycles,
    inter_speedup_without_intra, inter_speedup_with_intra,
    intra_speedup_without_inter, intra_speedup_with_inter,
    total_speedup, synergy,
    normal_cycles, congested_upper_bound_cycles, congestion_factor,
    simulation_signature, schedule_source, result_source,
    status, error

来源字段：

    T11_source = cycle_accurate_simulation
    T00/T10/T01_source = cycle_accurate_trace_replay

输出到现有：

    results/results.json
    results/results.csv
    figures/*.svg

## 13. 最小开发步骤与时间

| 工作 | 时间 |
|---|---:|
| 固定 Q=8 最大重叠 tile 程序 | 30~60 分钟 |
| trace 阶段提取 | 30~45 分钟 |
| 四种依赖回放器 | 30~60 分钟 |
| 176 case 映射、缓存和输出 | 30~45 分钟 |
| smoke/debug | 30~60 分钟 |
| 8~20 次周期精确仿真 | 4 路并行后约 1~3 小时 |

目标总时间：

    开发 2~3.5 小时
    周期精确调用 1~3 小时
    全部数据与绘图 4~7 小时

## 14. 验收标准

- 四种调度使用完全相同的工作量、padding、tile 和数据搬运量；
- T11 来自全 mesh、每 die 16 核、Q=8 的实际周期精确仿真；
- trace 至少提供 6 个稳定 tile 间隔；
- SRAM live set 不超过 3 MiB/core；
- 所有周期精确调用完成 drain；
- 反事实回放的直接校验误差不超过 20%；
- 176 个逻辑 case 均输出估算结果或明确的 algorithm_unavailable；
- 图中明确区分周期精确实测与 trace-calibrated replay，禁止将后者标为完整模型
  端到端周期精确仿真。
