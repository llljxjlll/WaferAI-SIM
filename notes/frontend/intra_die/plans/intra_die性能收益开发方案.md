# Intra-die 可测性能收益开发方案

> 依据：[die 内数据流编排完整方案](../refs/die内数据流优化方案.md)与
> [Intra-die 优化端到端开发方案](./intra_die端到端优化开发方案.md)。
>
> 本方案不再重复开发已经闭环的 refine、LOCAL_NOC、lowering、finalizer 和
> simulator 接口，目标是以最小增量在一个冻结负载上得到可重复的
> intra-die makespan 收益，并使无收益负载自动回退 identity。

## 1. 当前状态与无收益的直接原因

现有实现已完成功能闭环，但当前报告不能证明 intra-die 性能收益：

| 对照 | naive intra | optimized intra | 结论 |
|---|---:|---:|---|
| naive inter | 8964 cycles | 8964 cycles | intra-only 收益为 0 |
| swizzle inter | 2721 cycles | 2721 cycles | inter 之上的 intra 边际收益为 0 |

原因已经可以从实现和报告中闭合：

1. production config 的 `critical_path_order=false`，optimized 没有改变执行顺序；
2. optimized 先调用 naive scheduler，普通路径只改 SRAM offset、bank 和 lifetime，
   没有改变核数、任务数、opcode 或关键依赖；
3. 当前 simulator 没有把 bank conflict 反映到 makespan，而冻结负载的 SRAM
   high-water 也没有下降；
4. 现有 split-K E2E 用的负载与 2×2 baseline 不同，不能直接比较；
5. v2 evaluator 虽然预测 identity 比 split-K 更快，却因为显式请求而强制选择
   split-K；当前路径是 carrier 验证，不是性能择优；
6. 当前小负载 `H=32, I=64, prefill_tokens=2/4`中 GEMM 计算量太小，
   LOCAL_NOC、reduce 和 sync 的固定开销足以抵消多核并行收益。

因此最短路径不是继续扩充模板数量，而是先修复“公平对照—真实代价—
自动择优—可重叠执行”这条链。

## 2. 本轮范围与 Done 定义

### 2.1 必须交付

- 新增一个冻结的 compute-bound Dense TP2 性能负载；
- 同一负载、硬件、mapping、simulation config 下运行 identity 和 optimized；
- 同一 source IR/projection 生成 identity、schedule-only、split-K 和 split-K+pipeline
  候选；
- 用与 simulator 一致的每核资源时间线代价模型排序；
- `auto` 模式选择预计最快的合法候选，无收益时返回 identity；
- 至少一个冻结负载上 optimized 相比 identity 的 simulator makespan
  下降不少于 5%；
- 连续运行 3 次，schedule/artifact digest、makespan 和关键计数稳定；
- inter-die/intra-die 仍可独立开关，输出新的 2×2 消融报告。

### 2.2 公平对照的强制条件

baseline 和 optimized 之间只允许下列字段不同：

- `intra_die.mode = off | auto | force`；
- 候选或特性开关；
- 由上述选择派生的 refine/schedule/manifest/artifact digest。

以下内容必须完全一致：

- workload YAML 的模型、token/profile 和 parallel 字段；
- hardware、mapping 和 simulation 文件 SHA256；
- inter-die policy；
- source IR1 digest 和 refine 前 projection digest；
- finalizer 和 simulator 二进制 SHA256。

报告在上述任一对照条件不同时必须 fail closed，不计入性能结论。

### 2.3 收益验收

对每个负载记录：

$$\text{speedup}=\frac{T_{\text{identity}}}{T_{\text{auto}}},\qquad
\text{gain}=1-\frac{T_{\text{auto}}}{T_{\text{identity}}}$$

正式 Done 同时满足：

1. 功能：address/lifecycle/transport/control 全部 pass；
2. 性能：至少一个预先冻结的受支持负载 `gain >= 5%`；
3. 选择：所有其他负载 `T_auto <= T_identity`，否则必须选择 identity；
4. 稳定：3 次运行结果完全一致；
5. 预测：被选候选的解析 makespan 与 simulator 误差不超过 20%；
6. 预算：product compile 搜索内 simulator 调用为 0，单个 identity/auto
   正式收益对照的最终证据调用不超过 6。

本轮仍只做 timing 结论。当前 Dense primitive 不写数值输出，不宣称
end-to-end numerical correctness 或真实硬件绝对性能。

## 3. 性能负载与 break-even 条件

### 3.1 不再用小功能 fixture 作为性能主负载

现有 `H=32, I=64, tokens=2/4` 保留为 smoke test。新增两个用途分离的冻结负载：

| 负载 | 用途 | 要求 |
|---|---|---|
| `intra_die_perf_compute_bound.yaml` | 正向收益主负载 | 大 K/大 GEMM，至少 4 个兼容核，分块后单核 SRAM 合法 |
| `intra_die_perf_sync_bound.yaml` | 回退对照 | 小 GEMM，预期 split-K 不赢，`auto` 必须选 identity |

开发期用有界网格确定最小可盈利尺寸：

- `H in {128, 256, 512}`；
- `I in {4H}`；
- `prefill_tokens in {8, 32, 128}`；
- `split_k_parts in {2, 4}`；
- 最多枚举 12 个 workload grid point；每个 point 的 product 候选仍不超过 8；
- 只有解析 break-even 过滤后的前 2 个 grid point 进入开发期 simulator 终评。

不允许运行后任意更改单个尺寸美化结果。首个满足硬件约束和下述
break-even 式的尺寸将作为冻结主负载，连同选择规则一起入库。

### 3.2 split-K 只在预计收益为正时合法

对 $p$ 个核的 split-K，候选进入精排序的必要条件为：

$$T_{\text{gemm},1}-T_{\text{gemm},p} >
T_{\text{remote-input}}+T_{\text{reduce}}+T_{\text{sync}}+T_{\text{fill/drain}}$$

代价必须使用该硬件/simulation config 的参数，不能继续使用固定的
`1_000_000 ops/cycle`。若上式不成立，split-K 在可行性阶段就被淘汰。

对 double buffer，只有当时间分块数 $R\ge2$ 且 simulator 存在真实的 DTE/compute
并行资源时间线时才生成候选。只复用两个 SRAM slot 而没有时间重叠不能
记作 double-buffer 性能优化。

## 4. 最小目标架构

~~~text
IR2ProjectionResult
  │
  ├─ identity candidate
  ├─ schedule-only candidate
  │    └─ critical-path order with cycle weights
  └─ graph-refine candidates
       ├─ split-K(2/4)
       └─ split-K + temporal chunks + double buffer
             ↓
      feasibility + break-even filter
             ↓
      calibrated per-resource timeline estimator
             ↓
      choose minimum predicted makespan
             ↓
      existing schedule → GlobalAction → lowering → finalizer → simulator
~~~

不新增编译 pass。现有 `intra_die_refine` 仍是图改写边界，`intra_die_schedule`
仍负责 placement/order/buffer。新增内容只是候选生成、代价估计和更精确的
split-K materialization。

## 5. 实现设计

### 5.1 统一选项与选择语义

将 refine options 升级为版本化选项：

~~~python
class IntraDieOptimizationMode(Enum):
    OFF = "off"       # 只生成/选择 identity
    AUTO = "auto"     # 生成有界候选并选最小预测 makespan
    FORCE = "force"   # 仅测试/调试，强制指定候选

class IntraDieOptimizationOptions:
    mode: IntraDieOptimizationMode
    allowed_candidates: tuple[str, ...]
    max_candidates: int = 8
    split_k_parts: tuple[int, ...] = (2, 4)
    temporal_chunks: tuple[int, ...] = (1, 2, 4)
~~~

兼容规则：现有 `SplitKRefineOptions` 显式请求映射为 `FORCE`，保留功能 E2E；
新的性能 runner 必须用 `OFF/AUTO`。

每个 decision 报告必须包含：

- 全部生成候选及不可行原因；
- 每个代价分项：compute、local transport、reduce、sync、fill/drain；
- `selected_candidate_ref`、`selection_reason`和 baseline 预测收益；
- 候选上限、解析求值次数和 simulator 调用预算。

### 5.2 把 critical-path 从字节权重改为周期权重

现有 bottom-level 用 `task.bytes` 作为权重，无法表示 GEMM 和 wait 的时延。改为：

$$BL(v)=T_{\text{estimate}}(v)+\max_{u\in succ(v)}BL(u)$$

- COMP：使用 workload shape 和 simulator 计算吞吐；
- SEND/RECV：`setup + hops * router_latency + bytes / bandwidth`；
- REDUCE：reduce setup + bytes / reduce bandwidth；
- WAIT/BARRIER：使用配置的 issue/sync 延时；
- LOAD/STORE：按 SRAM/HBM 通路分开。

`critical_path_order` 不再作为 registry 声称支持但默认关闭的特性。AUTO 同时
评估 source-order 和 cycle-weighted order，只在预测严格下降时选新顺序。

### 5.3 修复 split-K 的净收益路径

优先做以下四个小改动，不扩展 SUMMA/脉动模板：

1. **只传远程 part**：CG0 的 local part 直接 alias 原 operand slice，不生成
   LOCAL_COPY/SEND/RECV；
2. **直接接收到 stage slot**：远端 RECV 直接绑定 split operand stage buffer，清理
   `RECV -> LOCAL_COPY -> COMP` 中间复制；
3. **流式 partial reduce**：reduce core 对到达 partial 立即累加，不等待所有
   partial 齐备后再串行处理；
4. **真实时间分块**：将 K 维再分为 $R$ 个 temporal chunks，形成
   `recv[i+1] || compute[i] || reduce[i-1]`流水，两个 slot 交替使用。

对每个候选显式证明：

- K 可整除、slice 精确覆盖且不重叠；
- local/remote part 所在核与 CG 归属一致；
- producer 到 stage、stage 到 compute、partial 到 reduce 的 event 一一闭合；
- slot 0/1 内不存在 lifetime overlap；
- 不断开普通 dependency，不放宽 exact coverage validator。

### 5.4 建立与 simulator 同构的代价模型

用小型微基准从当前 simulation config 获取或拟合以下表项：

| 微基准 | 变量 | 输出 |
|---|---|---|
| MATMUL | M/N/K、dtype、core count | setup、有效 ops/cycle、drain |
| LOCAL_NOC | bytes、hop count、direction | setup、router latency、effective bandwidth |
| LOCAL_REDUCE | bytes、part count | setup、bytes/cycle |
| WAIT/BARRIER | producer-consumer 间隔 | issue 和 sync cycles |
| DTE+COMP overlap | chunk bytes、GEMM cycles | 可隐藏周期与资源冲突 |

完整解析模型不再把所有分项相加，而是模拟四类资源时间线：

- 每核 issue stream；
- 每核 compute engine；
- 每个 DTE/LOCAL_NOC channel；
- reduce engine 和 event dependency。

任务开始时间为其 dependency 完成时间与所需资源可用时间的最大值。候选
预测 makespan 是所有终端任务完成时间的最大值。

标定产物版本化保存，并绑定 hardware/simulation SHA256。配置 digest 不同时
不允许复用。

### 5.5 最小 simulator 反馈

为快速定位无收益原因，在 run report 中增加：

- critical-path record IDs 与分项 cycles；
- per-core compute/DTE/issue busy cycles；
- LOCAL_NOC bytes、hop 和 busy cycles；
- reduce busy cycles；
- wait/barrier stall cycles；
- DTE/compute overlap cycles。

本轮不把 bank-conflict simulator 建模作为收益前置。在该建模完成前，
bank-stagger 只作为合法的布局候选，不计入预测收益。

## 6. 分阶段开发计划

### P0：公平对照和基线冻结（0.5–1 天）

1. 将 split-K CLI 扩展为 `off/auto/force` 三模式；
2. 同一 CLI 一次生成 identity 和 auto 两份报告；
3. 增加 input-equivalence validator；
4. 保留小负载作为 sync-bound 回退样例；
5. 按预定网格冻结 compute-bound 主负载。

**验收**：同一 compare report 中两格的输入 SHA 完全一致；小负载 auto 选
identity。

### P1：计时标定与 cycle-weighted schedule（1–2 天）

1. 增加 MATMUL、LOCAL_NOC、LOCAL_REDUCE、sync 微基准；
2. 生成版本化 timing calibration table；
3. 实现每核资源时间线 estimator；
4. critical-path ordering 改用 cycle weight；
5. AUTO 对 source-order 与 critical-path order 做真实择优。

**验收**：四类微基准和两个端到端候选的预测误差均不超过 20%；
AUTO 的 product 编译选择过程中不调用 simulator。

### P2：split-K 快路径去冗余（1–2 天）

1. local part zero-copy alias；
2. remote part direct-to-stage receive；
3. streaming partial reduce；
4. 按 break-even 条件筛选 `parts in {2,4}`；
5. 输出每个候选的传输/归约/同步差分。

**验收**：相比现有 split-K，不再为 local part 生成 LOCAL_NOC；每个 remote
part 只有一次必要发送和一次必要接收；所有 exact coverage/event/lifetime 测试通过。

### P3：真实双缓冲流水（1–2 天）

1. 生成 `temporal_chunks in {2,4}` 候选；
2. 将 issue、DTE、compute、reduce 按资源分离；
3. 实现 `recv[i+1] || compute[i] || reduce[i-1]`；
4. 证明两个 slot 的版本/lifetime/event 闭合；
5. 当 overlap 为 0 时不选择 double-buffer 候选。

**验收**：报告中 `overlap_cycles > 0`，且 pipelined split-K makespan 严格小于
非流水 split-K。

### P4：AUTO 择优和端到端收益（1 天）

1. 候选上限固定为 8；
2. 对不可行候选 fail closed，对不盈利候选正常淘汰；
3. AUTO 严格按 predicted makespan 选择，tie 时优先 identity；
4. 运行 identity/auto 各3 次；
5. 产出 intra-only 和 inter+intra 的 2×2 报告。

**验收**：compute-bound 负载 `gain >= 5%`；sync-bound 负载选 identity 且无回归；
预测误差不超过 20%；所有稳定性和预算断言通过。

预计开发时间为 4.5–8 天。P0–P2 可形成首个非流水收益版本；若已达
5% 收益，P3 可作为增益阶段，但 AUTO 回退和公平对照仍必须完成。

## 7. 代码改动清单

### 7.1 优先修改

| 文件 | 改动 |
|---|---|
| `schema/intra_die_refine.py` | OFF/AUTO/FORCE 选项和候选限额 |
| `schema/intra_die_v2_search.py` | 候选类型、分项代价、淘汰原因和选择证据 |
| `policies/intra_die_v2_search.py` | break-even filter 和 AUTO 择优 |
| `policies/optimized_intra_die.py` | cycle-weighted order 与候选化，不再全局固定关闭 |
| `policies/split_k_intra_die_refine.py` | zero-copy、direct receive、streaming reduce 和 temporal chunks |
| `compiler.py` | 传递统一 options 并记录 pre-refine digest |
| `runner.py` | 公平对照、标定证据、收益与稳定性报告 |

### 7.2 新增文件

| 文件 | 用途 |
|---|---|
| `schema/intra_die_timing_model.py` | 版本化 timing table 和资源时间线证据 |
| `policies/intra_die_timing_model.py` | 微基准拟合和完整候选估时 |
| `integration/run_intra_die_performance.py` | 同输入 identity/auto/FORCE 对照 runner |
| `examples/intra_die_perf_compute_bound.yaml` | 冻结正向负载 |
| `examples/intra_die_perf_sync_bound.yaml` | 冻结回退负载 |
| `reports/performance/` | 候选、标定、对照和 2×2 报告 |

C++ 仅在现有 simulator 无法输出资源 busy/stall/critical-path 时增加 instrumentation；
不新增 opcode，不为特定负载增加专用 runtime 分支。

## 8. 测试与报告

### 8.1 单元测试

- OFF 只选 identity，AUTO 选最小代价，FORCE 只用于显式候选；
- tie 时 identity 优先；
- 候选数 `<=8`，搜索中 simulator 调用为 0；
- break-even 不成立时 split-K 被淘汰；
- cost table 与 hardware/simulation digest 不匹配时 fail closed；
- cycle-weighted topological order 稳定且不违反 dependency；
- local part 不产生 LOCAL_NOC，remote slice 精确覆盖；
- temporal chunk 的 event triple、slot lifetime 和 reduce order 闭合；
- estimator 对同一候选重复计算得到同一 digest/makespan。

### 8.2 集成与 E2E

最小矩阵：

| 负载 | inter | intra | 预期 |
|---|---|---|---|
| compute-bound | naive | off | baseline |
| compute-bound | naive | auto | 至少 5% 收益 |
| sync-bound | naive | auto | 选 identity，无回归 |
| compute-bound | swizzle_topo | off | inter-only baseline |
| compute-bound | swizzle_topo | auto | 输出组合边际收益 |

每格 finalizer 1 次，simulator 3 次。保存：

- run report 和 SUCCESS marker；
- identity/auto 的 input-equivalence 证据；
- 候选全表与选择原因；
- timing calibration table 与相对误差；
- critical path 和 per-resource busy/stall/overlap；
- makespan、speedup、gain 和组合 interaction。

### 8.3 回归约束

- 现有 naive、Swizzle、ProgramIO、LOCAL_NOC 和 finalizer 测试继续通过；
- OFF 模式与现有 identity 的 schedule/artifact digest 保持一致；
- AUTO 在所有非正向负载上不得比 identity 慢；
- validator、event、route、SRAM capacity/lifetime 不允许放宽；
- 编译和运行输入不可变，stable ID 和 canonical order 保持确定。

## 9. 快速开发约束

为了尽快得到可信收益，本轮明确不开发：

- SUMMA、Systolic、Ring/Tree collective 新模板；
- push/pull 自动选择；
- HBM spill 和 reload；
- adaptive/source routing；
- 退火、大规模 DFS/beam 或 simulator-in-the-loop product search；
- 完整 bank-conflict/动态 NoC congestion 模型。

候选库首版只保留 identity、cycle-order、split-K(2/4) 和流水 split-K。
任何新特性只有在这四类候选无法达到 5% 收益，且 report 已证明瓶颈
确实需要新模板时才增加。

## 10. 风险与回退

| 风险 | 处理 |
|---|---|
| 大 GEMM 仍无收益 | 先查 critical path/busy/overlap；不扩搜索空间 |
| LOCAL_NOC 固定开销过大 | 提高 break-even 门槛，小粒度候选直接淘汰 |
| 归约核成为串行瓶颈 | streaming reduce；仍不盈利则减少 parts |
| double buffer 只省 SRAM 不省周期 | 要求 `overlap_cycles>0`，否则不选该候选 |
| 解析模型排序错误 | 修参数表/时间线，不增加 product simulator 调用 |
| AUTO 选到慢候选 | baseline 永远在候选集，tie 优先 baseline，E2E 设 no-regression gate |
| 为跑分特化负载 | 预定网格、冻结输入 digest、公开全部候选结果 |

最终回退总是 identity，但回退必须是明确的选择结果，报告
`selected_candidate=identity` 和原因，不允许静默跳过 refine/schedule pass。

## 11. 最终交付物

1. OFF/AUTO/FORCE 版本化接口；
2. cycle-weighted schedule 和有界 AUTO selector；
3. 去冗余、可流水的 split-K carrier；
4. 与 simulator 同构的 timing model 及微基准标定表；
5. compute-bound/sync-bound 两个冻结负载；
6. identity/auto 公平对照 runner；
7. 满足 `gain >= 5%`、预测误差 `<=20%`、3 次重复稳定的正式报告；
8. inter/intra 2×2 组合消融和无收益自动回退证据。
