# Intra-die 优化端到端开发方案

> 依据：[die 内数据流编排完整方案](../refs/die内数据流优化方案.md)。
>
> 本方案以“尽快得到一个真实、可替换、可端到端运行的 intra-die 优化实现”为第一目标，
> 同时保留 inter-die 与 intra-die 两个独立策略开关，支持后续 2×2 消融。完整 CG 划分、
> 算子重写和跨核流水属于第二阶段，不阻塞首个可运行版本。

## 1. 结论与最短实现路径

当前最短且风险最低的路线不是一次实现参考方案中的全部 P0–P5，而是分两层交付：

1. 先实现 schedule-only O2：
   保持 IR-2 的 task、dependency、flow、bytes 和 tensor slice 不变，只优化合法的
   core placement、core order、SRAM lifetime reuse、bank 错开和通信任务时序。
   该层复用现有 GlobalAction、lowering、manifest、finalizer 和 C++ simulator，不新增
   C++ opcode，也不新增编译 pass。
2. 再实现 graph-refine O2：
   在投影和调度之间增加版本化的 intra-die refine/plan 边界，支持 CG、retile、
   split-k、显式同 die 跨核传输、双缓冲和更完整的模板搜索。

组合优化采用两个正交策略选择，不做首版联合搜索：

~~~yaml
policy:
  partition: gemm_coll
  inter_die: naive          # 或 swizzle_topo
  intra_die: naive          # 或 optimized
~~~

关闭某一层优化时选择该层的 naive policy。不要增加“跳过 pass”的 enabled 布尔值：
inter-die plan 和 intra-die schedule 都是后续阶段所需的必备产物，跳过会破坏固定 pass
顺序、receipt 链和 provenance。

最快交付顺序如下：

~~~text
分支 A：naive inter → common IR2 → optimized intra → 通用 lowering/runtime
分支 B：swizzle_topo → common IR2 → naive intra → 通用 lowering/runtime
                              │
                              └→ optimized intra → 2×2 组合闭环
~~~

分支 A 是第一个可交付结果。分支 B 可以并行开发；两者完成后，组合格不应再引入
Swizzle 专用 scheduler 或专用 runtime 旁路。

## 2. 范围与完成标准

### 2.1 第一阶段必须完成

- 提供真实的 OptimizedIntraDiePolicy，实现冻结的 IntraDiePolicy v1 接口；
- 通过 production registry 激活 intra_die/optimized，不能用 naive factory 或简单包装
  naive 输出冒充优化实现；
- compiler 根据策略名显式选择 exact contract，策略和 contract 交叉组合必须 fail closed；
- 至少一个正式 Dense TP2/GEMM_RS 负载从 YAML 经过 compile、GlobalAction、lowering、
  manifest、finalizer、ProgramIo 和 simulator 完整运行；
- 同一输入重复编译和运行时，schedule/artifact digest、marker signature 和 makespan 稳定；
- optimized schedule 最终调用与 naive 相同的 validate_against，不降低任何依赖、容量、
  lifetime、route 或 runtime token 校验；
- 产生 A00/A10/A01/A11 四格报告，且两种 policy selection、implementation version、
  configuration digest 和 pass receipt 闭合。

### 2.2 优化能力的第一阶段边界

第一阶段允许改变：

- executable dependency 连通分量到 core 的映射；
- 同一 core 上满足原依赖的任务顺序；
- SEND/RECV 所在分量相对 C2C port 的物理位置；
- SRAM region、offset、bank 和不重叠 lifetime 的地址复用；
- 由 placement 唯一导出的本地 XY path；
- 由上述 schedule 合法派生的 GlobalAction order edge 和下游程序流。

第一阶段禁止改变：

- IR-2 task 集、task kind、dependency、flow、logical bytes 和 tensor slice；
- inter-die plan 的算法语义；
- 模型数学语义和 reduction 语义；
- task split/merge、retile、split-k、CG 划分；
- 任意同 die 跨核 producer-consumer 边；
- adaptive/non-XY routing；
- 为得到更好结果而在 policy 搜索内循环调用 simulator。

### 2.3 两类完成声明

功能完成和性能结论分开：

- 功能门槛：四格均合法、可执行、可重复。使用当前 simulator 参数即可完成。
- 优化门槛：至少一个冻结的受支持负载上，optimized 产生非 naive 的合法 schedule，
  并在 makespan 或明确的目标分项上优于基线；无改进时允许选择 naive 候选，但报告必须
  标记 selected_candidate=baseline。
- 论文级性能门槛：完成参考方案 §7.4 的硬件核实，解析时间与 simulator 的误差达到
  约 20% 以内。在此之前只报告功能闭环和相对趋势，不宣称硬件真实性能收益。

## 3. 当前接口、流水线与实际断点

### 3.1 已有可替换接口

主接口位于：

- llm/frontend/wafer_frontend/policies/interfaces.py
- llm/frontend/wafer_frontend/schema/policy.py

冻结签名为：

~~~python
class IntraDiePolicy(Protocol):
    def schedule(
        self,
        projection: IR2ProjectionResult,
        ir1: IR1,
    ) -> IntraDieScheduleSet: ...
~~~

当前固定主链位于 llm/frontend/wafer_frontend/compiler.py：

~~~text
build_ir0
  → logical_expand
  → placement
  → fusion_partition
  → inter_die_plan
  → project_to_ir2
  → intra_die_schedule
  → global_action_dag
  → lowering
  → manifest_link
~~~

因此 schedule-only O2 只需替换 intra_die_schedule 使用的 policy，不需要修改 PassManager
阶段数，也不需要增加 C++ 执行入口。

### 3.2 已有配置和 registry 状态

llm/frontend/wafer_frontend/schema/experiment.py 已定义：

- inter_die: naive | swizzle_topo；
- intra_die: naive | optimized；
- 两个字段彼此独立。

llm/frontend/wafer_frontend/policies/registry.py 已声明 intra_die/optimized，但截至本方案
编写时仍未绑定真实 factory。当前工作树已开始激活 inter_die/swizzle_topo，但这不等于
通用 2×2 链路已经完成。

### 3.3 必须先修的 contract 断点

compiler 当前实例化了两个独立 policy，却在创建 InterDiePlanningContext 和
IntraDieSchedulingContext 时依赖默认 contract。需要增加显式映射：

| Policy selection | Exact contract |
|---|---|
| inter_die/naive | DIRECT_NAIVE_V1 |
| inter_die/swizzle_topo | SWIZZLE_TOPO_V1 |
| intra_die/naive | NAIVE_COMPONENT_RR_XY_SEQUENTIAL_STATE_TRANSFER_V5 |
| intra_die/optimized | 新增 OPTIMIZED_CP_BANK_REUSE_XY_V1 |

映射必须由 compiler 显式传入 context；不得通过“默认 naive contract”碰运气，也不得在
validator 中只检查 kind、不检查 name 与 contract 的精确配对。

### 3.4 当前 v1 调度器的硬限制

llm/frontend/wafer_frontend/schema/ir2.py 的现有 validator 决定了第一阶段的上限：

- placements 和 core_orders 必须精确覆盖每个 executable task；
- 每条 executable dependency 的两端必须位于同一 core，因此放置单位只能是 executable
  dependency 的无向连通分量；
- SemanticFlow 不允许 source_die 与 destination_die 相同，当前没有通用的
  LOCAL_SEND/LOCAL_RECV；
- local_noc_path 必须等于 backend-v1 的 X-then-Y 路径；
- 不同 storage 可以复用相同物理 offset，但 lifetime 不能重叠；
- 每个 projection DAG 必须恰有一个 IntraDieSchedule。

所以第一阶段不能通过删除“dependency 同核”校验来伪造跨核流水。这样做会让两个
core-local SRAM 之间没有显式 transport，属于语义错误。

### 3.5 当前 inter + intra 组合的额外断点

截至本方案编写时，Swizzle 的专用投影产物是 SwizzleIr2Projection，并明确标记
current_ir2_compatible=False；通用 NaiveProjectToIR2 仍只接受 exact FusionPlan。
因此：

- registry 中 swizzle_topo 为 ACTIVE，不代表 A10/A11 可走通用编译链；
- Swizzle 专用 timing runner 不经过通用 IntraDiePolicy，不能计作组合闭环；
- 必须先把至少一个 Swizzle pattern 投影为通用 IR2ProjectionResult，并用
  NaiveIntraDiePolicy 跑通，之后才能接 optimized intra。

首个桥接范围固定为 Dense TP2 的 GEMM_RS。AG_GEMM、GEMM_AR 和更多 Swizzle pattern
在同一 carrier 契约稳定后逐个增加，不阻塞首个 2×2 小负载。

## 4. 目标架构

### 4.1 第一阶段：不增加 pass

~~~text
ExperimentSpec
  │
  ├─ inter_die = naive | swizzle_topo
  │       ↓
  │   FusedPlan
  │       ↓  pattern-aware ProjectToIR2
  │   IR2ProjectionResult
  │
  └─ intra_die = naive | optimized
          ↓
      IntraDieScheduleSet
          ↓
      GlobalActionDAG → lowering → manifest → simulator
~~~

OptimizedIntraDiePolicy 只依赖 IR2ProjectionResult 与 IR1，不允许 import Swizzle
内部实现，也不允许按 inter policy 名称分支。这样 A01 和 A11 使用同一套 O2 决策引擎。

### 4.2 第二阶段：完整图细化

当 schedule-only 的收益被同核限制卡住后，再引入：

~~~text
IR2ProjectionResult
  → intra_die_refine
  → RefinedIntraDieDAG
  → intra_die_schedule
  → IntraDieScheduleSet
  → GlobalActionDAG → lowering → simulator
~~~

建议把职责拆开：

- IntraDieRefinePolicy：CG、模板、retile、split、显式 local flow 和双缓冲；
- IntraDiePolicy：placement、order、buffer、bank 和固定路由映射。

新增 pass 时 naive 分支使用 identity refine，保证所有策略仍产生相同阶段数量和 receipt
形状。该阶段需要同步升级 PassManager phase、IR2 exact coverage、GlobalAction、
lowering、manifest 与必要的 C++ transport 支持，不进入第一阶段关键路径。

## 5. 策略配置与消融设计

### 5.1 第一层 2×2 消融

| Case | inter_die | intra_die | 目的 |
|---|---|---|---|
| A00 | naive | naive | 基线 |
| A10 | swizzle_topo | naive | 隔离 inter-die 收益，也是组合前置门禁 |
| A01 | naive | optimized | 隔离 intra-die 收益，也是首个 O2 交付 |
| A11 | swizzle_topo | optimized | 两层组合与交互效应 |

固定以下所有输入：

- workload、model、profile 和 shape；
- placement、fabric、hardware config 和 mapping；
- backend、simulation config、工具二进制；
- optimizer 的候选预算和 cost-table 版本；
- 随机种子；首版算法应完全确定性并尽量不使用随机数。

只允许两个 policy 字段及其合法下游产物变化。

### 5.2 成对不变量

- A00 与 A01 的 inter-die plan 和 project_to_ir2 digest 必须相同；
- A10 与 A11 的 inter-die plan 和 project_to_ir2 digest 必须相同；
- 在固定 inter policy 的一对 case 中，task/flow/bytes/slice 必须相同，只允许 schedule
  及其合法派生的 GlobalAction order、lowering 和 runtime timeline 变化；
- 四格的 PolicySelection 必须与 context、pass receipt 和 report 完全一致；
- 不要求 A00 与 A10 的 projection 相同，因为 inter-die 算法本来就允许改变合法 plan。

### 5.3 报告公式

设四格 makespan 为 T00、T10、T01、T11，报告：

- inter-only speedup = T00 / T10；
- intra-only speedup = T00 / T01；
- combined speedup = T00 / T11；
- interaction = T11 - T10 - T01 + T00。

makespan 是唯一优化目标。SRAM high-water、bank pressure、NoC byte-hop、DTE/core
利用率和 stall 只作为约束、同 makespan tie-break 或诊断项，不能混成任意加权总分。

### 5.4 内部子功能消融

首版为缩短开发时间，不扩展 ExperimentSpec 的 optimizer 子参数。使用一个冻结的
OptimizedIntraDieConfig，由 production registry 的零参 factory 捕获，并将完整配置写入
configuration_digest。

需要研究内部子功能时，ablation runner 用自定义 registry 构造以下 feature mask：

- critical_path_order；
- locality_placement；
- lifetime_reuse；
- bank_stagger；
- communication_shadow。

每个 mask 必须进入 configuration_digest 和报告。只有确实需要普通用户从 YAML 控制这些
细粒度参数时，才增加版本化的 typed ExperimentSpec 字段；不要在首个 E2E 前扩大输入 schema。

## 6. Schedule-only O2 算法

### 6.1 设计原则

- 永远把 NaiveIntraDiePolicy 产生的合法 schedule 纳入候选；
- 不支持的 task/fabric 组合明确 fail closed；
- 对受支持输入搜索不到更优候选时，可以选择 baseline，但不能伪报性能收益；
- 先生成有限个确定性候选，再用解析模型选择；policy 内不启动 simulator；
- 所有候选在比较前调用完整 validate_against；
- 主排序只看 predicted makespan；其他指标只做硬约束或同分 tie-break。

### 6.2 决策步骤

1. 验证 projection 和 IR1，按 die 读取 IntraDieDAG。
2. 使用 canonical Kahn 得到稳定拓扑序。
3. 把 executable dependency 当作无向边，求合法放置连通分量。
4. 计算每个 task 的静态 duration proxy 和 bottom-level criticality。
5. 按 criticality 处理分量，在兼容 core 集中选择预计完工最早的 core。
6. 对包含 SEND/RECV 的分量，在 predicted makespan 相同的情况下优先选择距离对应 C2C
   port 更近的 core。
7. 对每个 core 做 ready-list 排序：关键路径优先；可异步 DMA/RECV 在依赖允许时提前，
   WAIT/BARRIER 和输出 SEND/STORE 在不引入 stall 时后移。
8. core order 确定后再推导每个 buffer 的精确半开 lifetime。
9. 按 core/region 使用 aligned free-list 分配；回收已结束区间，允许不重叠 lifetime
   复用 offset；同分时选择能错开热点 bank 的 offset。
10. 根据最终 core/port 端点生成固定 X-then-Y local_noc_path 和 runtime binding。
11. 物化 IntraDieScheduleSet，运行 validate_against。
12. 在合法候选中按以下字典序选择：
    predicted makespan、peak SRAM、byte-hop、canonical candidate ID。

### 6.3 第一版解析代价

当前 IR1 CoreSpec 没有完整的 EXU/SFU/VEC/SA 吞吐与 capability 数据，所以首版采用
版本化静态 cost config：

- GEMM：由 rank/tile 的 m、n、k 计算 2mnk，再除以配置中的有效吞吐；
- 其他 COMP：按 workload shape 与配置查表；
- SEND/RECV/DMA：setup + latency + bytes / effective bandwidth；
- WAIT/BARRIER：固定 issue/sync 开销；
- NoC：固定 XY hop 延迟与 bytes/bandwidth；
- 同一 core 的任务按 core order 串行，独立 core 并行；
- 依赖、event 和资源可用时间共同决定预计开始时间。

配置必须包含 schema version，且进入 policy configuration digest。完成硬件标定前，
该模型只用于候选排序和相对趋势。

### 6.4 有界候选而非大搜索

第一版只生成少量候选：

- naive baseline；
- critical-path placement/order；
- critical-path + port locality；
- 上述 schedule 分别配 sequential allocator 与 lifetime-reuse allocator；
- 候选总数固定在 8–16 以内。

这比立即实现 DFS、退火和 simulator top-k 更快。离线标定工具可以把解析 top-k 送入
simulator 精评，但产品编译流程只消费冻结后的单趟规则；严禁把 simulator 放进每个
task/core 尝试的搜索内环。

### 6.5 共享 materializer

naive_intra_die.py 当前同时包含决策和完整 schedule 物化。推荐机械抽出一个中立 builder：

~~~python
ScheduleDecisions(
    component_to_core=...,
    core_task_order=...,
    allocation_mode=...,
)

materialize_schedule(dag, ir1, decisions) -> IntraDieSchedule
~~~

builder 统一负责 buffer view、binding、flow route、runtime token、stable ID 和最终验证；
naive 与 optimized 只产生 decisions。

为降低重构风险，先增加“naive 重构前后 canonical digest 完全相同”的 golden，再让 O2
复用 builder。若该机械抽取无法在一个微阶段完成，可以暂时让 optimized 复用
naive_intra_die.py 中的中立 helper，但禁止复制整份约 1300 行 materializer。

## 7. 分阶段开发与最短门禁

所有微阶段控制在 30–120 分钟，并保持可独立提交、可独立回滚。两个工作流可并行：

~~~text
P0 → I1 → I2 → I3 ─┐
                    ├→ C1 → C2
P0 → W1 → W2 → W3 ─┘

I：intra schedule-only
W：Swizzle common-IR2 bridge
C：组合与消融
~~~

### P0：冻结小负载与真实现状

产物：

- 选择 notes/frontend/examples/naive_dense_tp2.yaml 对应的 Dense TP2/GEMM_RS 为首个 case；
- 保存 A00 的 spec、fabric、projection、schedule、linked artifact 和 runtime report digest；
- 收口当前“registry 已认为 Swizzle active，但 compiler/test 仍认为 unavailable”的阶段不一致；
- 增加一条失败测试，证明当前 intra_die=optimized 会在 pass 前明确失败且不修改输入。

退出门禁：

- A00 通用 runner 双跑稳定；
- baseline 工作区测试状态明确；
- 后续任何阶段不得靠更新 A00 golden 掩盖 naive 行为漂移。

### I1：共享 builder 与 naive 等价重构

改动：

- 新增 policies/intra_die_schedule_builder.py；
- 从 naive_intra_die.py 迁移中立拓扑、合法 core、buffer、route 和 runtime 物化逻辑；
- NaiveIntraDiePolicy 继续产生 component round-robin、sequential allocation decisions。

退出门禁：

- 现有 test_naive_intra_die.py 全绿；
- 同一 fixture 的 schedule JSON 和 stable digest 完全不变；
- 输入 projection、IR1 digest 不变。

### I2：实现 OptimizedIntraDiePolicy

改动：

- 新增 policies/optimized_intra_die.py；
- 增加冻结 OptimizedIntraDieConfig 和 implementation schema version；
- 实现 component placement、critical-path ready-list、port locality、free-list reuse 和
  bank stagger；
- baseline 永远作为候选，所有候选逐个 validate。

退出门禁：

- deterministic、non-mutation、unsupported fail-closed 测试通过；
- dependency 同核、core order、容量、对齐、bank、lifetime、XY path 的正反例通过；
- 至少一个构造 fixture 的 optimized schedule 与 naive 不同；
- 不调用 simulator、不修改 projection。

### I3：contract、registry 与 compiler 接入

改动：

- schema/n5.py 增加 optimized exact contract 并升级 context schema；
- policies/registry.py 激活真实 optimized 零参 factory；
- compiler.py 显式选择 intra contract；
- passes/intra_die_schedule.py 在未注入 policy 时按 context contract 选择默认实现，
  或明确要求调用者注入；禁止 optimized context 静默落到 naive；
- policies/__init__.py 导出正式实现。

退出门禁：

- registry instantiate 返回 OptimizedIntraDiePolicy；
- selection 中 implementation ID、schema、capabilities、configuration digest 正确；
- naive/optimized policy-contract 交叉组合全部拒绝；
- pass receipt 记录真实 intra selection。

### I4：A01 通用端到端

改动：

- 从同一 base spec 只把 intra_die 改为 optimized；
- 复用现有 runner 的 compile、finalizer、ProgramIo 和 simulator 闭包；
- 增加不硬编码 optimized makespan 的 intra-die comparison runner。

退出门禁：

- A00、A01 均 compile → link → finalizer → simulator ×2；
- return code、DONE/ACK、drain、credit、marker signature 均闭合；
- A00/A01 的 project_to_ir2 digest 相同；
- runtime report 中 policy/context/receipt 一致；
- makespan > 0 且重复运行稳定。

完成 I4 即可声明“schedule-only intra-die optimized E2E complete”，无需等待完整 2×2。

### W1：修正 inter contract 与通用投影类型边界

改动：

- compiler.py 显式选择 naive/swizzle_topo 的 FusedInterDieContract；
- ProjectToIR2 接口输入从 legacy FusionPlan 扩展为版本化 FusedPlan；
- projector 对 plan 类型做显式 dispatch，未知类型 fail closed；
- 先保留 legacy FusionPlan 输出完全不变。

退出门禁：

- inter policy-contract 交叉组合拒绝；
- naive projection digest 不变；
- 选择 swizzle_topo 不再因默认 DIRECT_NAIVE_V1 失败，而是在尚未支持的 projection
  pattern 处给出精确错误。

### W2：GEMM_RS 的 common IR2 bridge

改动：

- 将 SwizzleFusionPlan 的 GEMM_RS rank action 投影为通用 COMP/SEND/RECV/WAIT/REDUCE/
  BARRIER；
- 从 IR1 与 typed plan 闭合 ComputeContract、ReductionContract、chunk slice、buffer
  view、dependency、flow、route 和 output ownership；
- 更新 IR2 对 FusedPlan 的 exact coverage/provenance validator；
- 不强转 SwizzleFusionPlan 为 FusionPlan，不从 shape/opcode 猜 pattern。

退出门禁：

- 生成的 IR2ProjectionResult 通过完整 validate；
- logical work、chunk union、flow bytes、rank ownership 与 Swizzle plan 精确闭合；
- 缺 ComputeContract、slice、dependency 或 ownership witness 时 fail closed。

### W3：A10 通用端到端

退出门禁：

- swizzle_topo + naive intra 经过同一 project_to_ir2、intra_die_schedule、
  GlobalAction、lowering、manifest、finalizer 和 simulator；
- NaiveIntraDiePolicy 能合法调度 Swizzle 投影；
- 不使用 Swizzle 专用 timing projection/runtime 冒充通用闭环；
- 双跑 digest、marker 和 makespan 稳定。

### C1：A11 组合闭环

工作：

- 让 OptimizedIntraDiePolicy 无条件消费 W2 产生的 common IR2；
- 不增加按 inter policy 名称特判；
- 运行 A10/A11 投影等价性和 schedule 差异测试。

退出门禁：

- A11 经通用链端到端运行；
- A10/A11 的 project_to_ir2 digest 相同；
- policy/context/receipt 同时记录 swizzle_topo 与 optimized；
- 组合 schedule 通过与其余三格相同的 validator/finalizer/runtime。

### C2：四格消融报告

产物：

- 一个 base spec；
- 一个 machine-readable 2×2 matrix；
- 四份独立 compile/runtime report；
- 一个 comparison JSON/Markdown，包含输入 digest、不变量 diff、T00/T10/T01/T11、
  speedup 和 interaction；
- 每格重复两次的 artifact/runtime 稳定性证据。

退出门禁：

- 四格使用同一 workload/fabric/profile/backend/tool epoch；
- machine diff 只出现允许变化；
- 性能好坏不作为 schema correctness 门禁；
- 未完成硬件标定时报告显式标记 calibrated=false。

## 8. 文件级改动清单

### 8.1 Schedule-only O2

| 文件 | 改动 |
|---|---|
| llm/frontend/wafer_frontend/policies/optimized_intra_die.py | 新策略、固定配置、解析 cost、候选和决策 |
| llm/frontend/wafer_frontend/policies/intra_die_schedule_builder.py | naive/optimized 共用合法 schedule 物化 |
| llm/frontend/wafer_frontend/policies/naive_intra_die.py | 改为产生 naive decisions；行为与 digest 不变 |
| llm/frontend/wafer_frontend/policies/registry.py | 激活 intra_die/optimized，记录真实 metadata/config digest |
| llm/frontend/wafer_frontend/policies/__init__.py | 导出 optimized API |
| llm/frontend/wafer_frontend/schema/n5.py | 新 optimized contract、exact pair、schema version bump |
| llm/frontend/wafer_frontend/compiler.py | 显式 inter/intra policy→contract 映射 |
| llm/frontend/wafer_frontend/passes/intra_die_schedule.py | 默认实现与 context 一致，保留注入入口 |

### 8.2 Swizzle 组合桥接

| 文件 | 改动 |
|---|---|
| llm/frontend/wafer_frontend/policies/interfaces.py | ProjectToIR2 接受版本化 FusedPlan |
| llm/frontend/wafer_frontend/policies/naive_project_to_ir2.py | legacy/swizzle 显式 dispatch；legacy 行为不变 |
| llm/frontend/wafer_frontend/passes/project_to_ir2.py | carrier 与 producer 支持 common projection |
| llm/frontend/wafer_frontend/schema/ir2.py | FusedPlan exact coverage/provenance；必要时升级 schema |
| llm/frontend/wafer_frontend/schema/swizzle_plan.py | 补足 common IR2 所需 typed witness，禁止猜测 |
| llm/frontend/wafer_frontend/passes/project_swizzle_plan.py | 复用/适配到 common IR2，而不是专用旁路 |

### 8.3 测试与 runner

| 文件 | 改动 |
|---|---|
| llm/test/frontend/unit/test_optimized_intra_die.py | 算法、确定性、non-mutation、负例 |
| llm/test/frontend/unit/test_naive_intra_die.py | naive digest 回归 |
| llm/test/frontend/unit/test_registry.py | optimized ACTIVE 与 metadata |
| llm/test/frontend/unit/test_n5_schema.py | policy-contract exact pair |
| llm/test/frontend/unit/test_n5_producers.py | policy 注入和 context 一致性 |
| llm/test/frontend/unit/test_compiler_policy_wiring.py | 四组合、context、receipt、无 fallback |
| llm/test/frontend/integration/run_intra_die_ablation.py | A00/A01 与最终 2×2 runner |
| llm/test/frontend/integration/test_intra_die_ablation.py | 快速 fake-runtime/结构门禁 |
| notes/frontend/examples/intra_die_ablation_base.yaml | 唯一 base workload，variant 只改 policy |

现有 run_naive/compile_naive 名称虽然不再准确，但只要其 validator 未硬编码 policy 名，
第一阶段不重命名，避免扩大改动面。完成四格后可增加 compile_experiment/run_experiment
别名，再单独迁移旧调用。

## 9. 测试策略

### 9.1 单元与 schema

- interface 签名保持 v1，不在 schedule-only 阶段偷偷扩参；
- OptimizedIntraDieConfig 序列化与 digest 稳定；
- 相同输入两次 schedule 完全相同；
- schedule 不修改 projection 或 IR1；
- policy/name/contract/implementation 交叉篡改被拒绝；
- executable dependency 跨核被拒绝；
- task 漏放、重复 order、非法 core、非法 region/initiator 被拒绝；
- buffer 超容量、错 alignment、错 bank、lifetime 物理重叠被拒绝；
- 非 XY path、错 port leg、错 runtime token/event 被拒绝；
- unsupported task/fabric 明确报错，不 fallback 到别的 registry policy。

### 9.2 编译集成

- A00/A01 的 pre-schedule digest 相同；
- A10/A11 的 pre-schedule digest 相同；
- 四格均生成 LinkedProgramBundle；
- receipts 的固定 pass 顺序不变；
- inter_die_plan receipt 和 intra_die_schedule receipt 各自只携带对应 selection；
- linked manifest 的 source digest 能追溯到选中的 schedule。

### 9.3 真实端到端

PR 快速门禁：

- TP2/GEMM_RS 四格串行运行；
- 每格 finalizer 一次、simulator 两次；
- makespan 正数、无 deadlock、DONE/ACK 闭合、drain=0、credit balanced；
- 四格保存报告但只对 A00 继续检查历史 golden，optimized 分支不硬编码固定性能值。

Nightly：

- TP4 和至少一个较长 profile；
- 全 2×2、重复运行、统一 comparison；
- 检查 SRAM max end、D2D packets/bytes 和可用的 NoC/DTE 指标；
- 扩展 AG_GEMM/GEMM_AR 仅在各自 common IR2 bridge 完成后加入。

### 9.4 性能证据

最低报告：

- makespan_cycles；
- per-core SRAM max end/high-water；
- task/action/opcode 数；
- D2D bytes 与 per-link packets；
- policy selection、configuration digest、projection/schedule/artifact digest。

若当前 simulator 未输出下列指标，不得从静态字段推断后声称“实测改善”：

- SRAM bank conflict；
- core/DTE busy 与 stall cycles；
- NoC byte-hop、链路利用率和拥塞；
- communication overlap/hidden latency。

这些指标作为后续 instrumentation 微阶段加入 simulator/report。makespan 始终是唯一主目标。

## 10. 提交顺序与开发纪律

建议按下列小提交推进：

1. baseline 与失败测试；
2. naive builder 机械抽取，digest 零变化；
3. optimized policy 单元实现，尚不 activate；
4. N5 contract 与 compiler mapping；
5. registry activate，A01 E2E；
6. inter contract 修复；
7. GEMM_RS common IR2 bridge，A10 E2E；
8. A11 与 2×2 runner；
9. 指标、标定和更多 pattern。

只有对应 E2E 通过后才把 registry 条目从 DECLARED 改为 ACTIVE。开发期可用测试 registry
注入尚未发布的实现，但 production registry 不能指向 pass-through stub。

每个阶段都必须保留：

- stable ID 与 canonical order；
- exact provenance；
- fail-closed；
- 输入不可变；
- naive regression；
- 同一错误不能靠放宽 validator 或更新 golden 消失。

## 11. 风险与处理

| 风险 | 处理 |
|---|---|
| v1 同核约束导致优化空间小 | 先交付 placement/order/reuse 的真实闭环；收益上限用 graph-refine v2 解决 |
| Swizzle 只有专用投影/runtime | GEMM_RS common IR2 bridge 是 A10/A11 的硬前置，不计旁路结果 |
| naive materializer 重构导致 ID 漂移 | 先做 canonical digest 等价测试，机械改动与算法改动分提交 |
| cost model 缺真实 core 吞吐 | 使用版本化静态 config；标定前不作真实性能宣称 |
| optimized 被激活但实际只返回 naive | baseline 仅作候选；要求至少一个 fixture 产生不同 schedule，并报告 selected candidate |
| 搜索时间失控 | 候选固定 8–16；同类型/同组件收缩；simulator 不进内环 |
| 组合实现按 policy 名称特判 | O2 只消费 common IR2；Swizzle 差异在投影边界闭合 |
| 细粒度 feature 开关污染 Experiment schema | 首版放 registry config；需要用户输入时再版本化扩 schema |
| schedule 指标好但 runtime 退化 | naive baseline candidate + 解析选择；最终以 simulator makespan 报告，不用代理指标替代 |

## 12. 第二阶段：完整 intra-die 数据流优化

第一阶段稳定后，按参考方案逐步补全：

1. P0 标定：核实远程读写、集合原语、XY/源路由、算力效率、带宽与 DTE 参数；
2. P1 模板：identity、只切 k fallback、P2P，再增加 Ring/SUMMA/Systolic；
3. P2 外层：GEMM anchor、同类型同 CG、BFS 定序、受限 DFS/beam、R1/R2/R3；
4. P3 内层：核数平衡、份数窄区间穷举、hoist/sink、合法 split；
5. P4 mapper：形状可嵌入、流量优先贴合、交换/滑移；仍默认 XY；
6. P5 精评：解析 top-k 后才调用 simulator，补链路负载与 water-filling。

首个 v2 模板集只实现：

- identity；
- 任意核数可行的 split-k + reduce fallback；
- P2P；
- 显式 LOCAL_SEND/LOCAL_RECV/WAIT；
- 双缓冲所需的版本化 buffer/event contract。

下列高级项继续延期，直到硬件能力被核实：

- push/pull 自动选择；
- 3×3 集合原语映射与环/树选择；
- SUMMA、脉动和嵌套模板；
- phase 化；
- HBM spill；
- source/adaptive routing；
- 退火与动态拥塞反馈。

## 13. 最终 Done 定义

### Schedule-only O2 Done

- intra_die/optimized 是 production registry 中的真实 ACTIVE factory；
- policy/context/contract/config/receipt/report provenance 全闭合；
- A00/A01 使用同一 projection，均走通通用 finalizer 和 simulator；
- optimized 确实产生过非 naive schedule，所有 validator 继续生效；
- 双跑稳定，无输入 mutation、无隐式 fallback、无 C++ 专用旁路。

### 组合优化 Done

- swizzle_topo 的目标 pattern 可生成通用 IR2ProjectionResult；
- A10 先在 naive intra 下端到端通过；
- A11 使用同一 OptimizedIntraDiePolicy，无 inter 特判；
- A00/A10/A01/A11 四格均有可复现报告和 machine diff；
- 能独立报告 inter-only、intra-only、combined 与 interaction；
- 四格只通过两个 policy 选择控制，未跳过任何必备 pass。

### 完整数据流优化 Done

- graph-refine v2 能显式表达 CG、split、local transport 和 double buffer；
- same-die 跨核依赖都有显式数据搬运与 event，不靠放宽 validator；
- 新 carrier 经 GlobalAction、lowering、manifest、finalizer 和 simulator 闭合；
- 搜索规模、解析模型误差和 simulator 调用预算满足参考方案的约束。
