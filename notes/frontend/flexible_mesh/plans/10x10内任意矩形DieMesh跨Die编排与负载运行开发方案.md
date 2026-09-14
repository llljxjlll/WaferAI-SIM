# 10×10 内任意矩形 Die Mesh 跨 Die 编排与负载运行开发方案

## 1. 目标与完成定义

本方案的目标是在尽量复用当前前端、Swizzle、Program Artifact 和 npusim
实现的前提下，支持以下能力：

- 物理 Mesh 为完整、连续矩形；
- Mesh 高、宽分别满足 1≤H≤10、1≤W≤10；
- 总 Die 数 R=H×W≤100；
- 默认每 Die 放置一个跨 Die rank；
- 任意合法尺寸均可完成硬件加载、placement、PairRoute、策略生成、
  projection、lowering、link、ProgramIO 和 timing simulator 运行；
- 在所有尺寸上始终保留一个可执行的安全 baseline，优化策略失败或无收益时
  可以确定性回退；
- Dense 优先，真正二维 MeshSlice 和 MoE 在 baseline 稳定后分阶段开放。

这里的“任意大小 Mesh”专指 10×10 envelope 内的完整矩形，不包括缺 Die、
洞状拓扑、故障绕行或 torus。

这里的“负载运行”首版固定为：

    timing_execution = true
    functional_execution = false

首版必须完成 finalizer、ProgramIO、resolver 和 npusim 的真实执行，但不声明数值
functional correctness。数值执行需另立里程碑。

### 1.1 分层完成状态

为避免把“可以建图”误报成“负载已支持”，完成状态分为：

    mesh_foundation_complete
      = 100 种尺寸均可加载、放置、建路由，且拓扑生成有界

    mesh_runtime_complete
      = 100 种尺寸均有 tiny timing canary 真实运行并通过 ProgramIO

    dense_workload_complete
      = 代表尺寸上的完整 Dense workload 进入同一 artifact 并真实运行

    wang_scale_complete
      = Wang 在大 R 下有界生成、可运行，并保持 executable fallback

    meshslice_rect_complete
      = 真二维 sharding 的矩形 MeshSlice production 链闭合

    flexible_mesh_dense_complete
      = mesh_runtime_complete
        && dense_workload_complete
        && wang_scale_complete
        && meshslice_rect_complete

MoE、functional execution 和任意非矩形拓扑不计入
flexible_mesh_dense_complete。

### 1.2 发布口径

最终发布声明必须区分：

- 所有 100 种矩形均“可加载、可编排、可 lower/link、可运行”；
- 只有经过重复测量的代表尺寸可以声明性能收益；
- 未经校准的尺寸只声明正确完成和确定性，不声明性能最优；
- timing-only 不得表述为数值 functional correctness。

## 2. 当前基础与只读验证结论

### 2.1 已经具备、应直接复用的能力

当前底层并非只支持 2×2：

1. llm/frontend/wafer_frontend/passes/load_fabric.py 已按独立的 Die X/Y
   参数构造完整矩形。
2. llm/frontend/wafer_frontend/schema/ir1.py 已校验完整矩形、row-major Die ID
   和 X-first XY PairRoute。
3. llm/frontend/wafer_frontend/passes/group_registry.py 的 expected-group 路径
   已支持任意 rank 数、rank-to-die override 和 ordered-pair routes。
4. C++ 数据面已经按 DIE_X×DIE_Y 构造 D2D links，并支持中间 Die ingress
   重新选择下一跳出口。
5. standard Swizzle projection、Core/Address ABI、Operand ABI、lowering、
   linker、ProgramIO、finalizer 和 simulator runner 大部分按 rank 数遍历，
   不要求精确四 rank。
6. C++ collective planner 已包含 child 展开、wave 切分、逐核容量 demand、
   image preflight 和运行期原子 wave admission。

使用当前测试构造器对 H,W∈[1,10] 的全部 100 种矩形做了硬件加载验证，均可成功
构造。10×10 的关键规模为：

| 项目 | 数值 |
|---|---:|
| Die/rank 数 R | 100 |
| 有向 D2D link | 360 |
| ordered PairRoute | 9,900 |
| 最远 Manhattan hop | 18 |

10×10 的完整 PairRoute 构建当前约为秒级，因此首版继续复用显式
R(R-1) routes，不提前引入 lazy route representation。

### 2.2 当前端到端阻断点

当前不能声明任意 Mesh 已支持，主要阻断不是 fabric，而是：

1. Hamiltonian cycle 在
   llm/frontend/wafer_frontend/policies/swizzle/topology.py 和
   policies/swizzle/wang_1d.py 中使用递归回溯 DFS；10×10 已出现超过
   10 秒仍不能完成的情况，奇×奇无环矩形风险更高。
2. production max_chunk_count=64，R=100 时 Wang 的最小 chunk=R
   都不能生成。
3. production max_actions=4096；100 rank 的 Wang、MeshSlice 和 executable
   naive 均可能产生数万 actions。
4. executable unfused comparison 对多 rank 仍有精确四 rank/XOR wave
   假设。
5. MeshSlice candidate generator 已支持矩形，但 production placement、
   standard lowering 和 audit 仍锁定 DP2×TP2/4 ranks。
6. main common-IR2 bridge 仍主要支持 1D、GEMM_RS、direct route，不适合直接
   承担任意矩形的第一阶段扩展。
7. C++ program loader 和 sync runtime 重复拒绝跨 Die core group，导致
   GROUP_SYNC 和复用同一 group table 的 collective 不能真正跨 Die。
8. 生产 P2P runtime 每核每 wave 实际只有 3 个 endpoint session；裸
   standalone P2P 没有 collective planner 的安全分波机制。
9. 100-rank collective 会遇到 planner derived-memory、artifact size 和
   record count 上限，不能只提高 Python action budget。
10. 现有 MoE schema、runtime marker 和 evidence 多处精确绑定 EP4/2×2。

## 3. 能力边界与设计决策

### 3.1 首版实现范围

- 完整且连续的 H×W 物理矩形；
- 1≤H,W≤10，R≤100；
- 物理 fabric 与请求 Mesh 尺寸相同；
- 每 Die 一个跨 Die rank；
- rank=row×W+column；
- X-first XY route；
- Dense inference timing workload；
- 逻辑一维 TP=R 映射到物理二维 Mesh；
- AllGather、ReduceScatter，随后由 RS+AG 组成 AllReduce；
- executable naive/unicast baseline；
- Wang 1D line/snake 优化；
- 真二维 sharding 合法时的 MeshSlice AG_GEMM；
- 统一 artifact、ProgramIO 和 npusim 运行；
- fail-closed 的 action、buffer、session、tag、record、文件容量预检。

### 3.2 后续扩展

- 在更大的物理 fabric 中选择确定性的连续 H×W 子矩形；
- MeshSlice RS/AR；
- Dense training forward/backward；
- MoE 任意 EP；
- 多 rank/Die；
- collective plan/template 去重与循环复用；
- 真实跨 Die barrier 控制流和延迟模型；
- functional execution。

### 3.3 明确不在首版范围

- 缺 Die、洞状、故障 Mesh；
- adaptive routing、torus；
- ragged shard、隐式 padding；
- 跨 Die multicast/DCA；
- 跨 Die HOST endpoint；
- 无限循环 workload 的 transport-tag 自动回收；
- 所有 100 种尺寸的性能收益保证；
- 把 MoE 中所有常量 4 机械替换为 R。

### 3.4 核心设计决策

1. 10×10 envelope 在 flexible-mesh feature/schema 层校验，不收窄底层
   PhysicalFabric 的通用能力。
2. 先支持“逻辑 1D、物理 2D”，再支持真正的逻辑 2D。
3. 每个合法 Mesh 必须有 executable baseline；Wang、MeshSlice 都不是唯一可运行路径。
4. collective 必须进入现有 group collective planner，不展开为同时连接所有 peer
   的裸 standalone P2P。
5. runtime 容量不足时在编译期 fail closed 或静态分 wave，不允许 runtime
   原地等待容量，因为 peer 尚未 post receive 时可能形成全局死锁。
6. 旧 S0-S3、C0-C4 和既有 stable artifact ID 不修改；新增并列的
   flexible-mesh conformance suite。

## 4. 复用后的目标架构

    Experiment / RectMeshSpec
      → existing LogicalExpand / sharding
      → existing load_fabric
      → rectangular placement
      → existing PhysicalGroup + complete XY PairRoute
      → shared RectMeshTopology
          ├─ executable cyclic-naive baseline
          ├─ Wang 1D snake/ring candidate
          └─ MeshSlice 2D candidate
      → existing Swizzle/naive projection
      → existing Core/Address/Operand ABI
      → existing standard lowering/linker
      → existing ProgramIO/finalizer
      → existing collective planner + wave admission
      → npusim timing execution

### 4.1 复用清单

| 能力 | 复用位置 |
|---|---|
| 矩形 fabric | passes/load_fabric.py |
| PhysicalFabric 与 XY 路由验证 | schema/ir1.py |
| rank placement 与 PairRoute | passes/group_registry.py |
| Swizzle plan 投影 | passes/project_swizzle_plan.py、passes/project_swizzle_ir2.py |
| timing IR2 lowering | lowering/swizzle.py |
| Core/Address ABI | lowering/swizzle_abi.py |
| Operand ABI | schema/swizzle_operand_abi.py |
| standard program link | lowering/swizzle_standard.py |
| ProgramIO/finalizer/simulator | 现有 scale runtime provider 和 runner |
| collective child/wave planner | src/dte/coll_plan_v1.cpp |
| collective image preflight | src/isa/collective_program_v1.cpp |
| runtime wave admission | include/dte/collective_wave_admission_v1.h |

### 4.2 不采用的第一阶段路线

第一阶段不在 main common-IR2 bridge 中逐条放宽 1D、GEMM_RS、direct-route
门禁。该路径当前同时缺少二维、状态转移和多类 fusion pattern 支持，逐点打补丁
容易形成第二套不完整语义。

优先稳定现有 generic standard timing chain，再由 main compiler 增加一个明确的
RectMesh dispatch，把完整 workload 中目标 region 交给该链路，并与普通 region
形成一个 artifact。

## 5. 详细设计

### 5.1 RectMeshSpec

新增一个唯一的 typed 矩形描述，建议位于：

    llm/frontend/wafer_frontend/schema/rect_mesh.py

建议字段：

| 字段 | 含义 |
|---|---|
| rows | H |
| columns | W |
| rank_order | 首版固定 row_major |
| route_policy | 首版固定 x_first |
| ranks_per_die | 首版固定 1 |
| origin | 首版固定 (0,0)，子矩形扩展时启用 |
| timing_execution | true |
| functional_execution | false |

统一提供：

- rank_count；
- rank(row,column)；
- coordinate(rank)；
- rows、columns 的 rank orders；
- snake rank order；
- 是否存在 Hamiltonian cycle；
- ordered rank pairs；
- directed-link 数；
- 最大 hop；
- stable serialization/digest。

校验：

- rows、columns 必须为整数；
- 1≤rows,columns≤10；
- rank_count≤100；
- rank_count 与 logical TP/group participants 精确一致；
- placement 无洞、无重复、无越界；
- route 不得离开选定矩形；
- endpoint 总数满足 16-bit ABI。

### 5.2 参数化硬件生成

不要维护 100 份硬件 JSON。将
llm/test/frontend/unit/test_fabric_loader.py 中的 minimal_hardware(x,y)
提取为 production/test 共享的确定性生成器。

生成器必须为每个 Die 派生：

- Die ID 和坐标；
- 四方向边界/邻接端口；
- HBM address space/home range；
- host/memory endpoint；
- D2D 对向 link；
- stable ID。

硬件仍由现有 load_fabric 加载和验证，生成器不重复实现 PhysicalFabric 规则。

首版要求物理 fabric 精确等于请求 H×W。后续若支持大 fabric 中的子矩形，
必须按 origin、rows、columns 选连续区域，不能简单选择“前 R 个 Die”；
例如 10×10 fabric 的前 12 个 Die 并不构成 3×4。

### 5.3 Placement 和 PairRoute

Baseline 使用现有一维 logical TP group：

    TP = R = H×W
    rank = row×W+column

调用 group_registry 的 expected-group 路径生成 placement、capacity/profile
和全部 ordered PairRoute。

对每个 rank pair：

- source/destination 必须属于同一 PhysicalGroup；
- expected path 必须是现有 X-first XY route；
- hop 数必须等于 Manhattan 距离；
- 每一跳必须存在正反向 physical link；
- path 不得穿出选定矩形。

首版保留 R(R-1) 显式 routes。只有 profiling 证明 route serialization/memory
成为瓶颈后，才引入按需 route 或 route template。

### 5.4 统一矩形拓扑

新增共享 helper，建议：

    llm/frontend/wafer_frontend/policies/swizzle/rect_mesh_topology.py

由 topology.py、wang_1d.py 和后续 MeshSlice 共用，删除重复的矩形识别与
Hamiltonian 搜索。

必须提供 O(R) 的确定性构造：

1. row orders；
2. column orders；
3. snake line；
4. Hamiltonian cycle；
5. 转置等价检查。

完整 H×W grid 存在 Hamiltonian cycle 当且仅当：

- H>1；
- W>1；
- H×W 为偶数。

其他情况立即返回无 cycle，并使用 snake/bidirectional line，不进入搜索。

不得保留回溯 DFS 作为大 R fallback。构造顺序、cycle 方向和 canonical start rank
必须稳定，保证候选顺序和 artifact digest 可重复。

### 5.5 Executable baseline

所有合法尺寸都必须有 baseline。当前 exact-four XOR wave 改为 cyclic offset：

    for delta in 1..R-1:
        rank i sends to (i+delta) mod R
        rank i receives from (i-delta) mod R

每个 delta 形成一个确定性 wave。每 rank 每 wave 一发一收，满足当前
3-session 上限，并避免 R=100 时同时打开 99 个 peer。

Baseline 顺序：

1. R=1：不生成 D2D action；
2. AG：cyclic wave；
3. RS：cyclic wave，RECV 后执行确定性 binary reduction；
4. AR：RS+AG；
5. 不合法 workload：typed fail；
6. 优化候选无收益或超预算：回退 baseline。

Baseline 首要目标是可执行和闭合，不以性能最优为目标。

### 5.6 跨 Die core group

当前以下位置拒绝 group 跨 Die：

- llm/src/monitor/config_helper_program.cpp；
- llm/src/dte/sync_runtime.cpp。

最小改造：

- 删除 same-die 限制；
- 保留 active-core、排序、唯一性、group size 和 U16 校验；
- 验证 rank-to-core 映射与 frontend group 完全一致；
- 保持现有 group registry 和 collective graph carrier；
- 增加 N=1、64、65、100 的跨 Die GROUP_SYNC 测试。

现有 barrier 是进程内全局同步状态，可以提供执行顺序正确性，但不会自然产生
真实 D2D barrier 控制流和延迟。首版 capability 必须显式报告：

    cross_die_group_sync_functional = true
    cross_die_group_sync_timing_model = provisional

在真实 D2D barrier 协议完成前，性能报告不得把 barrier 延迟当作已经精确建模。

### 5.7 Collective planner 与 session 安全

生产 runtime 每核每 wave 的 endpoint session 容量为 3，而非前端默认值 64。
collective 必须复用现有 planner/admission：

1. collective records 使用非零 group_id；
2. child flow 由现有 planner 展开；
3. planner 生成 per-core/per-wave session 和 receive-byte demand；
4. image 在 link/load 前复核；
5. runtime coordinator 原子 reserve/release；
6. 前一 wave 完成后再进入下一 wave。

当前 canonical greedy pair 顺序在 100-rank、单 chunk AllGather 上可能形成约
3,205 waves 和约 690,600 actions。建议将对称 collective 的 canonical schedule
改为 round-robin delta：

    wave delta: src → (src+delta) mod R

在每个 child payload 满足 receive-byte 容量的前提下，可形成 99 waves，每核每波
一发一收，action 量约 69,400。

必须保留精确容量预检：

- max sessions per core per wave≤3；
- receive bytes per core per wave 不超平台容量；
- planner children/actions/waves 不超 image 上限；
- derived bytes 使用精确 estimator，不继续固定假定 1 MiB；
- program records≤1,048,576；
- program file≤64 MiB；
- core/group/member ID 满足 U16；
- 每核 transport TX tag 生命周期≤65,535。

多 chunk 或多层 workload 可能超过 1M records。首版先用 tiny/single-layer
canary 建立闭环；完整 workload 必须通过 artifact preflight。若真实 production
profile 超限，再基于测量实现 collective plan/template 去重或循环复用，不允许
静默截断。

### 5.8 Standalone P2P

zero-key standalone records 不进入 collective planner，因此没有相同的 wave
scheduler。

首版任意 standalone flow 必须由前端静态分 wave，并证明：

    per_core_active_send + per_core_active_receive ≤ 3

每 wave 固定顺序：

    post receive
      → barrier/event
      → issue send
      → wait/fence
      → retire session

未经容量证明的 manifest 必须在 finalizer/load 前失败。禁止只把 runtime 改为
“容量满时等待”，否则所有 rank 都等待 peer post receive 时可能死锁。

### 5.9 Main compiler 与完整负载

现有独立 standard Swizzle timing 链先作为 conformance 和 runtime 闭环。
但 dense_workload_complete 必须满足：

- 输入来自真实 Experiment/IR0，而非手写单 collective fixture；
- ordinary region、standalone collective 和目标 fusion region 进入同一个 artifact；
- 不把两个 simulator makespan 相加；
- region replacement 不重复执行、不漏执行；
- output ownership、logical bytes、FLOPs、ALLOC/FREE 完整闭合；
- ProgramIO rank-local probes 数等于 R。

推荐在 main compiler 增加明确的 RectMesh standard-chain dispatch，而不是第一阶段
扩宽现有窄 common-IR2 bridge。独立链稳定后，再评估是否合并 common projector。

第一版完整 Dense workload 只承诺：

- AG+GEMM；
- GEMM+RS；
- GEMM+AR 可先由 RS+AG baseline 完成；
- tensor 维度、tokens、H/I/K 与 R/rows/columns/chunks 可整除。

### 5.10 Wang 扩展

Wang 的 rank 语义和 action builder 大部分已经参数化，应保留并做 scale-aware
改造。

首版大 R profile：

| 选项 | 建议值 |
|---|---|
| chunk_count | R |
| unroll | 1 |
| topology | snake line |
| max candidates | 1～2 |
| max_chunk_count | 至少 100 |
| ring | 仅构造式 cycle 存在时 |

100 rank 最小 chunk 下的 action 量级约为：

| pattern | 上界量级 |
|---|---:|
| AG_GEMM | 4R² |
| GEMM_RS | 5R² |
| GEMM_AR | 8R² |

若首版承诺 100-rank GEMM_AR，shape-aware max_actions 至少应覆盖约 80,000，
但不能简单把全局常量无限提高。

必须在构造 DAG 前使用公式预估：

- actions；
- buffers；
- chunks；
- SRAM；
- logical bytes；
- program records；
- derived bytes。

超预算候选在创建数万对象前被拒绝。另需：

- route index 每个 problem 只构建一次；
- ready queue 使用稳定 heap/Kahn queue，避免反复 pop(0)+sort；
- 大 R 不执行指数 DFS；
- terminal ownership 对任意 rank 使用 generic layout；
- 精确四 rank packed layout 仅保留为优化 fast path；
- 所有 Wang 失败都回退 executable baseline。

### 5.11 MeshSlice 扩展

MeshSlice candidate generator 已支持 2×4、4×2、4×4 等矩形。production
阻断集中在：

- passes/meshslice_2d_placement.py 的 DP2×TP2 假设；
- lowering/swizzle_meshslice_standard.py 的 4 ranks/2 rows/2 columns audit；
- problem physical-shape 在非方阵上的 rows/columns 转置风险；
- max_actions=4096。

参数化方案：

1. surrogate 一维 group 使用 TP=R，继续复用 expected-group route/capacity；
2. placement 增加 logical_shape=(H,W)；
3. logical_coord=(row,column)；
4. lowering 从 topology 派生 rows、columns 和 peer sets；
5. 每 rank row peers 为 W-1；
6. 每 rank column peers 为 H-1；
7. 一维 Mesh 明确不生成 MeshSlice；
8. 只有真实二维 OS sharding 和整除条件满足时才允许候选。

通用 flow 计数：

    row_flows    = slices × R × (W-1)
    column_flows = slices × R × (H-1)

AG_GEMM action 量：

    actions = slices × R × [3(H+W-2)+1]

10×10、slice=1 时约 5,500 actions，已经超过当前 production 4096。
因此 MeshSlice 必须使用 shape-aware budget。

交付顺序：

1. AG_GEMM；
2. slice=1；
3. RS；
4. AR；
5. 多 slice 性能搜索。

### 5.12 MoE

MoE 不与 Dense flexible-mesh 首版合并交付。当前 EP4/2×2 假设分布在 schema、
plan、state ABI、runtime markers 和 evidence 中。

后续第一版 MoE flexible-mesh 可限定：

- EP=R；
- 每 Die 一个 expert；
- top-k=1；
- static trace；
- balanced trace 先于 skewed trace；
- Direct-XY personalized A2A baseline；
- 关闭 2×2 专用 runtime marker，或按实际 H/W 重新生成 edge set。

禁止将 range(4)、EP4 和八条 2×2 directed edge 机械替换为 R；必须重新定义
流量闭合、expert ownership、session wave 和 evidence。

## 6. 分阶段开发计划

### 6.1 F0：冻结能力契约

#### 工作项

- 新增 RectMeshSpec；
- 固定完整矩形、10×10、每 Die 一 rank、row-major、X-first；
- 固定 timing=true、functional=false；
- 固定 Dense first；
- 定义 capability report；
- 定义 typed error/fallback 原因。

#### 产物

- schema；
- capability matrix；
- 正负例；
- 本文档中的完成状态进入测试报告。

#### 门禁

- 0、11、R>100、TP≠R、洞状 placement 必须 fail closed；
- 旧配置行为不变。

### 6.2 F1：矩形 fabric、placement 和有界拓扑

#### 工作项

- 提取参数化 hardware generator；
- 复用 load_fabric；
- 复用 expected-group 和 XY PairRoute；
- 新增共享 RectMeshTopology；
- 用 O(R) 构造替换两处 Hamiltonian DFS；
- 修复非方阵 rows/columns 转置风险；
- 增加 100-shape unit sweep。

#### 产物

- 100 种矩形的稳定 fabric/group/topology digest；
- route/link/hop 报告；
- bounded-time topology test。

#### 门禁

- 100 种尺寸全部通过；
- 10×10 topology 不得进入 DFS；
- 100-shape topology sweep 建议低于 10 秒；
- 同一输入重复构建 digest 完全一致。

### 6.3 F2：跨 Die group 和 capacity-safe collective

#### 工作项

- 解除 loader/runtime same-die group 限制；
- 增加跨 Die group validation；
- collective planner 改为 deterministic round-robin wave；
- 统一生产 session capacity=3；
- derived/action/wave/record/file 精确 preflight；
- standalone P2P 静态 wave 约束；
- 固定跨 Die baseline 为 unicast+endpoint reduce。

#### 产物

- N=1、3、10、64、65、100 collective plan；
- wave demand report；
- capacity negative tests；
- runtime residual report。

#### 门禁

- 每核每 wave session demand≤3；
- N=100 单 chunk collective 不死锁；
- endpoint/session/tag/barrier residual 全部归零；
- 超容量输入在运行前 typed fail。

### 6.4 F3：所有尺寸可运行的 executable baseline

#### 工作项

- exact-four XOR 改 cyclic offset；
- AG、RS、AR baseline；
- standard projection/lowering/link 适配 R streams；
- 泛化 ProgramIO rank-local probes；
- 抽取 scale runtime provider 的通用 prepare/lower/link 逻辑；
- 新增 flexible_mesh_cases/provider/report；
- 接入 finalizer、resolver、npusim 重复性检查。

#### 产物

- 100-shape tiny canary；
- representative runtime matrix；
- artifact、ProgramIO、makespan digest。

#### 门禁

- 所有 R>1 的合法 AG/RS 至少存在一个 executable baseline；
- R=1 无 D2D action；
- 所有 100 种尺寸至少一个 canary 完成；
- ProgramIO pass=1；
- 不出现 PROTO_WAIT；
- 两次运行 artifact 和 marker/makespan 稳定。

完成后标记：

    mesh_runtime_complete = true

### 6.5 F4：完整 Dense workload 接入

#### 工作项

- main compiler 增加 RectMesh standard-chain dispatch；
- ordinary、standalone、fusion region 合并为单 artifact；
- 限定首版 pattern 和可整除 workload；
- 完整输出 ownership、bytes/FLOPs 和 memory lifecycle audit；
- 增加 artifact 容量预检；
- 对超大重复 collective 评估 plan/template 去重。

#### 产物

- 代表尺寸真实 Dense workload；
- naive/AUTO 双分支；
- workload-level ProgramIO 和 provenance；
- manifest capacity report。

#### 门禁

- 不是手写单算子 fixture；
- 整个 workload 只运行一个 artifact；
- region replacement 无重复、无缺失；
- bytes、FLOPs、ALLOC/FREE、terminal outputs 闭合；
- artifact≤64 MiB、records≤1M；
- 代表尺寸两次运行稳定。

完成后标记：

    dense_workload_complete = true

### 6.6 F5：Wang 大 R 扩展

#### 工作项

- scale-aware candidate profile；
- cheap action/buffer/SRAM estimator；
- route-index cache；
- stable ready heap；
- line 优先、构造式 ring 后置；
- generic terminal ownership；
- AUTO 与 forced 分离。

#### 产物

- R=3、6、9、10、30、90、100 的 Wang candidate/runtime evidence；
- fallback reason report；
- planner latency/RSS report。

#### 门禁

- max candidates 始终有界；
- 10×10 prepare 建议低于 30 秒；
- 峰值 RSS 建议低于 1 GiB；
- 无指数拓扑搜索；
- 无候选或无收益时自动回退；
- forced 只证明可执行，AUTO 才参与收益声明。

完成后标记：

    wang_scale_complete = true

### 6.7 F6：矩形 MeshSlice

#### 工作项

- 参数化 2D placement；
- 参数化 standard lowering/audit；
- 修复 rows/columns 方向；
- 真二维 OS sharding 校验；
- AG_GEMM slice=1；
- shape-aware action/buffer budget；
- 扩展 RS/AR 和多 slice。

#### 产物

- 2×3、3×2、2×10、10×2、5×6、6×5、10×10 的 MeshSlice evidence；
- row/column flow closure；
- fallback report。

#### 门禁

- flow/action 计数符合公式；
- 每条 SEND/RECV 字节闭合；
- 一维 Mesh 不生成 MeshSlice；
- 缺真实二维 sharding 时 fail closed；
- 无收益时回退。

完成后标记：

    meshslice_rect_complete = true

### 6.8 F7：后续能力

- Dense training；
- MoE flexible mesh；
- functional execution；
- 子矩形 placement；
- 真实跨 Die barrier timing；
- plan/template reuse；
- 非矩形和故障 Mesh。

这些能力不得阻塞 F0-F6，但也不得被 F0-F6 的完成状态隐式覆盖。

## 7. 建议代码改动清单

### 7.1 Python frontend

| 文件/模块 | 改动 |
|---|---|
| schema/rect_mesh.py | 新增 RectMeshSpec 和 envelope |
| passes/load_fabric.py | 原则上不改拓扑语义，仅接生成器/补 capability |
| passes/group_registry.py | 复用 expected-group；补矩形 placement witness |
| policies/swizzle/rect_mesh_topology.py | 新增共享构造式拓扑 |
| policies/swizzle/topology.py | 删除 DFS，调用共享 helper |
| policies/swizzle/wang_1d.py | 删除重复 DFS，增加 scale profile |
| policies/swizzle/problem.py | rows/columns 从 topology 派生 |
| policies/swizzle/chunking.py | 允许 chunk=R≤100 |
| policies/swizzle_defaults.py | shape-aware limits，不盲目放大全局默认 |
| policies/swizzle/cost.py | route cache、stable ready queue |
| passes/project_unfused_comparison.py | exact-four XOR 改 cyclic waves |
| passes/meshslice_2d_placement.py | 从 DP2×TP2 参数化为 H×W |
| lowering/swizzle_meshslice_standard.py | 参数化 rank/row/column audit |
| passes/program_io.py | 验证任意 R 的 rank-local output/probe |
| compiler.py | 增加 RectMesh standard-chain dispatch |

### 7.2 C++ runtime/backend

| 文件/模块 | 改动 |
|---|---|
| monitor/config_helper_program.cpp | 允许合法跨 Die group；统一容量预检 |
| dte/sync_runtime.cpp | 删除 same-die group 假设 |
| dte/coll_plan_v1.cpp | deterministic round-robin wave |
| isa/collective_program_v1.cpp | 精确 image demand/derived preflight |
| dte/collective_wave_admission_v1.* | 原则上直接复用，补 N=100 证据 |
| workercore/workercore.cpp | 保持 session=3 fail-fast，不改为隐式等待 |
| isa/program_format.* | 原则上不改格式，继续保持 64 MiB/1M 门禁 |

### 7.3 测试与报告

新增并列目录/模块，避免修改旧 suite 的 frozen IDs：

    llm/test/frontend/unit/test_flexible_rect_mesh.py
    llm/test/frontend/unit/test_flexible_mesh_topology.py
    llm/test/frontend/unit/test_flexible_mesh_collective_capacity.py
    llm/test/frontend/integration/flexible_mesh_cases.py
    llm/test/frontend/integration/flexible_mesh_runtime_provider.py
    llm/test/frontend/integration/run_flexible_mesh_runtime.py
    llm/test/frontend/integration/flexible_mesh_runtime_report.py

具体文件名可按现有测试布局调整，但必须保持独立 case/provider/report 概念。

## 8. 规模与容量模型

### 8.1 Fabric

    R = H×W

    directed_links
      = 2×[H×(W-1) + W×(H-1)]

    ordered_pair_routes
      = R×(R-1)

    max_hops
      = H+W-2

10×10：

    R=100
    directed_links=360
    ordered_pair_routes=9900
    max_hops=18

### 8.2 Endpoint ABI

现有约束：

    die_count × (cores_per_die + host/memory endpoints per die) < 2^16

10×10 不需要扩大 core/source/destination ID，但必须在硬件加载前精确检查。

### 8.3 Program artifact

继续保留：

- 文件≤64 MiB；
- records≤1,048,576；
- cores/groups/member IDs 满足 U16；
- transport tag 生命周期满足 U16；
- barrier active-state 数有界。

禁止为了通过 10×10 测试直接删除这些上限。

### 8.4 Planner

所有策略在物化完整 DAG 前必须提供：

    estimate_actions
    estimate_buffers
    estimate_sram_bytes
    estimate_children
    estimate_waves
    estimate_derived_bytes
    estimate_program_records

估计超过硬上限时：

- AUTO：返回 typed no-candidate/fallback；
- forced：明确报容量错误；
- 不允许生成半成品 manifest；
- 不允许运行到 simulator 才因 session exhaustion 失败。

## 9. 测试与验收矩阵

### 9.1 100-shape unit sweep

每次提交枚举所有 H,W∈[1,10]，断言：

- rank 数=H×W；
- placement 数=R；
- rank=row×W+column；
- directed links 符合公式；
- PairRoute=R(R-1)；
- 每条 route hop=Manhattan 距离；
- route 为 X-first；
- rows、columns、snake 完整且无重复；
- cycle 存在条件精确；
- H×W 与 W×H 满足转置等价；
- 两次构建 digest 和候选顺序一致。

### 9.2 Strategy/lower-link 代表集

    1×1
    1×2, 2×1
    2×2
    1×10, 10×1
    2×3, 3×2
    3×3
    2×10, 10×2
    5×6, 6×5
    9×9
    9×10, 10×9
    10×10

该集合覆盖：

- 单 Die；
- 一维退化；
- 旧 2×2 回归；
- 非二次幂；
- 非方阵与转置；
- 奇×奇无环；
- 最大非对称；
- 最大尺寸。

### 9.3 Workload matrix

| pattern | 最低覆盖 |
|---|---|
| AG+GEMM | 所有代表尺寸 |
| GEMM+RS | 所有代表尺寸 |
| GEMM+AR | 2×3、3×3、10×10 |
| MeshSlice AG_GEMM | 2×3、3×2、5×6、6×5、10×10 |
| 1D Wang | 1×10、10×1、2×3、9×10、10×10 |

canary workload 必须显式选择可整除的 tokens、H、I、K 和 chunk，避免把
workload-shape 不合法误判为 Mesh 不支持。

### 9.4 Runtime 不变量

- manifest core streams=R；
- runtime core binding=R；
- ProgramIO rank-local probes=R；
- 每个 action 的 peer/route 属于真实 group；
- SEND/RECV logical bytes 成对；
- output ownership 精确；
- FLOPs 与 baseline 同工作量；
- ALLOC/FREE 全闭合；
- per-wave sessions≤3；
- endpoint/session/tag/barrier residual=0；
- npusim exit=0；
- ProgramIO pass=1；
- 不出现 PROTO_WAIT；
- 相同输入两次 artifact SHA、makespan 和关键 marker 一致。

### 9.5 负例

- rows/columns=0；
- rows/columns=11；
- R>100；
- TP≠R；
- endpoint 总数超过 65535；
- placement 缺 Die、重复、越界或有洞；
- 缺/重复 PairRoute；
- 缺 directed physical link；
- route 穿出 group；
- tensor extent 不可整除；
- K 不可被 rows/columns/slices 整除；
- 一维 Mesh 请求 MeshSlice；
- action、buffer、chunk、SRAM、HBM、derived bytes 超限；
- 每核第 4 个未退休 standalone session；
- transport TX tag 超过生命周期；
- artifact 超 64 MiB/1M records；
- 跨 Die multicast/DCA；
- 跨 Die HOST endpoint；
- forged plan/provenance/artifact SHA。

### 9.6 CI 层级

PR：

- 100-shape topology/placement/route unit；
- 旧 1×2/2×2 回归；
- 2×3、3×2 finalizer/resolver smoke；
- capacity negative tests。

Nightly：

- 全部代表尺寸真实 npusim；
- NAIVE/AUTO 双分支；
- 关键 case 重复两次；
- planner latency/RSS；
- ProgramIO 和 runtime residual。

Release：

- 100 种矩形各运行一次 tiny canary；
- 代表尺寸 AG/RS/AR；
- 代表尺寸完整 Dense workload；
- artifact/marker/makespan 重复性；
- capability report。

## 10. 兼容与上线策略

### 10.1 保持旧路径不漂移

- 不改旧 S0-S3/C0-C4 case 定义；
- 不改旧 stable ID；
- 不用新的 10×10 budget 覆盖旧默认；
- 旧 exact-four packed layout 保留为 fast path；
- 新 flexible-mesh schema/policy 显式 opt-in；
- 旧 2×1/2×2 输出若变化，必须有精确原因和 golden 更新。

### 10.2 Fallback 原因

报告至少区分：

- invalid_mesh；
- invalid_placement；
- incompatible_sharding；
- no_cycle_use_snake；
- candidate_action_budget；
- candidate_buffer_budget；
- candidate_sram_budget；
- runtime_session_budget；
- planner_derived_budget；
- artifact_record_budget；
- artifact_file_budget；
- no_economic_benefit；
- unsupported_functional_execution；
- unsupported_moe_mesh。

AUTO fallback 是正常能力，不应表现为异常退出。forced 超预算必须明确失败。

### 10.3 性能声明

所有尺寸可运行不等于所有尺寸优化都盈利。

性能报告只允许：

- 对同工作量 naive/AUTO 比较；
- 使用真实 runtime marker；
- 至少重复两次；
- 记录 planner time、artifact records、runtime makespan；
- 指明 exact H×W、workload 和策略；
- 不把 forced 分支当成收益证据；
- barrier timing 尚为 provisional 时明确标注。

## 11. 风险与缓解

| 风险 | 后果 | 缓解 |
|---|---|---|
| DFS 在无环矩形爆炸 | planner 超时 | O(R) 构造并立即判无环 |
| actions 随 R² 增长 | Python 时间/RSS、artifact 超限 | 物化前估算、候选裁剪、plan reuse |
| standalone 同时连接过多 peer | session exhaustion/死锁 | group collective 或静态 cyclic waves |
| same-die group 限制 | collective 无法跨 Die | 删除限制并保留严格 group validation |
| barrier 无真实 D2D latency | 性能高估 | capability 标 provisional，后续建协议 |
| 非方阵 rows/columns 颠倒 | tile/flow 错误 | topology 派生 shape、转置测试 |
| 大 fabric 选择前 R 个 Die | 子图非目标矩形 | 首版 exact fabric，后续 origin+rectangle |
| 100-rank 多层 workload 超 1M records | 无法 link/load | preflight，必要时 plan/template 去重 |
| MeshSlice 只删 2×2 校验 | audit/peer 数错误 | 全量参数化 flow 公式与 production bridge |
| MoE 机械泛化 | marker、state、ownership 错误 | 独立阶段和新 typed schema |
| 修改旧 suite | stable evidence 漂移 | 新增并列 conformance suite |

## 12. 建议 PR 拆分

### PR1：RectMesh schema、generator 和 100-shape topology

- RectMeshSpec；
- hardware generator；
- expected-group reuse；
- shared topology；
- 删除 DFS；
- 100-shape unit。

### PR2：跨 Die group 和 collective round-robin

- same-die restriction；
- round-robin waves；
- session=3；
- derived/image preflight；
- N=100 selftest。

### PR3：Executable cyclic baseline 与 generic runtime provider

- arbitrary-rank AG/RS；
- AR baseline；
- R-stream lower/link/ProgramIO；
- flexible mesh case/provider/report；
- representative npusim。

### PR4：100-shape canary 与 main compiler workload dispatch

- 100-shape runtime sharding；
- single-artifact Dense workload；
- capacity report；
- release gate。

### PR5：Wang scale

- shape-aware budget；
- route/ready-queue 优化；
- large-R profile；
- fallback/economic report。

### PR6：Rectangular MeshSlice

- generic 2D placement；
- standard lowering/audit；
- AG_GEMM slice=1；
- RS/AR 后续小 PR。

### PR7：MoE、training、functional 和 advanced topology

独立设计和验收，不与 Dense flexible-mesh 完成状态混用。

## 13. 最终 Definition of Done

### 13.1 mesh_foundation_complete

- 100 种 H×W 均加载成功；
- links/routes/hops 公式闭合；
- topology 有界且确定；
- 非法矩形 fail closed。

### 13.2 mesh_runtime_complete

- 100 种尺寸各至少一个真实 tiny canary；
- finalizer、resolver、npusim、ProgramIO 通过；
- 无 PROTO_WAIT；
- runtime residual 清零；
- 两次运行稳定。

### 13.3 dense_workload_complete

- 代表尺寸运行真实 Dense workload；
- ordinary、collective、fusion 在同一 artifact；
- bytes、FLOPs、ownership、lifecycle 闭合；
- artifact 容量通过；
- timing-only capability 如实报告。

### 13.4 wang_scale_complete

- R=100 不执行 DFS；
- 候选生成时间、内存有界；
- Wang 可运行；
- 超预算/无收益时 executable fallback；
- 性能声明只来自 AUTO 实测。

### 13.5 meshslice_rect_complete

- 真二维矩形 placement/lowering/audit 闭合；
- 非方阵方向正确；
- flow/action 公式闭合；
- 一维/不兼容 sharding fail closed；
- 代表尺寸真实运行。

当以上 Dense 状态全部满足后，才可将项目能力更新为：

    supports_full_rect_mesh_up_to_10x10 = true
    supports_timing_execution = true
    supports_functional_execution = false
    supports_irregular_mesh = false
    supports_general_moe_mesh = false

## 14. 一句话实施原则

先把当前已经通用的 fabric、XY route、standard lowering/linker 和 collective
wave admission 升格为矩形公共底座，用 capacity-safe executable baseline 保证
所有尺寸可运行；再增量推广 Wang 和 MeshSlice，而不是从 2×2 专用代码全面复制
出一套新的 10×10 路径。
