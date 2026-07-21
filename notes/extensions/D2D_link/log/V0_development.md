# V0 开发总结（多 die 基础设施重构）

> 记录 WaferAI-SIM D2D Link 建模 **V0** 的开发全过程：目标、增量顺序、每步的设计与验证、
> 途中发现并修复的 bug、以及最终状态。规划见 `../D2D_link_test.md` 与 `../D2D通信建模计划.md`，
> 测试与逐增量说明见 `llm/test/d2d_link/README.md`。
>
> **说明**：第 1–5 节是 V0b 增量开发（截至 V0b-2B1）的历史记录；此后 **V1-pre / V0-exit
> （Inc 1–4：配置驱动 HOST attachment + 基线冻结）** 的进展见第 6 节「最终状态」与 README 基线冻结节。
> 因此当时的 87/87、15/15 是历史快照，**最终数值以第 6 节为准（165/165、23/23）**。

## 1. 目标与约束

**目标**：拆除仿真器"单 die、方形 mesh、`GRID_SIZE` 多义"的基础假设，让它能实例化多 die、
在 die>0 上运行 workload——但**不引入真实 D2D 流量**（跨 die 传输属 V1）。

**贯穿全程的铁律**：
- **单 die 仿真时序（ns）逐位不变**——每一步都用 `llm/test/noc_congestion` 四场景
  （14781/29109、14833/45441）当回归基线，任何改动都必须保持它精确一致。
- **每步可独立验证**、**禁止一次性大改**——把大重构切成能独立回归的纵向增量。
- **诚实**——只声称验证过的；不把"未做/半成品"表述成"已完成"。

## 2. 增量顺序（最终形态）

V0 = **V0a 基础设施/纯函数** + **V0b 运行时改造**。V0b 又拆成 7 个纵向增量：

```
V0a  基础设施 + 纯函数 + 配置校验 + 统计接口
  ↓
V0b-1  编址/消息语义收口（HOST endpoint 分区、DATA 全局 source）
V0b-3  live torus 拆除（开边 mesh + 终结通道）
V0b-2 前置 hardening（溢出/delete[]/UAF）
V0b-2A 运行时多 die 模块实例化（die>0 存在但空闲）
V0b-4  D2D peer/link 结构级构造 + 配对校验
V0b-2C0 可靠跨 die 拒绝（raw-JSON preflight）
V0b-2C1 global-id 归一化（构造级）
V0b-2B0 HostEnvelope + LegacyHostEnqueue（解耦 config_helper 与物理 HOST lane）
V0b-2B1 per-die HOST lane —— workload 真正在 die>0 运行  ← 里程碑
```

> 这个切分不是一开始就有的，而是**在多轮 review 后收敛**出来的（见 §5）。早期曾试图"一次跑通
> die1 workload"，实测撞上段错误/挂死，才切成上面的纵向增量。

## 3. 各增量：做了什么 / 怎么验证

### V0a 基础设施 + 纯函数
- 编址常量拆分：`CORES_PER_DIE / DIE_X / DIE_Y / DIE_COUNT / TOTAL_CORES / HOST_ENDPOINT_ID`
  （`defs/spec.{h,cpp}`），单 die 时全部退化为旧 `GRID_SIZE`。
- 全局编址 helper（`utils/router_utils.h`）：`GlobalId/DieOfGlobal/LocalOfGlobal/…/IsDieEdge/
  DecodeEndpointType`。
- die 配置解析（`utils/config_utils.cpp`）：`"die":{x,y}` + 矩形 die 内 mesh（可选 `"y"`）。
- 端口数据结构 + 解析 + 校验（新模块 `die/port.{h}`、`die/port_config.cpp`）：`ParseDiePorts/
  ValidateDiePorts`、`port_for_host`；校验含 MVP `side==dir`、重复端口、越界 idx、缺方向 C2C、
  **HOST 可达性**、**参数范围**（`link_bw>=1/latency>=0/buffer_depth>=1`）、**endpoint 容量**
  （`core+host+mem <= 65536`，宽整数防溢出）。
- endpoint 三分解码（core/host/mem）。
- flow/link 统计接口（`D2DPortStats` + `TotalActivity()`）。
- `GetInputSource` Y 轴矩形修正（方阵无变化）、N/S 坐标定义固定（`NORTH=y+`）。
- 消息头：DATA 构造函数漏初始化 `source_`、`Msg` 全成员类内默认值、逐字段序列化往返测试。
- **验证**：`--d2d-v0-selftest` 纯函数自测（最终 87 项）。

### V0b-1 编址/消息语义收口
- HOST sentinel `des_/source_ == GRID_SIZE` → **每 die HOST endpoint 区间**：新增
  `IsHostEndpoint(ep)` 识别 `[TOTAL_CORES, TOTAL_CORES+DIE_COUNT)`，用于 `GetNextHop/GetNextHopReverse`
  与 `router.cpp` 的 output_lock（否则 `HostEndpointOfDie(1)` 会被当普通 core 坐标误算）。
- `logic.cpp` 的 **DONE/ACK 改发本核所在 die 的 host**（`HostEndpointOfDie(DieOfGlobal(cid))`）。
- **DATA 携带真实全局 `source_ = cid`**（原为 -1）。
- 保留 `GRID_SIZE` 的几何/容量语义，不机械替换。
- **验证**：单 die 逐位不变（这些在单 die 下都是别名，`HOST_ENDPOINT_ID==GRID_SIZE`）。

### V0b-3 live torus 拆除
- 新纯函数 `OpenMeshNeighbor(global, dir)`：无环绕、不跨 die，边缘返回 -1。
- `monitor.cpp` router-router 绑定改为**开边 mesh**：边缘方向输入侧接**共享终结通道**。
- **关键洞察 + 验证**：`GetNextHop` 目的导向、从不选 wrap 方向 → 环绕链路本就 inert。单 die
  逐位不变**从运行时证明**了 wrap 可安全物理移除（非"未使用"）。

### V0b-2 前置 hardening（扩容前的确定性 bug 修复）
- **地址校验防 int 溢出**：先宽整数校验维度/容量，**再**算 int 全局量（`x=100000` 在赋值前即拒）。
- 数组释放 `delete` → `delete[]`（`RouterMonitor::~`、`Monitor::~`）。
- **消除 `g_dram_kvtable` UAF/double-free**：`WorkerCoreExecutor::~` 原先 delete 整个数组后再访问
  其元素（UAF）+ 多核重复 delete（double-free）；改为只 delete 本核元素，数组由 Monitor `delete[]`。
- **发现（未修，隔离）**：启用逐个 `delete workerCores[i]` 会在退出时**段错误**——WorkerCore 析构链
  有既有 SystemC teardown 隐患。故保留 Worker 不逐个析构（退出即回收的泄漏），扩容也不触发
  double-free。此项标为独立清理，**不宣称完整生命周期修复**。

### V0b-2A 运行时多 die 模块实例化
- 核级数组/遍历/模块数按 **`TOTAL_CORES`**（`monitor.cpp`、`router.cpp` 的 `RouterMonitor`、
  `system_utils.cpp`、cache system）；**保留** `GRID_X/GRID_Y`（几何）、`CORES_PER_DIE`。
- **`GetCoreHWConfig` 同构映射**（`id % CORES_PER_DIE`）——避免 die>0 executor 取不到 HW 配置
  导致的**除零崩溃**（实测暴露）。
- **die>0 HOST 端口终结**：输入共享只读终结信号，**每个输出各自独立 sink**（避免 multi-writer）。
- **验证**：**独立层级模块计数**（`dynamic_cast` 遍历 SystemC 层级，非自报 `TOTAL_CORES`）——
  2×1/1×2 建 32、2×2 建 64；die>0 空闲；die0-only workload 仍 29109；**运行后 D2D 统计=0**。

### V0b-4 D2D peer/link 结构级构造 + 配对校验
- `BuildD2DLinks()`（`ParseDiePorts` 末尾）：从同构端口模板 + die-mesh 构造 `g_d2d_links`。
- **收口**：`g_d2d_links.clear()`（防重复解析残留）、**精确四元组互反**（原弱判断漏"远端指向第三
  端点"）、端口唯一性、bw/latency 兼容**标注为同构保证**（非已测的不兼容拒绝路径）。
- **验证**：2×1 满边 → 8 条有向 link、精确互反、单 die 0 link、**负向**（镜像坐标不匹配报错）。

### V0b-2C0 可靠跨 die 拒绝
- **根因**：原 guard 放在 `CoreConfigRemap`，但此阶段 `work.cast` **尚未填充**（casts 后转化）→
  guard 遍历到空 cast、不生效（实测 core0→core16 未被拒、DONE 处**永久等待**）。
- **修复**：`config_helper_core::PreflightValidateWorkload` 直接扫描**原始 JSON** 的
  core/cast/source/prim_copy id，在绘图/构造/elaboration **之前**拒绝跨 die cast、越界、die>0 目标。
- **绘图健壮化**：`plot_dataflow` 遇 `DIE_COUNT>1` 跳过（有 `GRID_SIZE` 网格假设，否则 die>0 id 段错误）。
- `id_space`（缺省 die0-local / `global`）消歧。
- **验证**：runner `core0→core16` 用例（10s 超时保护）→ 非零退出、报
  `cross-die traffic requires D2D Link`、**不进仿真、不挂死**。

### V0b-2C1 global-id 归一化（构造级，不跑仿真）
- 新模块 `monitor/workload_normalize.{h,cpp}`：`NormalizeWorkloadJson(j, die_id)`（JSON 层平移
  core-ref id 到 global + 置 `id_space:global`）+ `ValidateWorkloadStructure(j)`（从 preflight 拆出
  的纯函数结构校验）。
- **发现**：生产路径 `CoreConfigRemap` **本就用 `g_core_remap` map**（天然支持 global id）、用
  `tag==dest` 判 core-ref（非 `tag<GRID_SIZE`）；只有停用的 `random_core` 里有 `int o2r[GRID_SIZE]`
  → 顺手改 `vector(TOTAL_CORES)`。
- **验证**：11 项纯函数自测（0/1→16/17 各字段正确、die1-local 允许、cross-die 拒绝、越界拒绝、
  die_id=0 恒等）。

### V0b-2B0 HostEnvelope + LegacyHostEnqueue
- 新 `monitor/host_envelope.{h,cpp}`：`struct HostEnvelope{int dest_global_id; Msg msg;}` +
  `LegacyHostEnqueue`（按 `dest/GRID_X` 落 `write_buffer`）。
- `config_helper_core` 新增 **`BuildConfigMessages()/BuildStartMessages()`** 返回信封——只决定
  "发给哪个全局核 + 什么 Msg"，不含物理 lane；`fill_queue_config/start` 接口不变，内部 =
  `Build*Messages()` + `LegacyHostEnqueue`。
- **验证**：信封按原序生成、按 `dest/GRID_X` 落队（与原 `config.id/GRID_X` 一致）→ 消息
  dest/顺序/数量完全一致，单 die 逐位不变。（途中一个 brace 定位失误——`fill_queue_config` 实际在
  336 行结束、`return envs;` 误加到后面另一个方法——build 立刻暴露并修正。）

### V0b-2B1 per-die HOST lane —— 里程碑
- **HOST lane 数 `GRID_X` → `HOST_LANES = GRID_Y * DIE_COUNT`**（新全局量）：MemInterface 的
  host 端口/`write_buffer`、Monitor 的 host 信号/绑定循环全部按 `HOST_LANES`。
- **优雅的通用映射**：`lane i ↔ 全局行 i ↔ 西边缘 router i*GRID_X ↔ write_buffer[i]`，而
  `dest/GRID_X` **恰好就是 lane**——所以 `LegacyHostEnqueue`、Monitor 绑定循环**天然泛化到多 die**
  （单 die 方阵 `HOST_LANES==GRID_X`，逐位不变）。die>0 HOST 终结块删除，改由 lane 循环统一接入。
- **DONE 路由**：die>0 核 DONE 走 `HostEndpointOfDie(die)` → 本 die 西边缘（开边 mesh 不跨 die）→
  本 die HOST lane → MemInterface。
- **T3 暴露并修复的 2C-main 审计**：`config.cpp` 的 `c.id >= GRID_SIZE` → `>= TOTAL_CORES`
  （否则 die>0 核 id 被误判越界、`from_json` 提前 return → `send_global_mem` 未初始化 → 触发
  "Only one core can send global memory" 断言）。
- 移除 die>0-unrunnable guard（跨 die cast 仍被结构校验拒绝）。
- **验证（T1→T2→T3，小步不上整 GEMM 调试）**：
  - **T3** die1 平移 workload → **16 个 die1 核全部执行**、HOST1 收齐 DONE、sim-time=29109、D2D=0。
  - **T2** die0+die1 合并 → 两 die 各 16 核**并行、零干扰**、sim-time=29109、D2D=0。
  - **布局泛化**：die1@1×2（纵向）、die3@2×2（更高 die + 2D 布局，64 核，die3 核 48-63 全执行）。

## 4. 验证方法学

- **单 die 逐位一致**：每增量都跑 `llm/test/noc_congestion` 四场景，比对 **sim-time(ns)**——这是
  发现回归的第一道门。注意口径：比的是退出码 + sim-time + 独立层级模块计数 + 运行后 D2D 统计，
  **不是**完整 stdout/trace 逐字节（故文档统一用"sim-time 逐位一致"而非"byte-identical"）。
- **独立层级模块计数**：`dynamic_cast` 遍历 `sc_get_top_level_objects()` 数 `RouterUnit/WorkerCore`，
  不信自报 `TOTAL_CORES`。
- **运行后 D2D 统计**：`sc_start` 后 dump `[D2D] in/out/busy/stall`，runner 断言全 0。
- **die-run 断言"全部 16 核执行"**（不是 >=8）：grep 每 die 的核 id 集合 == `CORES_PER_DIE`。
- **超时保护**：跨 die 拒绝/die-run 用例设 10–120s `timeout`，把"挂死"变成失败而非卡住。
- 载体：`llm/test/d2d_link/run_test_d2d_v0.py`（端到端 15 组）+ `--d2d-v0-selftest`（纯函数 87 项）。

## 5. 关键教训 / 踩过的坑

1. **不要过度声称"完成"**：早期把"基础设施 + 纯函数"说成"V0 全部完成"，被 review 指出运行时仍只
   一块 die。此后严格区分"结构/纯函数层完成"与"运行时完成"，并逐条如实标注未做项。
2. **一次性大改会撞墙**：直接上"die1 跑整 GEMM"撞了段错误/挂死。切成纵向增量（每步单 die 逐位
   不变）后，每个失败都被小步隔离（除零、断言、config.cpp 越界都是这样精准定位的）。
3. **guard 位置取决于数据何时就绪**：跨 die 拒绝必须在**原始 JSON** 上做（casts 在 `CoreConfigRemap`
   后才填充），而非依赖已构造的 `work.cast`。
4. **既有隐患要隔离而非硬碰**：WorkerCore 析构段错误是既有问题，选择隔离（不析构、documented leak）
   而不是在多 die 工作里顺手引爆。
5. **好的抽象让泛化"免费"**：`dest/GRID_X == 全局行 == lane` 这个恒等式，让 HOST lane 从 die0 泛化
   到任意 die/布局几乎零额外逻辑。

## 6. 最终状态（含 V0b-2B1 之后的 V1-pre / V0-exit）

### 6.1 V0b 主体（截至 V0b-2B1，历史快照）
- **功能性目标全部达成并跨布局验证**：多 die 实例化（2×1/1×2/2×2）、**workload 在 die>0 运行**
  （die1@2×1/1×2、die3@2×2）、dual-die 并行零干扰、跨 die 启动期拒绝、单 die sim-time 精确一致。
- 当时测试：纯函数自测 **87/87**、端到端 **15/15**、noc_congestion 四场景一致。

### 6.2 V1-pre / V0-exit（Inc 1–4，V0b-2B1 之后）
- **配置驱动 HOST attachment**（原「非西侧 HOST」剩余项，已完成，不再是 polish）：
  - Inc 1–2a HOST 挂载表 `g_host_attach`（core↔lane↔tile 双向映射、`ValidateHostAttach`）；
  - Inc 2b `GetNextHop` 朝挂载 tile 收敛 + 显式拒绝跨 die HOST；
  - Inc 3a config 驱动挂载表构造器（`role=HOST` 端口）；Inc 3b-1 RouterUnit HOST 接口
    `IsHostAttachTile`；Inc 3b-2 **egress anchor 取自消息 `source_`**（非 `pos`）；
  - Inc 3c **W/S/E/N 四方向 × 方阵 4×4 + 矩形 4×2** die 内 mesh 端到端（mismatch=0、`(source,tag)`
    严格签名 + ACK 结构 + per-lane 分布 + 两次确定）。
- **修复**：CONFIG ACK tag 未初始化 UB（`Recv_prim(RECV_CONF)`）→ 类内默认值 + 显式 tag=0 契约。
- **Inc 4 基线冻结**：三道自动阻塞门（自测/端到端/NoC 精确数值）+ `run_v0_exit.py` 汇总。
- **最终测试数值（以此为准）**：纯函数自测 **165/165**、端到端 **23/23**、
  noc_congestion 四场景 **精确不变**（no_congestion 14781/29109、congestion 14833/45441）。

### 6.3 V0 基线冻结与发布记录（2026-07-21）

V0 没有直接在 `main` 上开发和打 tag，而是先在功能分支完成提交、验证后再 fast-forward 合并：

1. `feat/d2d-v0-baseline` 上形成 V0 功能提交
   `d260b609b6ddf68ee0c06d1831b4958efe3b9c00`
   （`feat(d2d): add V0 multi-die and config-driven HOST support`）。
2. 在同一分支追加冻结门提交
   `b6334397e43f704739692fcdc9e9ca450a469320`
   （`test(d2d): freeze V0 regression baseline`），其中包含 `run_v0_exit.py`、NoC 精确数值断言和
   本开发日志的冻结状态更新。
3. 确认本地 `main == origin/main` 且工作树干净后，以 **fast-forward** 将上述两个提交并入
   `main`；没有 squash/rebase，因此冻结提交的身份保持不变。
4. 在 `main` 上重新执行统一准入门 `python3 llm/test/run_v0_exit.py`，结果为：
   - D2D 纯函数自测 **165/165**；
   - D2D 端到端 runner **23/23**；
   - NoC 四场景 **4/4** 精确命中冻结值
     （no_congestion 14781/29109、congestion 14833/45441）；
   - 汇总退出码为 **0**。
5. 创建并推送 annotated tag **`d2d-v0-baseline`**。tag 的 peeled commit 精确为
   `b6334397e43f704739692fcdc9e9ca450a469320`；tag 注释同时记录 165/165、23/23 和 NoC 4/4。
6. tag 创建后，在 README 回填完整 tag/SHA，形成仅含文档变更的提交
   `1ca02e27efa2a6fefbf3b1148bde8de2dfd7e948`
   （`docs(d2d): backfill V0 baseline tag/SHA in README`）。tag 不包含这个后续提交是有意设计：
   提交无法记录自身尚未产生的 SHA，而真正的代码和阻塞门均已包含在冻结点中。

发布后核验状态：

- `origin/main` 指向 `1ca02e2`，本地 `main` 与其一致；
- `origin/feat/d2d-v0-baseline` 指向冻结提交 `b633439`；
- 远端 `refs/tags/d2d-v0-baseline` 存在，且解引用后指向 `b633439`；
- 重新运行冻结门后工作树仍为 clean，`git diff --check` 无错误；
- 可用 `git checkout d2d-v0-baseline` 精确恢复 V0 冻结代码与测试；该 tag 此后应保持不可变，
  V1 开发必须从新的功能分支继续。

这里的“已发布”仅指推送到个人 fork `origin`（`llljxjlll/WaferAI-SIM`）。当时
`upstream`（`IPADS-SAI/WaferAI-SIM`）仍停留在冻结前基线 `7940afa`，**尚未合入 V0**；若需要进入
上游，应另外创建 PR。上游若采用 squash/rebase，产生的新提交 SHA 不会改变本 fork 中
`d2d-v0-baseline` 的含义，也不应移动或重打该 tag。

### 6.4 真正剩余（纯 polish，不影响功能，不在 V0 冻结范围）
- ASan/UBSan 干净性、完整 WorkerCore teardown、pd/gpu 等非 dataflow 模式多 die 化、
  ACK 逐消息唯一序号（`ack_seq`）、mem-role at-edge 真实内存流量。

## 7. 主要改动文件

| 层 | 文件 |
| --- | --- |
| 编址/常量 | `include/defs/spec.h`、`src/defs/spec.cpp`、`src/utils/config_utils.cpp` |
| 编址 helper/路由 | `include/utils/router_utils.h`、`src/utils/router_utils.cpp` |
| 新模块 die/ | `include/die/port.h`、`src/die/port_config.cpp`、`src/die/v0_selftest.cpp` |
| 新模块 归一化/信封 | `include/monitor/workload_normalize.h`+`.cpp`、`include/monitor/host_envelope.h`+`.cpp` |
| 运行时多 die | `src/monitor/monitor.cpp`、`src/router/router.cpp`、`src/utils/system_utils.cpp` |
| HOST/config 分发 | `src/monitor/mem_interface.cpp`、`src/monitor/config_helper_core.cpp` |
| 消息/id 审计 | `include/common/msg.h`、`src/common/config.cpp`、`src/workercore/logic.cpp`、`src/workercore/workercore.cpp` |
| 绘图健壮化 | `src/utils/display_utils.cpp` |
| 测试 | `llm/test/d2d_link/`（`run_test_d2d_v0.py` + hardware 配置 + README） |

---
_下一阶段：V1（相邻 die 单链路 MVP）——第一条真实 NoC → D2D Link → NoC 通信。_
