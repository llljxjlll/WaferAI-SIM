# 10×10 内任意矩形 Die Mesh 开发落地报告

日期：2026-08-23

## 结论

本轮已经完成可独立合入的矩形 Mesh 基础、跨 Die collective、任意 rank
AG/RS/AR executable baseline、Wang 大规模有界生成，以及矩形 MeshSlice
AG_GEMM slice=1 的 production lower/link 链。

主 compiler 已公开 `compile_rect_mesh`，能够验证全部 100 种 `H×W` 参数，
AUTO 会确定性回退到现有 whole-workload NAIVE 单 manifest，并通过 typed
capability report 明确说明原因。

当前不能标记 `flexible_mesh_dense_complete=true`。主要缺口是：Swizzle
standard region 尚未合并进主 workload artifact；100 种尺寸尚未逐一经过
finalizer、ProgramIO 和 npusim 真实运行。

## 分阶段状态

| 阶段 | 本轮状态 | 已完成 | 尚缺门禁 |
|---|---|---|---|
| F0 能力契约 | 已实现 | `RectMeshSpec`、compile mode/fallback/capacity report、timing-only/Dense-first fail-closed | 最终 runtime report 仍无真实测量 |
| F1 Fabric/placement/topology | 已完成单元闭环 | 100-shape sweep；row-major/X-first；共享 O(R) rows/columns/snake/cycle；删除 DFS | 无 |
| F2 跨 Die collective | 代码和 C++ 自测完成 | 跨 Die group；cyclic-delta waves；N=100 精确 derived-byte preflight；每核每波 2 sessions≤3 | 真实 npusim residual/重复性矩阵 |
| F3 executable baseline | lower/link 闭环 | 任意 R 的 AG、RS；R>2 AR=RS+AG；R=2 和 R=4 兼容路径保留 | 100-shape finalizer/ProgramIO/npusim canary |
| F4 主 Dense compiler | 参数桥和安全 fallback 完成 | `compile_rect_mesh`；whole-workload NAIVE 单 manifest；AUTO typed fallback；1M record/64MiB 门禁载体 | standard Swizzle region replacement、finalizer 精确文件字节、代表尺寸真实运行 |
| F5 Wang scale | 生成/预检闭环 | R>16 bounded profile；构建前 action/buffer/SRAM/send-byte 预检；R≤16 顺序不变 | 大 R runtime evidence、latency/RSS 正式报告 |
| F6 矩形 MeshSlice | AG_GEMM slice=1 lower/link 闭环 | 通用 placement/audit/flow；peer 子视图和地址 addend；2×3、3×2 manifest；2×10、10×2、5×6、6×5、10×10 candidate evidence | 任意矩形 RS/AR、多 slice runtime evidence |
| F7 后续能力 | 未开始，明确 out of scope | capability report 明确不越界声明 | training、MoE、functional、子矩形、真实 barrier timing、非矩形 |

## 关键实现

### 矩形基础

- `RectMeshSpec` 固定 `1≤H,W≤10`、`R≤100`、一 Die 一 rank、row-major、
  X-first、timing-only、Dense-first。
- 共享 `RectMeshTopology` 以 O(R) 构造行、列、snake 和 Hamiltonian cycle。
- 100 种尺寸均动态验证，不提交 100 份 golden。
- 10×10 指标：100 dies、360 directed links、9,900 ordered routes、max hop 18。

### Runtime collective

- loader、sync runtime 和 collective graph 接受合法跨 Die core group，同时保留
  active、sorted、unique、range 和 U16 校验。
- symmetric collective 采用 deterministic cyclic-delta waves；rooted/P2P
  保持旧顺序。
- N=100 AllGather 为 9,900 children、99 waves，每核每波一次 send 和一次
  receive，即 2 sessions，低于生产容量 3。
- planner derived-state 硬上限与 64 MiB program file ceiling 对齐，继续做
  精确构建前校验。

### Executable baseline

- R=4 保留 XOR wave；其他 R>2 使用 cyclic offset。
- AG/RS 覆盖全部 `R(R-1)` ordered pairs。
- R>2 AllReduce 使用 `RS(R-1 waves)+AG(R-1 waves)`；连续 typed storage 为
  `(R+1)×chunk_bytes`，没有放宽 REDUCE contiguous-pair ABI。
- R>2 completion 使用确定性星型 event barrier；exact-2 行为保持不变。

### Wang

- R>16 只保留最小合法 `chunk=R`、`unroll=1`、line 优先/ring 后置，最多
  两个候选。
- R=100 AG 预估 49,700 actions；AR 预估 79,400 actions；预算不足时在
  action builder 前返回无候选。
- production `max_chunk_count=100`、`max_actions=80,000`，仍是显式硬上限。

### MeshSlice

- placement 和 standard audit 从 topology 派生真实 `rows/columns`。
- 通用 flow：

      row_flows    = slices × R × (W-1)
      column_flows = slices × R × (H-1)

- 多 peer RECV 写入同一聚合 operand 时，每个 peer 使用不重叠 typed subview；
  DTE relocation addend 携带实际 byte offset，MATMUL 继续读取完整聚合视图。
- exact-2×2 保留旧完整 view/零 addend fast path，避免旧 stable artifact ID 漂移。
- 构建前使用精确 action/buffer 估算；10×10、slice=1 为 5,500 actions。

### 主 compiler

- `compile_rect_mesh` 在任何编译 pass 前验证 fabric、TP、placement、静态 Dense
  inference 和 timing-only 合约。
- `NAIVE` 复用现有 whole-workload compile；`AUTO` 当前选择相同可执行链并报告
  `standard_chain_unavailable`。
- forced `STANDARD` 在编译前 typed fail，不产生半成品 artifact。
- 当前 AUTO 的单 manifest 已验证同时包含 coarse、ISA fusion 和 standalone
  collective fragments。

## 验证结果

- 最终组合 Python 回归：80 tests，通过。
  - RectMesh/compiler/harness/fabric/topology/Wang：46 tests；
  - arbitrary-rank unfused comparison + standard lower/link：20 tests；
  - MeshSlice standard + cost：14 tests。
- 额外旧 compiler policy 回归：4 tests，通过。
- C++：`isa_v1_selftest`、`sync_runtime_selftest`，2/2 通过。
- 全 frontend `compileall` 通过。
- `git diff --check` 通过。

本轮没有产生以下证据，因此 capability report 必须保持 `not_measured` 或
`false`：

- 100-shape finalizer/resolver/npusim 结果；
- ProgramIO pass=1 和 runtime residual 全归零；
- 重复运行的 artifact SHA、makespan、marker digest；
- 完整 Dense standard-chain performance benefit；
- functional correctness。

## 后续最短路径

1. 在主 compiler 中实现 standard Swizzle region replacement，并与 ordinary、
   ISA fusion、standalone fragments 合并成一个 manifest。
2. 先跑 1×1、1×10、10×1、2×3、3×2、3×3、10×10 的 NAIVE/AUTO
   finalizer+ProgramIO+npusim；闭合后再扩到 100-shape tiny canary。
3. 将 finalizer 的精确 artifact byte count 回填 capability report，验证
   `≤64 MiB` 和 `≤1M records`。
4. 补 MeshSlice 5×6、6×5、10×10 的 standard lower/link 与 runtime evidence，
   再单独推进矩形 RS/AR 和多 slice。
5. 采集 Wang R=30/90/100 的 prepare latency、RSS 和 runtime fallback evidence。

