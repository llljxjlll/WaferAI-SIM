# 分布式 HBM R0 开发记录

日期：2026-08-04
状态：配置、物理拓扑和地址映射契约已完成

## 1. 阶段范围

R0 建立 `memory_system` 的唯一配置真源、per-die HBM attachment、地址映射与
resolved memspec 校验，不改变生产访存路径。未配置 `memory_system` 时保持原每核
`DCache/DRAMSys` 的 `legacy_private` 行为。

## 2. 主要实现

- 新增 `HBMProfile/HBMStackConfig/HBMChannelConfig`，显式区分 compute die、stack、
  channel、pseudo-channel、MEM port 和 backend 粒度；
- 支持 `local_interleave`、`numa_local_interleave`、`global_interleave`；全局 UMA 使用
  “先 stack、后 stack 内 channel”的两级交织，避免按 channel 数错误加权 stack；
- `DecodeAddress(address,current_die)` 支持重叠的 per-die 本地地址空间；
  `ResolveMemRoute` 将地址归属与路径选择分离，并拒绝 local 模式的远端访问；
- 支持 channel port 和显式 aggregated port，校验 `channels_per_mem_port`、边缘跨度、
  HOST/C2C/MEM keep-out、corner 重复占用和 per-stack capacity；
- 解析 HBM2 resolved memspec，核对 generation、channel/pseudo-channel 数、实际数据
  总线宽度、`dataRate/tCK`、容量和带宽 cap。当前 DRAMSys 没有 HBM3 memspec 类，
  因此 HBM3 明确拒绝。

## 3. 评审修正

- 原 global 模式把 `(stack,channel)` 拍平；当 stack 的 channel 数不同，会给 channel
  更多的 stack 分配更多地址并破坏等容量前提，现已改为层次化交织；
- `backend_granularity` 是仿真实例粒度，不再被误作物理 port/PHY 数量；聚合端口必须
  显式声明，且聚合宽度不能超过物理 channel 数；
- 增加 leading gap、range overflow、逐 stack 等权容量和 pseudo-channel 粒度校验；
- topology/cache 字段未知值均拒绝，不能静默回退。

## 4. 测试结果

- `npusim --hbm-r0-selftest`：40/40；
- 覆盖合法单/多 die、NUMA/local/global UMA、确定性解码、聚合端口、pseudo-channel、
  range/容量/对齐、keep-out/corner、mislabeled memspec 和 legacy 复位；
- 已注册为 CTest `hbm_r0_selftest`。

## 5. 已知边界

R0 的 `MemRouteTable` 只给出本地 MEM anchor 或下一跳 C2C port，不发送真实 NoC flit。
生产 Router/WorkerCore 接入属于 R4。
