# SRAM S5/S6 最终验收记录

日期：2026-08-06
状态：SRAM S5/S6 与全计划最终条件 1–8 全部通过

## 验收范围

本轮关闭此前评审留下的生产路径缺口：完整 WorkerCore 手工调度 workload、`legacy_private` real-data transport、远端 NUMA、production role-label binding、旧 helper 旁路审计、完整 trace 和扩展关闭兼容。

## 开发结果

- 新增 `Sram_pipeline` 原语，支持 `engine=lsu|dte`、`schedule=blocking|double_buffer`、tile 数量/大小、计算周期、HBM 地址和双 buffer region 配置。
- distributed behavioral、distributed DRAMSys、legacy_private 都实现 HBM↔SRAM 非零 payload 往返；NUMA 路径经过 Router/NoC/C2C 到远端 home die。
- production label probe 覆盖 `input`、`intermediate`、`comm` 的 allocate、compatibility payload relocation、readback、free；rename 与 RegionTable metadata 同步。
- `events.json` 可观察 SRAM queue/read/write/bank-wait、LSU issue/HBM/SRAM/wait、DTE HBM/AXI/SPM/commit、Compute tile、region alloc/free；spill/reload 由 R6 定向覆盖。

## 测试结果

| 测试 | 结果 |
|---|---|
| 构建 `cmake --build build -j2` | PASS |
| SRAM R0-R6 | 7/7 PASS |
| HBM R0-R4 | 40/40、18/18、18/18、18/18、9/9 PASS |
| DTE V0/V3/V3b/V4 | 64/64、31/31、21/21、19/19 PASS |
| D2D V0/link | 308/308、37/37 PASS |
| behavioral WorkerCore pipeline | checksum 130560；LSU 2324/1765 ns；DTE 2320/1758 ns；trace overlap PASS |
| distributed DRAMSys | LSU/DTE round-trip 与 double-buffer PASS |
| legacy_private | LSU/DTE round-trip 与 double-buffer PASS |
| two-die remote NUMA | core0 LSU/core16 DTE checksum 130560；D2D 672/672 PASS |
| real-data off compat | legacy Relu_f/NpuBase/helper 正常完成，无 watchdog |
| `git diff --check` | PASS |

说明：HBM 自测必须从 `build/` 运行，因为 DRAMSys 配置使用相对路径。曾从仓库根运行造成 `Unsupported DRAM type`，已按正确工作目录重跑，不计为产品失败。

## 评审与修改

1. NUMA 初版 HBM 挂在 N0，与 W 边角 C2C tile 冲突；改挂 N1，并将 D2D 配为承载 memory flit 的 cycle backend。
2. NUMA workload 缺少 `id_space=global` 且 `pipeline=2` 导致重复执行；分别改为 global 和单 pipeline。
3. 生产 pipeline 原只覆盖 fixed double-buffer region；增加 input/intermediate/comm role-label probe，并在性能计时区间外执行。
4. trace runner 原只检查 compute/HBM overlap；升级为全部阶段存在且 B/E 平衡，并校验三类 role lifecycle。
5. label rename 原未更新 RegionTable allocation label；新增 API 和 R6 回归。
6. 默认 playground 在既有 rendezvous 层 watchdog；新增最小 compat workload作为确定性扩展关闭证据，不伪称修复或通过 playground。

## 旁路审计结论

real-data 开启时：

- HBM→SRAM 与 SRAM→HBM 由 `CoreLsuUnit`/`DteMemoryBridge` 经 `HbmByteTransport` 完成；
- compute、legacy read/append/temp helper 和 Send read 经 `SramAccessUnit`；
- `ram_array/temp_ram_array` 的 direct port 代码只在统一入口早返回之后的兼容 fallback 中可达；
- `temp_ram_array` 不再创建第二份功能容量，real path 的容量单一来源为每核 `SramStorage`。

## 最终结论

S5 和 S6 已完成开发、测试、评审、修改与记录，四项 SRAM 能力均有代码与可重复专项测试证据。后续 P0/P1 收口已补齐规范默认配置、启动 preflight、S_DATA 六阶段完成握手、有限工作 watchdog lease 与静态 rendezvous 校验；无参数 smoke 和完整 playground 均通过。因此最终条件 1–8 全部关闭。详细复测与评审修改见 `P0_P1_default_startup_closure_2026-08-06.md`。工作区仍包含用户已有的大量 NoC/DTE 改动以及 SRAM 新文件未跟踪状态；本轮未执行提交或清理这些用户改动。
