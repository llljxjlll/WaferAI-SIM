# SRAM 扩展复审记录

日期：2026-08-06

## 结论

此前“ S0-S6 已完成”的结论无效并已撤回。本轮已修复确定性 lease 死锁与 R5 假阳性，完成 beat/RR 仲裁、独立 LSU/DTE 并发配置、真实 SPM_TO_SPM、NpuBase manual schedule、region-aware lifetime/spill/clear，以及完整的 SRAM/LSU/DTE/region/compute 分阶段 trace。

当前判定：S0-S4 代码与组件验收闭环；S5 已完成 NpuBase 原语内调度和 double-buffer 组件验收；S6 仍为部分完成。未新增完整外部 WorkerCore workload fixture，也未给 `legacy_private` 增加 real-data backend，因此不能宣称计划最终验收条件全部满足。

## 本轮关键修改

- lease：实际请求持有 lease 时不被后发 reservation 反向阻塞；lease 依赖继续维持 WAW/RAW/WAR 顺序。
- R5：完成标志加 watchdog，必须真实执行到 `sc_stop()`。
- SRAM：按 port width/bank stripe 拆 beat，固定 RR，逐 bank/port 统计。
- LSU：`queue_depth` 控制可保留 descriptor，`max_outstanding` 控制并行 worker，并提供 issue/HBM/SRAM/wait trace。
- DTE：多 worker、严格长度、真实 `SPM_TO_SPM`、reserve/commit/abort 队列事务与 HBM/AXI/SPM/commit trace。
- lifecycle：legacy 标签 word 地址在统一边界换算为 byte；生产标签通过 AllocateAt/Resize 绑定 region allocation；clear 回写 word 游标并保护 non-spillable/layer/persistent。
- production：WorkerCore 绑定完整分阶段 trace，NpuBase manual 模式跳过自动装载、删标签和粗粒度输出写回。

## 有效回归证据

- build：`cmake --build build -j2` 通过。
- SRAM R0-R6：7/7 通过；R2-R6 均真实打印 `Simulation stopped by user`。
- HBM R0-R4：40/40、18/18、18/18、18/18、9/9；覆盖 behavioral、DRAMSys 和 Router/NUMA。
- DTE V0/V3/V3b/V4：64/64、31/31、21/21、19/19。
- R3：queue_depth=8 时可保留八个 descriptor、两个 worker 并行运行，第九个才因队列容量被拒绝。
- R4：SPM_TO_SPM、本地 copy、严格长度、两个 worker，以及 bridge 满时不创建第二个 control token/record 且 credit 完全回收。
- R6：512-bit SRAM 下 pos=1 映射 byte 64；生产标签获得 region allocation；Clear_sram 释放 task allocation、保留 non-spillable/layer 数据并把 byte high-water 换回 word 游标。

## 开放项

1. 增加可由 CLI 运行的完整 WorkerCore real-SRAM workload fixture，并校验 events.json 中四类 span 的真实重叠。
2. 决定并实现 `legacy_private` real-data HBM transport，或正式把它移出本计划最终条件。
3. 对全部旧 helper 做逐项旁路审计，并完成 communication/input/output/intermediate 标签的生产 workload 迁移。
4. 在开放项完成前，计划状态保持“基础实现完成，S6 未最终验收”。

## 最终关闭补充（2026-08-06）

上述四个 SRAM 生产开放项已全部关闭：CLI WorkerCore fixture、legacy_private real-data backend、production role-label binding、helper 旁路审计和完整 trace oracle 均已落地。全计划只保留独立的默认 workload 基线条件 7。最终证据与评审修改见 `FINAL_acceptance_2026-08-06.md`；本补充取代本文件早先的“S6 部分完成”状态，但保留原文作为评审历史。
