# S5 开发记录：原语内手动调度与双缓冲

日期：2026-08-05
状态：完成

## 开发

- 新增 `ComputeTimeline`，计算读请求通过统一 `SramAccessUnit`，随后执行指定 compute cycles。
- `TaskCoreContext` 暴露 `compute_timeline`，计算原语可直接组合 LSU/DTE issue、selective wait 和计算。
- 配置新增 `manual_memory_schedule`。
- R5 synthetic tiled workload 使用 `double_a/double_b` 两个手工 region，执行预热、稳态和排空。

## 测试

- 4 tile 阻塞版本满足 `N*(Tload+Tcompute)` oracle。
- 双缓冲版本满足 `Tload+N*max(Tload,Tcompute)` oracle，且短于阻塞版本。
- 非零 tile payload 在所有 slot 中保持正确。
- 增补 DTE load + compute、LSU load + compute + store、DTE store。
- 同一 slot 两次 load 的 WAW 被串行化，最终 payload 按 issue 顺序提交。
- `sram_r5_selftest` 通过。

## 评审

首版只验证 LSU load+compute，缺少 DTE、store 和错误调度覆盖。

## 修改

补齐 DTE 路径、两种 store、配置开关可见性和单 buffer 覆盖用例；确认 DTE/LSU/compute 使用同一 bank/port/hazard 时间线。

## 结论

S5 验收通过。手动调度是 callable API，不依赖独立 load primitive，能够在一个计算 primitive 的 tile 循环内表达真实重叠。

## 2026-08-06 P0/P1 复审修订

### 开发

- R5 增加 `finished` 标志与 10 us watchdog，只有 Run 线程完成并调用 `sc_stop()` 才可 PASS。
- `NpuBase` 接入 `manual_memory_schedule`：跳过自动输入装载、输入标签删除和粗粒度输出写回。
- `ComputeTimeline` 增加结构化 trace 与 `Compute_tile` Event_engine B/E。

### 测试

- R5 double-buffer、同 slot WAW、DTE/LSU/compute/store 全部完成；R6 的 ManualSchedulePrim 在一个 NpuBase primitive 内执行 load/compute/store。

### 评审

- 原 R5 在 SystemC 无事件时会提前退出并假 PASS；配置只解析未进入 NpuBase。

### 修改

- 旧的 R5 PASS 证据作废，以带完成标志的新回归结果替代。完整外部 WorkerCore workload fixture 仍作为 S6 开放项记录。

## 最终生产闭环（2026-08-06）

### 开发

- 新增注册的 `Sram_pipeline` synthetic compute primitive；同一个 primitive 的 tile 循环可选择 LSU 或 DTE，执行 blocking 或预热/稳态/排空 double buffer。
- 原语通过选定的生产 HBM transport 写入输入 pattern，再执行 load、`ComputeTimeline::RunTile`、XOR 计算、store 和 HBM 回读校验。
- `manual_memory_schedule` 的 NpuBase gate 保持启用，自动 input/output 管理不会与原语内调度重复收费。

### 测试

- CLI WorkerCore fixture 的 LSU blocking/double-buffer 为 2324/1765 ns，DTE 为 2320/1758 ns；四条路径校验和均为 130560。
- Python oracle 读取 `events.json`：blocking 无 compute/HBM overlap，double-buffer 的 LSU 与 DTE 均存在真实 overlap。
- R5 继续要求 `finished + sc_stop()`，保留同 slot WAW 和 watchdog 防假阳性。

### 评审

- 原记录只有组件级 `ManualSchedulePrim`，不能证明 workload parser、wire、WorkerCore 绑定和生产 trace sink 联合工作。

### 修改

- 以 `workload.json + run_test_sram_pipeline.py` 取代“组件测试可代表生产”的假设；脚本同时检查 marker 集合、非零数据、时序关系、trace 阶段与 B/E 平衡。

### 结论

S5 完成生产验收，不再仅是组件原型。
