# S6 开发记录：旧路径迁移与收口

日期：2026-08-05
状态：完成（最终生产验收）

## 开发

- real-data 模式下，`sram_first_write_generic` 通过阻塞 LSU load 搬运真实 payload。
- `sram_spill_back_generic` 接收标签对应的 SRAM 源位置，并通过 LSU store 写回真实 HBM 地址。
- 旧 SRAM read/append/temp helper 在 real-data 模式下进入 `SramAccessUnit`；关闭扩展时保留原实现。
- `Send_prim` 的通信 payload SRAM read 进入统一访问单元。
- 空壳 `Load_prim/Store_prim` 改为阻塞 `CoreLsuUnit` 兼容别名，`size=0` 保持历史 no-op。
- real-data 模式不再分配第二份 `temp_ram_array`；旧 temp 端口绑定到同一单行 compatibility shim，功能容量只由 `SramStorage` 计入。
- 默认 8x8 配置显式加入三个默认关闭开关；新增使用说明。

## 测试

- S6 非零 pattern 通过 `Load_prim` 完成 HBM→SRAM，再通过 `Store_prim` 完成 SRAM→HBM。
- 两个兼容原语 wire round-trip 和空描述符 no-op 通过。
- LSU 的 HBM/SRAM read/write byte 统计完全平衡，outstanding token 为零。
- S0-S6 共 7 个 CTest 全部通过。
- HBM R0-R4 共 5 个 CTest 全部通过。
- DTE V0/V3/V3b/V4 全部通过。

## 评审

- 编译评审发现 `Send_prim` 缺少硬件配置查询声明。
- 测试夹具评审发现 `TaskCoreContext` 没有默认构造。
- 静态旁路审查确认：real-data SRAM helper 均先走统一入口；保留的直接 HBM helper 属于 GPU 纯 HBM 路径或扩展关闭时的兼容 fallback。
- `git diff --check` 通过。

## 修改

- 为 `Send_prim` 补充 `system_utils.h`。
- 测试改用最小合法 `TaskCoreContext` 构造，避免依赖未初始化字段。
- 清理补丁格式并使用 Perl JSON parser 验证默认配置。

## 兼容边界

- 真实 SRAM 数据面当前要求 `distributed_hbm`；`legacy_private` 继续使用旧路径。
- 旧标签 helper 只能表达旧线性 SRAM 地址。使用手工 region 的新 workload 应通过 `Lsu_mem`/`Dte_async` 或 C++ region API 指定 region 名。
- 旧详细 SRAM module 在 real-data 模式只保留一行端口绑定 shim，用于满足既有 SystemC socket elaboration；它不再提供额外功能容量。

## 结论

S6 第一版收口完成：扩展关闭时旧路径不变，扩展开启时 DTE、LSU、compute、旧 helper 和通信 read 共享同一功能 SRAM 与统一仲裁入口。

## 2026-08-06 复审修订

### 开发

- non-manager spill 不再写固定地址 1024；标签保留已有 backing，缺失时通过 DramKVTable 分配真实地址。
- victim 选择排除 non-spillable region。
- `Clear_sram` 在 real-data 模式通过 `SramAccessUnit::kClear` 清 valid，保护 non-spillable 与 layer/persistent 标签，并释放 task allocation。
- 输入、静态数据与输出标签记录真实 HBM backing；新增四类生产 Event_engine trace。

### 测试

- R1 覆盖 lifetime/free/signature；R6 覆盖 NpuBase manual path 与 task clear、通信 buffer、layer cache。
- SRAM R0-R6、HBM R0-R4、DTE V0/V3/V3b/V4 均通过；`git diff --check` 在最终提交前执行。

### 评审

- 原 S6 结论忽略固定 spill 地址、Clear_sram 未清 storage、标签无 region/lifetime 以及 trace 缺失。

### 修改

- 撤销“S6 已完成”结论。当前旧标签核心契约已迁移，但完整 WorkerCore workload fixture、`legacy_private` real-data backend 与所有旧 helper 的逐一旁路审计仍开放。

## 当前状态

状态：部分完成，不得作为最终全闭环验收。


## 第二轮生产路径复审修改（2026-08-06）

### 开发

- 补齐 `SRAM_read/write/bank_wait`、LSU 四阶段、DTE 四阶段、`SRAM_region_alloc/free/spill/reload` 与带 region-relative offset 的 `Compute_tile` trace。
- WorkerCore 把生产 Event_engine 传入 RegionTable，allocation/free 生命周期不再只存在于组件内部。

### 测试

- `cmake --build build -j2` 通过。环境不提供 ctest，直接运行 npusim 自测入口。
- SRAM R0-R6 全部 PASS；HBM R0-R4 为 40/40、18/18、18/18、18/18、9/9；DTE V0/V3/V3b/V4 为 64/64、31/31、21/21、19/19。
- `git diff --check` 在最终文档修改后再次执行。

### 评审

- 上轮只有四个粗粒度事件，无法拆分 queue、bank、SRAM port、HBM/AXI 和 wait 的贡献。

### 修改结论

- 计划第 15 节列出的 trace 名称均已有生产事件；完整 CLI WorkerCore workload 和 legacy_private real-data backend 仍保留为最终验收开放项。

## 最终生产闭环（2026-08-06）

### 开发

- `LegacyPrivateByteTransport` 让 real SRAM 在 `legacy_private` 下复用 DCache/DRAMSys 时序，并维护 byte payload；默认 DRAMSys `NoStorage` 不再阻断功能往返。
- production label binding 按 `preferred_region` 调用 `RegionTable::Allocate/Resize/Free`；compatibility word 地址只在边界换算，必要时经统一 AccessUnit 搬移 payload。
- input、intermediate、comm 三类标签由生产 pipeline 实际分配、非零读回和释放；label rename 同步 allocation 元数据。
- 旧 first-load/spill/read/append/temp/send helper 在 real path 进入 LSU/HBM transport 或统一 SRAM access，保留的 direct port 代码只在扩展关闭 fallback 可达。

### 测试

- `run_test_sram_pipeline.py`：behavioral WorkerCore、完整 staged trace 和三类 role lifecycle 通过。
- `run_test_sram_dramsys.py`：distributed DRAMSys LSU/DTE 非零 round-trip 通过。
- `run_test_sram_legacy.py`：legacy_private LSU/DTE 非零 round-trip 与 overlap 通过。
- `run_test_sram_numa.py`：core0 LSU、core16 DTE 都访问远端 home HBM，校验和 130560，D2D in/out=672/672。
- `run_test_sram_compat.py`：real-data/manual-schedule 关闭的旧 Relu_f/NpuBase/helper workload 正常完成，无 protocol watchdog。
- SRAM R0-R6、HBM R0-R4、DTE V0/V3/V3b/V4、D2D V0/link 均通过；构建和 `git diff --check` 通过。

### 评审

- 最后一次旁路审计确认 direct `ram_array/temp_ram_array` 调用均位于 `context.sram_access` 早返回之后；real path 只有一份计容量的 `SramStorage`。
- 发现 `changePairName` 只改 SramPosLocator，RegionTable 中的 allocation label 会过期。
- 原默认 `playground.json` 本身在 rendezvous 层触发 watchdog，不能作为 SRAM 兼容成功证据。

### 修改

- 增加 `RegionTable::RenameAllocation` 与 R6 覆盖。
- 新增最小、确定性的 compat workload，不把既有 playground 死锁错误归因或掩盖为 SRAM 结果。
- README、计划状态和最终验收记录同步更新。

### 结论

S6 的 legacy_private、生产标签、helper 旁路、trace 和扩展关闭兼容路径全部闭环。此前“部分完成”状态由本节和最终验收记录取代。
