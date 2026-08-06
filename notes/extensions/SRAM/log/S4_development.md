# S4 开发记录：DTE 真实 HBM↔SRAM 路径

日期：2026-08-05
状态：完成

## 开发

- 新增 `DteMemoryBridge`，实现 `DRAM_TO_SPM` 和 `SPM_TO_DRAM` 的真实 payload 搬运。
- `DteAsyncTracker` 可绑定 bridge；wait/poll/fence/cancel 的完成条件同时覆盖 endpoint 与数据面。
- `Dte_async` 增加 `hbm_addr`、`sram_region`、`sram_offset`，保留旧两段/三段 wire。
- 每核 WorkerCore 在 real-data 模式下绑定 DTE bridge，并向 `TaskCoreContext` 暴露 DTE memory API。

## 测试

- 非零 pattern 完成 HBM→SRAM→HBM round-trip。
- token 不早于 HBM response 与 SRAM commit。
- bridge 未绑定时 SRAM 不变化，验证 V4 endpoint-only 兼容门禁。
- HBM/SRAM byte 统计逐笔平衡。
- DTE V0/V3/V3b/V4 回归分别通过 64、31、21、19 项检查。

## 评审

评审发现原有 hazard 只覆盖实际 SRAM request 生命周期；异步 descriptor 在 HBM 等待期间尚未进入 SRAM 队列，计算/LSU 可能抢先读取目标范围。

## 修改

在 DTE/LSU issue 时同步声明 descriptor-lifetime range lease，worker 提交访问时携带自身 lease，完成、失败或取消时统一释放。新增“DTE issue 后立即 LSU read”的 RAW 用例，确认 read 等待 commit 后得到新 payload。

## 结论

S4 验收通过。关闭 real-memory bridge 时严格保留旧 DTE 行为。

## 2026-08-06 复审修订

### 开发

- `DteMemoryBridge` 新增真实 `SPM_TO_SPM`：源 read lease 与目标 write lease 使用同一 group，避免自依赖。
- bridge 支持可配置多 worker/有界队列；real-memory 强制 byte 对齐且 payload bytes 等于 `spm_size`。
- 增加 copy/concurrency 统计和 `DTE_mem_commit` trace。

### 测试

- R4 覆盖真实 SRAM copy、长度不等负例、两个并行 DRAM_TO_SPM 和完整 token 生命周期。
- DTE V0/V3/V3b/V4 分别通过 64/64、31/31、21/21、19/19。

### 评审

- 原 bridge 只有两个 HBM 方向且单 worker，`payload_bytes <= spm_size` 会允许部分区间语义不一致。

### 修改

- wait/poll/release/cancel/hazard completion 的所有 memory-direction 判断均纳入 `SPM_TO_SPM`。


## 第二轮生产路径复审修改（2026-08-06）

### 开发

- DteMemoryBridge 增加 `Reserve/Commit/Abort` 两阶段提交；DteAsync 在创建 control record 和占用 DTE credit 前先预留 memory queue。
- 增加 `DTE_mem_hbm`、`DTE_mem_axi`、`DTE_mem_spm`、`DTE_mem_commit` 分阶段 trace，异常路径闭合。

### 测试

- R4 新增 queue_depth=1 bridge：第一笔保留队列，第二笔失败时 tracker/bridge 均只有第一笔记录；等待后 record、pending、active 和 credit 全部归零。
- SPM_TO_SPM、两 worker 并行、长度负例和 byte 统计继续通过。

### 评审

- 旧提交顺序先建立 DTE control context、后调用 bridge Issue；bridge 满异常可能遗留 token 和 credit。

### 修改结论

- bridge 容量失败现在发生在控制面变更之前，后续异常通过 Abort 释放 reservation 和 range lease。
