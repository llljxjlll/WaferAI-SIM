# S3：LSU 真实 HBM↔SRAM 路径闭环记录

日期：2026-08-05
状态：完成

## 开发

- 新增 HbmByteTransport 与 CoreMemByteTransport，生产请求继续经过
  CoreMemAdapter、NoC/MEM endpoint 和所选 HBM backend。
- 新增 per-core CoreLsuUnit SystemC worker，支持 issue/wait/poll/fence/cancel
  以及 blocking load/store 包装。
- HBM→SRAM 在 HBM payload 返回后经统一 AccessUnit 提交；SRAM→HBM 先经
  AccessUnit 检查 valid 并读取 payload，再等待 HBM write response。
- WorkerCore 在 real_data_path=true 且 distributed_hbm 下实例化 storage、
  access、transport 和 LSU；默认关闭时不分配新 payload SRAM，不改变旧路径。
- TaskCoreContext 暴露 lsu_memory、sram_regions、sram_access、sram_storage。
- 新增 Lsu_mem 原语，支持命名 region、absolute address、logical token、
  JSON/wire、issue/wait/poll/fence/cancel 与 blocking 简写。

## 测试

- sram_r3_selftest：
  - 73-byte 非零 pattern 执行 HBM→SRAM→compute 读取；
  - compute 改写后执行 SRAM→HBM 并逐 byte 对比；
  - token 不早于 HBM response 和 SRAM commit 完成；
  - invalid SRAM store 以 failed token 返回；
  - LSU/HBM/SRAM 四组 byte 统计守恒；
  - Lsu_mem 命名 region JSON→wire→decode 无损。
- sram_r0-r3 与 hbm_r0-r4 共 9 项 CTest 全通过。

## 评审

- logical token 最初仅由 engine 自动生成，无法让 workload 后续原语引用。
- 评审要求命名 region 必须进入 wire，不能只保存在 parseJson 的宿主对象中。
- 当前 LSU 为单 worker FIFO；允许多 outstanding 和 compute overlap，但同一
  LSU engine 内的多个 HBM descriptor 暂不并行服务。

## 修改

- Issue 增加 requested_token，并校验 outstanding token 唯一性；token 0 保留
  给 C++ API 自动分配。
- Lsu_mem wire 增加最多 64-byte region 名字编码和严格 segment 长度校验。
- core primitive queue 清空时若仍有 LSU token，显式报错要求 wait/fence。

## 结论

S3 验收通过。计算核 LSU 已具备可从 workload 和计算 primitive 内调用的真实
SRAM↔HBM 数据交换能力；兼容开关默认关闭。

## 2026-08-06 复审修订

### 开发

- LSU 使用独立 queue depth、max outstanding、issue latency，并按配置创建多个 SystemC worker。
- 增加 peak running/outstanding、issue latency 统计和 `LSU_hbm` trace。

### 测试

- R3 同时发起两个独立 load，验证 `peak_running=2`、第三个 descriptor credit 拒绝、非零 payload 与 trace。
- HBM R2 behavioral、R3 DRAMSys、R4 Router/NUMA 均通过。

### 评审

- 原“单 worker 也算 multiple outstanding”的结论不成立，无法表达无 hazard descriptor 并发。

### 修改

- WorkerCore 不再复用 SRAM queue depth；按核传入冻结的 LSU 配置。


## 第二轮生产路径复审修改（2026-08-06）

### 开发

- LSU `queue_depth` 现在限制可保留 descriptor/token 数，`max_outstanding` 仅决定 worker 数和最大并行执行数。
- 增加 `LSU_issue`、`LSU_hbm`、`LSU_sram`、`LSU_wait` B/E trace，异常路径闭合已开始的阶段。

### 测试

- R3 在 queue_depth=8、max_outstanding=2 下成功接收八笔 load，第九笔才拒绝；peak outstanding=8、peak running=2。
- R3 随后完成 store 和 invalid-read 负例，token、lease、统计全部排空。

### 评审

- 旧实现用 records 数直接套 max_outstanding，使六个排队槽永远不可用，且旧测试把第三笔拒绝误当正确。

### 修改结论

- admission capacity 和 execution concurrency 已解耦，示例中的 8/2 配置现在具有预期语义。
