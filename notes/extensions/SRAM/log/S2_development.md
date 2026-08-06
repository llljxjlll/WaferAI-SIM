# S2：统一 SRAM 端口与仲裁闭环记录

日期：2026-08-05
状态：完成

## 开发

- 新增 per-core SystemC AccessUnit，compute/DTE/LSU/NoC RX/legacy 使用统一入口。
- issue 时执行 region/权限/长度校验；完成时才提交 write 或返回 read payload。
- 实现 per-initiator read/write port、bank mapping、queue backpressure。
- 实现公共 RAW/WAR/WAW range hazard；read/read 不产生 range hazard。
- 导出 initiator/command、bank、queue wait、service 和 stall 分类统计。

## 测试

- sram_r2_selftest 的时序 oracle：
  - 32 B、128-bit/cycle、base=1 cycle，单请求服务 3 cycles；
  - 同 bank 两请求总计 6 cycles；
  - 异 bank、双端口两请求总计 3 cycles。
- DTE write 后并发 LSU read 被 RAW 阻塞，并读到提交后的 0xa5 pattern。
- sram_r0_selftest 至 sram_r2_selftest CTest 全通过。

## 评审

- 首次编译发现周期宏头文件错误，已改用 macros/macros.h。
- 当前 bank 模型保守地让请求在整个服务期占用所有触及 bank；功能和下界
  正确，但跨 bank 大请求尚未按 beat 流水化。
- 存储异常路径会释放 active/queue slot 并通知 waiter，不遗留死锁资源。

## 修改

- 修正周期宏 include 并重新运行完整阶段回归。
- 保留逐 beat 流水化为性能精化项；本阶段不改变已冻结的功能语义。

## 结论

S2 验收通过。后续 LSU 和 DTE 数据面必须经该统一入口，不另行估算 SRAM wait。

## 2026-08-06 P0 复审修订

### 开发

- 修复 lease 顺序：持有 descriptor lease 的实际请求忽略所有 hazard-only reservation，lease 间依赖继续保证 issue 顺序。
- 改为按端口宽度与 bank stripe 拆 beat；固定 5 initiator RR、initiator 内 FIFO；bank/port 逐 beat 获取释放。
- 增加逐 bank/port beats、bytes、service、stall 和 `SRAM_queue` trace。

### 测试

- R2 覆盖同 bank 串行、异 bank 并行、RAW、beat 统计和 trace。
- R5 连续写同 slot 能完成并真实到达 `sc_stop()`。

### 评审

- 原先“整个请求占全部 bank”的模型不满足冻结配置；后发 lease 会反向阻塞前发请求形成循环等待。

### 修改

- 撤销原文“保留逐 beat 为后续项”的结论；当前 S2 已按冻结配置实现并回归。
