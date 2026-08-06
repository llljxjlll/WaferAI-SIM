# S0：契约与配置闭环记录

日期：2026-08-05
状态：完成

## 开发

- 新增 byte 地址、半开区间、溢出安全的 range/alignment 类型。
- 新增 SRAM 全局配置与 cores[].sram/lsu 分层覆盖解析。
- 配置包含容量、bank/interleave、端口数与宽度、延迟、队列、兼容开关和 region。
- 旧的 memory.sram_size 自动生成覆盖全容量的 legacy block region。
- ParseHardwareConfig() 在 SystemC elaboration 前配置 per-core registry。

## 测试

- sram_r0_selftest：合法配置、旧配置、overlap、越界、错位、未知 initiator、per-core override。
- 构建：cmake --build build -j2，通过。
- CTest：sram_r0_selftest，通过。

## 评审

- 确认所有容量、地址、offset 和传输长度均为 byte。
- 复审确认 per-core override 应继承未显式覆盖的全局 bank/port 配置。
- 发现正式配置入口最初未调用 registry。

## 修改

- 将 registry 接入 ParseHardwareConfig()；非法 region 在构造 WorkerCore 前失败。
- 保持 real_data_path=false、manual_regions=false 的默认兼容门禁。

## 结论

S0 验收通过。配置契约可独立测试，并已进入生产配置解析路径。

## 2026-08-06 复审修订

### 开发

- 增加独立 `lsu.queue_depth/max_outstanding/issue_latency_ns` 与 `sram.dte_memory.workers/queue_depth`。
- per-core SRAM override 改为在全局 memory 配置上深合并，并支持 `cores[].lsu`。

### 测试

- R0 配置负例与全量构建通过。

### 评审

- 原实现错误地用 SRAM queue depth 代替 LSU 配置；per-core override 会丢失继承的 bank/port 字段。

### 修改

- 拆分配置、增加非零与 `max_outstanding <= queue_depth` 校验，并在 WorkerCore 按核接线。
