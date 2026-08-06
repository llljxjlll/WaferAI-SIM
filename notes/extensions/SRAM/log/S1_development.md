# S1：SRAM storage 与 region 闭环记录

日期：2026-08-05
状态：完成

## 开发

- 实现 per-core Storage：byte payload、逐 byte valid、byte-enable、clear、signature。
- 实现 RegionTable：相对/绝对 resolve、权限、fixed/block、allocate/free、label。
- region 校验覆盖名字唯一、alignment、overlap、capacity 和完整落区。

## 测试

- sram_r1_selftest：非零 pattern、两核隔离、invalid read、部分写、clear、越界。
- 覆盖权限错误、跨 region、aligned block 分配、复用、double free、fixed 分配失败。
- sram_r0_selftest、sram_r1_selftest CTest 全通过。

## 评审

- 发现 block allocator 在拆分 free span 时先插入 after 再使用旧 iterator，
  在存在前导 padding 时可能发生 vector iterator 失效。
- 确认 allocation 返回的是对齐后的保留范围，调用方不能越过该范围。

## 修改

- allocator 改为 index-based span replacement，避免任何 insert 后复用 iterator。
- 保留非法 free 的显式异常，不静默忽略。

## 结论

S1 验收通过。真实 payload、valid、手动 region 和子分配已有独立可用 API。

## 2026-08-06 复审修订

### 开发

- timing-only Storage 增加 payload fingerprint signature。
- allocation 与 `AddrPosKey` 增加 region、region allocation、spill policy 和 task/layer/persistent lifetime。
- `RegionTable::Free` 同时检查 lifetime 与 outstanding range。

### 测试

- R1 覆盖 timing-only signature、lease 持有期间 free 失败、persistent allocation 在 task 边界拒绝释放。

### 评审

- 旧标签只有线性位置，timing-only signature 恒为零，Free 不理解生命周期。

### 修改

- 标签插入真实 SRAM 时自动解析 region/spill policy；新增无权限副作用的 `LocateAbsolute`。


## 第二轮生产路径复审修改（2026-08-06）

### 开发

- 明确 `AddrPosKey.pos`/`context.sram_addr` 是 legacy SRAM word index，新增溢出安全的 word↔byte 边界 helper。
- `AnnotateRealSramKey` 使用 byte 地址定位 region；block region 通过 `AllocateAt` 绑定生产标签，tile 增长通过 `ResizeAllocation` 扩容。
- `Clear_sram` 以 byte 地址清理并把 byte high-water 向上换算回 word 游标；ETERNAL/static 标签分别绑定 persistent/layer lifetime。

### 测试

- R6 强制 512-bit SRAM，验证 `pos=1 -> byte 64`。
- R6 验证 task/layer 标签获得非零 region allocation，fixed 标签不伪造 allocation；tile 标签从 32B 增长到 48B 时原 allocation 定址扩容；clear 后 task 地址可重新分配。

### 评审

- 原 R6 使用 byte 风格 `pos`，无法暴露兼容路径单位错误；生产 addPair 也未调用 RegionTable allocator。

### 修改结论

- word/byte 契约只在 compatibility boundary 转换，标签、存储和 region 不再混用单位；生产 allocation/free 闭环已由 R6 覆盖。
