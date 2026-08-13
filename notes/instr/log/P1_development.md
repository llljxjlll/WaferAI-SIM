# P1 开发记录：稳定 opcode、线格式、工厂与分类

阶段：P1——稳定 opcode、线格式、工厂与分类
阶段状态：已完成
完成日期：2026-08-11（UTC）
提交/变更集：工作区实现，尚未提交

## 计划确认

- 依据 `编译产物指令集P0契约.md` 与 `isa_v1_manifest.md` 实现两个互不混用的编号空间：外部 `Opcode` 和内部 `PrimId`。
- 外部 record 是跨构建 ABI；内部 128-bit Prim wire 仅供同版本 loader/WorkerCore 使用。
- P1 不实现 program 文件、SRAM/event/DTE endpoint 运行时语义；这些能力分别属于 P2～P6，但必须在 P1 有稳定 schema 与明确 unavailable/gated 诊断。

## 实际完成事项

### 稳定 manifest 与工厂

- 51 个现存 `REGISTER_PRIM` 全部改为显式 `PrimId`，删除 production 链接顺序分配。
- 新增内部 Prim manifest，工厂严格校验 name/ID/creator 三者一致；重复、未知、0、52、255 均给出确定异常且不污染 registry。
- 新增 44 项外部 Opcode manifest，区分 available、unsupported、experimental；未分配 u8 为 reserved，超过 u8 为 unknown。
- npusim 在任何仿真或 selftest 前验证 opcode manifest、Prim manifest 与精确 51 项 factory 注册集合。

### 外部 record codec

- 固定 8-byte little-endian record header，为 44 个 Opcode 提供强类型 operand schema。
- 39 项可执行 schema 覆盖最小/典型/最大边界；5 项 reserved/unsupported 明确拒绝。
- 完成 16 类 operand shape 的完整 golden hex、round-trip、长度/枚举/保留位/地址 XOR/能力 gate 负例。
- 外部 NPU offset 严格为 u16，参数严格为 30-bit；artifact 不编码 PrimId、`sc_bv` 或 `g_addr_label_table` 数字 ID。

### 内部 Prim wire

- 所有单段 codec 校验精确段数、首段 PrimId、枚举、保留位、padding 与窄化。
- NPU/GPU、SetAddr、SetBatch 固定多段数量并校验每一段 PrimId；反序列化先写临时状态，再一次提交，重复 decode 不保留旧字段。
- 对 DTE async、LSU、Sram pipeline、collective data/marker、reduce compute 增加统一 `PRW1` wrapper/trailer 与 displaced-byte carry，使每个传输段低 8 位均为同一 PrimId。
- 宽 padding 改用 `.or_reduce()`，避免只检查低 64 位；DTE 直接序列化 region 长度超过 64 时拒绝。
- legacy Send/Recv/Load/Store/Clear 严格化非法输入，同时保留已冻结兼容：`des=-1`、stripe 0、packet 0 的历史编码行为。

### 分类与执行安全

- COMPUTE/COMM/MEM/SYNC 主类别唯一，NPU/GPU family bit 与主类别正交；DTE/LSU 可根据 op/direction 动态刷新类别。
- 四个数据准备原语改为 MEM；既有 helper/WorkerCore 用 `dynamic_cast<CompBase*>` 判断执行家族，避免类别变化破坏 legacy 调度。
- WorkerCore 解码使用临时 `unique_ptr`，完整 deserialize 和 context attach 成功后才提交到全局 stash；未知 creator 不再返回空指针。
- trace 按唯一主类别统计，拒绝无主类别或多主类别 Prim。

### 自动化入口

- `--isa-v1-selftest` 在 SystemC 平台初始化前早退出。
- CTest 注册 `isa_v1_selftest`，label=`isa`、`RUN_SERIAL=true`。
- P1 最终严格 codec 检查数为 1940：manifest/factory 815、external record 797、internal wire 328。接入 P2 后统一入口为 2808 项（另含 Program Format 620、record lowering 199、program helper 49）。

## 开发者自测与集成测试

- 增量 Release build 成功。
- `./build/npusim --isa-v1-selftest`：最终统一入口 2808/2808 通过。
- 最终主工作区 CTest：23/23 通过（含 P2 program 正负例）；P1 首次门禁的 17/17 同样通过。
- DTE V3 因有意采用 wrapper，期望段数更新为 4；31/31 通过。DTE V4 19/19 通过。
- 默认 workload、四源长 START、playground legacy smoke 均通过。
- `git diff --check` 通过；静态审计未发现隐式 `REGISTER_PRIM` 或 `next_id_` production 注册残留。

## Legacy 兼容纠正闭环

首次完整 Python runner 评审发现，strict wrapper 的 carry/trailer 和 `Set_addr` 3-label payload 会增加 legacy CONFIG 段数；关键 host lane 每新增一段增加 2ns，使冻结完成时刻出现 +4～+68ns。clean HEAD `139ef73a803dc3637b28000bbc3a34016b2d5e45` 对照证明改造前九个 standalone runner 全部命中旧值，因此没有更新 baseline，而是修复兼容层。

修复后：

- wrapper 只按明确 PrimId 白名单处理，不从任意 payload 猜 trailer；
- legacy wrapper 去除 framing，`Set_addr` 恢复 8→6 段，`Set_batch`、NPU、GPU 恢复旧位布局；
- core/GPU/GPU-PD/PD/PDS helper 的多段下发统一通过 legacy converter；
- `InitPlatform` 固定 strict，`InitGrid` 仅在成功完成 JSON 初始化后启用 legacy；
- strict/legacy 切换、trailer-like payload 碰撞、SetAddr/SetBatch/NPU/GPU golden 已纳入 Prim wire selftest。

修改→复测→再评审证据：

- unified ISA：2808/2808；Prim wire 子集 328/328；
- DTE V1/V2/V2b/V3/V3b/V4：全部通过并恢复全部冻结时刻；
- D2D V0：67/67；collective legacy pressure：3/3；NoC congestion：4/4；
- NoC 四个精确值：14781、29109、14833、45441ns；
- 全量 CTest：23/23；`git diff --check` 无输出。

## 跨构建稳定性

- 在最终 P1 源码快照上完成全新 Release 和 Debug configure/build。
- Release/Debug 的三条 ISA PASS 摘要完全相同，规范化 SHA256 均为 `d32b167e4d0fbb36f8eaa4cc29a36c880baf78563924b833e58ab61ef40d34ec`。
- Release CTest 17/17 通过；Debug 除既有 DRAMSys HBM2 checker 断言外 16/17 通过，ISA 门禁全绿。
- Release 再跑 P0 28 个 CLI/883 项与 SRAM R0～R6，全部通过。

## 评审、修改、复测

首次代码/测试评审发现并关闭：

- 缺少 external record codec/golden：新增完整 schema、codec 与 797 项测试。
- 多段后续段低 8 位不是 PrimId：原生格式修复或统一 wrapper/carry，增加逐段破坏矩阵。
- `SRAM_BIND` 错降到 legacy SetAddr：manifest 改为 `NEW_THIN`，留给 P3 one-shot 实现。
- reserved 与 unknown 混淆：u8 未分配、扩展保留区和大于 u8 分别诊断。
- capability 状态混淆：experimental 与 available 分离，MLA/PD 默认 gate。
- u16 外部 offset 被内部兼容改造扩大：外部 ABI 继续严格 u16；内部 legacy wire 明确为有符号 u32，两层不互相泄漏。
- 类别改动影响 legacy 执行：所有行为型消费者改用运行时类型，定向 smoke 与全量回归通过。
- deserialize stale state、SetAddr 无 context 重序列化、宽 padding 漏检：分别修复并加入重复 decode/全宽 corruption 测试。

再评审确认：全部 51 个内部 creator 和全部 44 个外部 Opcode 均有唯一编号、状态、分类与确定诊断；外部/内部格式边界符合 P0 契约，P2 可只凭外部 record 安全解析和 lowering。

## 已知非阻断项

- 工厂异常负例会经过既有日志宏打印 ERROR/SystemC `sc_stop` warning，断言与退出码仍正确；这是日志噪声，不影响功能门禁。
- Debug `hbm_r3_selftest` 的第三方配置断言已隔离记录，不作为 P1 ISA 失败。
- strict wrapper 使内部 program wire 有意增加 framing；legacy helper 在下发前恢复旧 wire，因此旧 JSON 的 CONFIG 段数、位串和冻结周期保持不变。

## 阶段退出条件

- [x] 计划确认完成
- [x] 显式 PrimId 与独立 Opcode manifest 完成
- [x] 外部 record 和内部 wire codec/golden 完成
- [x] 分类、工厂和失败原子性完成
- [x] 新增测试与受影响回归通过
- [x] Release/Debug clean build 稳定性通过
- [x] 评审意见完成修改、复测与再评审
- [x] legacy JSON 行为保持可用

是否允许进入下一阶段：是。
结论：P1 已完成；P2 可使用 `DecodeExternalRecord` 与 `LowerExternalRecord` 构建完整验证后再提交的 program loader。
