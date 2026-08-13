# P2 开发记录：编译产物格式、加载器与 CLI

阶段：P2——编译产物格式、加载器与 CLI
阶段状态：已完成
完成日期：2026-08-11（UTC）
提交/变更集：工作区实现，尚未提交

## 计划确认

- artifact 使用稳定外部 ISA record，不持久化内部 128-bit Prim wire 或 `g_addr_label_table` ID。
- loader 按“完整解析/校验→语义重定位→lowering→内部 wire 预验证→原子提交”执行。
- `--program` 与 legacy `--workload-config` 并存；program v1 固定为 DATAFLOW，仍使用公共 hardware/simulation/mapping 平台配置。
- MLA/PD、GLOBAL、P3～P7 尚未实现的 runtime 能力在加载期明确拒绝，不通过伪 workload 或不安全强转绕过。

## 实际完成事项

### Program Format v1

- 新增 64-byte `NPUPRG1\0` little-endian header，固定 format/ISA version、endianness、capability、section table、文件长度和 whole-file CRC-32C。
- 实现七个 required sections：字符串、符号、语义重定位、core group、每核 program index、外部 record stream、控制 envelope。
- 每个 section 带 offset/size/count/entry_size/CRC，拒绝截断、重叠、gap 垃圾、尾部垃圾、伪造 count、未知 required section 和所有非零保留位。
- 冻结 64 MiB 文件、64 sections、字符串/符号/重定位/group/core/record 数量上限；UTF-8 拒绝 NUL、非法编码、重复项和超长项。
- core/group/control list 必须有序唯一；ACK 集合由 empty-core policy 闭合，DONE 集合必须等于 terminal 集合。
- 外部 record 解码错误现在包含 core、record index、文件 offset、原始 opcode 及已知名称，不再只有无上下文 payload 错误。

### 重定位与 lowering

- semantic relocation 使用稳定 operand ID，支持计算地址、LSU、local DTE、symbol/region/label、resize/rename；先在临时 artifact 上应用。
- `LowerExternalRecord` 对当前阶段可用的计算、blocking LSU、三个 local `DTE_ISSUE` direction、`DTE_WAIT/FENCE/CANCEL` 构造 untracked Prim。
- lowering 前再次执行 opcode/capability/operand 校验；resolved symbol 不可为空或超过内部 wire 上限。
- P3～P7 的 `NEW_THIN_PRIM`/`MODE_DISPATCH` 路径返回 known-but-unavailable，禁止误当空操作成功。

### program helper 与平台注入

- `config_helper_program` 完整实现 CONFIG/WEIGHT/START/DONE 接口、clone、ACK/DONE 精确集合和 DONE 完成 `sc_stop()`。
- artifact core/group 再结合 `TOTAL_CORES/CORES_PER_DIE` 校验 active、范围和同 die；候选 `coreconfigs` 与 program 状态一次交换。
- helper 对所有 lowered Prim 先 serialize，任何异常都保持旧 helper/coreconfigs/strict mode 不变。
- START `tag/count` 在现有 host wire 边界内要求 `tag<=65535`、`1<=count<=255`，不截断；按 count 产生确定数量的 S_DATA。
- `InitGrid` 拆出 `InitPlatform`。platform 初始化先选择 strict wire；legacy JSON 只有全部初始化成功后才切为兼容 wire。
- 修复 injected `GlobalMemInterface` 构造器的 `assert(0)`；program v1 保留 chip memory endpoint 供 socket bind，但不加载未发布 GLOBAL 指令。
- `npusim --program` 现在实际将 `program_helper.get()` 注入 `Monitor`，不再错误进入空 workload path 的 JSON helper。

### CLI、参考编码器与文档

- `--program` 和显式 `--workload-config` 互斥；两者均缺失时保持原默认 JSON workload。
- 文件读取先检查存在、regular file 和 64 MiB 上限，再分配和 decode；解析失败在构造 Monitor/下发 CONFIG 前返回 2。
- 启动打印实际 Program Format、ISA version 和 capability。
- 新增 `npusim_program_fixture`，只链接 public opcode/record/container codec，不初始化 SystemC。
- reference artifact 含一个真实外部 `DTE_FENCE` record；CLI smoke 必须完成 lowering、strict wrapper decode、worker 执行和 DONE 后才能退出。
- fixture 支持确定性 CRC corruption，用于验证损坏文件在启动前拒绝。
- `notes/instr/program_format_v1.md` 记录每个 header/descriptor/section/control 字段、CRC、版本升级和 loader 顺序。

## 开发者自测、集成与回归

- Program Format selftest：620/620。
- record lowering selftest：199/199。
- program helper selftest：49/49。
- unified `--isa-v1-selftest`：2808/2808。
- Program CTest 6/6：Unicode+空格路径生成、真实 record 启动、CLI 互斥、缺文件、CRC 损坏生成与拒绝。
- 全量 Release CTest：23/23。
- legacy 默认、四 source START、playground 均通过。
- P1 冻结 runner 复测：DTE 六组全过、D2D 67/67、collective pressure 3/3、NoC 4/4；program 接入未改变 legacy 周期。
- `git diff --check` 无输出。

## 详细正负例覆盖

- 正例：单核、多核、空 core、不同 record 长度、全 sections、core group、symbol relocation、单/多 START、单/多 terminal、INCLUDE/EXCLUDE empty core、平台同 die group。
- 文件负例：magic/version/ISA/endianness/CRC、截断、bit flip、section 重叠/缺失/重复、record count/size/boundary、尾部垃圾、保留位和 capability 不一致。
- 引用负例：未知/wrong-kind symbol、relocation operand/kind/addend、重复 relocation target、group/member/core/envelope 闭合。
- helper 负例：零 platform core、core 越界/inactive、跨 die group、无 ACK/DONE、START 窄化、unsupported lowering、失败后状态不变。
- CLI 负例：program+workload、missing file、whole-file CRC corruption。

## 评审、修改、复测与再评审

首次评审的主要阻断及关闭方式：

- 只有预序列化 segments、没有外部 record codec：改为两层格式并完成独立 record/container golden。
- helper 注入会触发 GlobalMemInterface assert：修复 injected 构造并保持 GLOBAL capability 关闭。
- InitGrid 依赖伪 workload：拆 `InitPlatform`，program 明确选择 DATAFLOW。
- program helper 构造后 CLI 未使用：Monitor 改为按 mode 选择 helper constructor；真实 CLI smoke 已覆盖。
- helper 只校验 artifact、不校验实际平台：增加 TOTAL_CORES、active group membership、同 die和原子 coreconfigs。
- decode 错误缺 offset/opcode：增加 context formatter 和 corruption message selftest。
- reference fixture 只有空流：加入真实 `DTE_FENCE` 外部 record和 strict 多段执行。

再评审确认：program 模式不读取 workload JSON、不进入 PD/PDS 强转、不启用 legacy decoder；legacy 模式保持旧 helper 和旧 wire；损坏 artifact 不产生半配置状态。

## 阶段边界

- P2 的 JSON/program 等价证据覆盖公共 record→Prim 字段/wire，以及真实 DTE_FENCE program 执行与 legacy DTE 冻结回归。需要 one-shot binding 的计算链在 P3 完成 SRAM_BIND 后做完整 ops/cycle/标签差分；真实 byte P2P/collective 分别由 P5/P6 验收。
- format 已为后续 capability 保留版本机制，但当前不含 PD context section；四个 MLA/PD opcode 继续 gate，不能在 P3 静默开放。

## 阶段退出条件

- [x] program 完成版本、sections、CRC、资源上限与控制 envelope 校验
- [x] 符号重定位、lowering、内部 wire 预验证与两阶段提交完成
- [x] platform/Monitor/GlobalMemInterface 注入路径完成
- [x] START/ACK/DONE 和空 core 策略完成
- [x] CLI 正负例、真实 record smoke 与参考编码器完成
- [x] legacy JSON 回归和冻结周期不变
- [x] 评审问题完成修改、复测与再评审

是否允许进入下一阶段：是。
结论：P2 已完成；P3/P4 可只通过 `--program` 注入其新 opcode，不得回退到内部 segments artifact。
