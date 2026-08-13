# legacy JSON 到 `--program` 迁移与兼容指南

> 文档状态：P8 发布文档草案
>
> 目标版本：NPU ISA v1 / Program Format v1
>
> 兼容原则：JSON 与 `--program` 至少并存一个正式发布周期；迁移期间默认 JSON 行为不变；两种入口不得同时指定。
>
> P7 状态说明：P6 baseline collective 已完成真实字节验收。`broadcast_only`、`reduce_only`、`reduce_broadcast` 是稳定请求名称，但在 P7 最终运行时验收完成前不作为迁移可用能力承诺。

## 1. 迁移目标

迁移不是把 JSON 原语名字逐项换成数字 Opcode，而是把 workload 的稳定语义转成版本化 artifact：

```text
legacy JSON
  ├─ workload/helper 隐式状态
  ├─ persistent label binding
  ├─ 进程内 Prim/name 路径
  └─ source/DONE 等宿主约定

Program Format v1
  ├─ public external Opcode + typed operands
  ├─ strings/symbols/semantic relocations
  ├─ per-core record streams + static core_groups
  ├─ explicit START/ACK/DONE control envelope
  └─ 完整校验后原子 lowering/提交
```

迁移成功的判据不只是两个入口都退出 0，而是：

- 每核语义指令、操作数和依赖关系一致；
- 数据内容、有效范围和范围外 sentinel 一致；
- EXU/SFU/VEC ops、LSU/DTE overlap 和允许比较的周期一致；
- P2P/collective 的 bytes、checksum、rank/offset 和归约值一致；
- token、event、barrier、region、session 和 Router residual 全部归零；
- 已登记的行为差异确实来自入口契约，而不是漏 lowering。

## 2. 双入口周期

### 2.1 CLI 选择

JSON 入口：

```bash
build/npusim \
  --workload-config workload.json \
  --hardware-config hardware.json \
  --simulation-config simulation.json \
  --mapping-config mapping.spec
```

Program 入口：

```bash
build/npusim \
  --program model.npup \
  --hardware-config hardware.json \
  --simulation-config simulation.json \
  --mapping-config mapping.spec
```

规则：

- `--program` 与显式 `--workload-config` 同时出现时返回错误，不设优先级；
- 未指定 `--program` 且未显式给 JSON 时，继续使用既有默认 workload；
- program 模式仍读取同一套 hardware/simulation/mapping，但不读取或伪造 workload JSON；
- program v1 固定进入 `SIM_DATAFLOW`；需要 PD/PDS 调度上下文的 workload 不能据此假定可迁移；
- 成功完成 program 初始化后打印 `Loaded Program Format 1.0, ISA 1.0, capabilities=...`。

### 2.2 并存期要求

至少一个正式发布周期内：

- 原 JSON 文件、生成链、runner 和 frozen timing oracle 继续维护；
- 新编译器同时可产出 JSON 与 `.npup`，或保存能重建两者的统一中间表示；
- CI 对同一代表性 workload 串行运行两种入口，避免共享 `events.json`、VCD 或日志互相覆盖；
- 未迁移能力继续走 JSON，不能在 program 中编码内部 primitive 或 unsupported Opcode 规避门禁；
- 任何回滚只切换入口和产物，不改动同一轮使用的硬件、仿真和 mapping 配置。

## 3. 迁移前能力盘点

先把 JSON workload 中的操作分成四组：

1. **可直接映射**：已发布计算、blocking LSU、本地 async DTE、SRAM lifecycle、P2P、同步；
2. **需 whole-artifact lowering**：Scatter/Broadcast/Gather/Reduce/AllToAll/AllGather/ReduceScatter/AllReduce；
3. **仅 legacy/internal**：`load_expert`、`switch_data`、`parse_input`、`parse_output`、`Sram_pipeline`、`Set_batch`、legacy Load/Store、无参 clear-all；
4. **v1 禁止或 gated**：GPU、GLOBAL、LSU async、`DTE_POLL`、stub、实验融合、MLA/PD。

第 3 组必须在编译器中展开为已发布基础指令，或暂留 JSON；第 4 组必须暂留 JSON/旧模式或直接报告不支持，不能生成一个“看起来成功”的零成本 program。

## 4. 语义映射表

| legacy JSON/内部概念 | Program v1 表达 | 迁移注意事项 |
|---|---|---|
| NPU compute name + params | 对应 public compute Opcode + `ComputeOperands` | 参数顺序以 public schema 为准，offset 为 u16 byte，parameter 为 30-bit |
| `Set_addr` | `SRAM_BIND` | JSON persistent；program one-shot，连续 compute 必须重复 bind |
| blocking load/store | `LSU_LOAD` / `LSU_STORE` | program 只发布 blocking HBM↔SRAM |
| local DTE issue | `DTE_ISSUE` | 仅 SPM_TO_SPM、SPM_TO_DRAM、DRAM_TO_SPM |
| DTE wait/fence/cancel | `DTE_WAIT` / 无参 `DTE_FENCE` / `DTE_CANCEL` | token 非零；fence 不带 token；POLL 不发布 |
| legacy Send/Recv | `DTE_SEND(P2P)` / `DTE_RECV(P2P)` | 字段、配对和真实字节路径不同，不能复用 legacy wire |
| JSON collective descriptor/helper | keyed `DTE_SEND/DTE_RECV` + 必要 `REDUCE_COMPUTE` | loader whole-artifact 建图；external `0x42` 降到 strict `PrimId 0x3A`；不发 `COLLECTIVE_CALL`/internal barrier |
| group/barrier helper | static `core_groups` + `GROUP_SYNC` | group 只声明一次，`sync_seq` 单调；唯一 public barrier |
| 隐式/控制消息 event | `EVENT_SET` / `EVENT_WAIT` | 专用 EVENT 控制语义，不能伪装为零长度 DATA |
| region allocator 操作 | `SRAM_ALLOC/FREE/RESIZE/RENAME` | public rename 严格拒绝 duplicate target |
| legacy `Clear_sram` | `SRAM_CLEAR(label)` 或显式 `SRAM_FREE` | program CLEAR 是 targeted 且清字节；FREE 只释放 metadata |
| runtime label 数字 ID | UTF-8 string + `SRAM_LABEL/SRAM_REGION` symbol | artifact 禁止保存进程内 ID |
| 地址 patch | semantic relocation | 定位 operand，不做裸 byte patch |
| helper 推导 source/terminal | control envelope | active/start/terminal/ACK/DONE 全部显式闭合 |
| `FENCE_ALL` 别名 | `DTE_FENCE` | 不分配第二个 Opcode或重复 trace |
| internal collective barrier | 无 public record | 由 loader action image生成 |

完整 Opcode 和支持状态以 `isa_v1_manifest.md` 为准，不能以当前 C++ 类是否存在判断 public 可用性。

## 5. 关键行为差异

### 5.1 label binding

JSON `Set_addr` 保持 persistent，直到下一次 Set_addr；program `SRAM_BIND` 跳过中间 MEM/SYNC/COMM，但只由下一条 COMPUTE 消费。迁移器必须在每条 compute 前物化一条 bind，而不能照搬 JSON 中“一次设置、多次计算”的结构。

错误示例：

```text
SRAM_BIND A→B
MATMUL
GELU              # 失败：GELU 没有新的 binding
```

正确示例：

```text
SRAM_BIND A→B
MATMUL
SRAM_BIND B→C
GELU
```

### 5.2 SRAM lifecycle

- JSON legacy `parse_input` 的特定 replacement 路径可覆盖旧目标 label；public `SRAM_RENAME` 默认严格拒绝 duplicate target；
- JSON 无参 `Clear_sram` 是 legacy clear-all；program `SRAM_CLEAR` 必须带 label，清真实字节并释放符合条件的 allocation；
- `SRAM_FREE` 不清底层字节；依赖清零效果的 JSON 不能迁移成 FREE；
- pending one-shot bind 引用的 allocation 不能被 FREE/RESIZE/CLEAR，RENAME 则更新 pending name。

### 5.3 执行和计费

- 已发布 compute 复用同一生产成本模型；`DUMMY` 有固定成本，不是 NOP；
- LSU 是阻塞的，不能与下一条 compute 重叠；DTE ISSUE 可与 compute 重叠，依赖由 WAIT/FENCE 收束；
- program 使用 strict internal wire，JSON 使用 legacy compatibility wire。内部段数可能不同，不属于外部 ABI 差异；
- program control envelope 带来的 CONFIG/START 组织可以造成入口固定开销差异。比较计算区间时应归一化该开销，不能直接改 golden 掩盖语义差异；
- P3 已对 `MATMUL→ATTENTION→LAYERNORM→GELU` 代表链完成 ops/cycle/完成顺序差分；新增模型仍需独立差分。

### 5.4 通信与 collective

- program P2P 使用新的真实 payload endpoint，不等价于字段不足的 legacy Send/Recv wire；
- standalone P2P 可跨 die；非 P2P collective 的 group 必须同 die；
- program collective 统一 `N=group_size`，普通 Reduce 单 root，ReduceScatter/AllReduce 对称分解；
- collective external records 是逻辑角色，loader 生成 RX-first action image、child endpoint、wave 和 internal barrier；
- `REDUCE_COMPUTE(0x42)` 不能 record-local direct lowering；它必须作为闭合 collective graph 的逻辑角色生成 strict `Collective_data_v1_prim (PrimId 0x3A)`。legacy `Reduce_compute_prim (PrimId 0x2B)` 是 internal timing-only creator，不能用于 program 数据归约；
- baseline 已验证九种 TX×RX、真实字节和 1/8/32 KiB；JSON 若没有同等 whole-artifact 表达，应使用独立软件 byte oracle，而不是伪造“等价 JSON”。

### 5.5 capability 与 profile

- artifact capability bitmap 是硬要求；未知或关闭 capability 必须 fail-fast；
- 当前稳定迁移基线使用 `capabilities=0`；PD、GLOBAL 和 experimental 请求不通过 helper；
- `baseline`、`broadcast_only`、`reduce_only`、`reduce_broadcast` 由平台 NoC 配置请求，不改变高层 artifact；
- P7 最终验收前，迁移验收只把 `baseline` 作为已完成 production 依据。其他档位不能静默 fallback，也不能作为删除 JSON 的前置证据。

## 6. 推荐迁移流程

### 阶段 A：冻结 JSON 对照

1. 固定 workload、hardware、simulation、mapping 和随机种子。
2. 保存每核 primitive 顺序、参数、标签、开始/结束时刻。
3. 保存 EXU/SFU/VEC ops、LSU/DTE bytes、NoC/D2D stats、最终数据 checksum 和 residual。
4. 对共享日志 runner 使用独立输出目录或串行运行。
5. 把既有已知差异单独登记，禁止迁移时顺手修历史计费。

### 阶段 B：建立稳定中间表示

后端 IR 至少应显式包含：

- global core id 和每核有序操作；
- 稳定 symbol 名、kind、base、extent、local offset；
- compute 输入/输出 arity 与 one-shot binding；
- local DTE direction、token 和依赖；
- P2P peer/fsm/token/completion；
- collective group/rank/root/key/epoch/dtype/reduce_op；
- source/start/terminal/ACK/DONE；
- required capability 集合。

不要从 JSON helper 的进程内对象序列化内部 ID。

### 阶段 C：生成并自校验 artifact

1. 规范化 strings、symbols、groups、cores 和 envelope list。
2. 发射 public records；将仅 internal helper 展开或报告阻塞。
3. 应用/保留语义 relocation，检查地址 XOR 和 region span。
4. 运行 `ValidateProgramArtifact` 和 `EncodeProgramArtifact`。
5. `DecodeProgramArtifact(bytes)` 后再编码，要求逐字节一致。
6. 固定 golden hash，并在 compiler CI 中与仓库 codec 互解码。

### 阶段 D：分层差分

按以下顺序扩大范围，每层全绿后再继续：

1. 单核 `SRAM_BIND→DUMMY`；
2. 单条真实 compute；
3. compute 链和 one-shot bind；
4. blocking LSU；
5. local DTE double buffer；
6. same-die/cross-die P2P；
7. N=1/2/4 baseline collective；
8. 多核 compute+communication+sync 代表 program；
9. 长时间重复运行、mutation 和 sanitizer 门禁。

每层检查 record、内部 lowering 摘要、真实内存、计费、完成顺序和 drain，不只检查总仿真时间。

### 阶段 E：灰度和默认切换

1. 先在 CI 和离线仿真中双跑，不改变用户默认入口。
2. 对已覆盖模型按 allowlist 启用 program；未覆盖模型继续 JSON。
3. 收集错误分类、性能差异和回滚次数，修复后重跑完整 frozen matrix。
4. 只有满足第 10 节废弃条件后，才能讨论改变默认入口。

## 7. 失败诊断

### 7.1 失败发生在哪一层

| 现象 | 阶段 | 优先检查 |
|---|---|---|
| `--program and --workload-config are mutually exclusive` | CLI | 启动脚本只能选择一个 workload 入口 |
| `Configuration preflight failed` | 文件/公共配置预检 | 路径、regular file、64 MiB、header/CRC/section、平台配置文件 |
| `Configuration initialization failed` 且未打印 `Loaded Program Format` | loader 两阶段提交 | opcode、record schema、symbol/relocation、group、envelope、capability、whole-artifact graph |
| 已打印 `Loaded Program Format` 后失败 | CONFIG/执行/runtime | one-shot bind、token、event/barrier、endpoint 配对、数据范围、最终 drain |
| 退出 0 但差分失败 | 语义或 oracle | 参数顺序、byte/bit 单位、offset/span、legacy 允许差异、profile/backend |

### 7.2 常见诊断与修复

- `unknown`：原始 opcode 不在 v1 manifest；检查生成器版本或文件破坏。
- `reserved`：编号存在于保留区但未分配；不能猜测未来语义。
- `unsupported`：已知但 v1 不发布；展开为基础指令或留在 JSON。
- `capability is disabled` / helper 禁用 capability：artifact 请求了当前未开放能力；移除请求不是合法修复，必须改变模型选择或等待能力发布。
- `record ... offset/core/opcode`：利用错误中的 core、record index、文件 offset 定位后端 IR 指令；不要修改二进制绕过校验。
- CRC/section/file size/trailing bytes：重新使用 canonical encoder，whole-file CRC 计算时把 header `[56,60)` 视为 0。
- symbol kind/unknown/duplicate relocation：检查 operand ID、kind 和每目标唯一性。
- region span/outside/crosses configured regions：检查 base 只加一次、offset 是 local byte offset，以及 Scatter/Gather/Reduce 的 `N*L` span。
- missing/double/dangling bind：按每条 compute 重新物化 one-shot bind。
- endpoint pair、fsm、length、token mismatch：对同一 P2P/collective key 做全 artifact 配对检查。
- group unknown/non-member/cross-die：检查静态表、平台 active core 和同 die约束。
- residual 非零：不能把 watchdog 或析构清理当成功；检查 WAIT/FENCE、EVENT credit、GROUP_SYNC 成员、endpoint/session 和 terminal 过早 DONE。

loader 采用两阶段提交。任何加载失败都不应产生部分 CONFIG；修复 artifact 后应从新进程重试，以便诊断与生产执行保持一致。

## 8. 行为差异登记模板

每个迁移 workload 建议维护：

```text
workload/model:
JSON golden revision:
program compiler revision:
artifact SHA256:
hardware/simulation/mapping SHA256:

指令差异：
控制 envelope/固定开销差异：
允许的内部 wire 差异：
ops/cycle 差异：
数据/checksum 差异：
trace/stat 差异：
residual：
评审人/结论：
```

允许差异必须有源码契约或独立 oracle 依据。“program 比 JSON 快/慢”本身不是依据。

## 9. 回滚策略

### 9.1 可回滚资产

发布包在并存期必须保留：

- 已验证的 JSON workload；
- 对应 `.npup` 及其编译器版本、manifest/golden hash；
- 完全相同的 hardware/simulation/mapping；
- 两入口的 oracle、trace 摘要和已登记差异；
- 一键选择单一入口的启动配置。

### 9.2 回滚触发条件

发现以下任一项，立即将受影响 workload 切回已验证 JSON，并保留失败 artifact 用于复现：

- 错误解码、错误数据、死锁或 watchdog；
- token/event/barrier/tree/session/Router residual 非零；
- profile 请求发生 silent fallback；
- 无法解释的 ops/cycle 或资源计费变化；
- 新平台配置触发未覆盖 capability；
- parser/loader 崩溃、越界、超量分配或半提交迹象。

回滚不应执行：修改 ABI version 字段、手工改 CRC、把 unsupported opcode 换成 DUMMY、删除 WAIT/FENCE、扩大无界 buffer、在同一命令同时传两入口。

### 9.3 回滚后闭环

1. 归档失败 artifact、CLI、平台配置、stdout/stderr、trace 和退出码。
2. 在固定 binary 上确认 JSON golden 仍通过。
3. 最小化到 record/core/group 级复现。
4. 修复 encoder、loader 或 runtime，并补负例/golden。
5. 先复跑定向层，再复跑全部受影响 legacy 与 program 门禁。
6. 交叉评审关闭后重新进入灰度，不直接恢复全量默认。

## 10. 废弃策略

### 10.1 JSON 入口

双入口必须至少保留一个正式发布周期。之后只有同时满足以下条件，才可提出废弃 JSON 的评审：

- 所有目标 public Opcode 均可由 `--program` 执行并有自动化测试；
- 代表性计算、double buffer、多核 P2P/collective/sync program 全绿；
- parser mutation、重复运行、sanitizer、完整 Release/Debug 和 legacy 回归通过；
- 四档 profile 中计划对外开放的档位完成 production runtime 验收；未开放档位有稳定 fail-fast；
- 已迁移 workload 完成至少一个发布周期灰度，回滚路径被实际演练；
- 用户文档、编译器版本矩阵和支持窗口已提前公告；
- 没有依赖仅 legacy/internal primitive 而无公开替代的受支持 workload。

达到条件也只能先把 JSON 标记 deprecated，再经过公告期删除。删除 JSON 入口不能同时改变 ISA opcode、Program Format 或计算计费语义。

### 10.2 Opcode 和 artifact

- 已发布 Opcode 不改号、不复用；删除后保留 tombstone；
- unknown format/ISA major 永远 fail-fast，不猜测兼容；
- 不兼容容器字段提升 format version，不兼容指令语义提升 ISA version；
- 只有旧 loader 可安全忽略的新 section 才能标 optional；
- 内部 Prim wire 不承诺跨版本兼容，不能作为继续支持旧 artifact 的依据；
- ABI golden 的变化必须明确标记为 ABI、capability 或 internal-wire 变化，并按 [NPU ISA v1 ABI 与 Golden 变更策略](ISA_v1_ABI与Golden变更策略.md) 完成评审。

迁移器若改变 external record 的合法域、符号/重定位语义、lowering target 或运行完成点，必须和对应 manifest、codec、artifact、lowering、runtime、mutation 与 legacy golden 在同一评审中提交。不能把 `REDUCE_COMPUTE` 从 strict `0x3A` 改回 legacy `0x2B` 并称作内部实现替换。

## 11. 兼容验收清单

- [ ] JSON-only、program-only、两者冲突和两者均缺省行为符合 CLI 契约；
- [ ] program 不读取 workload JSON，公共平台资源与 JSON 入口一致；
- [ ] 17 个 available compute 的参数、ops、周期和副作用完成代表性差分；
- [ ] persistent Set_addr 已改写为逐 compute one-shot bind；
- [ ] legacy clear/rename 行为差异已显式处理；
- [ ] blocking LSU 与 async DTE overlap 关系正确；
- [ ] P2P same/cross-die 真实字节、sentinel 和 completion 正确；
- [ ] collective baseline 的 N、root/rank、offset、归约整数规则正确；
- [ ] source/start/terminal/ACK/DONE 显式闭合；
- [ ] unsupported/internal/gated 能力均 fail-fast，无 DUMMY 或 baseline 替代；
- [ ] 两入口共享 runner 串行或输出隔离；
- [ ] 成功、取消和失败路径 residual 全部归零；
- [ ] 回滚脚本、资产和触发条件已演练；
- [ ] 废弃提案满足完整发布门禁，而不是只依据单个 smoke。

## 12. 规范引用

- 外部 Opcode、状态和 lowering target：`notes/instr/isa_v1_manifest.md`
- Program Format、sections、CRC 和控制 envelope：`notes/instr/program_format_v1.md`
- JSON/program 并存、one-shot、能力和错误契约：`notes/instr/编译产物指令集P0契约.md`
- P8 迁移与发布退出条件：`notes/instr/编译产物指令集开发计划.md`
- 已完成的入口、计算、访存、P2P、collective 证据：`notes/instr/log/P2_development.md`～`notes/instr/log/P6_development.md`
- 编译器发射细节：`notes/instr/编译器后端lowering指南.md`
- ABI/Golden 分类、审批和自动门禁：[NPU ISA v1 ABI 与 Golden 变更策略](ISA_v1_ABI与Golden变更策略.md)
