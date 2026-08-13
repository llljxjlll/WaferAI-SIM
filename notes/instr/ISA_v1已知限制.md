# NPU ISA v1 已知限制

> 文档状态：P8 发布候选的限制清单
>
> 适用版本：NPU ISA v1 / Program Format v1
>
> 适用入口：legacy JSON 与 `npusim --program`
>
> 状态声明：本文记录当前 production 真相和发布边界。P7 四档 NoC collective backend 已按阶段证据标为 available；P8 发布总门是否完成仍以最终测试报告和 clean Release/Debug 结果为准。只有已有实现、测试和阶段验收证据的能力才标为 available；已有编号、配置名或纯函数测试不等于运行时可用。

## 1. 如何阅读本清单

本清单中的状态含义：

| 状态 | 含义 |
|---|---|
| available | 当前 production 路径可执行，但仍受平台、格式、地址和资源上限约束 |
| experimental/unavailable | 名称、编号或规划接口可能已冻结，但 release gate 尚未开放；请求时必须 fail-fast |
| unsupported | v1 明确不提供；已知项应返回 known-but-unsupported，而不是当作 unknown |
| internal-only | 只能由 loader/runtime/legacy helper 生成，外部 artifact 不可直接编码 |
| deprecated legacy | 仅保留旧 JSON/内部 wire 兼容，不是新的外部 ABI |

状态的唯一完整清单是 [NPU ISA v1 Manifest](isa_v1_manifest.md)。本文侧重用户和编译器后端必须知道的限制，不重复复制全部 Opcode 表。

## 2. 当前不可用的计算与控制能力

### 2.1 MLA/PD capability 关闭

以下 Opcode 已有稳定编号，但当前均为 `experimental/unavailable`：

| Opcode | 指令 | 当前限制 |
|---:|---|---|
| `0x02` | `MATMUL_MLA` | `pd_context` gate 关闭 |
| `0x03` | `MATMUL_PD` | `pd_context` gate 关闭 |
| `0x07` | `ATTENTION_PD` | `pd_context` gate 关闭 |
| `0x12` | `ROPE_PD` | `pd_context` gate 关闭 |

Program Format v1 当前没有已发布的 program-aware request、batch_info、job_type、chunk、KV cache 和 scheduler lifecycle adapter。通用 helper 不能强转为 PD/PDS helper，也不能用静态 shape 代替运行时上下文。当前稳定 program artifact 使用 `capabilities=0`；helper 对 PD、GLOBAL 或 experimental capability 请求执行启动期拒绝。

### 2.2 known unsupported / experimental 指令

以下 public 编号不可执行：

| Opcode | 指令 | 状态/原因 |
|---:|---|---|
| `0x16` | `BATCHNORM` | unsupported stub |
| `0x17` | `SPLIT_CONV` | unsupported stub |
| `0x18` | `MERGE_CONV` | unsupported stub |
| `0x19` | `GEMM_REDUCE_SCATTER` | experimental，v1 loader 拒绝 |
| `0xC6` | `DTE_POLL` | unsupported；结果没有可被后续指令消费的控制流接口 |

这些指令不能以零周期成功、不能替换为 `DUMMY`，也不能静默拆到行为不等价的序列。编译器应在生成 artifact 前报告不支持。

### 2.3 GLOBAL、GPU 和 internal helper

Program v1 不支持 `GLOBAL_SEND/RECV/LOAD/STORE`。Global memory 组件在 program 模式只保留安全的平台接线，不表示 GLOBAL Opcode 可用。

GPU 和 GPU-PD primitive 不进入 NPU ISA v1 外部空间。以下 helper 仍为 internal-only：

- `load_expert`、`switch_data`、`parse_input`、`parse_output`；
- `Sram_pipeline_prim`、`Set_batch`；
- legacy `Clear_sram`、`Load_prim`、`Store_prim`、`Send_prim`、`Recv_prim`；
- `COLL_BARRIER`/`Collective_prim`、tree batch marker、collective child/action primitive；
- strict `SRAM_BIND`/lifecycle/event/group-sync/P2P/collective lowering targets。

外部编译器不得把 internal `PrimId` 或 128-bit Prim wire 写入 artifact。若 helper 没有等价 public lowering，workload 必须继续使用 JSON 或明确拒绝。

## 3. JSON 与 `--program` 仍处于并存期

- JSON 与 `--program` 至少并存一个正式发布周期；当前不废弃 JSON。
- `--program` 和显式 `--workload-config` 互斥，同时指定时失败，不存在隐式优先级。
- 未指定 `--program` 时保留默认 JSON workload 行为。
- 两种入口共用 hardware/simulation/mapping；program 不读取或伪造 workload JSON，并固定使用 `SIM_DATAFLOW`。
- legacy JSON `Set_addr` 是 persistent；public `SRAM_BIND` 是 one-shot，每条 compute 都需要新的 binding。
- legacy 无参 `Clear_sram` 是 clear-all；public `SRAM_CLEAR(label)` 是 targeted，且只清理符合条件的 allocation。
- legacy `parse_input` 的特定 replacement 可覆盖旧 label；public `SRAM_RENAME` 严格拒绝 duplicate target。
- 内部 Prim wire 不承诺跨版本稳定；只有外部 Program Format/Opcode/operand/symbol/relocation 是编译器 ABI。

完整迁移、差分和回滚流程见 [legacy JSON 到 `--program` 迁移与兼容指南](JSON到Program迁移与兼容指南.md)。

## 4. 通信拓扑和数据路径限制

### 4.1 P2P 与 collective 的同/跨 die 边界

| 能力 | 同 die | 跨 die | 当前边界 |
|---|---:|---:|---|
| standalone P2P | 支持 | 支持 | 使用成对 `DTE_SEND(P2P)` / `DTE_RECV(P2P)`，真实字节走 NoC/D2D |
| Scatter/Broadcast/Gather/Reduce | 支持 | 不支持 | group 全部成员必须位于同一 die |
| AllToAll/AllGather/ReduceScatter/AllReduce | 支持 | 不支持 | whole-artifact lowering，同 die静态 group |
| `GROUP_SYNC` | 支持 | 不支持 | core group v1 必须同 die |
| `EVENT_SET/WAIT` | 支持 | 不支持 | 当前 event endpoint 限于同 die |

P2P 的协议允许 same-die 和 cross-die；当前发布证据分别覆盖 cross-die SRAM P2P 和 HBM source P2P，不据此承诺所有 source/backend/profile 组合都已形成独立性能 golden。

### 4.2 P2P operand 与完成语义

- standalone P2P 只支持 raw `UINT8` payload 和 `reduce_op=NONE`；
- send source 可为 SRAM 或 HBM，HBM source 必须是 absolute address；receive destination 是 SRAM；
- source/destination core 不能相同；同核搬运应使用 local DTE 或本地 copy；
- `fsm_id` 是非零 u32；ASYNC token 是非零 u32，SYNC token 必须为 0；
- remote 方向不能用 `DTE_ISSUE` 表达；`SPM_TO_REMOTE`、`REMOTE_TO_SPM`、`DRAM_TO_REMOTE` 在该 Opcode 下拒绝；
- RX 完整校验 seq、tail、length、source、fsm 和 CRC32C 后才提交 destination SRAM；错误 payload 不应部分可见；
- active P2P cancel 当前不是任意时刻可抢占的通用事务回滚接口，编译器应按已发布 tracker 状态约束使用 CANCEL。

### 4.3 collective 数据限制

- 已验收四档 production backend：`baseline=unicast+endpoint`、`broadcast_only=multicast+endpoint`、`reduce_only=unicast+DCA`、`reduce_broadcast=multicast+DCA`；P7 加速正向 production 证据集中于 N=4、32 KiB AllGather/AllReduce，其他 collective、dtype、tail 和 N 的广度主要由 P6 baseline 与纯层门覆盖，不自动构成独立性能 golden；
- 非归约 collective 搬运 raw UINT8 bytes；
- endpoint reduction 只支持 `UINT8/INT32/INT64` 的 `SUM/MAX`；FP reduction unsupported；
- INT32/INT64 使用 little-endian 二进制补码，SUM 按目标位宽取模，MAX 做有符号比较；
- 普通 Reduce 是单 root；只有 ReduceScatter/AllReduce 使用 per-target 对称分解；
- `N=group_size`，本地贡献不走网络，远端通常为 `N-1` 份；
- 当前 whole-artifact collective child source、staging 和 result 使用 SRAM。HBM 数据应先经 blocking LSU 或 local DTE 显式 stage 到 SRAM；
- P6 baseline runtime 证据覆盖 N=1/2/4、1 byte/tail/1/8/32 KiB 及 UINT8/INT32/INT64 代表矩阵；P7 四档 production 门覆盖 N=4、32 KiB AllGather/AllReduce。更大 group 或不同复合压力仍受同一 loader/resource gate，不因字段可编码就自动获得发布性能保证。

`REDUCE_COMPUTE(0x42)` 只能作为闭合 collective graph 的 external 逻辑角色，经 whole-artifact lowering 生成 strict `Collective_data_v1_prim (PrimId 0x3A)`。legacy `Reduce_compute_prim (PrimId 0x2B)` 是 internal timing-only creator，不读取/写回完整真实 reduction 数据，禁止作为 external `0x42` 的实现或 record-local lowering target。

## 5. P7 acceleration 可用范围与限制

四个 profile 的稳定请求名及当前发布状态：

| profile | 请求 backend | 当前发布状态 |
|---|---|---|
| `baseline` | unicast broadcast + endpoint reduce | available，P6 真实字节矩阵已验收 |
| `broadcast_only` | multicast broadcast + endpoint reduce | available，P7 production multicast 与 32 KiB 四档矩阵已验收 |
| `reduce_only` | unicast broadcast + DCA reduce | available，P7 production DCA 与 32 KiB 四档矩阵已验收 |
| `reduce_broadcast` | multicast broadcast + DCA reduce | available，P7 两 backend 正交组合与 32 KiB 四档矩阵已验收 |

限制与安全规则：

- profile 由平台配置选择，不改变 artifact 的高层 collective record；multicast 与 DCA 均走 production 真实 SRAM 字节路径，不是 metadata-only 或 synthetic-rank executor；
- profile/backend 配置不一致或资源无法证明安全时必须 fail-fast，禁止自动回落到 baseline；
- `ReduceScatter+DCA` capability 当前关闭；请求该组合必须拒绝，不能静默改用 endpoint；
- multicast 只适用于实际 Broadcast 语义及其 AllGather/AllReduce broadcast phase；Scatter 不分配 multicast tree；
- 每个 Router 的 collective tree table 上限为 64 entries；`max_trees_per_batch` 的 production 范围为 `[1,64]`。已有占用、冲突、批大小 K 和派生状态都纳入容量证明，tree 实际按批 program/erase，并以 occupancy/drain trace 验证回收；
- DCA/core 共享 production compute pool 的持续纯层门覆盖 32 CORE + 32 DCA、L=5、II=2、result depth=1，冻结 RR 严格交替、同源 service gap `<=2*II`、每 tag/completion exactly once 和最终 drain；
- 加速档的 available 是功能正确性承诺，不是相对 baseline 的性能 SLA；性能仍必须用 trace 分解 queue、stall、batch barrier、DCA latency/II 和尾部 drain。

P7 N=4、32 KiB AllGather/AllReduce 四档矩阵证明上述 production backend 可执行并保持真实值、生命周期和全 drain；它不等价于所有 N/chunk/dtype/tail 组合都具有独立 production 性能 golden。

## 6. Program Format 和编码上限

### 6.1 容器

| 项目 | v1 上限/要求 |
|---|---|
| 文件大小 | 64 MiB，包含 header、section table 和全部 section |
| header | 固定 64 bytes，magic=`NPUPRG1\0` |
| format / ISA | 只接受当前支持的 1.0；不做猜测性版本兼容 |
| endianness | 只接受 little-endian |
| sections | canonical encoder 生成 7 个 required sections；decoder 总数上限 64 |
| strings | 最多 65,536 项；单项最多 255 UTF-8 bytes；总 string table 最多 16 MiB |
| symbols | 最多 1,048,576 项 |
| semantic relocations | 最多 1,048,576 项；同一语义 operand 目标不可重复 |
| core groups | 最多 65,536 项 |
| program cores | 最多 65,536 项，仍需落在实际平台 active core 范围内 |
| external records | 全 artifact 最多 1,048,576 条 |
| checksum | 每 section 和 whole file 都使用 CRC-32C；whole CRC 计算时 header `[56,60)` 视为 0 |

字符串必须是规范 UTF-8、不得含 NUL、不得重复。已知 section 的 flags/reserved 必须符合 v1；section 重叠、gap 垃圾、尾部垃圾、count/offset/size 溢出或 CRC 不符都会在 CONFIG 前拒绝。

### 6.2 字段和名称

- core/rank/root/tree/phase 是 u16；group/collective/epoch/sync_seq、fsm/token/event tag 是 u32；
- `collective_id=0xFFFFFFFF` 永久保留给纯 `GROUP_SYNC`；group_id 0 非法；
- compute 的 input/data/output offset 是 u16 byte offset；shape/parameter 最大 `2^30-1`；
- HBM/SRAM/remote address、offset、length/size 是 u64 byte 单位，但仍受实际 region、memory 和 transport 上限；
- SRAM region 名最多 64 UTF-8 bytes；SRAM label 最多 255 UTF-8 bytes；均不得含 NUL；
- start host wire 进一步限制 `tag<=65535`、`1<=count<=255`；
- absolute SRAM address 与 `region+offset` 必须恰好二选一；不允许隐式窄化或 `sc_bv` 截断。

### 6.3 控制 envelope

- active/source/terminal/ACK/DONE 必须显式声明，不能从最后一条指令推导；
- `INCLUDE_EMPTY` 时 ACK 集合等于全部 program cores；`EXCLUDE_EMPTY` 时只含非空 record stream cores；
- DONE 集合必须等于 terminal 集合；
- core/group/list 必须规范有序且唯一；
- v1 只支持 `ABORT_ALL` failure policy；没有 per-core 局部继续执行策略。

完整 byte layout 和 CRC 规则见 [NPUSim Program Format v1](program_format_v1.md)。

## 7. 地址、payload 和资源上限

### 7.1 region/span

- `SRAM_REGION.value` 是物理 byte base，`size_bytes` 是有界 extent；relocation addend 和 named address offset 都是 region-local byte offset；
- named-region 保留形式不能把 base 加进 offset；转 absolute 时只能加一次；
- 普通 endpoint/reduction result 的 span 是 `L`；Scatter source、Gather/Reduce destination 和 reduction staging 的 span 是 `N*L`；
- 负 addend、零 region size、`base+size`、`offset+span` 或 `N*L` 溢出，以及跨 region 访问都在 lowering/运行前拒绝；
- unaligned 地址可由已发布路径验证，但是否产生 bank/port wait 取决于平台 SRAM/HBM 配置，不存在统一零代价保证；
- `SRAM_ALLOC.alignment` 必须是非零 2 的幂；zero-size allocation/transfer 拒绝。

### 7.2 endpoint 和 planner

- 每个 endpoint DATA fragment 携带 16 bytes，sequence 是 u16；standalone P2P 单次 `length_bytes` 最大 `65535*16 = 1,048,560` bytes；
- collective 逻辑 payload 可被 planner 分成 child chunk，每个 child 最大 1,048,560 bytes；所有 chunk、action、wave 和 staging span 仍需通过整体上限；
- production loader 当前把每 rank/core 每 wave 的 endpoint session 上限设为 `MAX_BUFFER_PACKET_SIZE=3`，每 core wave 的 receive bytes 上限为 1,048,560；
- baseline planner 默认最多 1,048,576 children、1,048,576 actions、32,768 waves，并限制派生状态为 1,048,576 bytes；实际 program image 还有 plans、issue sites、tokens 和 wave demands 的独立上限；
- endpoint transport tag 是 lifetime-unique u16，不 wrap、不复用；耗尽时明确失败，而不是冒险接受 stale ACK；
- Router、D2D、endpoint、event、token、reassembly 和 SRAM/HBM 队列都是有限容量。队列满只能 backpressure、分 wave 或 fail-fast，不能假设无界缓冲。

上述数值是当前 production/default gate，不是编译器可用来预分配同等大内存的建议值。后端应尽早估算派生规模，并把超限诊断定位到 group、collective key 和 record。

## 8. SRAM lifecycle 限制

- `SRAM_ALLOC/FREE/RESIZE/RENAME` 是 RegionTable 的薄封装，不提供独立虚拟内存或跨 core共享 allocation；
- `SRAM_FREE` 只释放 metadata，不清底层字节；
- public `SRAM_CLEAR(label)` 只接受唯一有效、task-lifetime、spillable、non-busy allocation；persistent/layer/nonspillable/missing/double clear 拒绝；
- pending one-shot binding 引用的 allocation 不能 FREE/RESIZE/CLEAR；RENAME 会更新 pending label；
- metadata 操作本身不应产生虚假数据面周期；CLEAR 必须通过 SRAM AccessUnit 体现真实清零时序；
- program 结束允许按契约保留 persistent/layer/output region，但临时 task state 和 residual 必须按对应测试清理。

## 9. 性能模型的历史语义

ISA v1 冻结当前 simulator 成本行为，不表示这些公式等同于所有真实硬件：

- `RELU` 计 EXU；
- `MAXPOOL` 计 SFU；
- `MERGE_MATMUL` 按 `B*T*C` 计 EXU；
- `RMSNORM` 保持当前历史 SFU/结果行为；
- `DUMMY` 固定 `exu_ops=10`，不是零成本 NOP；
- compute 最终周期按现有 EXU/SFU/VEC 最大值和 DRAM overlap 规则结算一次；ISA/lowering 不重复计费；
- LSU blocking 与后续 compute 不重叠；local DTE/endpoint async 只有在显式 WAIT/FENCE 之前可重叠；
- 仿真周期和 trace 是模型输出，不是 wall-clock 性能 SLA；未形成 trace/oracle 的差异不能通过放宽阈值接受。

历史成本若要修正，必须作为独立性能语义变更，更新 oracle/golden 并按 ABI/发布策略评审，不能混入迁移或 P7 加速开放。

## 10. 安全和 fail-fast 行为

以下情况必须在下发任何 CONFIG 或进入对应 runtime 前拒绝：

- magic、format/ISA version、endianness、file/section size、CRC 或 reserved 字段错误；
- unknown、reserved、unsupported、experimental/capability-disabled Opcode；
- operand variant、payload size、enum、地址 XOR、范围、span 或窄化错误；
- duplicate/unknown/wrong-kind symbol，非法 relocation 或 addend 溢出；
- core/group/envelope 不闭合、跨 die collective group、P2P/collective role 不匹配；
- standalone record-local `REDUCE_COMPUTE`；
- planner/image/tree/DCA/session/tag/buffer 资源无法证明安全；
- profile/backend 配置不一致、`ReduceScatter+DCA` 或其他明确 unsupported 组合。

loader 采用“完整解析与校验→重定位→whole-artifact lowering→全部内部 wire 预验证→原子提交”。任一步失败都不得留下半份 `coreconfigs`、label intern、group registry、collective image 或 CONFIG 状态。

运行期 checksum、seq/tail、fsm/token、event/barrier、tree/session 或最终 drain 错误是程序失败，不能以 watchdog、析构清理、扩大无界 buffer或 silent fallback 当作成功。unknown、reserved、known unsupported 和 capability disabled 应保持不同错误类别，便于编译器和用户诊断。

## 11. 尚未由本清单宣称完成的发布工作

本文不宣称以下工作已经完成：

- `ReduceScatter+DCA` 正向能力开放；
- 更广 N/chunk/dtype/tail，以及 P2P、DCA、multicast production 重叠场景的独立性能 golden；
- 最新工作树上的 clean Release/Debug、sanitizer、重复运行/状态泄漏和完整 CI 门禁最终汇总；
- 最终性能报告、测试报告和发布评审。

这些项目只有在对应自动化证据、交叉评审和发布记录完成后，才能从限制清单移除或改为 available。

## 12. 相关文档

- 状态、Opcode、PrimId 与 strict/legacy lowering 边界：[NPU ISA v1 Manifest](isa_v1_manifest.md)
- Program byte layout、section、CRC 与资源上限：[NPUSim Program Format v1](program_format_v1.md)
- 编译器发射、one-shot、P2P/collective lowering：[NPU ISA v1 编译器后端 lowering 指南](编译器后端lowering指南.md)
- JSON/program 并存、行为差异与回滚：[legacy JSON 到 `--program` 迁移与兼容指南](JSON到Program迁移与兼容指南.md)
- ABI、Golden、版本和审批门禁：[NPU ISA v1 ABI 与 Golden 变更策略](ISA_v1_ABI与Golden变更策略.md)
- 冻结的字段、能力和同步契约：[编译产物指令集 P0 契约](编译产物指令集P0契约.md)
- 阶段测试与最终发布条件：[编译产物指令集开发计划](编译产物指令集开发计划.md)
