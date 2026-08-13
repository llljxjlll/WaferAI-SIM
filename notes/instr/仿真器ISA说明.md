# NPUSim 仿真器 ISA 说明

> 适用版本：NPU ISA v1 / Program Format v1
>
> 适用入口：`npusim --program <artifact.npup>`
>
> 文档定位：本文面向仿真器使用者、编译器后端开发者和运行时调试人员，说明“一个 ISA 程序如何被编码、加载、执行和观测”。精确字节布局以 [Program Format v1](program_format_v1.md) 和 [ISA v1 Manifest](isa_v1_manifest.md) 为准。

## 1. ISA 在仿真器中的位置

NPUSim 同时保留两种工作负载入口：

- legacy JSON：`--workload-config`，用于兼容既有 workload；
- ISA program：`--program`，读取稳定的 Program Format v1 二进制 artifact。

两个入口互斥，但共用 hardware、simulation 和 mapping 配置。ISA program 不包含仿真器对象指针、C++ 类名或进程内 label ID；它只保存稳定的外部 Opcode、operand、符号、重定位、core group 和控制 envelope。

从 artifact 到执行的路径是：

```text
Program Format v1
  ├─ header / section table / CRC-32C
  ├─ strings / symbols / semantic relocations
  ├─ core groups / per-core external records
  └─ control envelope
            │
            ▼
loader 全文件校验
  ├─ 版本、边界、CRC、reserved
  ├─ Opcode/operand schema
  ├─ symbol/relocation/address span
  ├─ platform core/group/capability
  └─ collective graph 与派生资源上限
            │
            ▼
lowering
  ├─ record-local lowering
  ├─ whole-artifact collective lowering
  └─ internal Prim strict-wire 验证
            │
            ▼
每核 Worker 队列 + runtime
  ├─ compute / LSU / local DTE
  ├─ P2P / collective / Router / D2D
  ├─ SRAM lifecycle / event / barrier
  └─ token、session、tree、credit 与 drain
```

loader 使用两阶段提交。任何校验或 lowering 失败都发生在正常 CONFIG/START 之前，不允许把半个程序提交给一部分 core。

## 2. 外部 ISA 与内部 Prim 的区别

这是使用本 ISA 时最重要的边界：

| 名称 | 是否是外部 ABI | 谁生成 | 稳定性 |
|---|---:|---|---|
| Program Format | 是 | 编译器/fixture | v1 冻结 |
| Opcode | 是 | 编译器 | 编号、语义 append-only |
| external record | 是 | 编译器 | 8-byte header + Opcode payload |
| symbol/relocation | 是 | 编译器 | v1 冻结 |
| PrimId | 否 | loader/helper | 仅内部 manifest 管理 |
| 128-bit Prim wire | 否 | lowering/runtime | 不供 artifact 直接编码 |
| Router/P2P/DCA wire | 否 | runtime | transport 内部协议 |

一个外部 Opcode 可以 lowering 成多个内部 Prim；collective 甚至会根据完整 artifact 生成 child endpoint、phase barrier、launch marker、tree schedule 和 action image。因此：

- 编译器不得把 PrimId 当 Opcode；
- 编译器不得直接编码内部 128-bit segment；
- 编译器不得自行分配 Router tree/session；
- runtime 内部重构只有在不改变外部可见语义时才不构成 ABI 变化。

## 3. Program Format 概览

### 3.1 文件头

Program Format v1 文件头固定 64 bytes：

- magic：`NPUPRG1\0`；
- format/ISA version：`1.0 / 1.0`；
- endianness：little-endian；
- 文件大小上限：64 MiB；
- section 数量上限：64；
- 每个 section 和全文件均使用 CRC-32C。

canonical encoder 生成七个 required sections：

1. `STRING_TABLE`
2. `SYMBOL_TABLE`
3. `SEMANTIC_RELOCATION_TABLE`
4. `CORE_GROUP_TABLE`
5. `CORE_PROGRAM_INDEX`
6. `EXTERNAL_RECORD_STREAM`
7. `CONTROL_ENVELOPE`

未知 required section、section 重叠、gap 垃圾、尾部垃圾、CRC 错误、count/offset/size 溢出都会被拒绝。

### 3.2 external record

每条 record 的公共头固定为 8 bytes：

```text
opcode:u8
record_version:u8 = 1
flags:u16 = 0
payload_size:u32
payload[payload_size]
```

所有整数均为 little-endian。payload 必须与 Opcode 的唯一 schema 完全匹配；长度过短、过长、非法枚举、非零 reserved 或隐式窄化均失败。

### 3.3 符号与重定位

v1 支持三类 symbol：

- `ABSOLUTE_ADDRESS`
- `SRAM_REGION`
- `SRAM_LABEL`

`SRAM_REGION.value` 是物理 byte base，`size_bytes` 是有界 extent。该 symbol 的 relocation addend 是 region 内 byte offset。loader 转成绝对地址时只计算一次 `base + offset`，并在 lowering 前检查整个访问 span，而不只检查起始地址。

artifact 保存字符串或 symbol index，不保存 `g_addr_label_table` 等进程内编号。

## 4. Opcode 空间

外部 Opcode 按功能分区：

| 区间 | 类别 | 主要语义 |
|---|---|---|
| `0x00～0x3F` | COMPUTE | NPU 计算 |
| `0x40～0x7F` | COMM | P2P 与 collective 逻辑角色 |
| `0x80～0xBF` | MEM | blocking LSU、local DTE、SRAM 生命周期 |
| `0xC0～0xEF` | SYNC | token、event、group barrier |
| `0xF0～0xFF` | reserved | v1 一律拒绝 |

### 4.1 COMPUTE

当前 available：

| Opcode | 指令 | 参数摘要 |
|---:|---|---|
| `0x01` | `MATMUL` | `B,T,C,OC` |
| `0x04` | `CONV` | `B,W,H,C,pX,pY,sX,sY,kX,kY,F` |
| `0x05` | `MAXPOOL` | `B,W,H,C,pX,pY,sX,sY,kX,kY` |
| `0x06` | `ATTENTION` | `B,T,C,NH,R` |
| `0x08` | `GATE` | `B,T,C,E_N,K` |
| `0x09` | `MOE_MATMUL` | `B,T,C,OC,K,E_N,is_merge,need_choose` |
| `0x0A～0x0E` | `GELU/SILU/SWIGLU/RELU/RESIDUAL` | `N` |
| `0x0F～0x10` | `LAYERNORM/RMSNORM` | `B,T,C` |
| `0x11` | `ROPE` | `B,T,C,NH` |
| `0x13～0x14` | `SPLIT_MATMUL/MERGE_MATMUL` | `B,T,C,dim,slice` |
| `0x15` | `DUMMY` | 无参数；固定 `exu_ops=10` |

计算 operand 还包含：

```text
datatype
input_offset_bytes
data_offset_bytes
output_offset_bytes
parameters[]
```

当前 compute datatype 为 `INT8` 或 `FP16`。offset 是已绑定 label 内的 byte offset。

每条 compute 前必须有一条独立 `SRAM_BIND`。该 binding 是 one-shot：

```text
SRAM_BIND(inputs=[activation, weight], output=result)
MATMUL(...)
SRAM_BIND(inputs=[result], output=next)
GELU(...)
```

第一条 compute 成功开始时立即消费第一份 binding；MEM/SYNC 指令可以插在 bind 与 compute 之间，但不会消费 binding。缺 bind、连续两次 bind、arity 不匹配或程序结束仍有 dangling bind 都会失败。

### 4.2 COMM

| Opcode | 指令 | 用途 |
|---:|---|---|
| `0x40` | `DTE_SEND` | standalone P2P 或 collective send 角色 |
| `0x41` | `DTE_RECV` | standalone P2P 或 collective receive 角色 |
| `0x42` | `REDUCE_COMPUTE` | collective graph 的归约角色 |

standalone P2P 可 record-local lowering；collective 必须由 whole-artifact lowering。外部 `REDUCE_COMPUTE` 不能独立执行，也不能映射到 legacy timing-only reduce Prim。

### 4.3 MEM

| Opcode | 指令 | 语义 |
|---:|---|---|
| `0x80` | `LSU_LOAD` | blocking HBM→SRAM |
| `0x81` | `LSU_STORE` | blocking SRAM→HBM |
| `0x82` | `DTE_ISSUE` | local async copy |
| `0x83` | `SRAM_CLEAR` | targeted clear |
| `0x84` | `SRAM_BIND` | one-shot compute binding |
| `0x85` | `SRAM_ALLOC` | 建立 region/label |
| `0x86` | `SRAM_FREE` | 释放 metadata |
| `0x87` | `SRAM_RESIZE` | grow/shrink，保留有效前缀 |
| `0x88` | `SRAM_RENAME` | 原子修改 label |

`DTE_ISSUE` 只支持：

- `SPM_TO_SPM`
- `SPM_TO_DRAM`
- `DRAM_TO_SPM`

remote 传输必须使用 `DTE_SEND/RECV`，不能借 `DTE_ISSUE` 表达。

### 4.4 SYNC

| Opcode | 指令 | 语义 |
|---:|---|---|
| `0xC0` | `DTE_WAIT(token)` | 等待指定本核 token |
| `0xC1` | `DTE_FENCE` | 等待本核全部 local DTE token |
| `0xC2` | `DTE_CANCEL(token)` | 按 tracker 状态取消 |
| `0xC3` | `EVENT_SET` | 向指定同 die endpoint 增加 event credit |
| `0xC4` | `EVENT_WAIT` | 消耗匹配 credit，支持 SET 提前到达 |
| `0xC5` | `GROUP_SYNC` | public group barrier |

`GROUP_SYNC` 是唯一 public barrier；collective phase barrier 是 loader 生成的 internal action。相同 group 的 `sync_seq` 必须单调，每个成员对每一序号恰好到达一次。

## 5. 执行与并发语义

每个 core 的 external record stream 保持程序发射顺序，但并非所有工作都串行完成：

- compute 在当前 Worker 上执行并生成成本/trace；
- `LSU_LOAD/STORE` 是 blocking，完成前不放行后续依赖；
- `DTE_ISSUE` 可异步运行，使用 token 与 `WAIT/FENCE` 建立依赖；
- P2P/collective 可由独立 endpoint 或 acceleration worker 推进；
- event、barrier、Router 和 D2D 都可能引入 backpressure；
- SEND_DONE 前必须确认本核及全局相关 queue/session/token/tree 已 drain。

正确程序应显式表达依赖。不能因为仿真器当前某次调度顺序“恰好先完成”而省略 WAIT、FENCE 或 barrier。

## 6. 内存模型

### 6.1 SRAM

SRAM 访问可使用：

- absolute byte address；
- `SRAM_REGION + region_offset`；
- compute 使用 `SRAM_LABEL + bind 内 offset`。

absolute 与 region 形式必须二选一。region 访问检查 `offset + access_bytes <= size_bytes`。生命周期操作失败时必须保持旧 metadata 和数据可见性不变。

`SRAM_FREE` 释放 metadata，不隐式清除底层字节；`SRAM_CLEAR(label)` 是 targeted clear。legacy JSON 的无参 clear-all 与 public targeted clear 不是同一语义。

### 6.2 HBM

public LSU 使用 byte address 和 byte length。仿真器后端把逻辑小请求适配为硬件合法的 physical burst，并保持逻辑 byte-enable、读回聚合和一次 logical completion。统计仍按逻辑请求计费。

HBM source P2P 只接受 absolute address；collective 当前先把 HBM 数据显式 stage 到 SRAM。

### 6.3 地址和数据正确性

仿真器在不同层检查：

1. artifact symbol/relocation 边界；
2. lowering 的操作语义 span；
3. runtime region 权限和 busy 状态；
4. media read/write；
5. 测试 probe 的 payload/checksum/sentinel。

`--p5-memory-probe`、`--p6-memory-probe`、`--p8-double-buffer-probe` 是测试 sidecar，不属于应用 ISA ABI。

## 7. P2P 协议

standalone P2P 使用匹配的 send/receive 描述符。核心字段包括：

- source/destination core；
- `fsm_id`；
- completion：`SYNC` 或 `ASYNC`；
- async token；
- length；
- source/destination address；
- datatype/reduce operator。

当前 standalone P2P 是 raw `UINT8`、`reduce_op=NONE`。source 可在 SRAM 或 HBM，destination 在 SRAM。same-die 走 NoC，cross-die 经 D2D。

数据被切为 16-byte fragments，带 sequence、tail 和 CRC32C。接收端只有在完整校验 source、flow、sequence、length、tail 和 CRC 后才提交 destination；错误包不会作为完整成功事务可见。

transport tag 在 runtime 生命周期内单调分配且不复用。REQUEST duplicate 可幂等抑制，conflict 或 malformed ACK 会精确终止对应事务并清理资源。

## 8. Collective 与四档 profile

collective 不使用单独 `COLLECTIVE_CALL` Opcode。编译器以 `DTE_SEND`、`DTE_RECV` 和必要的 `REDUCE_COMPUTE` 声明逻辑角色，所有角色共享：

```text
(group_id, collective_id, epoch)
```

loader 在 whole-artifact 级别验证 graph 闭合、rank/root、source/destination span 和资源上限，然后生成 immutable action image。

支持的高层组合来自发送/接收语义：

- Unicast / Scatter / Broadcast
- Unicast / Gather / Reduce
- 由此构成 P2P、Scatter、Broadcast、Gather、Reduce、AllToAll、AllGather、ReduceScatter 和 AllReduce。

profile 来自 simulation/platform 配置，不写进 artifact：

| profile | broadcast backend | reduce backend |
|---|---|---|
| `baseline` | unicast | endpoint |
| `broadcast_only` | multicast | endpoint |
| `reduce_only` | unicast | DCA |
| `reduce_broadcast` | multicast | DCA |

所有 profile 的 external collective record 仍使用逻辑 `tree_id=0`；具体 tree、session、batch 由 loader/runtime 分配。请求的 backend 不可用、容量无法证明、跨 die group 或 profile 不一致时启动失败，禁止静默回退。

当前限制：

- collective group 必须同 die；
- endpoint integer reduction 支持 `UINT8/INT32/INT64` 的 `SUM/MAX`；
- FP reduction 不支持；
- `ReduceScatter+DCA` 当前明确 unsupported；
- tree table 每 Router 最多 64 entries；
- `max_trees_per_batch` 范围为 1～64。

## 9. 控制 envelope

control envelope 明确声明：

- active cores；
- host START events；
- terminal cores；
- expected CONFIG ACK cores；
- expected DONE cores；
- empty-core ACK policy；
- failure policy。

仿真器不会从“最后一条 record”猜 terminal。DONE 集合必须等于 terminal 集合；ACK 集合必须与 empty-core policy 一致。v1 failure policy 为 `ABORT_ALL`。

这使程序结束条件与指令内容解耦：某个 core 可以没有 record，但仍根据 envelope 参与 ACK；另一个 core 可以有 record，却不是 terminal core。

## 10. Trace、统计和 drain

默认 trace 文件为运行目录下的 `events.json`。常见观测包括：

- compute B/E 和 `Compute_cost`；
- LSU/DTE transfer B/E；
- `P6_collective_action` 与 wave；
- `[P7 PROFILE]`；
- `[P7_TREE_BATCH]` begin/end；
- `[P7_MULTICAST_TX]` / `[P7_MULTICAST_COMMIT]`；
- DCA arm/TX/result；
- P2P、collective、timing 和 global drain。

成功结束不只要求收到 DONE，还要求相关 residual 为零。重点 residual 包括：

- local DTE token/descriptor/hazard；
- P2P session/reassembler/pending wire；
- Router data/control queue、pulse cooldown、flow lock；
- collective child/FSM/barrier/aggregate；
- tree entries、DCA stream/compute、batch schedule；
- SRAM busy reference 与 lifecycle state。

调试死锁时，优先查看最后一个未完成 record、token/session key、Router/endpoint residual 和 `[PROTO_WAIT]`，不要先提高 watchdog 掩盖状态错误。

## 11. 运行一个 ISA 程序

### 11.1 构建

```bash
/opt/cmake/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
/opt/cmake/bin/cmake --build build -j2
```

### 11.2 生成参考 artifact

```bash
build/npusim_program_fixture build/minimal.npup
```

该 fixture 是参考 encoder，不初始化完整仿真。第二个位置参数可选择场景，例如：

```bash
build/npusim_program_fixture build/p8a.npup --p8-double-buffer
build/npusim_program_fixture build/p6.npup \
  --p6-broadcast-reduce-n4-l1024-i32-sum
```

不带输出路径运行会打印当前支持的 fixture 选项并返回非零 usage 状态；它没有独立的 `--help` 模式。

### 11.3 执行

```bash
build/npusim \
  --program build/minimal.npup \
  --hardware-config llm/test/default/hardware.json \
  --simulation-config llm/test/default/simulation.json \
  --mapping-config llm/test/default/mapping.spec \
  --trace-window 1000000
```

`--program` 与 `--workload-config` 不能同时出现。

### 11.4 自检

```bash
build/npusim --isa-v1-selftest
/opt/cmake/bin/ctest --test-dir build -j1 -L isa --output-on-failure
```

如果修改了 Program Format、Opcode、operand、lowering 或 internal wire，还应运行 mutation、Program runtime、P5/P6/P7/P8 和相关 R/V compatibility 门。

## 12. 常见错误

| 错误类型 | 常见原因 | 修复方向 |
|---|---|---|
| format/version/CRC | header、section 或 whole-file checksum 错 | 使用 canonical encoder，重新计算 CRC-32C |
| truncated/extra payload | record payload_size 与 schema 不同 | 按 Opcode 固定 schema 编码 |
| unknown/unsupported | 使用 reserved、experimental 或 internal-only 能力 | 查询 manifest，改用 available Opcode |
| address XOR/span | absolute 与 region 同时设置，或访问越界 | 二选一并按完整 access span 校验 |
| missing/dangling bind | compute 前没有 one-shot bind，或 bind 未消费 | 每条 compute 发独立 `SRAM_BIND` |
| token duplicate/unknown | token 重用过早或 WAIT/CANCEL 不匹配 | token 单调管理，显式 WAIT/FENCE |
| collective graph | key/rank/root/role/span 不闭合 | whole-artifact 构图后再编码 |
| profile rejected | capability、跨 die、tree/DCA 容量不满足 | 修平台配置或选择受支持的拓扑 |
| non-zero drain | 漏 WAIT、漏 endpoint completion、tree/session 未退役 | 根据 residual 与 trace 找生命周期缺口 |
| `[PROTO_WAIT]` | 等待条件不可能满足或控制依赖形成环 | 检查 START、event、barrier、P2P post 和 DONE |

## 13. 编译器与工具接入检查表

生成 artifact 前至少确认：

- [ ] 只使用 manifest 中 `public + available` 的 Opcode；
- [ ] record version、flags、payload length、enum 和 reserved 位正确；
- [ ] 所有整数在编码前完成范围检查，没有静默窄化；
- [ ] strings、symbols、relocations、groups、cores 均规范排序且唯一；
- [ ] 地址形式二选一，完整访问 span 在 symbol/region 内；
- [ ] 每条 compute 有且只有一个待消费 `SRAM_BIND`；
- [ ] local async DTE 的 token、WAIT/FENCE/CANCEL 生命周期闭合；
- [ ] P2P send/recv、fsm、length、completion 和 endpoint 匹配；
- [ ] collective graph 的 key、rank、root、角色和地址闭合；
- [ ] control envelope 的 active/START/ACK/DONE/terminal 集合一致；
- [ ] capability 与平台资源在启动前可证明；
- [ ] 编码结果通过 decode→encode byte-stability 和 golden/mutation 测试。

## 14. 相关文档

- [NPU ISA v1 Manifest](isa_v1_manifest.md)：Opcode、PrimId、状态与编号的唯一清单。
- [NPUSim Program Format v1](program_format_v1.md)：容器、section、CRC 和控制 envelope 的精确布局。
- [编译器后端 lowering 指南](编译器后端lowering指南.md)：operand、参数顺序、重定位和完整生成示例。
- [P0 契约](编译产物指令集P0契约.md)：架构决策与不变量。
- [ABI 与 Golden 变更策略](ISA_v1_ABI与Golden变更策略.md)：版本和兼容性评审规则。
- [JSON 到 Program 迁移指南](JSON到Program迁移与兼容指南.md)：双入口迁移、差分和回滚。
- [ISA v1 已知限制](ISA_v1已知限制.md)：当前不可用能力、资源上限和发布边界。
- [ISA v1 最终测试报告](ISA_v1最终测试报告.md)：Release/Debug、兼容矩阵和 sanitizer 证据。
