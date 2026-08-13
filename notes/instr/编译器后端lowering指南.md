# NPU ISA v1 编译器后端 lowering 指南

> 文档状态：P8 发布文档草案
>
> 适用接口：NPU ISA v1 / Program Format v1 / `npusim --program`
>
> 事实来源：`编译产物指令集P0契约.md`、`isa_v1_manifest.md`、`program_format_v1.md`、P0～P7 开发记录及当前 public codec
>
> P7 状态说明：四档 profile 的 production multicast/DCA、批级 tree 生命周期和 N=4、32 KiB AllGather/AllReduce 运行时矩阵已验收；profile 由平台配置选择，不改变 artifact 高层 record。`ReduceScatter+DCA` 仍为 known unsupported 并 fail-fast。available 表示功能正确性，不是性能 SLA。

## 1. 后端的交付边界

编译器后端输出的是带版本的 `ProgramArtifact`，不是内部 Prim wire。完整流程为：

```text
高层图/调度结果
  → 每核 public external records
  → strings / symbols / semantic relocations
  → core groups / control envelope
  → EncodeProgramArtifact
  → .npup
  → config_helper_program 完整校验、重定位和 whole-artifact lowering
  → 内部 Prim / collective action image
  → WorkerCore / DTE / NoC / SRAM
```

后端负责：

- 选择 public、available 的外部 Opcode；
- 按 opcode schema 填写强类型 operand，并使用 byte 单位；
- 为每个 core 给出逻辑 record 顺序；
- 建立稳定字符串、符号和语义重定位；
- 声明静态 core group 和显式控制 envelope；
- 分配不碰撞的 public token、FSM、collective key、epoch 和同步序号；
- 在写文件前执行与 public codec 等价的范围、能力和闭包校验。

后端不得：

- 把 `PrimId`、`vector<sc_bv<128>>`、内部 segment 或 `g_addr_label_table` ID 写入 artifact；
- 直接生成 `COLLECTIVE_CALL`、`COLL_BARRIER`、tree batch marker、child endpoint token 或其他 internal primitive；
- 依赖未定义的 record 顺序推导 source、terminal、ACK 或 DONE；
- 对 unknown、reserved、unsupported 或 capability-disabled 指令做静默降级；
- 通过 `DTE_ISSUE` 表达 remote 传输，或通过 public artifact 编码 LSU async、`DTE_POLL`、GLOBAL、GPU、数据准备 helper。

## 2. 外部 record 与 Opcode

每条 external record 的公共头固定为 8 bytes，所有字段 little-endian：

```text
opcode:u8, record_version:u8=1, flags:u16=0, payload_size:u32
```

payload 必须由 opcode 专属 schema 编码。后端应直接复用 `record_codec.h` 中的 `ExternalRecord`、operand struct 和 `EncodeExternalRecord`，或在其他语言中逐字段实现同一 ABI，并用仓库 golden 做互操作测试。

### 2.1 可直接发射的 Opcode

| 类别 | v1 available public Opcode | lowering 要点 |
|---|---|---|
| COMPUTE | `MATMUL`、`CONV`、`MAXPOOL`、`ATTENTION`、`GATE`、`MOE_MATMUL`、`GELU`、`SILU`、`SWIGLU`、`RELU`、`RESIDUAL`、`LAYERNORM`、`RMSNORM`、`ROPE`、`SPLIT_MATMUL`、`MERGE_MATMUL`、`DUMMY` | 每条 compute 前必须有独立 `SRAM_BIND` |
| COMM | `DTE_SEND`、`DTE_RECV`、`REDUCE_COMPUTE` | 仅 standalone P2P 可 record-local lowering；collective 和 `REDUCE_COMPUTE` 必须 whole-artifact lowering |
| MEM | `LSU_LOAD`、`LSU_STORE`、`DTE_ISSUE`、`SRAM_CLEAR`、`SRAM_BIND`、`SRAM_ALLOC`、`SRAM_FREE`、`SRAM_RESIZE`、`SRAM_RENAME` | LSU 仅 blocking；DTE_ISSUE 仅三个 local direction |
| SYNC | `DTE_WAIT`、`DTE_FENCE`、`DTE_CANCEL`、`EVENT_SET`、`EVENT_WAIT`、`GROUP_SYNC` | `DTE_FENCE` 无参；`GROUP_SYNC` 是唯一 public barrier |

数值 Opcode 与 operand schema：

| Opcode | 指令 | operand 类型 | payload bytes |
|---:|---|---|---:|
| `0x01` | `MATMUL` | `ComputeOperands`，4 个 parameters | 24 |
| `0x04` | `CONV` | `ComputeOperands`，11 个 parameters | 52 |
| `0x05` | `MAXPOOL` | `ComputeOperands`，10 个 parameters | 48 |
| `0x06` | `ATTENTION` | `ComputeOperands`，5 个 parameters | 28 |
| `0x08` | `GATE` | `ComputeOperands`，5 个 parameters | 28 |
| `0x09` | `MOE_MATMUL` | `ComputeOperands`，8 个 parameters | 40 |
| `0x0A`、`0x0B`、`0x0C`、`0x0D`、`0x0E` | `GELU`、`SILU`、`SWIGLU`、`RELU`、`RESIDUAL` | `ComputeOperands`，1 个 parameter | 12 |
| `0x0F`、`0x10` | `LAYERNORM`、`RMSNORM` | `ComputeOperands`，3 个 parameters | 20 |
| `0x11` | `ROPE` | `ComputeOperands`，4 个 parameters | 24 |
| `0x13`、`0x14` | `SPLIT_MATMUL`、`MERGE_MATMUL` | `ComputeOperands`，5 个 parameters | 28 |
| `0x15` | `DUMMY` | `ComputeOperands`，空 parameters | 8 |
| `0x40` | `DTE_SEND` | `DteSendOperands` | 72 |
| `0x41` | `DTE_RECV` | `DteRecvOperands` | 72 |
| `0x42` | `REDUCE_COMPUTE` | `ReduceComputeOperands`；whole-artifact only | 80 |
| `0x80`、`0x81` | `LSU_LOAD`、`LSU_STORE` | `LsuOperands` | 40 |
| `0x82` | `DTE_ISSUE` | `DteIssueOperands` | 80 |
| `0x83`、`0x86` | `SRAM_CLEAR`、`SRAM_FREE` | `SymbolOperands` | 4 |
| `0x84` | `SRAM_BIND` | `SramBindOperands` | 72 |
| `0x85` | `SRAM_ALLOC` | `SramAllocOperands` | 32 |
| `0x87` | `SRAM_RESIZE` | `SramResizeOperands` | 16 |
| `0x88` | `SRAM_RENAME` | `SramRenameOperands` | 8 |
| `0xC0`、`0xC2` | `DTE_WAIT`、`DTE_CANCEL` | `TokenOperands` | 4 |
| `0xC1` | `DTE_FENCE` | `NoOperands` | 0 |
| `0xC3` | `EVENT_SET` | `EventSetOperands` | 8 |
| `0xC4` | `EVENT_WAIT` | `EventWaitOperands` | 12 |
| `0xC5` | `GROUP_SYNC` | `GroupSyncOperands` | 8 |

表中的 payload bytes 不含 8-byte external record header。`ComputeOperands` 的大小为 `8+4*N`，其中 N 是 parameters 数量。

`REDUCE_COMPUTE(0x42)` 是 external collective graph 的逻辑角色，不是可独立执行的 record。它只有在完整 artifact 的 send/recv/key/rank/span 闭合后，才能由 whole-artifact lowering 生成 strict `Collective_data_v1_prim (PrimId 0x3A)`。对它调用 record-local `LowerExternalRecord` 必须拒绝；legacy `Reduce_compute_prim (PrimId 0x2B)` 仅是 internal timing-only creator，禁止作为 external `0x42` 的数据实现或 lowering target。

必须拒绝而不能生成的已知项：

- `BATCHNORM(0x16)`、`SPLIT_CONV(0x17)`、`MERGE_CONV(0x18)`；
- `GEMM_REDUCE_SCATTER(0x19)`；
- `DTE_POLL(0xC6)`；
- GPU、GLOBAL、legacy Load/Store、`load_expert`、`switch_data`、`parse_input`、`parse_output`、`Sram_pipeline`、`Set_batch`；
- `MATMUL_MLA(0x02)`、`MATMUL_PD(0x03)`、`ATTENTION_PD(0x07)`、`ROPE_PD(0x12)`，直到 `pd_context` 的 program-aware adapter 和 golden 被正式开放。

### 2.2 计算 operand

所有已发布计算指令使用：

```text
datatype:u8, input_offset_bytes:u16, data_offset_bytes:u16,
output_offset_bytes:u16, parameters:u32[]
```

`datatype` 仅 `INT8=0`、`FP16=1`。三个 offset 是绑定 label 内的 byte offset，最大 65535。每个 parameter 有效范围是 `0..2^30-1`；参数数量和顺序必须与下表完全一致。

| Opcode | parameters 顺序 |
|---|---|
| `MATMUL` | `B,T,C,OC` |
| `CONV` | `B,W,H,C,pX,pY,sX,sY,kX,kY,F` |
| `MAXPOOL` | `B,W,H,C,pX,pY,sX,sY,kX,kY` |
| `ATTENTION` | `B,T,C,NH,R` |
| `GATE` | `B,T,C,E_N,K` |
| `MOE_MATMUL` | `B,T,C,OC,K,E_N,is_merge,need_choose` |
| `GELU`、`SILU`、`SWIGLU`、`RELU`、`RESIDUAL` | `N` |
| `LAYERNORM`、`RMSNORM` | `B,T,C` |
| `ROPE` | `B,T,C,NH` |
| `SPLIT_MATMUL`、`MERGE_MATMUL` | `B,T,C,dim,slice` |
| `DUMMY` | 空数组；它仍有固定 `exu_ops=10`，不是 NOP |

后端还应在发射前检查 shape 派生乘法、输入/权重/输出布局、除数、卷积输出范围和内部现有 `int` 存储上限。不要依赖 loader 的窄化检查代替编译器诊断。

## 3. 字符串、符号与重定位

### 3.1 稳定表构造

字符串必须是无 NUL 的规范 UTF-8，单项不超过 255 bytes，表内不得重复。建议后端按 UTF-8 byte lexicographic order 规范化字符串和符号，再重写所有索引，以得到确定性 artifact。

symbol kind：

- `ABSOLUTE_ADDRESS=1`：`value` 是物理 byte address；
- `SRAM_REGION=2`：`value` 是物理 byte base，`size_bytes` 是有界 extent；
- `SRAM_LABEL=3`：稳定 label 名，不能用进程内数字 ID。

同一名字不得重复定义。符号插入顺序变化不得改变 lowering 语义。

### 3.2 地址 operand

`SramAddressOperand` 必须精确选择一种形式：

```text
ABSOLUTE: absolute_address_bytes 有效，其余地址字段为 0
REGION:   region_symbol_index + region_offset_bytes 有效，absolute 为 0
NONE:     仅 opcode schema 明确允许缺少该端时使用，三个数据字段均为 0
```

对 `SRAM_REGION`：

- symbol `value` 只表示物理 base；
- relocation `addend` 或 address operand 的 offset 只表示 region-local byte offset；
- 保留 named-region 表达时，不能把 base 再加到 offset；
- 转绝对地址时只能计算一次 `value + offset`；
- 必须检查 `offset + access_bytes <= size_bytes` 及全部加法溢出。

span 规则：普通 endpoint/归约结果是 `L`；Scatter source、Gather/Reduce destination 和 reduction staging 是 `N*L`。

### 3.3 语义重定位

重定位定位的是 `{core_index,instruction_index,operand_id}`，不是文件 byte offset。常用 operand ID：

| 语义 | operand ID |
|---|---|
| compute input/data/output 地址 | `COMPUTE_INPUT_ADDRESS` / `COMPUTE_DATA_ADDRESS` / `COMPUTE_OUTPUT_ADDRESS` |
| endpoint/DTE source、destination | `SOURCE_ADDRESS` / `DESTINATION_ADDRESS` |
| HBM 地址 | `HBM_ADDRESS` |
| lifecycle 通用 symbol | `SYMBOL` |
| alloc region 名与 label | `REGION_NAME` / `LABEL_SYMBOL` |
| rename | `OLD_SYMBOL` / `NEW_SYMBOL` |
| bind 输入 0～15 与输出 | `SRAM_BIND_INPUT_0..15` / `SRAM_BIND_OUTPUT` |

每个目标最多一个 relocation；kind 必须与 symbol kind 匹配。未知符号、负 region offset、重复目标或 addend/span 溢出必须在编码前拒绝。

## 4. `SRAM_BIND` 与计算 lowering

public `SRAM_BIND` 是 one-shot，不是 legacy JSON `Set_addr` 的 persistent binding。

规则：

1. `input_count` 必须在 1～16；未用的 input slot 必须为 0；output symbol 必须存在。
2. bind 后的 MEM、SYNC、COMM 指令不消费 binding。
3. 下一条成功开始的 COMPUTE 消费 binding，随后立即失效。
4. 连续两条 compute 必须有两条 bind。
5. 重复 bind、compute 缺 bind、输入 arity 不匹配、程序结束仍有 dangling bind 都是错误。
6. pending binding 引用的 region 不能被 `FREE/RESIZE/CLEAR`；`RENAME` 会原子更新 pending label。

推荐基本块内调度模板：

```text
MEM/SYNC 准备输入
SRAM_BIND(inputs=[...], output=...)
COMPUTE(...)
MEM/SYNC/COMM
SRAM_BIND(inputs=[...], output=...)
COMPUTE(...)
```

不要为了复用 legacy 调度而合并 bind，也不要把 label 数字 intern ID 固化到 artifact。

## 5. 访存 lowering

### 5.1 blocking LSU

- `LSU_LOAD`：HBM→SRAM；
- `LSU_STORE`：SRAM→HBM；
- operand 为 `hbm_address_bytes`、非零 `size_bytes` 和一个存在的 SRAM address；
- LSU 完成后才放行下一条指令，后端不能把它标记为可与后续 compute 重叠。

v1 不存在 public LSU issue/wait/poll/fence/cancel。

### 5.2 local async DTE

`DTE_ISSUE` 只允许：

| direction | source | destination | 禁用字段 |
|---|---|---|---|
| `SPM_TO_SPM` | SRAM | SRAM | `hbm_address_bytes=0` |
| `SPM_TO_DRAM` | SRAM | HBM/DRAM | destination SRAM=`NONE` |
| `DRAM_TO_SPM` | HBM/DRAM | SRAM | source SRAM=`NONE` |

`token` 非零且为本核命名空间；`size_bytes>0`；`payload_bits` 必须恰好等于 `size_bytes*8`。异步序列使用 `DTE_WAIT(token)`、无参 `DTE_FENCE` 或 `DTE_CANCEL(token)`。不要生成 `FENCE_ALL` 独立 Opcode；它只能是编译器对 `DTE_FENCE` 的别名。

### 5.3 SRAM lifecycle

- `SRAM_ALLOC`：region name、label、size、2 的幂 alignment、lifetime、spillable；
- `SRAM_RESIZE`：symbol 和非零新尺寸；
- `SRAM_RENAME`：不同的 old/new symbol；
- `SRAM_FREE`：只释放 metadata，不清字节；
- `SRAM_CLEAR(label)`：targeted 清真实字节后释放 task-lifetime、spillable region。

需要清数据时不能把 `FREE` 当作 `CLEAR`。legacy 无参 clear-all 不可从 artifact 编码。

## 6. P2P lowering

standalone P2P 使用一对 `DTE_SEND(mode=P2P)` / `DTE_RECV(mode=P2P)`：

- send/recv `fsm_id` 相同且非零；
- `length_bytes`、source/destination、datatype、completion 配对；
- standalone key 固定 `group_id=collective_id=epoch=0`；
- send `peer_core=destination`，recv `peer_core=source`；
- `tree_id=0`、`expected_sources=0`、`reduce_op=NONE`、datatype=`UINT8`；
- ASYNC 的 token 非零且各 core 本地唯一，SYNC 的 token 必须为 0；
- send source 可为 SRAM 或 HBM；HBM source 只能使用 absolute address；recv destination 为 SRAM；
- source core 和 destination core 必须不同。

P2P 可以同 die或跨 die，真实字节经 SRAM/HBM read、NoC/D2D、CRC/reassembly、`kNocRx` write；后端不能生成直接目的 SRAM copy。

接收 descriptor 应在依赖图上先于可能阻塞的发送。即使 record 表的逻辑角色顺序不同，也不能构造“所有核先同步 send”的循环等待。

## 7. Collective lowering

### 7.1 高层语义到 TX×RX

| 高层语义 | TX | RX | 获得结果的 rank |
|---|---|---|---|
| P2P | UNICAST | UNICAST | 单 destination |
| Scatter | SCATTER | UNICAST | 全部成员，各自不同 slice |
| Broadcast | BROADCAST | UNICAST | 全部成员，同一数据 |
| Gather | UNICAST | GATHER | 单 root，按 rank 拼接 |
| Reduce | UNICAST | REDUCE | 单 root |
| AllToAll | SCATTER | GATHER | 全部成员 |
| AllGather | BROADCAST | GATHER | 全部成员 |
| ReduceScatter | SCATTER | REDUCE | 全部成员，各自不同归约 slice |
| AllReduce | BROADCAST | REDUCE | 全部成员，相同归约结果 |

`N=group_size`，每个成员贡献一份；本地贡献不走网络。普通 Reduce 只有 root 获得结果，不能编译成对称 reduce。ReduceScatter/AllReduce 才由 loader 按 target 对称展开。

### 7.2 后端发射规则

1. 在 `core_groups` 中只声明一次升序、唯一、同 die成员表；record 只引用非零 `group_id`。
2. 同一 collective 的所有逻辑角色共享 `{group_id,collective_id,epoch}`；`collective_id=0xFFFFFFFF` 永远保留给 `GROUP_SYNC`。
3. 对应 rank 按语义发射 `DTE_SEND`、`DTE_RECV`，归约目标另发射 `REDUCE_COMPUTE`。三类角色必须在同一 whole-artifact graph 中闭合；send/recv 使用 ASYNC token。
4. keyed record 的 `peer_core=0`；所有 profile 的 external record 始终使用 `tree_id=0`，具体 child peer、tree/session 和 batch 由 whole-artifact loader/profile sidecar 确定性分配。
5. Gather/Reduce 的 `expected_sources=N-1`；P2P/UNICAST_RX 为 0。
6. 非归约 send/recv 使用 `UINT8/NONE`。Reduce/ReduceScatter/AllReduce 的 recv 与 `REDUCE_COMPUTE` 使用一致的 `UINT8/INT32/INT64` 和 `SUM/MAX`。
7. `REDUCE_COMPUTE.element_count = length_bytes / dtype_bytes`，必须整除；source 为 N 份 staging，destination 为 L-byte result。
8. 对每个 public send/recv token发射 `DTE_WAIT`，最后可发射 `DTE_FENCE`；不要发射 internal phase barrier。

loader 会对整个 artifact 建图，并生成 receive-post→send-issue→receive completion→send token→internal phase barrier 的不可变 action image。因此 external record 是逻辑角色声明，不是让后端手工编码 child endpoint 或 barrier 的接口。

非 P2P collective 的 group 不允许跨 die。所有 child flow、wave、offset、fsm 和内部 token 由 loader 确定性分配并受容量上限约束。

尤其不能先对单条 `REDUCE_COMPUTE` 做 direct lowering 再拼装 collective；record-local 门禁必须拒绝它，否则就会落入 legacy timing-only 语义并绕过真实 staging 读写、整数归约和最终可见性检查。

### 7.3 整数归约

- payload/SRAM 是 little-endian；
- UINT8 SUM 模 `2^8`，MAX 无符号比较；
- INT32/INT64 是二进制补码；SUM 每一步模 `2^32/2^64`；MAX 有符号比较；
- FP endpoint reduce 在 v1 unsupported；
- 结果只有写到最终 destination 后才完成，中间 staging 不算完成。

## 8. 控制 envelope 与同步

每个 artifact 必须显式给出：

- `active_cores`；
- 每个 source 的 `{target_core,tag,count}` start event；当前 host 约束 `tag<=65535`、`1<=count<=255`；
- `terminal_cores`；
- `expected_ack_cores` 和 `expected_done_cores`；
- `INCLUDE_EMPTY` 或 `EXCLUDE_EMPTY`；
- `ABORT_ALL` failure policy。

DONE 集合必须等于 terminal 集合。`INCLUDE_EMPTY` 时 ACK 集合等于全部 program cores；`EXCLUDE_EMPTY` 时等于 record stream 非空的 cores。空 core若被排除，不能是 source 或 terminal。

同步 lowering：

- `GROUP_SYNC(group_id,sync_seq)`：每个 group member 对同一序号恰好到达一次，序号单调；
- `EVENT_SET(source,destination,tag)` / `EVENT_WAIT(...,count)`：完整 key 必须匹配，WAIT 消耗 credit，支持先 SET；
- collective internal phase barrier 由 loader 生成，不能用额外 public barrier 模拟；
- program 结束时 token、event、barrier、tree/session 等 residual 必须为零。

## 9. Capability、profile 与 fail-fast

artifact `capabilities` 只声明 program 必需能力，不是“允许模拟器猜测 fallback”的提示。当前稳定发布基线要求 bitmap 为 0；PD、experimental、GLOBAL 等非零请求由 helper 拒绝。

NoC collective 的稳定 profile 请求名称为：

| profile | 请求 backend | 状态 |
|---|---|---|
| `baseline` | unicast broadcast + endpoint reduce | production available |
| `broadcast_only` | multicast broadcast + endpoint reduce | production available |
| `reduce_only` | unicast broadcast + DCA reduce | production available |
| `reduce_broadcast` | multicast broadcast + DCA reduce | production available |

profile 来自平台 NoC 配置，不改变 artifact 的高层 collective 记录。编译器不得生成 internal tree/action，也不得按 profile 改写 external record；`max_trees_per_batch` 是范围为 `[1,64]` 的平台资源参数，不是 artifact 字段。`ReduceScatter+DCA`、跨 die collective、tree/DCA 容量不足或 profile/backend 不一致时必须启动期失败，禁止回落到 baseline 后继续运行。四档 available 是真实字节和完成语义的功能正确性承诺，不保证加速档一定快于 baseline。

## 10. 完整最小例

以下示例构造一个 core 0 的 `SRAM_BIND → DUMMY` artifact。它展示稳定 symbol、one-shot bind、record 和 envelope；生产后端应复用相同 API，并增加自己的确定性排序和错误上下文。

```cpp
#include "isa/program_format.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

int main() {
    ProgramArtifact artifact;
    artifact.capabilities = 0;
    artifact.strings = {"dram_label input", "result"};
    artifact.symbols = {
        {0, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0},
        {1, ProgramSymbolKind::SRAM_LABEL, 0, 0, 0},
    };

    SramBindOperands bind;
    bind.input_count = 1;
    bind.input_symbol_indices[0] = 0;
    bind.output_symbol_index = 1;

    ComputeOperands dummy;
    dummy.datatype = ExternalDataType::INT8;
    dummy.input_offset_bytes = 0;
    dummy.data_offset_bytes = 80;
    dummy.output_offset_bytes = 80;
    dummy.parameters = {};

    artifact.cores = {{
        0,
        {
            {Opcode::SRAM_BIND, bind},
            {Opcode::DUMMY, dummy},
        },
    }};
    artifact.envelope.active_cores = {0};
    artifact.envelope.start_events = {{0, 1, 1}};
    artifact.envelope.terminal_cores = {0};
    artifact.envelope.expected_ack_cores = {0};
    artifact.envelope.expected_done_cores = {0};
    artifact.envelope.empty_core_ack_policy =
        EmptyCoreAckPolicy::INCLUDE_EMPTY;

    const std::vector<uint8_t> bytes = EncodeProgramArtifact(artifact);
    std::ofstream out("minimal.npup", std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("cannot write minimal.npup");
}
```

仓库参考生成器和运行方式：

```bash
build/npusim_program_fixture "build/minimal.npup"
build/npusim --program "build/minimal.npup"
```

成功启动必须先打印 `Loaded Program Format 1.0, ISA 1.0`，随后完成 CONFIG ACK、START 和 DONE。若在该行之前失败，表示容器、record、引用、capability、平台 group 或全局 lowering 校验未通过，不能把失败 artifact 交给运行时重试。

## 11. 后端提交前检查清单

- [ ] format/ISA 为 1.0、little-endian，七个 required sections 完整，CRC32C 正确；
- [ ] strings/symbols/cores/groups/list 规范排序且唯一；
- [ ] 只使用 manifest 中 available public Opcode；
- [ ] 每条 record header、payload size、enum、保留位、地址 XOR 合法；
- [ ] compute 参数顺序、30-bit 上限、offset 和 shape 派生范围合法；
- [ ] 每条 compute 有且只有一个可消费的 one-shot bind；
- [ ] local DTE token、P2P fsm/token、collective key/epoch/sync_seq 无碰撞；
- [ ] P2P 配对、collective 成员/角色/count/span/dtype/reduce_op 闭合；
- [ ] source/start/terminal/ACK/DONE 集合可推导且一致；
- [ ] 不生成 internal Prim、进程内 label ID、第二个 fence/barrier 或静默 fallback；
- [ ] `REDUCE_COMPUTE` 只进入 whole-artifact→strict `PrimId 0x3A`，未落入 legacy `0x2B`；
- [ ] 用 `DecodeProgramArtifact` 重解并再编码后逐字节相同；
- [ ] 用目标平台做 `--program` smoke，并核对数据、trace 和最终 drain，而不只看退出码。

## 12. 规范引用

任何 Opcode、operand、relocation、lowering target、Prim wire 或 golden 的变化，都必须按 [NPU ISA v1 ABI 与 Golden 变更策略](ISA_v1_ABI与Golden变更策略.md) 分类、审批和复测。特别是把相同 external artifact 改降到不同的正确性相关内部目标，不能作为“内部重构”处理；必须先判断是否需要提升 ISA/Program Format 版本，并同步 compiler、loader、runtime、mutation 与 legacy golden。

- ABI 与 golden 变更审批：[NPU ISA v1 ABI 与 Golden 变更策略](ISA_v1_ABI与Golden变更策略.md)
- 外部编号与支持状态：`notes/instr/isa_v1_manifest.md`
- 容器字段、section、CRC 和 loader 顺序：`notes/instr/program_format_v1.md`
- 字段单位、one-shot、同步和算法冻结结论：`notes/instr/编译产物指令集P0契约.md`
- 阶段目标、测试与发布门禁：`notes/instr/编译产物指令集开发计划.md`
- 真实 P2P 和 collective 验收边界：`notes/instr/log/P5_development.md`、`notes/instr/log/P6_development.md`
