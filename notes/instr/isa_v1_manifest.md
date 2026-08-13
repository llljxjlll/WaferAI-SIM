# NPU ISA v1 Manifest

> 状态：冻结
> ISA major：1
> Program format major：1
> 契约正文：`编译产物指令集P0契约.md`

## 1. 状态模型

本清单同时维护三个正交状态维度：

| 维度 | 值 | 含义 |
|---|---|---|
| visibility | `public` | 有稳定外部 Opcode 或由该 Opcode 直接引用 |
| visibility | `internal` | 只能由 loader/runtime/helper 生成，artifact 不可直接编码 |
| lifecycle | `stable` | 编号与语义已分配，不得静默改变 |
| lifecycle | `deprecated` | 仅为 legacy 兼容保留，不供新程序使用 |
| lifecycle | `reserved` | 外部编号已占用但 v1 不可执行 |
| lifecycle | `tombstone` | 曾发布后删除；编号永久不可复用 |
| support | `available` | 目标 v1 实现必须支持；仍可受硬件 capability 校验 |
| support | `unsupported` | 已知项，必须报 known-but-unsupported |
| support | `experimental` | 编号稳定，但默认 capability gate 关闭 |

`public` 不表示 C++ 类名或内部 PrimId 是外部 ABI。外部 record 只携带 Opcode；loader 通过显式 lowering 构造一个或多个内部 primitive。

## 2. 内部 PrimId inventory

以下 60 项来自当前全部 `REGISTER_PRIM`。`PrimId=0x00` 永久非法。编号按本表冻结，不依赖链接或静态初始化顺序。

| PrimId | 注册类 | factory name | 主类别 | visibility | lifecycle | support | 外部映射/备注 |
|---:|---|---|---|---|---|---|---|
| `0x01` | `Attention_f` | `Attention_f` | COMPUTE | public | stable | available | `ATTENTION` |
| `0x02` | `Batchnorm_f` | `Batchnorm_f` | COMPUTE | public | stable | unsupported | 外部保留 `BATCHNORM` |
| `0x03` | `Conv_f` | `Conv_f` | COMPUTE | public | stable | available | `CONV` |
| `0x04` | `Dummy_p` | `Dummy_p` | COMPUTE | public | stable | available | `DUMMY`；不是零成本 NOP |
| `0x05` | `gate_forward` | `gate_forward` | COMPUTE | public | stable | available | `GATE` |
| `0x06` | `Gelu_f` | `Gelu_f` | COMPUTE | public | stable | available | `GELU` |
| `0x07` | `Gemm_rs_swizzle` | `Gemm_rs_swizzle` | COMPUTE | public | stable | experimental | 外部保留 `GEMM_REDUCE_SCATTER` |
| `0x08` | `Layernorm_f` | `Layernorm_f` | COMPUTE | public | stable | available | `LAYERNORM` |
| `0x09` | `Matmul_f` | `Matmul_f` | COMPUTE | public | stable | available | `MATMUL` |
| `0x0A` | `Matmul_f_mla` | `Matmul_f_mla` | COMPUTE | public | stable | experimental | `MATMUL_MLA`；PD context gate |
| `0x0B` | `Max_pool` | `Max_pool` | COMPUTE | public | stable | available | `MAXPOOL` |
| `0x0C` | `Merge_conv` | `Merge_conv` | COMPUTE | public | stable | unsupported | 外部保留 `MERGE_CONV` |
| `0x0D` | `Merge_matmul` | `Merge_matmul` | COMPUTE | public | stable | available | `MERGE_MATMUL` |
| `0x0E` | `parse_input` | `parse_input` | MEM | internal | stable | available | 数据准备 helper |
| `0x0F` | `parse_output` | `parse_output` | MEM | internal | stable | available | 数据准备 helper |
| `0x10` | `Recv_global_memory` | `Recv_global_memory` | MEM | internal | stable | unsupported | GLOBAL v1 禁用 |
| `0x11` | `Relu_f` | `Relu_f` | COMPUTE | public | stable | available | `RELU` |
| `0x12` | `Residual_f` | `Residual_f` | COMPUTE | public | stable | available | `RESIDUAL` |
| `0x13` | `rmsnorm_forward` | `rmsnorm_forward` | COMPUTE | public | stable | available | `RMSNORM` |
| `0x14` | `rope_forward` | `rope_forward` | COMPUTE | public | stable | available | `ROPE` |
| `0x15` | `Send_global_memory` | `Send_global_memory` | MEM | internal | stable | unsupported | GLOBAL v1 禁用 |
| `0x16` | `silu_forward` | `silu_forward` | COMPUTE | public | stable | available | `SILU` |
| `0x17` | `Split_conv` | `Split_conv` | COMPUTE | public | stable | unsupported | 外部保留 `SPLIT_CONV` |
| `0x18` | `Split_matmul` | `Split_matmul` | COMPUTE | public | stable | available | `SPLIT_MATMUL` |
| `0x19` | `swiglu_forward` | `swiglu_forward` | COMPUTE | public | stable | available | `SWIGLU` |
| `0x1A` | `switch_data` | `switch_data` | MEM | internal | stable | available | 数据准备 helper |
| `0x1B` | `Attention_f_gpu` | `Attention_f_gpu` | COMPUTE | internal | stable | available | legacy GPU 模式 |
| `0x1C` | `attention_forward_gpu_pd` | `attention_forward_gpu_pd` | COMPUTE | internal | stable | available | legacy GPU-PD 模式 |
| `0x1D` | `Gelu_f_gpu` | `Gelu_f_gpu` | COMPUTE | internal | stable | available | legacy GPU 模式 |
| `0x1E` | `Layernorm_f_gpu` | `Layernorm_f_gpu` | COMPUTE | internal | stable | available | legacy GPU 模式 |
| `0x1F` | `Matmul_f_gpu` | `Matmul_f_gpu` | COMPUTE | internal | stable | available | legacy GPU 模式 |
| `0x20` | `matmul_forward_gpu_pd` | `matmul_forward_gpu_pd` | COMPUTE | internal | stable | available | legacy GPU-PD 模式 |
| `0x21` | `Residual_f_gpu` | `Residual_f_gpu` | COMPUTE | internal | stable | available | legacy GPU 模式 |
| `0x22` | `load_expert` | `load_expert` | MEM | internal | stable | available | 数据准备 helper |
| `0x23` | `matmul_forward_moe` | `matmul_forward_moe` | COMPUTE | public | stable | available | `MOE_MATMUL` |
| `0x24` | `Clear_sram` | `Clear_sram` | MEM | internal | stable | available | legacy 无参 clear-all；不是 public targeted clear |
| `0x25` | `Collective_data_prim` | `Collective_data_prim` | COMM | internal | stable | available | collective lowering |
| `0x26` | `Collective_prim` | `Collective_prim` | SYNC | internal | stable | available | internal phase barrier/arrival |
| `0x27` | `Dte_async_prim` | `Dte_async` | MEM/SYNC | internal | stable | available | local ISSUE 与 WAIT/FENCE/CANCEL lowering |
| `0x28` | `Load_prim` | `Load_prim` | MEM | internal | deprecated | available | legacy 16-bit load wire |
| `0x29` | `Lsu_mem_prim` | `Lsu_mem` | MEM | internal | stable | available | public 仅 blocking LOAD/STORE |
| `0x2A` | `Recv_prim` | `Recv_prim` | COMM | internal | stable | available | legacy host/NoC 协议；非 public DTE_RECV |
| `0x2B` | `Reduce_compute_prim` | `Reduce_compute_prim` | COMM | internal | stable | available | legacy timing-only creator；不得作为 external `REDUCE_COMPUTE` 的数据实现 |
| `0x2C` | `Send_prim` | `Send_prim` | COMM | internal | stable | available | legacy host/NoC 协议；非 public DTE_SEND |
| `0x2D` | `Set_addr` | `Set_addr` | MEM | internal | stable | available | legacy persistent bind |
| `0x2E` | `Set_batch` | `Set_batch` | MEM | internal | stable | available | program/PD helper |
| `0x2F` | `Sram_pipeline_prim` | `Sram_pipeline` | MEM | internal | stable | experimental | overlap/selftest oracle |
| `0x30` | `Store_prim` | `Store_prim` | MEM | internal | deprecated | available | legacy 16-bit store wire |
| `0x31` | `attention_forward_pd` | `Attention_f_pd` | COMPUTE | public | stable | experimental | `ATTENTION_PD`；注意 factory name 不同于类名 |
| `0x32` | `matmul_forward_pd` | `matmul_forward_pd` | COMPUTE | public | stable | experimental | `MATMUL_PD` |
| `0x33` | `rope_forward_pd` | `rope_forward_pd` | COMPUTE | public | stable | experimental | `ROPE_PD` |
| `0x34` | `Sram_bind_oneshot` | `Sram_bind_oneshot` | MEM | internal | stable | available | `SRAM_BIND` 的 strict one-shot lowering；不复用 legacy persistent `Set_addr` |
| `0x35` | `Sram_lifecycle` | `Sram_lifecycle` | MEM | internal | stable | available | `SRAM_ALLOC/FREE/RESIZE/RENAME/CLEAR` 的 strict lowering |
| `0x36` | `Group_sync_prim` | `Group_sync_prim` | SYNC | internal | stable | available | public `GROUP_SYNC` 的独立保留 key 命名空间 |
| `0x37` | `Event_control_prim` | `Event_control_prim` | SYNC | internal | stable | available | `EVENT_SET/WAIT` 专用 control FIFO/mailbox |
| `0x38` | `Dte_send_endpoint_prim` | `Dte_send_endpoint_prim` | COMM | internal | stable | available | public `DTE_SEND` P2P endpoint；collective child 也由 whole-artifact lowering 生成 |
| `0x39` | `Dte_recv_endpoint_prim` | `Dte_recv_endpoint_prim` | COMM | internal | stable | available | public `DTE_RECV` P2P endpoint；接收 descriptor 先发布 |
| `0x3A` | `Collective_data_v1_prim` | `Collective_data_v1_prim` | COMM | internal | stable | available | strict real-byte local copy/reduction；不对外直接编码 |
| `0x3B` | `Collective_phase_barrier_v1_prim` | `Collective_phase_barrier_v1_prim` | SYNC | internal | stable | available | whole-artifact collective phase barrier；不是 public `GROUP_SYNC` |
| `0x3C` | `Collective_launch_v1_prim` | `Collective_launch_v1_prim` | COMM | internal | stable | available | immutable collective program image 的 per-core launch marker |

### 2.1 内部编号追加规则

1. 新内部 primitive 从 `0x3D` 起顺序追加；不得填补 `0x01～0x3C` 中因删除形成的空位。
2. 删除已登记 creator 时保留 PrimId tombstone，decoder 报 retired/internal-unsupported。
3. 新 public Opcode 不得假设数值等于 lowering target 的 PrimId。
4. 一个 Opcode 可以 lowering 为多个 PrimId；多个 Opcode 也可以 lowering 到同一 PrimId 的不同 op variant。
5. 所有内部 128-bit segment 的低 8 位必须带相同 PrimId；decoder 校验段数、ID 一致性和保留位。
6. 新增内部 ID 必须同步更新本表、factory selftest、golden wire 和全量 `REGISTER_PRIM` 覆盖测试。

## 3. 外部 Opcode 表

### 3.1 计算区 `0x00～0x3F`

| Opcode | 指令 | visibility | lifecycle | support | lowering target | capability/备注 |
|---:|---|---|---|---|---|---|
| `0x00` | `INVALID` | public | reserved | unsupported | 无 | 永久非法 |
| `0x01` | `MATMUL` | public | stable | available | PrimId `0x09` | — |
| `0x02` | `MATMUL_MLA` | public | stable | experimental | PrimId `0x0A` | `pd_context` 默认关闭 |
| `0x03` | `MATMUL_PD` | public | stable | experimental | PrimId `0x32` | `pd_context` 默认关闭 |
| `0x04` | `CONV` | public | stable | available | PrimId `0x03` | — |
| `0x05` | `MAXPOOL` | public | stable | available | PrimId `0x0B` | — |
| `0x06` | `ATTENTION` | public | stable | available | PrimId `0x01` | — |
| `0x07` | `ATTENTION_PD` | public | stable | experimental | PrimId `0x31` | `pd_context` 默认关闭 |
| `0x08` | `GATE` | public | stable | available | PrimId `0x05` | — |
| `0x09` | `MOE_MATMUL` | public | stable | available | PrimId `0x23` | — |
| `0x0A` | `GELU` | public | stable | available | PrimId `0x06` | — |
| `0x0B` | `SILU` | public | stable | available | PrimId `0x16` | — |
| `0x0C` | `SWIGLU` | public | stable | available | PrimId `0x19` | — |
| `0x0D` | `RELU` | public | stable | available | PrimId `0x11` | 继承当前 EXU 计费 |
| `0x0E` | `RESIDUAL` | public | stable | available | PrimId `0x12` | — |
| `0x0F` | `LAYERNORM` | public | stable | available | PrimId `0x08` | — |
| `0x10` | `RMSNORM` | public | stable | available | PrimId `0x13` | 继承当前计费 |
| `0x11` | `ROPE` | public | stable | available | PrimId `0x14` | — |
| `0x12` | `ROPE_PD` | public | stable | experimental | PrimId `0x33` | `pd_context` 默认关闭 |
| `0x13` | `SPLIT_MATMUL` | public | stable | available | PrimId `0x18` | — |
| `0x14` | `MERGE_MATMUL` | public | stable | available | PrimId `0x0D` | 继承当前计费 |
| `0x15` | `DUMMY` | public | stable | available | PrimId `0x04` | 固定成本，不是 NOP |
| `0x16` | `BATCHNORM` | public | reserved | unsupported | PrimId `0x02` | stub |
| `0x17` | `SPLIT_CONV` | public | reserved | unsupported | PrimId `0x17` | stub |
| `0x18` | `MERGE_CONV` | public | reserved | unsupported | PrimId `0x0C` | stub |
| `0x19` | `GEMM_REDUCE_SCATTER` | public | reserved | experimental | PrimId `0x07` | v1 loader 拒绝 |

`0x1A～0x3F` 未分配并保留。GPU 计算 primitive 不进入 NPU ISA v1 外部空间。

### 3.2 通信区 `0x40～0x7F`

| Opcode | 指令 | visibility | lifecycle | support | lowering target | capability/备注 |
|---:|---|---|---|---|---|---|
| `0x40` | `DTE_SEND` | public | stable | available | PrimId `0x38`（P2P）；collective 由 whole-artifact lowering 展开 | P2P 与 P6 baseline 均为真实字节；P7 加速按 profile capability gate |
| `0x41` | `DTE_RECV` | public | stable | available | PrimId `0x39`（P2P）；collective 由 whole-artifact lowering 展开 | RX descriptor 必须先发布；无静默 profile fallback |
| `0x42` | `REDUCE_COMPUTE` | public | stable | available | whole-artifact lowering→PrimId `0x3A` | 仅 endpoint reduction graph；direct record-local lowering 拒绝 legacy `0x2B` |

`0x43～0x7F` 未分配并保留。`COLLECTIVE_CALL`、GLOBAL_SEND/RECV 和 public `COLL_BARRIER` 不存在。

### 3.3 访存区 `0x80～0xBF`

| Opcode | 指令 | visibility | lifecycle | support | lowering target | capability/备注 |
|---:|---|---|---|---|---|---|
| `0x80` | `LSU_LOAD` | public | stable | available | PrimId `0x29`, `LOAD_BLOCKING` | 只允许 HBM→SRAM |
| `0x81` | `LSU_STORE` | public | stable | available | PrimId `0x29`, `STORE_BLOCKING` | 只允许 SRAM→HBM |
| `0x82` | `DTE_ISSUE` | public | stable | available | PrimId `0x27`, `ISSUE` | 仅 SPM_TO_SPM/SPM_TO_DRAM/DRAM_TO_SPM |
| `0x83` | `SRAM_CLEAR` | public | stable | available | 新 targeted clear primitive | 必须带 label |
| `0x84` | `SRAM_BIND` | public | stable | available | 新 one-shot bind wrapper | legacy PrimId `0x2D` 仍 persistent |
| `0x85` | `SRAM_ALLOC` | public | stable | available | 新 RegionTable thin primitive | metadata 操作 |
| `0x86` | `SRAM_FREE` | public | stable | available | 新 RegionTable thin primitive | metadata 操作 |
| `0x87` | `SRAM_RESIZE` | public | stable | available | 新 RegionTable thin primitive | metadata 操作 |
| `0x88` | `SRAM_RENAME` | public | stable | available | 新 RegionTable thin primitive | metadata 操作 |

`0x89～0xBF` 未分配并保留。Load/Store legacy、数据准备 helper、Sram_pipeline 和 Set_batch 不可由 artifact 直接编码。

### 3.4 同步区 `0xC0～0xEF`

| Opcode | 指令 | visibility | lifecycle | support | lowering target | capability/备注 |
|---:|---|---|---|---|---|---|
| `0xC0` | `DTE_WAIT` | public | stable | available | PrimId `0x27`, `WAIT` | token 必须非零 |
| `0xC1` | `DTE_FENCE` | public | stable | available | PrimId `0x27`, `FENCE` | 无参；token 必须为零 |
| `0xC2` | `DTE_CANCEL` | public | stable | available | PrimId `0x27`, `CANCEL` | token 必须非零 |
| `0xC3` | `EVENT_SET` | public | stable | available | 新 event control primitive | 专用 EVENT control message |
| `0xC4` | `EVENT_WAIT` | public | stable | available | 新 event control primitive | 支持提前到达和 count |
| `0xC5` | `GROUP_SYNC` | public | stable | available | 新 thin sync primitive→internal barrier | 唯一 public barrier |
| `0xC6` | `DTE_POLL` | public | reserved | unsupported | PrimId `0x27`, `POLL` | v1 loader 拒绝 |

`0xC7～0xEF` 未分配并保留。`FENCE_ALL` 只是 `DTE_FENCE` 的编译器别名，不占号；`COLL_BARRIER` 仅为 internal lowering marker。

### 3.5 全局保留区 `0xF0～0xFF`

全部为 `visibility=public, lifecycle=reserved, support=unsupported`，v1 不得注册或执行。

## 4. 分类位

`PRIM_TYPE` 在现有位基础上追加：

```text
COMM_PRIM = 1 << 5
MEM_PRIM  = 1 << 6
SYNC_PRIM = 1 << 7
```

主分类规则：

- COMPUTE：NPU/GPU/PD/MoE 计算；
- COMM：DTE_SEND/RECV、collective data、endpoint reduce；
- MEM：blocking LSU、local DTE_ISSUE、SRAM 生命周期和数据准备；
- SYNC：DTE WAIT/FENCE/CANCEL、EVENT、GROUP_SYNC、internal phase barrier；
- 每条 decoded 指令只计一个主类别，继承产生的辅助 bit 不重复计数。

## 5. Manifest 验证规则

自动化 selftest 至少断言：

1. 60 个当前注册类全部出现且恰好一次；
2. 内部 PrimId 和 factory name 唯一；
3. public Opcode 唯一并落入正确类别区间；
4. reserved/tombstone 不可作为 available creator 注册；
5. 每个 available public Opcode 有 schema、lowering、capability 和至少一个 golden；
6. 每个 unsupported/experimental Opcode 给出稳定诊断；
7. 随机改变注册顺序不改变本表映射；
8. 生成的 manifest 按数值排序，不依赖 `unordered_map`；
9. release/debug 和两次 clean build 的 manifest 逐字节相同；
10. manifest 变化必须在代码评审中明确标记为 ABI、capability 或 internal-wire 变更。
