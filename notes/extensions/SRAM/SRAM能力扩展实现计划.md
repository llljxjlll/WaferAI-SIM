# SRAM 能力扩展实现计划

> 状态：S0–S6 与最终验收条件 1–8 全部闭环
> 日期：2026-08-06
> 使用说明：`notes/extensions/SRAM/README.md`
> 阶段记录：`notes/extensions/SRAM/log`
> 关联文档：`notes/extensions/DTE/DTE建模计划.md`、`notes/extensions/DRAM/HBM建模计划.md`

## 评审回退与当前判定（2026-08-06）

此前错误的“ S0-S6 全部完成”结论已撤回，并按两轮评审逐项重做。lease 死锁、R5 假阳性、beat/RR、LSU 队列、DTE 回滚与 SPM_TO_SPM、word/byte、region 标签、spill/clear 和细粒度 trace 均已有定向回归。

## 最终生产路径收口（2026-08-06）

新增注册原语 `Sram_pipeline` 和完整 CLI WorkerCore fixtures，覆盖 LSU/DTE blocking 与 double buffer、behavioral/DRAMSys/legacy_private、远端 NUMA、非零 payload、真实 overlap trace，以及 input/intermediate/comm 标签生命周期。扩展关闭的旧 NpuBase/helper workload、默认 smoke、完整 `playground.json` 与 D2D 冻结回归同时通过。P0/P1 收口新增规范默认配置、启动 preflight、S_DATA 六阶段完成握手、有限工作 watchdog lease 和 rendezvous 静态校验；当前判定为 S0-S6 及最终条件 1-8 全部完成。

## 0. 目标与决策速览

本计划为模拟器增加以下四项能力：

1. DTE 能访问每核 SRAM，并在 SRAM 与 HBM 之间搬运真实数据；
2. 计算核 LSU 能绕过 DTE，直接执行本核 SRAM 与 HBM 之间的 load/store；
3. DTE 和 LSU 两条路径都提供可在 workload 中调用的 primitive，并同时提供可在计算 primitive 内调用的 C++ API；
4. 用户能在硬件配置中为输入、中间结果、通信 buffer、double buffer 等用途手动划分 SRAM 区域。

本轮冻结以下架构决策：

- SRAM 是 **per-core private、byte-addressable** 的片上存储；SRAM 地址统一为核内 byte 地址。
- HBM 地址是系统物理 byte 地址，继续由 `CoreMemAdapter`、地址映射、NoC、MEM endpoint 和 HBM backend 处理。
- DTE 与 LSU 共用同一份本核 SRAM、同一 HBM 数据面，但拥有独立的发起队列、端口和统计。
- DTE 复用现有 `DteAsyncTracker` 的 token、credit、issue/wait/poll/fence/cancel 和区间 hazard 语义，不另造第二套异步控制协议。
- LSU 新增与 DTE 对称的异步 token API；阻塞式 load/store 只是 `issue + wait` 的语法糖。
- DTE/LSU/计算单元对 SRAM 的访问全部进入统一 `SramAccessUnit` 仲裁，禁止继续在 helper 中各自直接 `wait()` 估算 SRAM 时间。
- 分区是容量与地址所有权契约；bank/port 是时序资源契约，两者不能混为一谈。
- 目标架构每核只有一份计入容量的物理 SRAM。现有 `temp_ram_array` 在迁移期保留兼容，最终映射为主 SRAM 中的命名 `temp` region，不能继续作为一份不计容量的额外 SRAM。
- `distributed_hbm` 与 `legacy_private` 均通过统一 `HbmByteTransport` 接入真实数据面；前者支持 behavioral/DRAMSys 和 NUMA，后者保留 DCache/DRAMSys 时序并提供 byte payload backing。
- 不采用 `std::thread`/`std::async`。所有并发、完成和 backpressure 都由 SystemC process、event 和有界队列表达。

## 1. 当前基线与缺口

### 1.1 当前 SRAM

- 每个 `WorkerCore` 创建独立 `ram_array` 和 `temp_ram_array`。
- `memory.sram_size` 是全核共享的容量配置；`cores[].sram_bitwidth` 被用于访问次数估算。
- 默认 `USE_SRAM_MANAGER=0`，主路径使用递增 `sram_addr` 与 `SramPosLocator` 标签表管理数据。
- `SramManager` 只支持按大小 first-fit 分配，不支持指定 base 的 `allocate_at()`、命名 region 或访问权限。
- 默认 `DUMMY_SRAM=1`，SRAM 不保存实际 payload；读返回零。
- `sram_address` workload 字段表示标签关系，不表示物理地址分区。
- 行为级 SRAM 与 detailed SRAM 的数据宽度、端口、延迟责任不统一；部分输出写 SRAM 当前为零延迟。

### 1.2 当前 DTE

- 已有 `Dte_async` 的 `issue/wait/poll/fence/cancel`、logical token、credit、区间 RAW/WAR/WAW 检查。
- 已有 `DRAM_TO_SPM`、`SPM_TO_DRAM`、`SPM_TO_SPM` 等方向。
- DTE V4 当前只模拟 SPM/AXI endpoint 端口服务时间，不调用真实 SRAM，不调用 `CoreMemAdapter`，也不产生 HBM 请求。
- `spm_addr/spm_size` 当前只是范围元数据，不会预留或更新 SRAM 容量与数据有效位。

### 1.3 当前 LSU/HBM helper

- `sram_first_write_generic()` 与 `sram_spill_back_generic()` 是同步 helper。
- distributed HBM 下它们直接调用阻塞式 `CoreMemAdapter::Access()`；primitive 线程在请求返回前不能继续计算。
- `Load_prim`、`Store_prim`、`Send_global_memory`、`Recv_global_memory` 尚不能承担通用 SRAM↔HBM 搬运。
- 当前 `NpuBase` 在 primitive 前统一装载输入，在末尾用 `max(dram_time, compute_time)` 做整 primitive 粗粒度重叠，不能表达逐 tile double buffering。

## 2. 范围与非目标

### 2.1 本轮必须交付

- 每核 SRAM 真实 byte backing store、valid 状态和容量边界。
- 静态 region 配置、启动期校验、运行时 region 查询和子分配。
- 统一 SRAM read/write 请求、端口仲裁、bank 冲突、排队和统计。
- DTE `DRAM_TO_SPM`/`SPM_TO_DRAM` 接入真实 SRAM 与真实 distributed HBM。
- 本核 LSU 的 HBM↔SRAM load/store。
- DTE 和 LSU primitive，以及计算 primitive 内部可直接调用的 API。
- issue/wait 形式的异步搬运与 compute overlap。
- double buffer 示例 workload 和端到端验收。
- behavioral HBM 与 DRAMSys HBM 两种 backend 的数据正确性和时序回归。

### 2.2 明确暂缓

- 跨核直接访问对方 SRAM；跨核通信仍由 NoC/DTE remote 方向独立建模。
- cache coherence、shared LLC、虚拟内存、页表和地址翻译。
- 自动 tile 搜索、自动 double-buffer 切分、编译器自动调度。
- scatter/gather、二维 stride、transpose、shuffle；第一版只支持连续 byte range。
- 多播 HBM→多核 SRAM；后续可在统一 descriptor 上扩展。
- 硅后绝对功耗/面积标定。

## 3. 统一术语、地址和单位

### 3.1 地址空间

| 名称 | 类型 | 含义 |
|---|---:|---|
| `hbm_addr` | `uint64_t` | 系统物理 byte 地址，由 `CoreMemAdapter` 解码 |
| `sram_addr` | `uint64_t` | 本核 SRAM 内 byte 地址，范围 `[0, capacity_bytes)` |
| `region_offset` | `uint64_t` | 命名 region 内 byte 偏移 |
| `size_bytes` | `uint64_t` | 本次传输或分配的有效字节数 |

所有区间使用半开形式 `[base, base + size)`。内部禁止以 bitwidth word index 冒充 byte 地址。

### 3.2 对齐

- region 的 `base_bytes` 和 `size_bytes` 必须满足 `allocation_alignment_bytes`。
- DTE/LSU descriptor 至少 byte 对齐；若 backend 或端口要求更严格对齐，由配置解析给出明确错误。
- 非 burst 整数倍的尾部访问必须保留真实 `size_bytes`，以 byte-enable 处理，不能静默放大功能写入范围。

### 3.3 数据语义

- SRAM backing store 使用 byte payload 和逐 byte valid 状态。
- HBM→SRAM：HBM response payload 全部到达并提交 SRAM 后，目标区间才变为 valid，token 才完成。
- SRAM→HBM：源 SRAM 全区间必须 valid；读取 payload 后通过 HBM write 和 byte-enable 写回。
- timing-only 模式可以不保存大数组，但必须保留可检测的数据签名和 valid 状态；功能 selftest 必须使用真实 payload 模式。
- 未初始化 SRAM read、越界访问、region 权限错误默认抛异常，不以零值掩盖错误。

## 4. 目标架构

```text
                                  per-core
  compute primitive                 │
       │                            │
       ├── compute SRAM read/write ─┤
       │                            ▼
       ├── LSU issue ───────┐  +-------------------+
       │                    ├─►|  SramAccessUnit   |──► SRAM banks/backing
       └── DTE issue ───────┘  | arbiter + hazard  |
                               +-------------------+
                                  ▲           ▲
                                  │           │
                    +-------------+--+     +--+-------------+
                    | CoreLsuUnit    |     | DTE data plane |
                    +-------+--------+     +--------+--------+
                            │                       │
                            +-----------+-----------+
                                        ▼
                                CoreMemAdapter
                                        ▼
                              NoC / MEM endpoint / HBM
```

建议新增模块：

- `SramStorage`：byte backing、valid、clear、功能读写；
- `SramRegionTable`：静态 region、权限、spill policy、地址解析；
- `SramRegionAllocator`：region 内子分配、释放和标签绑定；
- `SramAccessUnit`：统一端口、bank、仲裁、排队、完成事件和统计；
- `CoreLsuUnit`：本核 LSU descriptor、token、HBM transport；
- `DteMemoryBridge`：把现有 DTE descriptor 与真实 SRAM/HBM 数据面连接；
- `MemoryTransferTracker`：可选公共 token 基类；DTE 继续使用现有 `DteAsyncTracker`，LSU 复用相同生命周期契约。

## 5. SRAM 存储与访问模型

### 5.1 `SramStorage`

建议接口：

```cpp
struct SramByteRange {
    uint64_t address;
    uint64_t size_bytes;
};

class SramStorage {
public:
    std::vector<uint8_t> Read(SramByteRange range) const;
    void Write(SramByteRange range,
               const std::vector<uint8_t>& payload,
               const std::vector<uint8_t>& byte_enable = {});
    void Clear(SramByteRange range);
    bool IsValid(SramByteRange range) const;
    uint64_t CapacityBytes() const;
};
```

要求：

- 不再用模板 `sc_bv<SRAM_BITWIDTH>` 决定功能数据宽度；runtime bitwidth 只属于端口时序。
- detailed/behavioral 模式共用同一功能存储与 valid 语义，仅时序实现不同。
- `DUMMY_SRAM` 改为明确的 runtime `store_payload=false` 模式，不能改变边界、valid、hazard 和统计。

### 5.2 `SramAccessUnit`

统一请求：

```cpp
enum class SramInitiator { COMPUTE, DTE, LSU, NOC_RX, LEGACY };
enum class SramCommand { READ, WRITE, CLEAR };

struct SramAccessRequest {
    uint64_t request_id;
    int core_id;
    SramInitiator initiator;
    SramCommand command;
    uint64_t address;
    uint64_t size_bytes;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> byte_enable;
    sc_event done;
};
```

统一完成条件：请求经过 region/权限检查、hazard 检查、bank/port 仲裁、存储读写后才通知 `done`。

### 5.3 bank 和端口

配置至少包括：

- `bank_count`；
- `bank_interleave_bytes`；
- compute read/write port 数；
- DTE read/write port 数；
- LSU read/write port 数；
- 每类端口宽度 bit/cycle；
- read/write base latency；
- arbitration policy：第一版固定 round-robin；
- queue depth 与超限 backpressure。

bank 映射冻结为纯函数：

```text
bank = floor(sram_addr / bank_interleave_bytes) % bank_count
```

跨 bank 请求拆成 beat；同 bank/同端口请求排队，不同独立端口可以并行。behavioral 模式也必须经过同一资源仲裁，不能只按请求大小单独 `wait()`。

### 5.4 hazard

至少阻止：

- 未完成 HBM→SRAM write 与 compute/LSU/DTE read 的 RAW；
- compute/DTE/LSU read 与覆盖同区间的 write 的 WAR；
- 两个重叠 write 的 WAW；
- 同一 double-buffer slot 尚未消费完成就被下一次 load 覆盖。

read/read 可并发。DTE 已有的范围 hazard 逻辑应抽为公共纯函数，避免 LSU 再实现一份不一致版本。

## 6. 手动 SRAM region 划分

### 6.1 配置格式

保留旧 `memory.sram_size`，新增结构化配置：

```json
{
  "memory": {
    "sram_size": 33554432,
    "sram": {
      "bank_count": 16,
      "bank_interleave_bytes": 256,
      "allocation_alignment_bytes": 1024,
      "regions": [
        {
          "name": "input",
          "base_bytes": 0,
          "size_bytes": 8388608,
          "allocator": "block",
          "spillable": true,
          "access": ["compute", "dte", "lsu"]
        },
        {
          "name": "intermediate",
          "base_bytes": 8388608,
          "size_bytes": 8388608,
          "allocator": "block",
          "spillable": false,
          "access": ["compute", "dte", "lsu"]
        },
        {
          "name": "comm",
          "base_bytes": 16777216,
          "size_bytes": 4194304,
          "allocator": "block",
          "spillable": false,
          "access": ["compute", "dte", "noc_rx"]
        },
        {
          "name": "double_a",
          "base_bytes": 20971520,
          "size_bytes": 4194304,
          "allocator": "fixed",
          "spillable": false,
          "access": ["compute", "dte", "lsu"]
        },
        {
          "name": "double_b",
          "base_bytes": 25165824,
          "size_bytes": 4194304,
          "allocator": "fixed",
          "spillable": false,
          "access": ["compute", "dte", "lsu"]
        }
      ]
    }
  }
}
```

如果 `regions` 缺省，则自动创建覆盖完整容量的 `legacy` region，保持旧 workload 可运行。

如需不同核心采用不同布局，在 `cores[].sram.regions` 中覆盖全局默认；禁止部分字段静默合并造成不同核心布局不可预测。

### 6.2 region 规则

- 名字在单核内唯一；保留名：`legacy`、`input`、`output`、`intermediate`、`comm`、`temp`。
- region 不得重叠、越界或整数溢出。
- `fixed` region 不允许通用 allocator 子分配，调用方直接使用 region+offset。
- `block` region 使用独立 block allocator；分配不会跨 region。
- `spillable=false` 的 region 永不成为自动 victim。
- descriptor 必须完整落在一个 region 内；第一版不允许一次传输跨 region。
- 权限在 issue 时检查；例如 LSU 不能写只有 `noc_rx` 权限的通信区。
- 所有静态 region 在 SystemC elaboration 前完成解析和校验。

### 6.3 运行时接口

```cpp
struct ResolvedSramRange {
    uint64_t address;
    uint64_t size_bytes;
    int region_id;
};

ResolvedSramRange ResolveRegion(
    int core_id, std::string_view name,
    uint64_t offset, uint64_t size_bytes,
    SramInitiator initiator, SramCommand command);
```

标签表保存 `region_id + address + size + allocation_id + valid/spill`，不能只保存一个递增位置。`Clear_sram` 改为按 region 或标签清理，不得无条件破坏 non-spillable 通信/double-buffer 区。

## 7. DTE 访问 SRAM/HBM

### 7.1 复用现有控制面

继续使用：

- `Dte_async issue/wait/poll/fence/cancel`；
- logical token 与 physical transfer；
- descriptor credit、command slot、aggregation；
- SPM read/write、AXI read/write 端口；
- SRAM 区间 hazard。

新增 descriptor 解析字段：

- `hbm_addr`：DRAM 方向的明确字段；旧 `remote_addr` wire bits 可继续承载 resolved 数值以兼容编码；
- `sram_region` + `sram_offset`：JSON 便利写法，配置加载时解析为绝对 `spm_addr`；
- `size_bytes`：作为 `payload_bits/8` 的显式校验值；二者不一致时报错。

### 7.2 真实数据面

新增 `DteMemoryBridge`：

- `DRAM_TO_SPM`：通过 `CoreMemAdapter` read HBM payload，再向 `SramAccessUnit` 发 DTE write；
- `SPM_TO_DRAM`：向 `SramAccessUnit` 发 DTE read，再通过 `CoreMemAdapter` write HBM；
- `SPM_TO_SPM`：从源区读取 payload 并写目标区；
- remote 方向保留既有 endpoint 语义，本轮不借 SRAM-HBM 改造偷换为新网络协议。

`CoreMemAdapter::Access()` 可以保持阻塞接口，但只能由 DTE bridge 的独立 SystemC worker 调用，不能阻塞 compute primitive 线程。

### 7.3 时序唯一责任

DTE real-memory 模式的阶段定义：

```text
HBM→SRAM：descriptor launch → HBM/NoC read → AXI_READ → SPM_WRITE
SRAM→HBM：descriptor launch → SPM_READ → AXI_WRITE → HBM/NoC write
```

- descriptor launch、slot、credit、SPM/AXI endpoint 端口归 DTE 负责；
- HBM queue/media 与 NoC/C2C 归现有 HBM 路径负责；
- SRAM bank/port 归 `SramAccessUnit` 负责；
- 同一阶段不得在 `memory_utils.cpp` 再额外估算一次；
- 第一版允许 descriptor 级 store-and-forward；后续按 `transfer_chunk_bytes` 做 chunk pipeline，使稳态吞吐由最慢阶段决定而不是所有带宽倒数简单相加。

token 完成必须同时满足 DTE 控制面、真实数据面和目标数据提交完成。

## 8. LSU 访问 SRAM/HBM

### 8.1 `CoreLsuUnit`

每核新增一个 LSU 模块：

```cpp
enum class LsuDirection { HBM_TO_SRAM, SRAM_TO_HBM };

struct LsuDescriptor {
    uint64_t token;
    LsuDirection direction;
    uint64_t hbm_addr;
    uint64_t sram_addr;
    uint64_t size_bytes;
};

class CoreLsuUnit : public sc_module {
public:
    uint64_t Issue(const LsuDescriptor& descriptor);
    void Wait(uint64_t token);
    bool Poll(uint64_t token) const;
    void Fence();
    void Cancel(uint64_t token);
};
```

### 8.2 LSU 与 DTE 的差异

| 项目 | DTE | LSU |
|---|---|---|
| descriptor launch | 有 DTE `gamma/tau_launch` | 使用 LSU issue latency |
| 多 channel/slot | 使用现有 DTE 配置 | 独立 `lsu_queue_depth/outstanding` |
| SRAM 端口 | DTE read/write port | LSU read/write port |
| HBM 数据面 | `CoreMemAdapter` | 同一个 `CoreMemAdapter` |
| token API | 现有 `DteAsyncTracker` | 新增对称 tracker |
| aggregation | 可用现有 DTE aggregation | 第一版不支持 |

LSU 必须只能访问 `descriptor.core_id` 对应的本核 SRAM；HBM 地址可能 home 在本地或远端 die，由地址映射自然决定。

### 8.3 LSU 配置

```json
{
  "cores": [
    {
      "id": 0,
      "lsu": {
        "queue_depth": 8,
        "max_outstanding": 2,
        "issue_latency_ns": 2,
        "sram_read_width_bits": 512,
        "sram_write_width_bits": 512
      }
    }
  ]
}
```

LSU 与 DTE 的 HBM 请求在相同 channel/backend 中自然竞争；两者的 SRAM 请求由 `SramAccessUnit` 按实际端口配置竞争。

## 9. primitive 与可调用 API

### 9.1 两层接口

必须同时提供：

1. workload primitive：可在 workload 中显式列出 issue/wait/fence；
2. C++ callable API：计算 primitive 能在自己的 tile 循环中发起 load/store，实现原语内 double buffering。

仅实现独立 `Load_prim/Store_prim` 不足以满足“一个计算原语内手动搬运并与计算重叠”。

### 9.2 DTE primitive

复用并扩展 `Dte_async`：

```json
{
  "type": "Dte_async",
  "op": "issue",
  "token": 17,
  "direction": "DRAM_TO_SPM",
  "hbm_addr": 1048576,
  "sram_region": "double_a",
  "sram_offset": 0,
  "spm_size": 65536,
  "payload_bits": 524288
}
```

后续以 `wait token=17`、`poll token=17` 或 `fence` 同步。legacy remote DTE workload 保持原编码和行为。

### 9.3 LSU primitive

新增 `Lsu_mem`：

```json
{
  "type": "Lsu_mem",
  "op": "issue",
  "token": 31,
  "direction": "HBM_TO_SRAM",
  "hbm_addr": 2097152,
  "sram_region": "input",
  "sram_offset": 65536,
  "size_bytes": 65536
}
```

支持 `issue/wait/poll/fence/cancel`。可选 `op=load_blocking/store_blocking` 仅作为 issue 后立即 wait 的简写，不建立第三套时序路径。

### 9.4 计算 primitive 内部 API

在 `TaskCoreContext` 中暴露：

```cpp
DteMemoryApi* dte_memory;
LsuMemoryApi* lsu_memory;
SramRegionTable* sram_regions;
ComputeTimeline* compute_timeline;
```

double-buffer 示例：

```cpp
auto a = context.sram_regions->Resolve("double_a", 0, tile_bytes);
auto b = context.sram_regions->Resolve("double_b", 0, tile_bytes);

auto current = context.dte_memory->IssueLoad(hbm_base, a);
for (uint64_t tile = 0; tile < tile_count; ++tile) {
    context.dte_memory->Wait(current);

    auto next = InvalidToken;
    if (tile + 1 < tile_count) {
        auto dst = ((tile + 1) & 1) ? b : a;
        next = context.dte_memory->IssueLoad(
            hbm_base + (tile + 1) * tile_bytes, dst);
    }

    auto src = (tile & 1) ? b : a;
    context.compute_timeline->RunTile(src, tile_compute_cycles);
    current = next;
}
context.dte_memory->Fence();
```

DMA worker 与 compute primitive 必须是不同 SystemC process；`RunTile()` 在 compute 线程推进时间时，DTE/LSU worker 可继续完成请求。

### 9.5 与 `NpuBase` 的关系

新增显式调度模式，例如 `manual_memory_schedule=true`：

- 跳过 `NpuBase::checkInputData()` 的整 tensor 自动装载；
- 跳过末尾 `max(dram_time, compute_time)` 的粗粒度重叠；
- 计算与搬运时间由 primitive 内部分段推进；
- 若输出需要保留为标签，由 primitive 显式 `BindLabel()`；
- primitive 返回前必须消费所有本 primitive 创建的 token，或把所有权显式转交给 core tracker。

默认模式保持旧 workload 行为不变。

## 10. 生命周期、一致性与错误处理

### 10.1 token

- token 在单核、单 engine 内唯一；DTE 与 LSU token namespace 可独立。
- `wait` 消费并释放 token；`poll` 不消费；`fence` 按 issue 顺序等待并释放全部。
- 只允许取消 pending descriptor；已经进入 SRAM/HBM 数据面的请求拒绝 cancel。
- primitive/core/task 结束时存在未消费 token 必须报错，禁止静默丢弃。

### 10.2 region 与分配生命周期

- 静态 region 生命周期等于模拟器实例。
- 动态 allocation 具有 task、layer、persistent 三种 lifetime；默认 task。
- allocation 仍有 outstanding read/write 时禁止 free。
- non-spillable region 不参与 `SramPosLocator` victim 选择。
- spillable label 的 HBM backing address 必须真实记录，禁止继续使用固定地址 `1024` 作为通用写回位置。

### 10.3 并发顺序

- 同一 engine 内无 hazard 的 descriptor 可并发；有 hazard 时按 issue sequence 建依赖。
- DTE 与 LSU 之间也要通过公共 range tracker 发现 hazard。
- HBM write response 返回后才算 store 完成；不能在 request 注入时提前完成。
- HBM→SRAM 的目标 valid 位在完整 commit 前不可见；若未来支持 chunk streaming，再改为逐 chunk valid。

### 10.4 配置门禁

以下情况启动失败或 issue 失败：

- region 重叠、越界、未对齐、重复命名；
- descriptor 跨 region 或超出 region；
- initiator 无 region 访问权限；
- `distributed_hbm` 未启用却请求 real-memory DTE/LSU 模式；
- `Dte_async` DRAM 方向未开启 DTE async/fine-grained resources；
- SRAM/HBM 地址加长度溢出；
- payload/size/byte-enable 长度不一致；
- 对 invalid SRAM 数据执行 store 或 compute read。

## 11. 配置兼容与迁移

### 11.1 默认关闭策略

新增 runtime 开关：

```json
{
  "memory": {
    "sram": {
      "real_data_path": false,
      "manual_regions": false
    }
  }
}
```

扩展默认关闭时：

- 旧 `sram_size/sram_bitwidth/use_beha_sram` 继续工作；
- 旧 workload 的 DONE 时间和 trace 保持冻结；
- DTE V0-V4、HBM R0-R4 和 NoC/D2D 回归不得变化。

功能成熟后再单独提交默认值迁移，不在首个实现提交中切换。

### 11.2 旧 helper 迁移

- `sram_first_write_generic()`：先改为阻塞包装器，内部调用 LSU 或 DTE real API 的 `issue + wait`；
- `sram_spill_back_generic()`：改为真实 `SRAM_TO_HBM`，使用标签记录的 HBM 地址；
- `sram_read_generic()`/`sram_write_append_generic()`：统一进入 `SramAccessUnit`；
- 删除 helper 内重复的 SRAM `wait()` 和 distributed HBM 旁路；
- `Load_prim`/`Store_prim` 可废弃或变成 `Lsu_mem` 的兼容别名；
- `temp_ram_array` 用户逐个迁入 `temp/intermediate` region，迁移完成后移除第二份容量。

## 12. 代码改造清单

### 12.1 新增文件

| 文件 | 责任 |
|---|---|
| `llm/include/memory/sram/sram_types.h` | byte range、initiator、command、request、stats |
| `llm/include/memory/sram/sram_storage.h` | backing store 与 valid 接口 |
| `llm/src/memory/sram_storage.cpp` | 功能读写实现 |
| `llm/include/memory/sram/sram_region.h` | region config、table、allocator |
| `llm/src/memory/sram_region.cpp` | 解析后校验、resolve、allocate/free |
| `llm/include/memory/sram/sram_access_unit.h` | per-core SRAM 仲裁器 |
| `llm/src/memory/sram_access_unit.cpp` | bank/port/queue/SystemC 状态机 |
| `llm/include/memory/core_lsu_unit.h` | LSU descriptor/token/API |
| `llm/src/memory/core_lsu_unit.cpp` | LSU worker 与 HBM/SRAM 数据面 |
| `llm/include/dte/dte_memory_bridge.h` | DTE 到真实 memory system 的桥接 |
| `llm/src/dte/dte_memory_bridge.cpp` | DTE real-memory worker |
| `llm/src/prims/norm_prims/lsu_mem.cpp` | `Lsu_mem` primitive |
| `llm/src/memory/sram_r*_selftest.cpp` | 各阶段 SystemC selftest |

### 12.2 修改文件

| 文件 | 修改 |
|---|---|
| `llm/include/common/config.h`、`llm/src/common/config.cpp` | SRAM/LSU per-core 配置类型 |
| `llm/include/defs/spec.h`、`llm/src/defs/spec.cpp` | real data path/manual region 开关 |
| `llm/src/utils/config_utils.cpp` | region、bank、port、LSU 参数解析和校验 |
| `llm/include/common/include.h` | `TaskCoreContext` 暴露 DTE/LSU/region/timeline API |
| `llm/include/workercore/workercore.h`、`llm/src/workercore/workercore.cpp` | 每核实例化、binding、生命周期 |
| `llm/include/dte/dte_async.h`、`llm/src/dte/dte_async.cpp` | 公共 hazard 与 real completion gate |
| `llm/include/dte/dte_unit.h`、`llm/src/dte/dte_unit.cpp` | bridge hook、真实数据面完成状态 |
| `llm/src/prims/norm_prims/dte_async_prim.cpp` | `hbm_addr/region/offset` 解析 |
| `llm/src/prims/base/npu_base.cpp` | manual schedule 模式和旧路径兼容 |
| `llm/src/utils/memory_utils.cpp` | 旧 helper 包装到统一 API |
| `llm/src/common/memory.cpp` | region-aware label/spill/地址管理 |
| `CMakeLists.txt`、`llm/unittest/npusim.cpp` | 新源码、自测入口、CTest 注册 |

## 13. 分阶段实施与验收

### S0：冻结契约与配置

- [x] 定义 byte 地址、区间、对齐、valid、completion 和错误语义。
- [x] 定义 region JSON schema、全局默认与 per-core override。
- [x] 定义 DTE/LSU descriptor、token 和 primitive wire/JSON。
- [x] 明确 DTE endpoint、SRAM、HBM/NoC 三层时序责任。
- [x] 增加纯配置 parser 和负例测试。

验收：不存在 word/byte、SPM/SRAM、remote/HBM、端口带宽/HBM 带宽的歧义；所有非法 region 在 elaboration 前失败。

### S1：SRAM storage 与 region

- [x] 实现 `SramStorage` byte payload、valid、byte-enable、clear。
- [x] 实现 `SramRegionTable`、fixed/block region、权限和 resolve。
- [x] 实现 region-aware allocation/free/label binding。
- [x] 旧配置自动映射为完整 `legacy` region。

验收：写入非零 pattern 后可逐 byte 读回；重叠、越界、权限、非法 free 全有负例；不同 core SRAM 数据隔离。

### S2：统一 SRAM 端口与仲裁

- [x] 实现 `SramAccessUnit` 的 read/write queue、bank mapping 和 RR。
- [x] 接入 compute、legacy helper、DTE、LSU initiator 类型。
- [x] 实现公共 RAW/WAR/WAW range tracker。
- [x] 导出 per initiator/bank/port 的 bytes、requests、queue wait、service、stall。

验收：单端口带宽符合 oracle；同 bank/端口串行、独立端口并行；DTE 与 LSU 对同区间 hazard 正确阻塞；无重复计时。

### S3：LSU 真实 HBM↔SRAM 路径

- [x] 实现 `CoreLsuUnit` issue/wait/poll/fence/cancel。
- [x] HBM read payload 提交本核 SRAM；SRAM payload 写回 HBM。
- [x] 实现 `Lsu_mem` primitive 和 C++ API。
- [x] 接 behavioral HBM 与 DRAMSys HBM。

验收：HBM→SRAM→HBM round-trip 非零 pattern 一致；HBM backend stats 的 read/write bytes 与 descriptor 一致；远端 NUMA HBM 仍经过真实 NoC/C2C。

### S4：DTE 真实 HBM↔SRAM 路径

- [x] 实现 `DteMemoryBridge`。
- [x] 扩展 `Dte_async` DRAM 方向的字段解析和 completion gate。
- [x] 复用 DTE credit、slot、SPM/AXI port、aggregation 和 token。
- [x] real-memory 关闭时严格保持 V4 endpoint-only 行为。

验收：DTE round-trip 数据正确；token 不会早于 HBM 和 SRAM commit 完成；DTE V0-V4 全回归；endpoint 与 HBM/SRAM 统计可逐笔对账。

### S5：原语内手动调度与 double buffering

- [x] 实现 `manual_memory_schedule` 和 `ComputeTimeline::RunTile()`。
- [x] 增加一个 tiled matmul 或 synthetic compute primitive。
- [x] 使用 `double_a/double_b` 手动 region，执行预热、稳态、排空。
- [x] 增加 DTE load + compute、LSU load + compute、load/compute/store 三阶段示例。

验收：trace 显示真实 compute 与 DTE/LSU worker 重叠；双缓冲稳态接近 `max(Tload, Tcompute)`，阻塞版本接近 `Tload + Tcompute`；单 buffer 覆盖 hazard 负例失败。

### S6：旧 SRAM 路径迁移与收口

- [x] 旧 helper 全部包装到统一 SRAM/LSU/DTE API。
- [x] 自动 spill/reload 使用真实 HBM 地址和 payload。
- [x] communication buffer、input/output/intermediate 标签迁入 region。
- [x] 移除或兼容别名化空壳 `Load_prim/Store_prim`。
- [x] 迁移 `temp_ram_array`，消除额外未计容量 SRAM。
- [x] 更新用户文档、默认配置和 trace viewer 字段。

验收：新旧 workload 结果一致；扩展关闭时旧时序冻结；扩展开启时所有 SRAM/HBM 流量只有统一入口，不存在直接旁路。

## 14. 测试矩阵

| 类别 | 必测内容 |
|---|---|
| region | 合法布局、holes、重叠、越界、对齐、权限、fixed/block、per-core override |
| 数据 | 非零 pattern、尾部 byte-enable、部分覆盖、invalid read、clear、round-trip |
| SRAM 时序 | 单 port 饱和、多 bank 并行、同 bank 冲突、队列深度、RR 公平 |
| LSU | load/store、多个 outstanding、selective wait、fence、pending cancel、越界 |
| DTE | 六方向回归、DRAM 两方向真实数据、credit、slot、aggregation、hazard |
| HBM | behavioral/DRAMSys、local/remote NUMA、多 channel、多核争用、统计对账 |
| 并发 | DTE vs LSU、DMA vs compute、read/read、RAW/WAR/WAW、double buffer |
| primitive | JSON parse/serialize/deserialize、issue/wait/poll/fence、未消费 token 负例 |
| 兼容 | real path on/off、manual region on/off、DTE on/off、legacy/distributed topology |
| 回归 | DTE V0-V4、HBM R0-R4、D2D、NoC、默认 workload DONE 时间 |

独立 Python oracle 至少覆盖：

- SRAM beat/bank/port 服务时间；
- region address resolve；
- descriptor 分块与尾部 byte-enable；
- blocking 与 double-buffer 总时长；
- DTE/LSU/HBM/SRAM 各层 byte 统计守恒。

## 15. trace 与统计

新增 trace 阶段：

- `SRAM_queue`、`SRAM_read`、`SRAM_write`、`SRAM_bank_wait`；
- `LSU_issue`、`LSU_hbm`、`LSU_sram`、`LSU_wait`；
- `DTE_mem_hbm`、`DTE_mem_axi`、`DTE_mem_spm`、`DTE_mem_commit`；
- `SRAM_region_alloc/free/spill/reload`；
- `Compute_tile`，包含输入 region、offset、tile id。

统计守恒：

```text
HBM→SRAM completed bytes
  = HBM read response effective bytes
  = DTE/LSU destination SRAM committed bytes

SRAM→HBM completed bytes
  = source SRAM effective read bytes
  = HBM write effective bytes
```

每笔 descriptor 记录 issue、admit、HBM begin/end、SRAM begin/end、complete、wait begin/end，能够解释排队、媒体、NoC、端口和 hazard 各自贡献。

## 16. 风险与规避

1. **重复计时**：DTE endpoint、SRAM helper、HBM backend 都可能收费。通过统一阶段所有权和统计对账禁止旁路。
2. **双份容量**：现有 `ram_array/temp_ram_array` 会让可用 SRAM 翻倍。迁移期明确标记，S6 必须收敛为同一容量池。
3. **配置 bitwidth 与模板宽度不一致**：功能存储改为 byte backing，端口宽度 runtime 化。
4. **SystemC 并发错误**：所有异步 worker 都使用有界 SystemC process/event；不使用宿主线程。
5. **token 提前完成**：completion gate 必须等待控制面和数据面共同完成。
6. **自动 spill 破坏手动 buffer**：non-spillable region 从 victim 集合中彻底排除。
7. **HBM 地址伪造**：每个 spillable allocation 保存真实 backing address，禁止默认固定写回地址。
8. **回归范围过大**：所有新能力以 runtime 开关接入，每阶段都运行冻结回归后再进入下一阶段。

## 17. 最终验收条件

只有同时满足以下条件，四项需求才算完成：

1. DTE 发起 `DRAM_TO_SPM/SPM_TO_DRAM` 时，真实 HBM backend 和真实 per-core SRAM 都观察到对应 payload 与统计，不再只是端口时间。
2. LSU primitive/API 能在不经过 DTE 的情况下完成本核 SRAM↔HBM round-trip，并与 DTE/HBM/SRAM 正确竞争。
3. workload 可通过 primitive 控制 issue/wait/fence；计算 primitive 可通过 C++ API 在 tile 循环中控制 load/store，与 compute 发生真实 SystemC 时间重叠。
4. 配置可定义 input/intermediate/comm/double_a/double_b 等不重叠 region，访问权限、spill policy、对齐和容量均被强校验。
5. double-buffer workload 的数据结果、总时间和 trace 同时通过 oracle；不能只凭公式或日志宣称重叠。
6. behavioral HBM、DRAMSys HBM、本地/远端 NUMA、多核争用均通过测试。
7. 新能力关闭时，既有 DTE/HBM/NoC/D2D 与默认 workload 回归无变化。
8. 所有生产 SRAM/HBM 流量进入统一访问路径，旧同步 helper 不再保留独立计时或数据旁路。

## 18. 最终验收记录（2026-08-06）

- `Sram_pipeline` 在真实 WorkerCore 中完成 LSU/DTE 两套 blocking 与 double-buffer 调度，四种执行结果校验和均为 130560，double-buffer 均快于 blocking。
- `run_test_sram_pipeline.py` 直接解析 `events.json`，要求 SRAM、LSU、DTE、compute、region 的全部阶段存在且 B/E 平衡，并证明 LSU/DTE 的 compute/HBM overlap。
- production label probe 把 compatibility-word payload 分别搬移到 `input`、`intermediate`、`comm` block region，读回非零数据后经 `deletePair` 释放；rename 同步 RegionTable allocation 元数据。
- distributed behavioral、distributed DRAMSys、legacy_private 和双 die remote NUMA fixtures 全部通过；NUMA 用例的 D2D in/out 均为 672 包。
- real-data 关闭的旧 `Relu_f`/NpuBase/helper workload 正常结束且无 watchdog；D2D V0 与 link 自测分别 308/308、37/37。
- 完整命令、结果和评审修改记录见 `log/FINAL_acceptance_2026-08-06.md`。
- 最终条件 7 已关闭：README 和 CLI 使用 `llm/test/default` 的规范四配置；无参数 smoke 与 real-data 关闭的完整 `playground.json` 均通过。playground 在 7270718 ns 完成 4 个 DONE，S_DATA 六阶段均为 4，router/D2D residual 为 0 且 credit balanced。P0/P1 开发、测试和评审记录见 `log/P0_P1_default_startup_closure_2026-08-06.md`。
