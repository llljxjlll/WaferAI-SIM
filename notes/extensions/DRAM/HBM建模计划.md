# 分布式 HBM 建模计划（修订版）

## 0. 建模结论与范围

本计划建模的目标系统是：一个或多个 compute die，每个 compute die 物理上连接一颗或多颗 HBM stack，compute die 之间通过现有 C2C/D2D 网络互连。

本文统一使用以下术语，避免把 compute die、HBM DRAM die 和 HBM stack 混淆：

- `compute_die`：包含 core/tile/NoC 的计算裸片；
- `hbm_stack`：一颗完整 HBM 封装，由 base die 和多层 DRAM die 组成；
- `channel`：stack 内独立的 HBM channel/MC 粒度；
- `pseudo_channel`：channel 内的半独立数据通道，具体数量由 HBM profile 决定；
- `mem_port`：compute die 边缘连接 MC/PHY 的逻辑接入位置，不等同于 HBM channel、pseudo-channel 或 stack。

默认内存语义确定为：

1. **单个 compute die 内是地址交织 UMA**：该 die 上所有 core 都可以访问其本地一颗或多颗 HBM stack，目标 stack/channel/pseudo-channel 由物理地址决定，不按 core ID 固定拥有私有 DRAM。
2. **多 compute die 系统是共享地址空间 NUMA**：物理地址先确定 `home_compute_die`，再在 home die 的本地 HBM 中交织；远端 core 可以经 C2C 访问该地址，但延迟更高并消耗 C2C 带宽。
3. **“核到最近 MEM 端口”只负责路径选择，不负责数据归属**。同一物理地址无论由哪个 core 访问，都必须解码到同一个 home die/stack/channel。
4. 分布式 HBM 第一版默认 `cache_policy=none`，所有 DRAM 访问都进入 NoC/HBM；原每核私有 `DCache` 作为 `legacy_private` 兼容模式保留，不能直接搬到 MEM 端口充当 HBM controller。
5. 第一阶段不建模 PHY 电路细节、信号完整性和热仿真；只把 floorplan、端口跨度和可达性作为配置约束。

## 一、基本需求与验收含义

### 1. HBM 与 compute die 边缘的连接

基本物理/逻辑通路为：

```text
core tile
  → on-die NoC
  → edge router
  → MEM endpoint / memory controller
  → HBM PHY
  → micro-bump / interposer
  → HBM stack
```

配置必须能表达：

- 哪个具体 `compute_die` 挂了哪些 HBM stack；
- stack 位于哪一侧、占用哪段连续边缘位置；
- stack 下有哪些 channel/pseudo-channel，以及它们如何映射到 MEM 端口；
- HOST、C2C、MEM 对边缘 tile/PHY keep-out 区间的竞争；
- 一颗 stack 的所有端口是否物理聚簇、连续且不跨边；
- 逻辑配置与实际 DRAM backend 的 channel 数、容量和带宽是否一致。

### 2. 核发起读写请求

请求必须携带至少：

- 64-bit byte address；
- read/write command；
- transaction ID；
- transfer length、burst sequence 和必要的 byte enable；
- write data 或 read response data；
- response status。

系统必须支持多个 core 对同一地址读写、多个 outstanding transaction、NoC/MC 背压和响应匹配，不能只传递一个“访问发生了”的延迟事件。

### 3. NoC、C2C 与 HBM 带宽匹配

需要分别计算并验证：

- HBM channel/pseudo-channel 的理论及可实现带宽；
- MEM endpoint 接入带宽；
- on-die NoC 每条有向链路的 raw wire bandwidth 和 useful payload bandwidth；
- 远端 NUMA 访问经过的每条 C2C link bandwidth；
- 多个 MEM endpoint 共享 router/link 时的聚合瓶颈。

读路径和写路径的请求/数据方向不同，必须分别计算，不能只用一个 `bandwidth_gbps` 与某个 C2C 端口做比较。

## 二、内存语义与地址映射

### 1. 默认：本地交织 UMA + 系统级 NUMA

地址解码顺序固定为：

```text
physical address
  → home compute die
  → HBM stack
  → channel
  → pseudo-channel（若由上层显式选择）
  → backend local address / bank / row / column
```

默认策略 `numa_local_interleave`：

1. 以较大连续地址范围确定 `home_compute_die`；
2. 在该 die 挂载的多个 stack/channel 间按 `stripe_bytes` 细粒度交织；
3. 同一地址的 home 不随请求 core 改变；
4. home 相同则走本地 NoC，home 不同则先走 C2C 到 home die，再走 home die 的本地 NoC/MEM endpoint。

示例：两个 compute die，每个挂两颗 HBM stack，每个 NUMA node 暴露 32 GiB：

```text
[0, 32 GiB)   → home die 0 → stack 0/1 交织
[32, 64 GiB)  → home die 1 → stack 2/3 交织
```

### 2. 可配置地址策略

至少支持：

- `local_interleave`：每个 die 只有独立本地地址空间，禁止远端访问；
- `numa_local_interleave`：全局共享地址空间，地址有 home die，允许远端访问；默认；
- `global_interleave`：跨所有 die/stack 细粒度交织，仅用于实验，不作为默认，因为会让大量顺序流量穿过 C2C。

建议配置字段：

```json
"address_policy": {
  "mode": "numa_local_interleave",
  "home_ranges": [
    {"die_id": 0, "base": 0, "size_bytes": 34359738368},
    {"die_id": 1, "base": 34359738368, "size_bytes": 34359738368}
  ],
  "stack_interleave_bytes": 256,
  "channel_interleave_bytes": 256
}
```

启动期必须校验：地址范围无重叠、无空洞（除非显式允许）、对齐满足 burst/stripe 要求、暴露容量不超过 backend 容量。

### 3. 地址映射与路由映射分离

新增两个独立抽象，不能再用一张 `core_id → port_id` 表同时承担两种职责：

```text
MemoryAddressMap:
  address → {home_die, stack_id, channel_id, pseudo_channel_id, local_address}

MemRouteTable:
  {current_die, ingress_tile, target_stack/channel}
    → local MEM port 或下一跳 C2C exit port
```

“就近”只用于 `MemRouteTable` 在多个等价入口间选路；选择结果不得改变地址的 home。

## 三、物理拓扑与配置结构

### 1. stack 级配置与 port 级配置分离

`dram_config`、容量、代际、PHY span 都是 stack/channel 级属性，不应在每个端口重复填写。建议结构为：

```json
"memory_system": {
  "topology": "distributed_hbm",
  "cache_policy": "none",
  "profiles": {
    "hbm2_2gt": {
      "generation": "HBM2",
      "channels_per_stack": 8,
      "pseudo_channels_per_channel": 2,
      "data_rate_gbps_per_pin": 2.0,
      "stack_bus_width_bits": 1024
    }
  },
  "hbm_stacks": [
    {
      "stack_id": 0,
      "compute_die_id": 0,
      "profile": "hbm2_2gt",
      "side": "E",
      "start_idx": 2,
      "phy_span_tiles": 8,
      "capacity_bytes": 8589934592,
      "backend_granularity": "channel",
      "channel_dram_config": "hbm2-example.json"
    }
  ]
}
```

最终 schema 可调整，但必须保持以下单一真源：

- stack 数量 = `hbm_stacks` 中不同 `stack_id` 的数量；
- channel 数由 profile 与 stack 配置共同校验；
- 每个 port 只引用一个 stack/channel，不复制 stack 属性；
- `stack_id` 不能用“端口数除以常数”事后猜测；端口分组用于校验物理连接，而不是推断一个含糊的 HBM die 数。

允许端口聚合多个 pseudo-channel，但必须显式配置 `port_granularity` 或映射列表，不能根据带宽接近程度自动改变硬件拓扑。

### 2. per-die floorplan

当前 `g_die_ports` 是所有 compute die 共用的同构模板；这不足以表达“只有外围 die 挂 HBM”。需要增加以下方案之一：

- `die_ports.per_die_overrides`；或
- 由 `hbm_stacks[].compute_die_id + mem_port` 建立每个 die 的实例化端口表。

推荐后者：保留 C2C/HOST 模板，同时由 stack attachment 为具体 die 增加 MEM 角色和 keep-out。

物理校验至少包括：

- 同一 stack 的 channel/MEM port 在同一边、连续或符合显式允许的间隔；
- `phy_span_tiles` 是 stack/PHY 级连续区间；
- MEM PHY 区间不与 HOST/C2C PHY 区间重叠；
- corner tile 同时属于两条边时不能被重复占用；
- 两颗 stack 的 interposer footprint/PHY span 不重叠；
- compute die 没有可行 attachment 时不得创建本地 stack。

“朝向相邻 compute die 的整条边一定不能挂 HBM”只作为可选 floorplan 规则，不能由逻辑 die mesh 自动假定。实际判断应基于离散端口/keep-out 预算。

### 3. C2C lane 配置

C2C 实占 lane 数已经可以由 `die_ports.overrides` 中该方向 C2C port 的数量得到。不要再增加一份独立且可能冲突的 lane-count 真源。

如需要简化配置，可以增加 count 语法糖并在解析时展开成端口列表；展开后仍以端口表为唯一真源，并遵守现有 multi-port/link-group 契约。

## 四、运行时模块与请求路径

### 1. 模块边界

新增独立模块，而不是复用 `DCache`：

```text
CoreMemAdapter
  - 接收现有 blocking/non-blocking 访存调用
  - 做地址解码、分包、transaction ID 分配
  - 保存 outstanding transaction 状态
  - 收齐响应后完成原 TLM/event 请求

MemEndpointUnit
  - 挂在具体 edge router/MEM port
  - 接收并重组 MEM request/write data
  - 维护有界请求队列和公平性
  - 向 HBM backend 发 TLM transaction
  - 将 read data/completion 封装成 NoC 响应

HBMBackend
  - BehavioralHBMBackend 或 DRAMSysHBMBackend
```

`DCache` 当前包含每核 tag、dirty、坐标、统计和 DRAMSys wrapper，不能整体移动到 die 级端口。分布式模式下只复用/重构 `DRAMSysWrapper`，不复用 DCache cache-state 部分。

### 2. MEM endpoint 编址与路由

仓库已经有 `EP_MEM` 和 `MemEndpointOfDie(die_id)`。当前缺的是 EP_MEM 的实际路由和交付逻辑，不是简单再增加一个未被消费的 `MEM_ENDPOINT_ID` 常量。

第一版保持“每 compute die 一个 MEM endpoint ID”，具体 stack/channel 放入 MEM wire header，并由 home die 的 `MemRouteTable` 解析到目标 MEM tile。这样 endpoint 地址预算仍为 `DIE_COUNT` 个 MEM endpoint。

若后续需要每个 controller 独立 endpoint，再把 endpoint 数量改为按实际 controller 数动态预算，并同步扩展 `ValidateAddressSpace()`；不能在仍只预留 `DIE_COUNT` 个地址时直接编码任意多个 controller。

路由需要新增：

- `ResolveMemAnchor()`：把 EP_MEM + target stack/channel 解析成目标 tile；
- 本地 MEM 路由：片内 XY/已有 NoC 算法收敛到目标 MEM tile；
- 远端 MEM 路由：先按 home die 走现有 die-level C2C 路由，进入 home die 后重新解析本地 MEM anchor；
- response 以原请求 core 为目标，经 NoC/C2C 返回，但不要求物理路径严格对称。

### 3. 独立 MEM wire protocol

当前通用 `Msg.offset_` 在 wire 上只有 8 bit，且 `REQUEST/ACK/DATA` 已承担既有 send/collective 协议语义，不能直接拿来表示 64-bit 内存地址。

新增独立 codec/消息类型，例如：

```text
MEM_REQ    : txid, source, home, stack/channel, 64-bit address, command, length
MEM_WDATA  : txid, sequence, end, byte-enable, payload
MEM_RDATA  : txid, sequence, end, payload
MEM_RESP   : txid, completion/status
```

可以通过 tagged-union 复用 256-bit wire，但必须使用独立 serializer/deserializer 和静态位宽检查，不能把地址截断到 `Msg.offset_`。

必须定义并测试：

- read/write 的请求、数据和 completion 顺序；
- 相同 transaction ID 内有序，不同 ID 是否允许乱序；
- 最大 outstanding 数和 ID 回收；
- 请求头、write data、read response使用何种控制/数据通道；
- request/response queue 分离，避免响应被请求阻塞形成协议死锁；
- NoC/C2C 背压时不得丢包、重复包或提前完成 TLM transaction。

## 五、cache 策略与旧路径兼容

### 1. 默认策略

分布式 HBM 初版配置：

```text
memory_system.topology = distributed_hbm
memory_system.cache_policy = none
```

此时所有 DRAM 读写都经过 CoreMemAdapter → NoC/C2C → MemEndpointUnit → HBM backend，便于验证真实共享带宽与争用。

### 2. legacy 与未来扩展

- `legacy_private`：保留原每核 DCache/DRAMSys 直连，确保旧配置行为不变；
- `private_dcache`：未来可支持每核私有 cache，只有 miss/writeback 进入分布式 HBM；必须同时定义 write policy、replacement 和 coherence/non-coherent 契约；
- `shared_llc`：属于后续独立扩展，不放入本轮 R0-R4 的完成条件。

R2 以后也不能无条件删除 `WorkerCore::dcache`。应根据 topology 模式选择实例化 legacy DCache 或 CoreMemAdapter，待所有默认配置迁移完成后再考虑移除 legacy 代码。

## 六、HBM backend、实例粒度与仲裁

### 1. DRAMSys 配置必须解析为 resolved profile

`DRAMSys/configs/hbm3-example.json` 当前实际引用 `HBM2_WSC.json`，不能作为“HBM3 默认配置”。R0 必须增加配置一致性校验：

- wrapper 文件名/声明 profile 与 memspec `memoryType` 一致；
- `nbrOfChannels`、`nbrOfPseudoChannels`、`width`、`nbrOfDevices`、`dataRate`、`tCK` 可被完整读取；
- address mapping 与 memspec 匹配；
- profile 声明的 channel 数与 backend 实际实例数匹配。

在没有经过验证的 HBM3 memspec 前，不设置伪 HBM3 默认值。测试默认可以显式使用经过校验的 HBM2 配置。

### 2. 实例粒度必须唯一确定

推荐第一版采用：

```text
一个 DRAMSys instance = 一个 HBM channel（内部含该 channel 的 pseudo-channel）
一颗 HBM2 stack = profile 指定数量的 channel instance
```

前提是使用的 memspec 确实描述单 channel。若未来采用 `nbrOfChannels=N` 的完整 stack memspec，则改为一个 DRAMSys instance 表示整颗 stack，并让 DRAMSys 内部 address decoder 选 channel；两种模式不能同时叠加。

配置增加显式 `backend_granularity = channel | stack`，启动期拒绝实例数与 `nbrOfChannels` 不一致的组合。

### 3. 仲裁位置

DRAMSys 内部已经有按 channel 的 arbiter/controller queue。`MemEndpointUnit` 仍需要一个 NoC→TLM 接入队列，用于：

- 汇聚多个 core/入口请求；
- 限制 outstanding；
- 在多个 MEM port 映射到同一 controller 时仲裁；
- 建模 endpoint/MC 前端带宽和公平性。

不要直接复用 `memory/sram/arbiter_ram_bank.h`：它是 SRAM `ram_if`/semaphore 模型，不是带 transaction ID、TLM phase 和响应重排的内存控制器前端。

### 4. behavioral 与 DRAMSys 共用入口

现有 `memory_utils.cpp` 中行为级 DRAM 直接 `wait()`，会绕过 NoC。改造后两种 backend 必须都位于 MemEndpointUnit 后面：

```text
MemEndpointUnit → BehavioralHBMBackend
                or DRAMSysHBMBackend
```

`HW_BEHA_DRAM_UTIL` 只用于 behavioral backend。DRAMSys backend 不再额外乘利用率，避免重复限速。

## 七、带宽、容量与单位

### 1. 统一单位

配置中避免含糊的 `bandwidth_gbps`：

- pin data rate：`data_rate_gbps_per_pin`，单位 Gbit/s/pin；
- 总带宽：`bandwidth_GBps` 或内部统一为 `bandwidth_bytes_per_second`；
- 容量：`capacity_bytes`；
- NoC/C2C：保留 packet/cycle 时必须同时知道 packet/flit 位宽和 cycle 时间。

### 2. HBM 理论带宽

物理 profile 的理论带宽：

```text
B_stack = data_rate_per_pin × stack_bus_width / 8
```

例如 HBM2 2 Gbit/s/pin、1024-bit stack 为 256 GB/s；一个 64-bit pseudo-channel 为 16 GB/s。

但 DRAMSys 实例的有效数据总线使用 resolved memspec 语义，当前代码中 `dataBusWidth = width × nbrOfDevices`。因此不能只读取 `width × dataRate / tCK` 就认定实例带宽；R0 解析器应按 DRAMSys 实际 MemSpec 规则计算，并通过 R3 microbenchmark 校准。

### 3. NoC useful bandwidth

对每条有向 NoC link：

```text
B_raw = wire_bits / cycle_time
B_useful = B_raw × useful_payload_bits / wire_bits × protocol_efficiency
```

当前 cycle 配置的例子是 256-bit wire、`CYCLE=2ns`，raw 上限为 16 GB/s；若内存数据每 flit 只能携带 128 bit useful payload，则未计其他开销时只有 8 GB/s。最终校验必须使用实际 MEM codec 的 payload，而不是通用 `Msg` 的假设值。

`HW_NOC_PAYLOAD_PER_CYCLE` 若表示行为级压缩/缩放，不得被误当成 cycle router 的物理多 lane 数；cycle 模式大于 1 packet/cycle 必须有真实多 lane/多通道实现。

### 4. 路径瓶颈

分别计算：

```text
local read  = min(request NoC path, HBM read, response NoC path)
local write = min(request NoC path, write-data NoC path, HBM write)
remote read/write = 上述路径再加入所有 C2C link
```

还必须检查多个 MEM endpoint 汇聚到同一 router/link、多个 core 共享同一 channel 的聚合 offered load。MEM port 不能与 `D2DPort.bw` 做“一端口对一 C2C 端口”的静态比较；C2C 只在远端 NUMA 路径上作为额外瓶颈。

### 5. override 规则

- 从 profile/memspec 推导的理论带宽是物理上限；
- `bandwidth_cap_GBps` 可以显式配置更低的实现上限；
- cap 超过理论值应报错，而不是允许 2 倍偏差；
- behavioral `efficiency` 必须在 `(0,1]`；
- DRAMSys 不使用 behavioral efficiency。

### 6. 容量

带宽不能推导容量。容量来源优先级：

1. DRAMSys memspec/address mapping 可实际寻址的容量；
2. profile/产品配置声明的 stack 容量；
3. `capacity_bytes` 可以限制暴露容量，但不能超过 backend 容量。

若显式容量大于 memspec 容量，必须更换/生成匹配的 memspec 和 address mapping，不能只修改 metadata。地址解码、越界检查和 local address compaction 都以实际暴露容量为准。

## 八、代码改造清单

### 1. 配置与拓扑

- `llm/include/die/port.h`
  - 保留 `ROLE_MEM`；
  - 增加 per-die 实例化端口查询；
  - 不把所有 stack 属性塞入通用 `D2DPort`；新增独立 `HBMProfile/HBMStackConfig/HBMChannelConfig`。
- `llm/src/die/port_config.cpp`
  - 解析具体 die 的 MEM attachment；
  - 建立 per-die port occupancy/keep-out；
  - 新增 `BuildMemAttach()`/`MemRouteTable`；
  - 保留现有 C2C multi-port/link-group 语义。
- `llm/src/utils/config_utils.cpp`
  - 解析 `memory_system`、address policy、profiles、stacks；
  - 解析并校验 resolved DRAMSys memspec；
  - topology 未配置时保持 `legacy_private`。

### 2. 地址与路由

- 新增 `MemoryAddressMap` 纯函数模块；
- `llm/include/utils/router_utils.h`、`llm/src/utils/router_utils.cpp`
  - 消费现有 `EP_MEM/MemEndpointOfDie()`；
  - 新增本地/远端 MEM anchor 解析和 repin；
  - 若未来改成 per-controller endpoint，再动态扩展 endpoint budget。
- `llm/include/defs/spec.h`/`llm/src/defs/spec.cpp`
  - 第一版无需再创建重复的单值 `MEM_ENDPOINT_ID`；如需要可增加语义清晰的 `MEM_ENDPOINT_BASE`，其值与现有 `MemEndpointOfDie(0)` 一致。

### 3. 内存数据路径

- 新增 `CoreMemAdapter`、`MemEndpointUnit`、`HBMBackend` 接口；
- 新增 MEM wire codec 和 transaction/outstanding table；
- `llm/include/router/router.h`、`llm/src/router/router.cpp`
  - 在 MEM attach tile 创建 endpoint 接口；
  - 接入 MEM request/response 队列、背压和统计；
- `llm/include/workercore/workercore.h`、`llm/src/workercore/workercore.cpp`
  - 按 topology 选择 legacy DCache 或 CoreMemAdapter；
  - 将 `MaxDramAddr/defaultDataLength/g_dram_kvtable` 初始化迁到统一 memory-system metadata；
- `llm/src/utils/memory_utils.cpp`
  - behavioral 和 DRAMSys 请求都通过统一 adapter；
  - 删除 distributed 模式下直接按 `dram_bw` 做 `wait()` 的旁路。

### 4. backend 生命周期与统计

- 新增 `BuildHBMBackends()`，由 memory-system/monitor 统一持有生命周期；
- 不在 `BuildD2DLinks()` 中创建 SystemC module；拓扑纯数据构建与 SystemC elaboration 分开；
- 统计至少包括 per core/port/channel/stack 的 requests、bytes、queue wait、service time、NoC/C2C stalls、row hit/miss（DRAMSys 可用时）和 remote NUMA traffic。

## 九、分阶段开发与验证计划

### R0：配置、物理拓扑和地址映射契约

目标：不改变运行时，先建立唯一、可验证的系统描述。

开发内容：

1. 新增 `memory_system` schema、HBM profile/stack/channel、per-die attachment；
2. 新增 `MemoryAddressMap` 和 `MemRouteTable` 纯函数；
3. 新增 resolved memspec 解析和单位换算；
4. 新增 stack/port/keep-out/address range/capacity/backend granularity 校验；
5. 保持所有旧配置默认走 `legacy_private`。

测试：

- 合法的单 die 多 stack、本地交织配置；
- 合法的多 die NUMA home range；
- 地址映射确定性：不同 requester 访问同一地址得到相同 home；
- range 重叠/空洞、stripe 未对齐、容量越界；
- per-die MEM/C2C/HOST keep-out 冲突和 corner 重复占用；
- mislabeled memspec、channel 数/实例粒度不匹配；
- 旧配置与全部已注册回归保持不变。

**实现状态：R0 契约已完成，`--hbm-r0-selftest` 40/40 通过。** 已实现并验证：

- `local_interleave` 使用 `DecodeAddress(address,current_die)` 消除独立地址空间歧义，允许各 die 的 range 重叠；`MemRouteTable` 明确拒绝该模式的远端访问；
- `numa_local_interleave` 的全局 home range，以及 `global_interleave` 的两级 UMA 映射（先 stack、后该 stack 内 channel）。禁止把 `(stack,channel)` 拍平，否则 channel 更多的 stack 会被错误分配更多地址份额；
- channel/聚合 MEM port 显式粒度、`channels_per_mem_port`、per-die keep-out、corner/端口冲突和容量/对齐校验；物理 port 数与 `backend_granularity` 解耦；
- resolved HBM2 memspec 的 channel、pseudo-channel、总线宽度、pin rate、容量和带宽交叉校验。当前 DRAMSys checkout 没有 `MemSpecHBM3` 实现，HBM3 配置明确拒绝；
- 未配置 `memory_system` 时完整复位并保持 `legacy_private`。

### R1：功能性 MEM wire 与本地 UMA 路径

目标：先用简单 backing store 跑通真实 NoC 请求/响应，不引入 DRAMSys 时序复杂度。

开发内容：

1. `CoreMemAdapter`、MEM codec、transaction ID/outstanding table；
2. `MemEndpointUnit` 和无/固定时延 functional backend；
3. EP_MEM 本地路由、目标 MEM tile 交付和响应返回；
4. topology switch：distributed 模式走新路径，legacy 保持旧路径。

测试：

- 单核 read/write/read-after-write；
- 两个 core 对同一地址交叉读写，证明不是每核私有副本；
- 不同地址交织到多个 stack/channel；
- 多 outstanding、响应乱序、txid 回收；
- burst、非对齐、channel/容量边界；
- 有界 buffer 背压下无丢包、重复包、提前完成或死锁。

**实现状态：wire/adapter/endpoint 合成链路已完成，`--hbm-r1-selftest` 18/18 通过；R4 已在保持该接口不变的前提下接入真实 Router/NoC/C2C。**

- MEM transaction 使用独立的 256-bit tagged wire：REQ、WDATA、RDATA、RESP；支持 64-bit 地址、14-byte 数据 payload、逐 flit 路由元数据、sequence/end、byte-enable 和最长 65535-byte transaction；
- `CoreMemAdapter` 使用 collision-free txid 集合，所有合成请求/响应都强制经过 codec，并在地址 range、stack/channel stripe 边界自动拆分和重组；
- `MemEndpointUnit` 使用有界 FIFO 和独立 dispatch/completion 进程，支持多个调用方并发、max outstanding、队列背压和响应匹配；
- 测试覆盖同址跨核共享、双 channel 隔离、40-byte 多 flit、跨 stripe burst、并发请求和 txid 回收。

R1 的隔离自测仍保留直接 endpoint binding，供 codec/endpoint 单元回归；生产配置由 R4 的 `MemTransport/HBMNetwork` binding 替换该交付方式，请求和响应实际占用 Router、NoC 与 C2C。

### R2：共享队列与 behavioral HBM

目标：建立共享 HBM 竞争、带宽和延迟的快速模型。

开发内容：

1. MemEndpoint/MC 前端有界队列、公平仲裁、read/write turnaround；
2. `BehavioralHBMBackend` 的 latency、bandwidth cap、efficiency；
3. on-die NoC request/response useful-bandwidth 统计；
4. 所有 distributed 模式的行为级访问移除直接 `wait()` 旁路。

测试：

- 单 channel 饱和吞吐收敛到配置上限；
- N 核共享 channel 时总吞吐不变成 N 倍，且无永久饥饿；
- 多 channel/stack 交织获得预期并行扩展；
- 分别构造 NoC 瓶颈、endpoint 瓶颈和 HBM 瓶颈；
- 读写混合、短 burst 与长 burst 的开销差异。

**实现状态：behavioral backend 与共享 endpoint 已完成，`--hbm-r2-selftest` 18/18 通过；R4 已补齐 NoC/C2C hop、双向流量、注入 stall 和共享输出争用统计。**

`BehavioralHBMBackend` 维护共享 byte-address backing store，串行化同一 channel 的可用时间，带宽取 `bandwidth_GBps × efficiency`，并实现 read↔write turnaround 和 byte-enable。`MemEndpointUnit` 导出 requests/bytes/completed、queue stalls、queue wait 和 response time。测试覆盖单 channel 饱和、四核共享不放大带宽、双 channel 近线性扩展、队列深度与 backend 服务时间解耦、长短 burst、部分/重叠写、两个方向的 turnaround 及统计一致性。由于尚未接真实 Router，计划中的 NoC useful-bandwidth 与 NoC 瓶颈不能在 R2 合成链路中冒充完成。

### R3：DRAMSys backend

目标：在相同 CoreMemAdapter/MemEndpoint 路径下替换为 DRAMSys 时序模型。

开发内容：

1. 按 `backend_granularity` 创建 channel 或 stack 实例；
2. NoC transaction 与 TLM phase/response 生命周期桥接；
3. 使用 DRAMSys 内部 channel arbiter，不重复应用 behavioral efficiency；
4. 导出实际 service time、吞吐和 row/bank 行为统计。

测试：

- 单 instance 合成 trace 与直接 DRAMSys baseline 一致；
- sequential/random、row-hit/row-miss、read/write turnaround；
- 实测饱和带宽与 resolved memspec 理论值对比，并记录协议效率；
- 多 channel 实例的容量和吞吐只计算一次，不因 port/pseudo-channel 重复实例化。

**实现状态：DRAMSys backend 核心与统一生命周期构建器已完成，`--hbm-r3-selftest` 18/18 通过；R4 已完成 NoC flit 组包、endpoint/TLM 交付与 response reinjection。**

- `HBMBackend::Submit` 是异步完成契约；`DRAMSysHBMBackend` 使用 `nb_transport_fw/bw`、PEQ 和完整 BEGIN_REQ/END_REQ/BEGIN_RESP/END_RESP 生命周期，不再调用 `b_transport`，也没有固定 60ns 延迟；
- END_REQ 后继续发下一笔，请求可在 DRAMSys 内同时 outstanding；测试一次提交 8 笔并验证 peak in-flight > 1；
- `BuildHBMBackends()` 在 elaboration 阶段按 `backend=behavioral|dramsys` 和 `backend_granularity=channel|stack` 创建唯一 backend/endpoint，由 `HBMRuntime` 统一持有并绑定 adapter；
- 导出每笔访问的 decoded channel/rank/bankgroup/bank/row/column、submit/issue/complete/service/status，以及 aggregate request/byte/service stats；row-locality 测试使用 DRAMSys 自己的 AddressDecoder 构造地址；
- 强制 `StoreMode=Store`，保证参考配置不仅计时也实际保存 payload；resolved memspec 与真实 wrapper 容量交叉一致。

当前 trace 能观察 row/bank 地址及实际 service time，但不是 DRAMSys controller 内部“最终调度 row-hit/miss counter”；如最终报告要求严格 row-hit 计数，需要在 DRAMSys controller instrumentation 层增加原生计数器。真实 NoC flit 到 TLM transaction 的组包、response reinjection 和 NoC/C2C stalls 仍属于 R4。

### R4：多 die NUMA、floorplan 与全量迁移

目标：完成本地/远端 HBM 访问、物理约束和所有生产访存路径迁移。

开发内容：

1. 地址 home die 解码和跨 C2C MEM 路由；
2. per-die HBM attachment，覆盖有/无本地 HBM 的 compute die；
3. `SIM_DATAFLOW/PD/PDS/GPU` 等生产模式逐项迁移；未迁移模式在 distributed 配置下启动期明确拒绝，不能静默绕过 NoC；
4. legacy 模式继续保留，直到配置迁移完成。

**实现状态：R4 已完成，`--hbm-r4-selftest` 9/9 通过。**

- `HBMNetwork` 将每核 adapter 的 REQ/WDATA 注入源 Router 的 data VC，在 home die 的 MEM attachment 组包后调用唯一 backend/endpoint，再把 RDATA/RESP 注入独立 ctrl VC；
- 每个数据 flit 自带 source/home/stack/channel，因此 header 与 data 分离仲裁后仍可独立路由；组包器按 source/txid/kind/sequence 重排，允许 Router 背压下乱序到达；
- MEM 跨 die 逐 die 选择稳定 C2C 出口，支持无本地 HBM 的 compute die 和多跳路径；functional、behavioral、bounded SAF 三种 D2D backend 均识别 MEM wire，bounded 模式仍占有限 SAF/inflight/RX/credit 与 token 资源；
- `Monitor` 在 WorkerCore 前统一创建 `HBMRuntime/HBMNetwork`。distributed topology 不再创建每核私有 `DCache`，而是创建并绑定 `CoreMemAdapter`；同时不构造旧 `NB_DcacheIF/DcacheCore`、GPU L1/L2 cache system 及其独立 DRAMSys，避免未绑定 socket 和旁路；legacy topology 的 DCache/TLM binding 保持原样；
- GPU 地址元数据 `GpuPosLocator` 与 cache/data path 解耦，distributed 模式仍保留 locator 供原语分配、查询地址；
- `sram_first_write_generic`、`sram_spill_back_generic`、`gpu_read_generic`、`gpu_write_generic` 是 SIM_DATAFLOW/PD/PDS/GPU 共用的 DRAM 入口，distributed 分支统一调用 adapter；`SPEC_USE_BEHA_DRAM` 不再能绕过 NoC；
- 统计包括 logical read/write/bytes、request/response flit、NoC hop、双向 C2C hop、注入 stall 和 HBM/collective DATA 的共享 Router 输出争用。

测试：

- 至少 3×3 compute-die mesh，包含真正的内部 die；
- 无本地 HBM die 经一跳/多跳 C2C 访问远端 home；
- 同一地址从本地和远端 core 读取相同数据；
- remote latency/traffic 包含请求与响应的 NoC+C2C 路径；
- HBM 流量与 collective/DTE 流量共享 NoC/C2C 时的争用；
- 所有已注册 selftest、功能测试和集成回归，不再把版本号硬编码为 V0-V6。

R4 自测使用 3×3 compute-die mesh 和真实 `RouterUnit + D2DLinkUnit`，仅 die 0/8 挂 HBM，覆盖内部 die 远端访问、同址本地/远端一致、多跳读写、远端额外延迟、并发共享队列、请求/响应双向 C2C、collective data wire 与 HBM 的 Router 争用及最终 residual=0。CTest 已扩展为 R0-R4 自动发现，不再停在 R0-R3。

## 十、最终验收条件

1. **地址正确性**：同一物理地址只有一个确定 home，任何 core 的读写都观察同一 backing store；
2. **拓扑正确性**：每颗 stack、channel、MEM port 和 compute die 的物理/逻辑映射唯一且可校验；
3. **协议正确性**：请求/响应在背压和乱序下无丢失、重复、截断、txid 泄漏和死锁；
4. **带宽正确性**：单 channel、单 stack、多 stack 的饱和曲线符合 backend 与路径瓶颈，NoC/C2C/HBM 不重复计带宽；
5. **NUMA 正确性**：本地和远端访问数据一致，远端访问额外消耗 C2C 并表现出更高延迟；
6. **容量正确性**：暴露地址范围不超过实际 backend，容量不由带宽反推；
7. **兼容性**：未配置 `distributed_hbm` 时原 legacy 行为保持不变；已启用 distributed 模式时不存在绕过新路径的行为级/特殊仿真模式。
