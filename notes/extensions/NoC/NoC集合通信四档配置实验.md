# NoC 集合通信四档配置实验

日期：2026-08-03

实验使用 2×2 mesh、group `[0,1,2,3]`、root 0，依次执行 Broadcast 和
UINT8/SUM AllReduce。物理 flit payload 为 128 bit，DCA vector 为 512 bit，`L=7`、
`II=1`。命令：

```bash
cd build
python3 ../llm/test/noc_collective/run_experiment_profiles.py
```

| payload | profile | time(ns) | normal hops | collective hops | B | DCA issues | fill | steady | vs baseline |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 640 B | baseline | 3466 | 480 | 0 | 10 | 0 | 0 | 0 | 1.000x |
| 640 B | broadcast_only | 2712 | 320 | 120 | 10 | 0 | 0 | 0 | 1.278x |
| 640 B | reduce_only | 2398 | 320 | 123 | 10 | 30 | 7 | 29 | 1.445x |
| 640 B | reduce_broadcast | 890 | 0 | 363 | 10 | 30 | 7 | 29 | 3.894x |
| 1 KiB | baseline | 5038 | 768 | 0 | 16 | 0 | 0 | 0 | 1.000x |
| 1 KiB | broadcast_only | 3900 | 512 | 192 | 16 | 0 | 0 | 0 | 1.292x |
| 1 KiB | reduce_only | 3454 | 512 | 195 | 16 | 48 | 7 | 47 | 1.459x |
| 1 KiB | reduce_broadcast | 1178 | 0 | 579 | 16 | 48 | 7 | 47 | 4.277x |
| 8 KiB | baseline | 34382 | 6144 | 0 | 128 | 0 | 0 | 0 | 1.000x |
| 8 KiB | broadcast_only | 26076 | 4096 | 1536 | 128 | 0 | 0 | 0 | 1.319x |
| 8 KiB | reduce_only | 23166 | 4096 | 1539 | 128 | 384 | 7 | 383 | 1.484x |
| 8 KiB | reduce_broadcast | 6554 | 0 | 4611 | 128 | 384 | 7 | 383 | 5.246x |
| 32 KiB | baseline | 134990 | 24576 | 0 | 512 | 0 | 0 | 0 | 1.000x |
| 32 KiB | broadcast_only | 102108 | 16384 | 6144 | 512 | 0 | 0 | 0 | 1.322x |
| 32 KiB | reduce_only | 90750 | 16384 | 6147 | 512 | 1536 | 7 | 1535 | 1.487x |
| 32 KiB | reduce_broadcast | 24986 | 0 | 18435 | 512 | 1536 | 7 | 1535 | 5.403x |

结论：新 DCA 的固定 latency 只出现在 pipeline fill，payload 增长后的斜率由 link
flits、tree 每 beat pairwise issue 和 `II` 决定。reduce_only 隔离了纯 reduce 收益；
reduce_broadcast 进一步消除 Broadcast/AllReduce-result 的重复 unicast。结果不表示所有
拓扑和拥塞场景都必然加速，但每个数据点的流量与 DCA 工作量均可由 oracle 解释。

## 负载与结果分析

### 1. 实验测量的不是单个 Reduce

每个 case 在相同的 2×2 NoC 上依次执行：

1. 一次 Broadcast；
2. 一次 UINT8/SUM AllReduce。

因此总时间同时包含 standalone Broadcast、AllReduce 的 Reduce 阶段和 AllReduce result
distribution。四档配置对这三个阶段的映射如下：

| profile | standalone Broadcast | AllReduce Reduce | AllReduce result |
|---|---|---|---|
| `baseline` | ordinary unicast | endpoint/Tier0 collection | ordinary unicast |
| `broadcast_only` | Router multicast | endpoint/Tier0 collection | ordinary unicast |
| `reduce_only` | ordinary unicast | stream DCA tree reduce | ordinary unicast |
| `reduce_broadcast` | Router multicast | stream DCA tree reduce | Router multicast |

`broadcast_only` 只加速 standalone Broadcast。由于它仍使用 endpoint AllReduce，不会为
AllReduce result 单独创建一个 orphan multicast tree。

### 2. 四档网络流量公式

令：

```text
F = ceil(payload_bits / 128)    # 每份 payload 的 physical data flits
B = ceil(payload_bits / 512)    # 每份 payload 的 DCA vector beats
```

root 0 到其他三个 core 的固定 XY 路径边数之和为 4：`0->1` 和 `0->2` 各 1 hop，
`0->3` 为 2 hops。multicast/reduce tree 覆盖四个 rank，需要 3 条 tree edges。

#### baseline

Broadcast、Reduce collection 和 result distribution 都使用普通 unicast：

```text
normal hops = 4F + 4F + 4F = 12F
collective hops = 0
```

例如 640 B 时 `F=40`，所以 normal hops 为 `12*40=480`。

#### broadcast_only

只有 standalone Broadcast 变为三边 multicast；AllReduce 的收集和结果分发仍是普通
unicast：

```text
normal hops = 4F + 4F = 8F
collective hops = 3F
```

640 B 时分别为 `320` 和 `120`。因此该档只消除一段重复发送，端到端加速稳定在
1.278～1.322x。

#### reduce_only

standalone Broadcast 和 AllReduce result 仍为 ordinary unicast：

```text
normal hops = 4F + 4F = 8F
```

Reduce stream 在每条 tree edge 上传输一个 header 和 `F` 个 data flits：

```text
collective hops = 3(F+1)
```

640 B 时分别为 `320` 和 `3*41=123`。除减少网络流量外，该档还把 source-phased endpoint
collection 替换为沿 tree 的流式向量归约，因此端到端加速为 1.445～1.487x。

#### reduce_broadcast

三个阶段全部使用相应的 tree backend：

```text
Broadcast multicast       = 3F
Reduce stream             = 3(F+1)
AllReduce result multicast = 3F

normal hops = 0
collective hops = 9F + 3
```

640 B 时 collective hops 为 `9*40+3=363`；32 KiB 时 `F=2048`，为
`9*2048+3=18435`，均与 production trace 精确一致。

### 3. DCA 计时为什么不会随 chunk 异常增长

UINT8 在 512-bit DCA 上有 64 个并行 lanes，所以：

```text
B = ceil(payload_bytes / 64)
```

四个 rank 的每个 vector beat 需要总计 `N-1=3` 次 pairwise SIMD issues。它们分布在 tree
的不同 Router 节点，而不是对每个 element 串行累加：

```text
DCA issues = 3B
```

| payload | vector beats B | DCA issues |
|---:|---:|---:|
| 640 B | 10 | 30 |
| 1 KiB | 16 | 48 |
| 8 KiB | 128 | 384 |
| 32 KiB | 512 | 1536 |

DCA 是 latency/II 分离的流水线。无额外资源争用时，固定 latency 只形成一次 pipeline
fill：

```text
T_dca_pipeline = L + (issues-1)*II
```

32 KiB 时为 `7+1535*1=1542`，而不是旧模型式的 `1536*7=10752`。因此 payload 增大后
性能斜率由 data flits、pairwise issue 数、II 和真实 CORE/DCA contention 决定，不再出现
“每个 chunk 重复支付固定 DCA latency”的伪影。

### 4. 为什么 combined 的提升明显更大

`reduce_broadcast` 不是把 broadcast_only 和 reduce_only 的独立加速比简单相加，而是同时
缩短三个相互依赖的关键路径：

- standalone Broadcast 从 root 对多个目标分别发送变为 single injection；
- Reduce 从 source-phased endpoint collection 变为 Router tree 上的流式聚合；
- AllReduce result 从 root-to-peer 多次发送变为 single-injection multicast；
- 网络传输、Router assembly 和 DCA pipeline 可以重叠；
- payload 增大后，barrier、header 和 pipeline fill 等固定项被进一步摊薄。

对应地，端到端时间下降和加速比为：

| payload | broadcast_only | reduce_only | reduce_broadcast |
|---:|---:|---:|---:|
| 640 B | 21.8% / 1.278x | 30.8% / 1.445x | 74.3% / 3.894x |
| 1 KiB | 22.6% / 1.292x | 31.4% / 1.459x | 76.6% / 4.277x |
| 8 KiB | 24.2% / 1.319x | 32.6% / 1.484x | 80.9% / 5.246x |
| 32 KiB | 24.4% / 1.322x | 32.8% / 1.487x | 81.5% / 5.403x |

大 payload 下 combined 的加速比持续增大，说明其稳态流量和串行阶段斜率显著低于 baseline；
同时 DCA fixed latency 没有随 beat 数重复累加。

### 5. 结果边界

- 表中的加速比属于“Broadcast + AllReduce”组合负载，不能直接当作单独 Broadcast 或单独
  Reduce 的性能。
- 实验使用 2×2、固定 XY、同 die、无 OpenSMART bypass；其他 topology 和拥塞模式需要重新
  计算 tree/link oracle。
- 总时间还包含 endpoint、barrier、配置和仲裁开销，不能只用总 flit-hop 比例线性换算。
- `reduce_broadcast` 在本实验中最快不表示它在所有短消息、CORE 饱和或单分支长期背压场景
  都必然最快；这些场景需要结合 `COLL_SHARED`、`COLL_DCA` 和 backpressure trace 分析。
