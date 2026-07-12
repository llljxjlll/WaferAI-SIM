# test_noc_congestion — 评估 WaferAI-SIM 的 NoC 拥塞建模

在 **4×4 (=16 核)** 核阵列上跑同一个**分布式 GEMM**，用两种片上通信模式（无拥塞 /
严重拥塞），并分别在**行为级 NoC** 与**周期精确 NoC** 下运行，量化仿真器对片上网络
拥塞的建模能力。

## 目录结构

```
llm/test/noc_congestion/
├── hardware/core_4x4.json          # 4x4 核阵列，noc_payload_per_cycle=4（两场景相同）
├── sim/sim_cycle.json              # use_beha_noc=false —— 周期精确 router（建模拥塞）
├── sim/sim_beha.json               # use_beha_noc=true  —— 行为级 roofline NoC（对照）
├── mapping/identity.spec           # 恒等映射（逻辑核=物理核）
├── gen_workloads.py                # 生成下面两份 workload（可改规模/匹配）
├── workload/gemm_no_congestion.json    # 无拥塞：奇->偶近邻匹配（链路不相交）
├── workload/gemm_congestion.json       # 严重拥塞：奇->偶远端匹配 15-i（共享中心链路）
├── run_test_noc_congestion.py      # 运行器：4 次运行(2 场景 × 2 NoC 模型) + 汇总
└── noc_congestion_summary.txt      # 运行器输出的对照表（运行后生成）
```

## 如何运行

需要已编译的 `npusim`（仓库根目录 `build/npusim`）。若尚未编译：

```bash
cd build
cmake -DBUILD_DEBUG_TARGETS=OFF -DL1CACHESIZE=4194304 -DL2CACHESIZE=15099494 ..
make -j
```

一键跑全部 4 组并打印对照表（可在任意目录执行，脚本会自动切到 `build/` 运行）：

```bash
python3 llm/test/noc_congestion/run_test_noc_congestion.py
```

单独手动运行某一组（必须在 `build/` 下，配置里用了相对 `build/` 的 `../font`、`../DRAMSys`）：

```bash
cd build
./npusim \
  --workload-config  ../llm/test/noc_congestion/workload/gemm_congestion.json \
  --hardware-config  ../llm/test/noc_congestion/hardware/core_4x4.json \
  --simulation-config ../llm/test/noc_congestion/sim/sim_cycle.json \
  --mapping-config   ../llm/test/noc_congestion/mapping/identity.spec
```

结束时日志会打印 `All requests finished ... <N> ns` / `Catch test finished`，
其中 `<N> ns` 即总仿真时间（= 总周期数 × 周期，`CYCLE=2 ns`），是我们对比的指标。

## 两种通信场景的设计

两份 workload **算力与每核收发消息数完全一致**，差异**只在物理路由是否共享链路**，
从而把延迟差异**纯粹**落在 NoC 链路争用上：

* 16 个核各做一次 `Matmul_f` 分片（同样的 `B/T/C/OC`）。
* 通信是一个 **二分匹配**：8 个奇数核各把结果发给 1 个偶数核，每个偶数核恰好收 1 条。
  每个接收核的负载都是 1 —— **没有"多发一"的接收端串行化**，两场景在这点上完全对称。

| 场景 | 匹配 | 物理效果 |
| --- | --- | --- |
| `no_congestion` | 奇 `i` → 偶 `i-1` | 同行相邻、每条流 1 跳、链路互不相交 → 无拥塞 |
| `congestion` | 奇 `i` → 偶 `15-i` | 关于网格中心点对称的最长对角流，XY 路由下 8 条流大量共享中心链路 → 严重争用 |

> **为什么用二分结构而不是任意置换？** 核间传输是 `REQ→ACK→DATA` 握手，发送会阻塞
> 等待接收方回 `ACK`。如果让每个核都"先发再收"（如反射置换这种对合），会形成环形等待
> 而**死锁**（已实测）。让奇数核只发、偶数核只收（偶数核先 post 接收即可回 ACK），
> 既避免死锁，又保证每核负载对称。

## 结果与结论

`noc_payload_per_cycle=4`，`B=1,T=256,C=64,OC=1024`（cast 负载 = `B*T*OC` 足够大，
使 NoC 主导延迟）。总仿真时间（ns，越小越快）：

| 场景 | 行为级 beha (roofline) | 周期精确 cycle | cycle − beha (≈ NoC 部分) |
| --- | --- | --- | --- |
| no_congestion | 14781 | 29109 | 14328 |
| congestion | 14833 | 45441 | 30608 |

**核心结论：**

1. **行为级 NoC 对拥塞是"盲"的**：无拥塞 14781 ns ≈ 严重拥塞 14833 ns（比值 **1.00×**）。
   行为级是 roofline 模型，延迟只取决于传输负载大小，与路由/距离/链路争用无关 ——
   因此两种通信模式在它眼里几乎没区别。
2. **周期精确 router 能建模拥塞**：严重拥塞 45441 ns vs 无拥塞 29109 ns（比值 **1.56×**）。
   周期精确 router 有 XY 维序路由 + 有限缓冲（`MAX_BUFFER_PACKET_SIZE`）+ 每条输出链路
   按 flow 的 `tag` 上锁串行化，中心链路上的多流争用被真实建模。
3. **隔离出的片上争用开销** = 拥塞场景比无拥塞场景多出的 `cycle−beha` ≈ **16280 ns**，
   这部分正是"链路拥塞"本身、且**只有周期精确 NoC 才能反映**。

> 一句话：**想评估/观测片上 NoC 拥塞，必须用 `use_beha_noc=false`（周期精确 router）；
> 默认的行为级 NoC 会完全忽略链路争用。** 这是本测例最重要的发现。

## 拥塞建模机制（源码定位）

* XY 维序路由：`llm/src/utils/router_utils.cpp:34-62`（`GetNextHop`，先 X 后 Y）
* 逐链路按 tag 上锁串行化（拥塞核心）：`llm/src/router/router.cpp:344-416`
* 有限缓冲背压：`MAX_BUFFER_PACKET_SIZE`，`llm/include/defs/spec.h`
* 行为级 roofline（无争用）：`llm/src/workercore/logic.cpp:58-61,147,563-564`
* 结束判定 / 总时间：`llm/src/monitor/config_helper_core.cpp:543-551`（`sc_stop()` 时的 `sc_time_stamp`）

## 进一步实验

* 改 `gen_workloads.py` 里的 `B/T/C/OC` 放大/缩小 cast 负载：负载越大，拥塞比值越大。
* 改 `hardware/core_4x4.json` 的 `noc_payload_per_cycle`（链路带宽）：越小，拥塞越严重。
* 改 `gen_workloads.py` 里的 `match_congestion` 尝试其它匹配（注意保持奇发偶收的二分结构以免死锁），
  观察不同流分布下中心链路争用的变化。
* 想看更细的拥塞证据：调高 `sim_cycle.json` 的 `log.log_level`，`grep NETWORK` 看 router
  上锁/停顿；或用 `streaming_trace_viewer/` 打开运行生成的 `build/events.json` 看 flow 停顿。
