# NPU ISA v1 性能报告

> 文档状态：**FINAL**
>
> 口径：本文报告模拟器模型时间、ops、队列/链路/存储等待与 backend 工作量；runner wall-clock 不是硬件性能 SLA。功能可用性也不承诺任一加速档在所有负载上快于 baseline。

## 1. 候选与环境

| 字段 | 值 |
|---|---|
| 基线 commit | `139ef73a803dc3637b28000bbc3a34016b2d5e45` + 当前 P0～P8 工作树 |
| build | fresh Release，GCC 11.4.0，CMake 3.31.3 |
| SystemC | 2.3.3；sanitizer 使用 pthread coroutine |
| hardware | P8-B `9993c0e5adc66740c97d498be6f5a71243e0e922720e87fcc1d828ada429ab3e` |
| simulation | `32bce414f159454eff62612d46ffb2d9b85d75fad8e31973af94329bb7b5ce3c` |
| mapping | `99a357b646bc6d0d81ac188c8bfffcbf6ab8f8f72a5d262fe81624f6f9a9a66c` |
| 重复 | correctness repeat3；release stability repeat20；额外 raw timing repeat1 |

同一横向比较只改变 profile，artifact、hardware、mapping、payload、group、root 与 seed 均相同。

## 2. 冻结的计算与存储语义

ISA v1 保留既有成本公式：`RELU` 计 EXU，`MAXPOOL` 计 SFU，`MERGE_MATMUL` 计 `B*T*C` EXU，`RMSNORM` 保留既有行为，`DUMMY` 固定 `exu_ops=10`。compute 周期按 EXU/SFU/VEC 最大值及 DRAM overlap 规则只结算一次。

代表 compute oracle：

| Opcode | `(exu_ops,sfu_ops,vec_ops,compute_cycle_ns)` | final 状态 |
|---|---|---|
| MATMUL | `(0,0,32768,512)` | Release/Debug diff gate PASS |
| ATTENTION | `(4096,64,128,5)` | PASS |
| LAYERNORM | `(0,4,2060,32)` | PASS |
| GELU | `(0,256,1024,16)` | PASS |

P8-A 的 trace 门证明：public LSU_LOAD 完成后才开始首个 MATMUL；两次 DTE 传输均与对应 compute 窗口重叠；token 81/82 各 issue/wait 一次。其目标是验证调度关系，不将该小样本的总 ns 作为跨机器性能 golden。

## 3. P8-B 四档 production 比较

额外 Release raw run 保留了每个 profile 的完整 stdout。表中 total ns 取日志最后一个模拟时间；repeat3/repeat20 的 normalized digest 与 raw run 一致。

| profile | broadcast | reduce | total ns | 相对 baseline | multicast TX | DCA result | tree-batch markers | raw log SHA-256 |
|---|---|---|---:|---:|---:|---:|---:|---|
| baseline | unicast | endpoint | 1978 | 1.000x | 0 | 0 | 0 | `7d5bba326f048d5ece20e0f94e5f3dc27ffbf28b06f8d64c2a7610cb21f6d87b` |
| broadcast_only | multicast | endpoint | 1280 | 1.545x | 8 | 0 | 16 | `f3ab64faf1d44c5f6750df76a5330bf68ab970e64bc90979dce74c09cbcf35ca` |
| reduce_only | unicast | DCA | 1942 | 1.019x | 0 | 4 | 8 | `d4066692825bdeac6d0f8de5f365f97bee51446b388d2ff49ee9533d3951f1c1` |
| reduce_broadcast | multicast | DCA | 1828 | 1.082x | 8 | 4 | 16 | `2eb0665fcad567fb255f373594286fc9b1ab431fe54ecf4b7db205e5d8f47088` |

解释：

- broadcast_only 减少 endpoint normal flow，并由四棵 multicast tree 分批执行，因此该样本改善最大。
- reduce_only 将 AllReduce reduce phase 真实 offload 到 DCA，但仍保留 AllGather unicast；小样本的启动/同步成本使改善有限。
- reduce_broadcast 同时使用 multicast 与 DCA；组合路径严格执行 DCA→multicast 依赖，因而不是两个单项加速的简单乘积。
- 四档均 byte/sentinel/order/SUM 正确、无 fallback、全 residual=0；上述差异没有正确性影响。

稳定性 digest：

| profile | repeat3 | repeat20 | digest |
|---|---:|---:|---|
| baseline | PASS | PASS | `c20a0d033dcf037c51e1f117f8ad2314d993fff7fe820c35d5ecd34803661843` |
| broadcast_only | PASS | PASS | `e04b620a13a6c69ed33ab585842cc86e447677f604d141679dd038ca13f4c39f` |
| reduce_only | PASS | PASS | `fdadebe1de80785066a8a04e0287ee6ea98dd6ad3451196916c6151b465eb295` |
| reduce_broadcast | PASS | PASS | `f067f45becf35be858ded5fc1e919fd87c55f6200b0fe6a4703e71cd1bbf64e3` |

repeat20 共 80 个隔离进程，日志 SHA `beadafe416e299ca53dbf4af5e6f72295bf7ddfd0c27517bd08590949d1ef96c`。

## 4. NoC profile 曲线

R8 的 behavioral compatibility 曲线使用 2×2、Broadcast + UINT8/SUM AllReduce。DCA production 要求真实 SRAM，因此 behavioral R8 对 reduce 档验证的是精确 fail-fast；DCA 正向性能/正确性由 P7/P8 program gates 提供。

| payload | profile | time ns | normal hops | collective hops | speedup |
|---:|---|---:|---:|---:|---:|
| 640 B | baseline | 4142 | 480 | 0 | 1.000x |
| 640 B | broadcast_only | 3162 | 320 | 120 | 1.310x |
| 1024 B | baseline | 6146 | 768 | 0 | 1.000x |
| 1024 B | broadcast_only | 4638 | 512 | 192 | 1.325x |
| 8192 B | baseline | 43554 | 6144 | 0 | 1.000x |
| 8192 B | broadcast_only | 32190 | 4096 | 1536 | 1.353x |
| 32768 B | baseline | 171810 | 24576 | 0 | 1.000x |
| 32768 B | broadcast_only | 126654 | 16384 | 6144 | 1.357x |

四个 payload 均把 normal hops 减少 33.33%；payload 增大后固定启动成本摊薄，speedup 从 1.310x 上升至 1.357x。R8 完整结果为 16/16 PASS。

## 5. NoC 拥塞回归

| 场景 | behavioral ns | cycle ns |
|---|---:|---:|
| no congestion | 14777 | 37295 |
| congestion | 14829 | 61763 |

behavioral 比值约 1.00x；cycle 模型比值 1.66x，isolated contention 增量 24416 ns。控制输出显式 low-cycle 修复改变了旧绝对时间，但数据、credit、drain 和拥塞定性均保持；golden 已按同一候选的结构性 oracle 更新。

## 6. P7 32 KiB 与资源压力

`program_p7_accelerated_32k_runtime_matrix` 在 fresh Release/Debug 通过 8/8，并在 sanitizer 下再次通过 8/8：

- AllGather 与 AllReduce，各四档，N=4、32 KiB。
- `max_trees_per_batch=1`，每个 N=4 tree plan 被确定性拆为四批。
- begin/end 一一配对，programmed=erased，end `occupancy_after=0`，per-router capacity=64 从未超限。
- multicast commit 与 DCA completion exactly once，session/tag 不复用，flit/credit/tree/DCA/Worker residual 全 0。
- production DcaComputePool 另有 32 CORE + 32 DCA、L=5、II=2、result depth=1 的持续 RR 压力门：严格交替、同源 service gap≤4、64 completion exactly once、最终 drain=0。

精确 per-router peak 是正确性上界观测，不冻结为跨拓扑性能 golden；更换 topology 或 K 时必须重新生成 profile 报告。

## 7. 唯一计费对账

| 资源/事件 | owner | final 对账 |
|---|---|---|
| source memory read | SRAM/HBM media | probe bytes = media read bytes，PASS |
| endpoint wire | Router/NoC/D2D | DATA fragments ×16/tail = payload bytes，PASS |
| destination write | SRAM `kNocRx` | committed bytes = oracle bytes，PASS |
| compute | EXU/SFU/VEC | opcode formula = unit ops；cycle 结算一次，PASS |
| multicast | Router tree | 每 target exactly once，无 endpoint duplicate，PASS |
| DCA | stream/compute pool | issue=completion=expected chunks，tag 不复用，PASS |
| barrier | sync runtime | phase/seq 释放一次，最终 state=0，PASS |

## 8. 结论与限制

性能证据支持四档 backend 的 production 可用性，并显示此代表程序与 R8 广播曲线上的改善；它不是硬件性能承诺。ReduceScatter+DCA、跨 die collective 和未列出的更广 N/dtype/chunk 组合继续按 [ISA v1 已知限制](ISA_v1已知限制.md) fail-fast 或需要独立测量。最终结论：**PASS**。
