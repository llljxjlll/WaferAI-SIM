# NoC 集合通信 V3 开发日志

日期：2026-07-26

## 交付

V3 完成 Tier0 归约族的通信、接收同步和显式计算闭环：

- Reduce：`N-1` 个非 root 源按 source-rank phase 单播到 root，root 对 expected bitmap 到齐后执行一次显式计算；
- ReduceScatter：固定 `REDUCE_ROOT_SCATTER`，先 root reduce，再按 quotient/remainder 分片散发；
- AllReduce：固定 `REDUCE_ROOT_BROADCAST`，先 root reduce，再以 Tier0 单播 baseline 广播结果；
- 每个通信 phase、root compute 后以及结果分发后均有全组 barrier，保持 V1 顺序 dispatcher 下的结构性无死锁；
- Reduce RX 状态按 `(CollectiveKey,self_rank)` 隔离，跟踪 expected/received bitmap，拒绝重复源和 root 网络 operand；
- 最后一个远端 operand 到齐后计一次 1-cycle alignment/endpoint completion，随后状态擦除；
- `REDUCE_ARRIVAL` marker kind 使用 `Collective_prim` wire bits `[47:40]`，反序列化做范围检查；
- root ALU 只由独立 `Reduce_compute_prim` 计费，通信 marker 不计 ALU；
- compute 计时为 `ceil(count*(N-1)/(vec_x*vec_cnt))` cycle，吞吐读取 root core 的实际硬件配置，并输出 elements、operations、cycles 和 lanes trace；
- 支持 UINT8/INT32/INT64 的 SUM/MAX；FP32、缺失 reduce_op、payload shape 不一致和地址未按 dtype 对齐均启动期拒绝；
- reduction 要求 `chunk_bits == count*dtype_bits`。

## 运行时分派与唯一计费边界

- `Collective_prim::taskCoreDefault` 对三种 marker 显式分派：`GATHER_ARRIVAL` 进入 reorder、`REDUCE_ARRIVAL` 进入 Reduce RX，只有 `BARRIER` 进入 barrier；wire 往返后不会把 Reduce arrival 静默降级为 barrier。
- Reduce RX marker 只在全部远端 operand 到齐时执行一次 `wait(CYCLE)`，表示 alignment/endpoint completion；该路径不计算 reduction operations。
- `Reduce_compute_prim` 不属于 WorkerCore 的 Send/Recv/Collective 特殊分支，而是进入通用 compute/task_logic；task_logic 只应用一次 primitive 返回的 delay。
- 因此一次归约的 ALU 时间只有一个来源：root 上唯一的 `Reduce_compute_prim`；通信 phase、RX marker 和结果分发均不重复计 ALU。
- compute 后的 `barrier(n)` 保证所有 rank 在结果就绪后才进入 ReduceScatter/AllReduce 的 phase `n+1` 分发。

## Schema 示例

```json
{
  "op": "allreduce",
  "collective_id": 300,
  "group": [0, 1, 2],
  "root": 0,
  "count": 10,
  "dtype": "int32",
  "reduce_op": "sum",
  "chunk_bits": 320,
  "stride_bits": 320,
  "src_addr": 4096,
  "dst_addr": 8192,
  "terminal": true
}
```

## 关键文件

- `llm/include/dte/coll_types.h`
- `llm/include/dte/coll_plan.h`
- `llm/include/dte/coll_runtime.h`
- `llm/src/dte/coll_runtime.cpp`
- `llm/src/dte/coll_v3_selftest.cpp`
- `llm/src/prims/norm_prims/collective_prim.cpp`
- `llm/src/prims/norm_prims/reduce_compute_prim.cpp`
- `llm/src/monitor/config_helper_core.cpp`
- `llm/test/noc_collective/oracle.py`
- `llm/test/noc_collective/run_test_coll_v3.py`

## 验收

- build：PASS。
- V3 planner/RX/wire/compute self-test：14/14。
- 独立 Python oracle：PASS，分别核对 flow、phase 和 compute cycles。
- V3 runner：19/19。
- 在 `vec_x=64, vec_cnt=1, count=100` 验收配置下，cycle/behavioral 完成值：Reduce 768/672 ns，ReduceScatter 1028/900 ns，AllReduce 1252/1060 ns。
- 三种整数 dtype × SUM/MAX：6/6 端到端通过。
- 负例：缺 reduce_op、FP32、payload shape、alignment、Gather depth on Reduce RX 全部通过。
- V2：8/8；V1：24/24 且冻结周期不变；V0：32/32。
- DTE V0：64/64；DTE V3b：21/21；DTE V4：19/19。
- NoC congestion：4/4，冻结值 14781/29109、14833/45441。
- D2D V0 runner：67/67 test groups（含 pure self-test 308/308）。

## 边界

- V3 是 timing/traffic/metadata 模型：校验 dtype/op、operand 来源、流量、同步和计算工作量，但现有 simulator 不搬运真实 tensor 数值，因此不宣称数值 SUM/MAX bit-accurate。
- V3 compute 使用普通 endpoint vector unit，不是 V5 DCA；V5 in-network reduce 使用独立的 `max(comp,p/128)+54` 契约。
- Tier0 继续使用 source-rank phase 串行算法，不宣称 ring/tree 的并行效率。
- FP32 在舍入、NaN、溢出和结合顺序冻结前保持拒绝。
