# 分布式 HBM 多核端口与片上拥塞实验设计

日期：2026-08-04

## 1. 实验目标

本实验使用真实的 `CoreMemAdapter → RouterUnit → HBMNetwork → MemEndpointUnit → BehavioralHBMBackend` 路径，验证：

1. HBM stack/channel 能按物理边端口配置落到确定的 MEM attachment tile；
2. 多个核同时读写同一个 HBM channel 时，请求和响应确实产生片上 NoC 流量；
3. 多核流量在通往 MEM attachment 的共享 Router 输出发生拥塞，而不是每核访问私有 DRAM 副本；
4. 在拥塞条件下仍保持读写数据一致、请求计数完整并最终完全排空。

本实验重点验证单个 compute die 内部的 HBM 挂载和 NoC 汇聚拥塞。跨 die C2C/NUMA 正确性由 `--hbm-r4-selftest` 的 3×3 die 测试覆盖。

## 2. 假设

- H1：改变 HBM attachment 位置会改变 NoC hop 和访问延迟；
- H2：并发核数增加时，MEM/MEM Router 输出竞争、注入阻塞和访问延迟会上升；
- H3：达到共享路径或 HBM 瓶颈后，并发核数翻倍不会带来线性吞吐增长；
- H4：所有场景的读回数据、请求数和 residual 必须保持正确。

## 3. 实验拓扑

- compute die：1 个；
- 片上网络：4×4 mesh，共 16 个真实 `RouterUnit`；
- Router 周期：`CYCLE=2 ns`；
- flit：256 bit，MEM 数据有效载荷为 14 byte/flit；
- HBM：1 个 HBM2 stack、1 个 channel、2 个 pseudo-channel；
- backend：behavioral HBM；
- channel 原始传输带宽：16 GB/s；
- 每次访问基础延迟：20 ns；
- read/write 方向切换开销：4 ns；
- cache policy：`none`。

端口变量：

| 配置 | 物理端口 | attachment tile | 说明 |
|---|---:|---:|---|
| N0 | north edge index 0 | 12 | 靠近左侧请求核集合 |
| N3 | north edge index 3 | 15 | 远离左侧请求核集合 |

attachment tile 由生产函数 `BuildMemAttach()` 生成，并经过 `ValidateMemAttach()` 校验，不在实验驱动中直接指定。

## 4. 负载

并发核从以下固定列表取前 N 个：

`[0, 4, 8, 1, 5, 9, 2, 6]`

并发度：

`N = 1, 2, 4, 8`

每个核执行 8 组操作，每组为：

1. 向该核的私有物理地址区域写入 1024 byte 确定性 payload；
2. 从相同地址读取 1024 byte；
3. 比较读回 payload 与写入 payload。

因此每核产生：

- 8 次 write；
- 8 次 read；
- 16 个逻辑 memory transaction；
- 16 KiB 应用有效访问量。

不同核使用不重叠地址，但全部地址映射到同一个 stack/channel，从而隔离“共享通路和共享 HBM channel 竞争”，避免数据竞争影响结果。

## 5. 实验矩阵

| attachment | 并发核数 | 每核 write/read 对数 | 单次访问 |
|---|---:|---:|---:|
| N0 | 1/2/4/8 | 8 | 1024 B |
| N3 | 1/2/4/8 | 8 | 1024 B |

总计 8 个独立 SystemC 进程。每个场景重新 elaboration，避免全局 topology 和统计状态污染。

## 6. 采集指标

- `mem_tile`：实际 MEM attachment tile；
- `logical_requests/reads/writes`：逻辑请求完整性；
- `elapsed_ns`：全部并发核完成时间；
- `useful_GBps`：`request_bytes / elapsed_ns`，其中读和写各计一次应用有效字节；
- mean/P95/max latency；
- `noc_hops`；
- request/response flit；
- `injection_stalls`：源 Router CENTER 队列满导致的注入等待次数；
- `mem_mem_noc_contention`：两个 MEM flit 竞争同一 Router data 输出的次数；
- endpoint queue stall/wait；
- backend service time；
- network/router residual；
- data errors。

`mem_mem_noc_contention` 是本实验新增的直接观测量；它与 HBM/普通或 collective 流量的 `shared_noc_contention` 分开统计。

## 7. 通过条件

每个场景必须同时满足：

1. attachment tile 与端口规则一致；
2. `logical_requests = cores × 8 × 2`；
3. read/write 各占一半；
4. endpoint/backend completed 数与逻辑请求一致；
5. `data_errors = 0`；
6. network residual 和全部 Router residual 均为 0；
7. 多核场景出现非零 `mem_mem_noc_contention`；
8. 增加并发核后出现延迟增长和吞吐次线性增长。

## 8. 复现方法

先构建：

```bash
/opt/cmake/bin/cmake --build build -j2
```

运行完整矩阵：

```bash
llm/test/hbm/run_hbm_contention_experiment.sh > hbm_contention.csv
```

也可以运行单个场景：

```bash
cd build
./npusim --hbm-contention-experiment \
  --hbm-experiment-port=N0 \
  --hbm-experiment-cores=8 \
  --hbm-experiment-pairs=8 \
  --hbm-experiment-bytes=1024
```

原始结果保存在同目录的 `HBM多核端口拥塞实验数据.csv`。
