# V5 mixed D2D / on-chip NoC congestion experiment

该目录提供一个独立、可重复运行的周期精确实验，验证 V5 2-way striping
跨 die 流量与片内流量共享源 die NoC 链路时产生的拥塞。

## 对照场景

两个场景均使用 2×1 dies（每 die 4×4 cores）、`bounded_saf`、两个相同
GEMM 和 32-packet 跨 die DATA flow。跨 die flow 拆成两个 16-packet
subflow，分别使用两条独立 C2C link；本地 flow 固定为 `core5→core7`。

```text
shared:
  local       5 -> 6 -> 7
  stripe 0  4 -> 5 -> 6 -> 7 -> 3  -> D2D(row 0) -> 16 -> 20
  stripe 1  4 -> 5 -> 6 -> 7       -> D2D(row 1) -> 20

disjoint:
  local       5 -> 6 -> 7
  stripe 0  8 -> 9 -> 10 -> 11 -> 15 -> D2D(row 3) -> 28 -> 24
  stripe 1  8 -> 9 -> 10 -> 11      -> D2D(row 2) -> 24
```

两场景的 GEMM、总包数、stripe 配额、源/目的 mesh packet-hop、D2D 参数和
HOST 布局完全相同；只改变跨 die 路径是否与本地 flow 共享 `5→6`、`6→7`。

## 运行

先构建 `build/npusim`，再从仓库根目录运行：

```bash
python3 llm/test/d2d_link/mixed_noc_congestion_v5/run_experiment.py
```

程序会让 shared/disjoint 各运行两次，检查：

- 确实使用 `use_beha_noc=false` 的周期精确 NoC；
- V5 两个 subflow 分别固定到两条 link，各传输 16 个 DATA 包；
- 每 subflow 包数、顺序 hash、payload checksum、序号和 tail 完整；
- 两次运行的全部已解析指标一致；
- disjoint 的源 die NoC stall 为 0，shared 大于 0；
- 两场景成功发送数、片内 flow 完成周期、D2D backpressure 与活动总量相同；
- SAF、credit、router 和 link 状态全部排空；
- 理想链路容量模型预测 32-cycle 拥塞增量，周期仿真偏差不超过 4 cycle。

测试成功后自动更新
[`mixed_noc_congestion_v5_report.md`](mixed_noc_congestion_v5_report.md)；任何契约
失败都会返回非零状态。本实验是 V5 冻结后的独立性能验证，不修改冻结 tag，
也不自动纳入全版本 correctness gate。

## 文件

- `hardware/{shared,disjoint}.json`：两条 C2C 端口的共享/不相交放置；
- `workload/{shared,disjoint}.json`：等价双 GEMM 编排；
- `run_experiment.py`：输入审计、重复仿真、完整性检查、理论模型和报告生成；
- `mixed_noc_congestion_v5_report.md`：当前实测报告。
