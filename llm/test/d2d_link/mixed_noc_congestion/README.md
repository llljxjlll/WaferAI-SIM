# Mixed D2D / on-chip NoC congestion experiment

这个目录提供一个独立、可重复运行的周期精确对照实验，用于验证跨 die
流量与片上流量共享 NoC 链路时产生的拥塞。它补充 V3 回归门中的动态
mixed-traffic 用例：这里固定保存输入配置，并自动生成可交付的结果报告。

## 对照设计

两个场景均使用 2×1 dies（每 die 4×4 cores）、V3 `bounded_saf`、相同的
两个 GEMM 和 32-packet DATA flow。本地流始终为 `core5 -> core7`。

```text
shared:   cross 4 -> 5 -> 6 -> 7 -> D2D -> 20
                         +---- local 5 -> 6 -> 7

disjoint: cross 8 -> 9 -> 10 -> 11 -> D2D -> 24
                         local 5 -> 6 -> 7
```

shared 的跨 die 流与本地流共享 die0 的 `5->6`、`6->7` 两条链路；
disjoint 的两条路径不相交。两者只改变跨 die flow 所在行与 C2C 端口行，
C2C hop 数、链路速率/延迟/容量及 HOST 距离保持一致。

## 运行

先确保 `build/npusim` 已构建，然后从仓库根目录执行：

```bash
python3 llm/test/d2d_link/mixed_noc_congestion/run_experiment.py
```

runner 会让两个场景各运行两次，并检查：

- 使用 `use_beha_noc=false` 的周期精确模式；
- 两次运行的全部已解析指标一致；
- disjoint 的 die0 NoC stall 为 0，shared 的 stall 大于 0；
- 两场景成功发送数、本地 flow 完成周期和 D2D DATA/ACK 数一致；
- 32 个 DATA 包无丢失、无重复/乱序，credit 平衡且全部状态排空；
- D2D source、端口和链路限速均未触发，排除 D2D 限速这一混杂因素；
- 总时间差等于跨 die flow 的 cycle 差乘以 2 ns/cycle。

成功时报告写到
[`mixed_noc_congestion_report.md`](mixed_noc_congestion_report.md)。runner 任一契约
不满足都会返回非零状态。

## 文件

- `hardware/shared.json`、`hardware/disjoint.json`：仅 C2C 端口行不同；
- `workload/shared.json`、`workload/disjoint.json`：同一 GEMM 的共享/不相交编排；
- `run_experiment.py`：运行、重复性检查、完整性检查与报告生成；
- `mixed_noc_congestion_report.md`：当前基线实测报告。
