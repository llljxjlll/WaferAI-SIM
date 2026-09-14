# exp1-1：GEMM + Collective 两层编排实验

本目录评估 AG+GEMM 和 GEMM+RS 在 4 种 die mesh、6 类模型、Attention/MLP
层与两档序列长度下的 inter-die 与 intra-die 编排收益。DeepSeek-V3 不包含
Attention case，共 176 个逻辑 case。

## 文档入口

- 实验原始定义：[plans/plan.md](plans/plan.md)
- 当前开发方案：[plans/development_plan.md](plans/development_plan.md)
- 开发与结果报告：[reports/development_report.md](reports/development_report.md)
- 调试经验与排障手册：[reports/debugging_guide.md](reports/debugging_guide.md)

## 当前结论边界

当前 `results/results.csv` 的 176 个结果全部来自
`analytical_resource_replay_fallback`。它们是与架构行为对齐的两层解析/资源回放估算，
不是 176 个完整模型的端到端周期精确仿真。代码支持用 Q=8 周期精确 trace 摘要替换
fallback，但目前没有与 8 个 mesh/operator 签名一一匹配的当前硬件校准文件。

HBM 流量与 exp1-2 统一使用 `fused_boundary_tile_replay_v1`，按融合边界与 M/N tile
重放次数计算，并在结果中输出 `hbm_bytes_per_die`、`hbm_cycles` 和模型标识。

## 一键复现

从仓库根目录运行：

```bash
python3 -B exps/exp1/exp1_1/estimate_trace_replay.py
python3 -B exps/exp1/exp1_1/result_adapter.py
python3 -B exps/exp1/exp1_1/plot_results.py
python3 -B exps/exp1/exp1_1/plot_dual_axis.py
PYTHONPATH=/workspace/exps/exp1/exp1_1:/workspace \
  python3 -B -m unittest discover -s exps/exp1/exp1_1/tests -v
```

主要产物：

- `results/results.csv`、`results/results.json`：规范化后的 176 个结果；
- `results/trace_replay_results.*`：估算器原始输出；
- `results/dual_axis_data.json`：双轴图绘图数据；
- `figures/`：16 张基础/指标图；
- `figures/dual_axis/`：8 张最终双轴图。
