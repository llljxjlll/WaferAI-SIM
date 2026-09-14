# exp1-2：MoE Dispatch/Combine + GEMM 双算力实验

本目录实现并运行 plans/development_plan.md 中的探索性实验。结果覆盖单 die
128 TFLOP/s（H128）与 2000 TFLOP/s（H2000），采用少量当前仿真器周期精确烟测验证
执行链路，再用分 expert 的解析物理资源回放扩展到完整矩阵。

当前结果可用于设计空间筛选，不能作为目标硬件的正式周期精确结论。原因是现有仿真器
不能闭合“6×6 wafer、四个边缘 HBM stack、每条 D2D 1 TB/s”的目标绑定，且已有 tiny
fixture 是 H16/I32/top-1，并非目标 top-2/top-8。所有结果均明确保存
simulator_unit_closure=false、calibration_status、source 和 digest。

## 已生成结果

- results/h128/architecture/：H128、四 stack architecture，16 cases；
- results/h2000/architecture/：H2000、四 stack architecture，16 cases；
- results/h128/hbm_free_compute_comm/：H128 计算/通信消融，16 cases；
- results/h2000/hbm_free_compute_comm/：H2000 计算/通信消融，16 cases；
- results/loaded_groups/{h128,h2000}/architecture/：9 个显式 EP group 的
  DeepSeek-V3 长序列敏感性，共 8 cases；
- figures/<profile>/<mode>/：四组主图，每组各含 Dispatch 和 Combine 两张 SVG；
- calibration/：当前 build 的 Flexible MoE 与 GroupGEMM 周期精确烟测证据。

主矩阵为 32 个 architecture cases，另有 32 个完全配对的 HBM-free 消融和 8 个
loaded-groups 敏感性 cases，总计 72 条解析结果。

## 复现

从仓库根目录执行：

~~~bash
python3 -B exps/exp1/exp1_2/run_experiment.py \
  --profile all \
  --memory-mode all \
  --network-scenario all
~~~

生成四组图：

~~~bash
python3 -B exps/exp1/exp1_2/plot_results.py \
  --input exps/exp1/exp1_2/results/h128/architecture/results.csv
python3 -B exps/exp1/exp1_2/plot_results.py \
  --input exps/exp1/exp1_2/results/h128/hbm_free_compute_comm/results.csv
python3 -B exps/exp1/exp1_2/plot_results.py \
  --input exps/exp1/exp1_2/results/h2000/architecture/results.csv
python3 -B exps/exp1/exp1_2/plot_results.py \
  --input exps/exp1/exp1_2/results/h2000/hbm_free_compute_comm/results.csv
~~~

运行回归：

~~~bash
PYTHONPATH=/workspace/exps/exp1/exp1_2:/workspace \
  python3 -B -m unittest discover \
  -s exps/exp1/exp1_2/tests -v
~~~

## 数据口径

- S 是整个 EP=4 group 的 token 数；
- assignment 以整数矩阵 A[source_rank, expert] 保存；
- padding 在每个 expert 内独立执行；
- Dispatch logical FLOPs 包含 gate/up 两个 GEMM，系数为 4；
- Combine 只包含 down GEMM，系数为 2；
- local assignment 不进入 D2D；
- D2D、HBM route 与 local NoC 均按有向链路累计，local NoC 遍历完整 Tk；
- HBM 使用位于 (1,0)、(4,0)、(1,5)、(4,5) 的四个 16 GiB stack，地址空间从
  0/16/32/48 GiB 开始且互不重叠；
- optimized_tflops 是 focus EP group 吞吐；loaded 模式另保存
  scenario_optimized_tflops，表示 9 个 group 的总吞吐；
- HBM-free 仅是计算/通信消融，不是可部署硬件。

## 文档

- 原始要求：plans/plan.md
- 开发与验收方案：plans/development_plan.md
- 实现与验证记录：reports/development_report.md
- 实验结果与边界：reports/experiment_report.md
- 调试经验：reports/debugging_notes.md
