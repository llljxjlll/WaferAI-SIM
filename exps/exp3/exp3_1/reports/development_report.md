# Exp3.1 开发与实测 GPU 数据验收报告

## 当前状态

Exp3.1 的 48 个逻辑实验点已使用实测 GPU YAML 正式运行：

- 48 个 logical cases；
- 每个 case 的 `W00/W11/C00/C10/G00/G10` 六状态，共 288 条状态记录；
- `native_full`、`native_inter_only`、`gpu_inter` 三组配对，共 144 条 speedup；
- 五个 GPU LUT 分组各 24 项，共 120 个精确 `(M,N,K)` key；
- 120 个 key 全部有实测 `p50` 延迟，单位为 `ns`，无 placeholder。

三条主指标固定为：
- GPU 为 `NVIDIA A100-SXM4-80GB`，BF16 输入/FP32 accumulation、p50（5 次 warmup、21 次测量）；clock policy 在输入数据中如实记录为 `not_recorded`。D2D 依赖与通信仍由本实验的 wafer 拓扑模型回放。

- `native_full = W00/W11`，单 die 16 核 canonical baseline 对 16 核 intra+inter 优化实现；
- `native_inter_only = C00/C10`，固定 16 核 canonical 实现和相同基础 HBM 工作量下，仅改变 inter-die 串行/流式编排；
- `gpu_inter = G00/G10`，两侧都使用同一 GPU LUT decomposition。


本次结果的顶层状态为 `measured_gpu_run`。GPU 本地 GEMM 延迟来自实测 LUT；native 轨道及跨 die D2D 依赖仍属于模型回放，不能误表述为真实多 GPU 端到端测量。

## 16-core 对齐与 HBM-neutral inter-only 重跑

- 结果 schema 升为 v6，并对 48 个 case 重新运行：288 条状态、144 个配对比较；Exp1 源 W-state 的逐周期 compatibility audit 仍通过，但不再把历史 W 值直接作为本轮输出。
- `native_full` 均值为 `1.4786×`（范围 `1.0286–1.9526×`）。Dispatch+GEMM 在 D=6/9/36 的均值为 `1.7250/1.7260/1.7313×`；GEMM+RS 为 `1.0865/1.1391/1.4640×`。
- `native_inter_only` 均值为 `1.0852×`（范围 `1.0043–1.3596×`）。Dispatch+GEMM 在 D=6/9/36 的均值为 `1.0214/1.0262/1.0523×`；GEMM+RS 为 `1.0590/1.0903/1.2619×`。
- `gpu_inter`（实测 GPU LUT replay）均值为 `1.1346×`（范围 `1.0011–1.5901×`）。Dispatch+GEMM 在 D=6/9/36 的均值为 `1.0140/1.0134/1.0118×`；GEMM+RS 为 `1.1602/1.2703/1.3379×`。每条 native state 输出 `inter_port_time_ns`、共同的 `hbm_time_ns`、端口/共享 NoC 链路负载与 fabric 时间；不再输出中间 HBM materialization 的节省字段。
- 结果图现输出 16 张逐 shape SVG：每个算子族的 8 个 shape 各自按 D=6/9/36 显示三条指标；另保留 4 张明确标为 `aggregate over 4 shapes` 的补充均值图。`figures/speedup_by_shape.csv` 保存全部 48 case × 3 comparison 的绘图源数据。

## 复用与证据边界

- Dense 的 `W00/W11` 均为 16 核：W00 使用 `4×4×1` canonical 映射，W11 使用 Exp1.1 派生的 adaptive 16-core intra schedule 与 inter streaming；两者都显式回放 16 core 经 4×4 X-first NoC 汇聚到两个 256 GB/s DTE 端口、跨 D2D fabric 后再分发到 core 的拥塞。独立的 `C00/C10` 仍固定 canonical 映射、相同端口/路由和相同基础 HBM 工作量，只允许 C10 对同一路径分段流水。
- Exp1.1 原始四状态的八个 Dense fixture 与 D=36、S=36864 的四个主矩阵重叠 case 仍逐周期审计，用于验证本轮复用的源公式；它们不再直接构成本轮 `native_full` 的数值输出。
- MoE 直接调用 Exp1.2 的 Mixtral/DeepSeek 真实 profile，以 `D=4 compact+H128+architecture` 为锚点；`D=6/9/36` 明确按 `compute×4/D`、`communication×sqrt(4/D)` 解析外推。
- Exp1 中已提交的周期精确小 motif/smoke 只用于约束调度结构和 setup 合理性。其硬件、shape 或 target unit 尚未闭合到本轮大模型主矩阵，因此 audit 明确保留 `structural prior only` / `target calibration pending` 边界。
- Mixtral 在 `D=9/36` 使用均衡等效 replay，结果标记 `production_expert_placement_closed=false`。

## GPU 配对规则

Dense 会分别回放 1D Ring 和 2D Row-Column 候选，并按 inter-on makespan 选择候选。`G00` 必须使用所选候选自己的 paired-off 结果；coarse GEMM 只作 coverage 诊断。MoE 同理使用 source-expert chunk 的 paired off/on。

因此 `G00/G10` 的 algorithm、lookup key、execution count 和 compute duration 完全相同，只改变 inter-die 依赖与重叠。该不变量由单元测试和 `audit.json` 双重检查。

## 使用实测 GPU YAML 重跑

将实测文件放到 `inputs/gpu_measurements.yaml`，保留模板中的 shape、BF16/FP32 accumulation 和 p50/ns 口径，然后运行：

```bash
python3 -B exps/exp3/exp3_1/gpu_lut.py validate \
  --required exps/exp3/exp3_1/generated/required_gpu_shapes.yaml \
  --measurements exps/exp3/exp3_1/inputs/gpu_measurements.yaml

python3 -B exps/exp3/exp3_1/run_experiment.py \
  --gpu-yaml exps/exp3/exp3_1/inputs/gpu_measurements.yaml \
  --output-dir exps/exp3/exp3_1/results

python3 -B exps/exp3/exp3_1/plot_results.py \
  --results exps/exp3/exp3_1/results/exp3_1_results.json \
  --output-dir exps/exp3/exp3_1/figures
```

生成单张“算子类型 → mesh → shape”总览折线图：

```bash
python3 -B exps/exp3/exp3_1/plot_overview.py \
  --results exps/exp3/exp3_1/results/exp3_1_results.json \
  --output exps/exp3/exp3_1/figures/speedup_overview_by_shape.svg
```

正式数据不得带 `data_status: placeholder_analytical`，也不要传 `--allow-placeholder`。缺 shape、重复 shape 冲突、非法 latency 或元信息不完整都会在实验运行前失败。

## 验收

标准库 `unittest` 共 27 项，覆盖 case/shape、LUT、历史对齐、same-decomposition replay、结果 schema、端到端生成和 byte-for-byte 确定性重跑，当前全部通过。
