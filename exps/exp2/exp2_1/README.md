# exp2-1：整模型训练与 P/D 推理 overlap 实验

本目录实现并执行了 `plans/development_plan.md`。最终发布状态是
`analytical_only_target_binding_unclosed`：现有周期精确 workload 被复用为结构与生命周期
锚点，但目标 6×6 配置的 D2D 单 lane 无法表达 1 TB/s，因此不能发布目标硬件绝对周期。

## 复现

在本目录执行：

```bash
python3 hardware_binding.py
python3 run_experiment.py
python3 build_operator_amdahl.py
python3 build_prefill_operator_amdahl.py
python3 plot_results.py
python3 -m unittest discover -s tests -v
```

`plot_results.py` 也会自动重建训练和 prefill 的算子 Amdahl 对照文件。最后一次验收为
59/59 tests PASS。
主运行生成：

- 12 个训练 case：6 models × S={2304,36864}
- 12 个训练算子均值/Amdahl 对照 case
- 12 个 prefill 算子均值/Amdahl 对照 case：6 models × S={2304,36864}
- 12 个 two-token local decode case：6 models × B={64,512}
- 12 个请求级联合 case：S=2304、G=512、6 models × B={64,512}
- 12 个 prefill/PD TTFT breakdown：6 models × S={2304,36864}
- 24 个 MoE routing-skew case：2 models × 2 workloads × 2 conditions × 3 λ
- 8 个 DeepSeek routed-only / routed+shared expert sensitivity case

## 结果入口

- `results/training_e2e.{json,csv}`
- `results/training_operator_amdahl.{json,csv}`
- `results/prefill_operator_amdahl.{json,csv}`
- `results/inference_request_e2e.{json,csv}`
- `results/inference_decode_e2e.{json,csv}`
- `results/inference_prefill_pd_breakdown.{json,csv}`
- `results/moe_skew_sensitivity.{json,csv}`
- `results/deepseek_shared_expert_sensitivity.{json,csv}`
- `results/capacity_audit.json`
- `results/calibration_summary.json`
- `figures/training_e2e.svg`
- `figures/inference_request_e2e.svg`
- `figures/inference_decode_e2e.svg`
- `figures/inference_prefill_e2e.svg`
- `figures/inference_prefill_pd_breakdown.svg`

## 证据分层

周期精确复用证据包括 Stage4 PDS 双跑 6187/6187 cycles、Stage3 prefill/decode
双跑观察，以及 exp1-2 Flexible MoE 与 GroupGEMM smoke。每条解析结果保存这些证据的
SHA-256，但它们只证明 action/lifecycle、ProgramIO、drain、字节守恒和可重复性，不能校准
目标硬件绝对周期。

解析回放显式建模 HBM ingress、DTE、directional D2D、local NoC、SRAM、tensor/vector、
reducer 与 control 资源。训练输出 base、forward-only 和 full-train 三种状态，三态逐
action 的算子、工作量、字节数、route、资源和 duration 相同，只改变 dependency edge。
forward-only 只放宽前向；full-train 还允许 backward dX、WGRAD 与逐层 gradient
collective 合法重叠，但每层 sync 仍等待本层全部 WGRAD，AdamW 仍等待全部梯度同步。
12/12 个主训练 case 的 full-train forward、backward、WGRAD 均严格短于 base。
DeepSeek MLA 始终标记 `analytical_only_mla`。

训练图另叠加 exp1 算子平均加速折线，用于观察 Amdahl 稀释：

- Dense：exp1-1 的 Attention/MLP × AG_GEMM/GEMM_RS，共 4 项；
- Mixtral：上述 4 项加 exp1-2 的 DISPATCH_GEMM/GEMM_COMBINE，共 6 项；
- DeepSeek：exp1-1 只有 routed-expert MLP × AG_GEMM/GEMM_RS，再加 exp1-2 两项，共 4 项；
  MLA Attention 因没有 exp1-1 anchor 而显式标记缺失。

exp1-1 固定取与训练相同的 `3x3` TP mesh；其可用序列档只有 2048/32768，故分别作为
2304/36864 的同 mesh 邻近代理（目标/来源均为 1.125）。exp1-2 取 H128、noncompact、
精确 2304/36864，作为四 rank EP noncompact 代理。每个 case 的折线值是所选项的未加权
算术平均，范围 2.3360×–4.1334×，系统 full-train 为 1.0158×–1.0869×。该均值只做
描述性单算子/系统对照，不是按时间占比加权的 Amdahl 预测；TP 子算子与 EP 复合算子
可能存在嵌套，也不应把两者差值直接解释为可优化比例。

Prefill 主图采用纯 prefill 延迟而非 TTFT：柱为 `prefill_cycles_base/overlap`，系统折线为
`prefill_cycles_base/prefill_cycles_overlap`，原来的 P/D TTFT breakdown 继续单独保留。
Prefill 的 Dense/Mixtral/DeepSeek 算子集合与实际 DAG 对齐：Dense 取 attention/MLP 的
AG_GEMM/GEMM_RS，Mixtral 再加入 DISPATCH_GEMM/GEMM_COMBINE，DeepSeek 只取有来源锚点的
routed-MLP 与 MoE 两项，MLA attention 缺失并附限制标签。exp1-1 使用精确 2x3 mesh、
以 2048/32768 分别代理目标 2304/36864；exp1-2 使用 H128、compact、精确
S={2304,36864}，作为连续 2x3 prefill instance 的四 rank 拓扑代理。12 个 case 的算子
未加权均值为 2.7101×–4.0476×，系统 prefill speedup 为 1.0290×–1.4339×；两者只用于
描述性 Amdahl 对照。请求级联合结果仍固定使用 S=2304，不受新增长序列点影响。

Decode 采用两步自回归展开，以每个 D instance 自身的 commit-to-commit 间隔估计固定 KV
上下文附近的 local TPOT；它没有冒充已验证收敛的严格稳态。B64/B512 分别按
`min(8,ceil(B/128))` 使用 1/4 个窗口，tensor service 按每个窗口的小 M 计算。KV-cache
read 只进入 HBM stream，KV append 显式写回 HBM，reducer 仅处理 activation；collective
必须等待本窗口 HBM 数据 ready。通信包含 2-cycle DTE launch 和 1 cycle/hop，HBM 包含
10-cycle first-byte；这些目标配置参数均带 unvalidated tag。修正后 B64 六个模型均为
1.0000×，B512 为 1.0003×–1.0504×。

请求级联合评价显式冻结输出长度 G=512，并按
`T_request = TTFT + 512 × local_TPOT` 从上述两份分项结果派生。联合 speedup 为
1.0000×–1.0504×；原始 decode 与 prefill/PD 文件和图仍独立保留。联合项把固定
KV=36864 上下文附近的 local TPOT 常数外推 512 次，不重放生成过程中逐 token 的 KV 增长，
因此带有专门 limitation tag，不能替代分项结果。

容量门禁是结果状态而非图注。主训练和主 decode 的 24 个 case 全部标为
`capacity_infeasible_projection`；prefill-only 的 12 个长短序列 case 中有 3 个通过
64 GiB resident-state 审计：LLaMA-2-7B 短序列和 LLaMA-3-8B 长短序列。

详细方法、结果与已知限制见 `reports/`。
