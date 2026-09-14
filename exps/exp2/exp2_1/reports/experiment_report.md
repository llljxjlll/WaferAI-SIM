# exp2-1 实验报告

## 结论摘要

本轮得到的是有周期精确结构证据约束的解析 E2E 投影，不是目标硬件周期精确结果。
发布状态为 `analytical_only_target_binding_unclosed`。

- forward-only 训练 speedup 范围 1.0048×–1.0270×，均值 1.0190×；
- full-train 训练 speedup 范围 1.0158×–1.0869×，均值 1.0604×；
- exp1 算子未加权平均 speedup 范围 2.3360×–4.1334×，均值 3.5769×，相对系统
  full-train speedup 的比值为 2.30–3.92；
- 12/12 个训练点的 full-train forward、backward、WGRAD 都严格快于 base；
- two-token local decode speedup 范围 1.0000×–1.0504×，均值 1.0055×；
- S=2304、G=512 联合请求 speedup 范围 1.0000×–1.0504×，均值 1.0056×；
- S={2304,36864} prefill/PD TTFT speedup 范围 1.0290×–1.4336×，均值 1.1982×；
- 12/12 训练和 12/12 主 decode case 均超过 64 GiB，不能解释为单 wafer 可运行结果；
- prefill-only 容量审计 3/12 可行：LLaMA-2-7B 的短序列与 LLaMA-3-8B 的长短序列。

所有秒数使用 500 MHz；decode 主指标是固定 KV 上下文附近的两 token local interval，
未验证严格稳态收敛。报告同时给出请求级联合指标，并继续独立报告 decode 与 TTFT。

## 训练主结果

| 模型 | S | forward-only | full-train | full step (s) | resident (GiB) |
|---|---:|---:|---:|---:|---:|
| LLaMA-2-7B | 2304 | 1.0270 | 1.0869 | 6.2552 | 404.41 |
| LLaMA-2-7B | 36864 | 1.0178 | 1.0558 | 23.9257 | 445.94 |
| GPT-3-175B | 2304 | 1.0227 | 1.0725 | 162.1371 | 10428.74 |
| GPT-3-175B | 36864 | 1.0162 | 1.0506 | 426.8638 | 10751.47 |
| LLaMA-3-8B | 2304 | 1.0245 | 1.0784 | 7.3073 | 481.52 |
| LLaMA-3-8B | 36864 | 1.0173 | 1.0541 | 25.4378 | 524.77 |
| LLaMA-3.1-405B | 2304 | 1.0220 | 1.0700 | 376.0591 | 24228.58 |
| LLaMA-3.1-405B | 36864 | 1.0158 | 1.0494 | 928.1896 | 24796.00 |
| Mixtral-8x7B | 2304 | 1.0238 | 1.0762 | 11.9142 | 768.62 |
| Mixtral-8x7B | 36864 | 1.0167 | 1.0524 | 35.1959 | 782.33 |
| DeepSeek-V3 | 2304 | 1.0048 | 1.0158 | 166.7218 | 10766.88 |
| DeepSeek-V3 | 36864 | 1.0199 | 1.0629 | 296.2503 | 10805.70 |

相对 base，full-train 的 forward、backward、WGRAD phase 平均分别缩短 15.99%、
15.98%、15.98%（各 case 的下降范围约 6.55%–30.58%）。forward 与 full-train 使用
同一前向依赖，因此 full-train 相对 forward-only 的新增收益来自反向/WGRAD/梯度同步重叠；
optimizer 的 all-gradients-ready barrier 和 phase 时间保持不变。这里的 resident 包含
FP16 weights/gradients、FP32 master/m/v AdamW、per-block checkpoint activation。

## 单算子与整体系统：Amdahl 对照

每个训练点按实际模型结构选择 exp1 算子：Dense 使用 exp1-1 的
Attention/MLP × AG_GEMM/GEMM_RS（4 项）；Mixtral 在这 4 项上加入 exp1-2 的
DISPATCH_GEMM/GEMM_COMBINE（6 项）；DeepSeek 因 exp1-1 没有 MLA Attention anchor，
只取 routed-expert MLP × AG_GEMM/GEMM_RS 和 exp1-2 两项（4 项）。

exp1-1 使用与训练 TP 完全相同的 3x3 mesh，但 2304/36864 分别只能匹配到 2048/32768，
两组都是目标/来源=1.125 的同 mesh 邻近代理。exp1-2 使用 H128、noncompact、精确 seq_len，
作为训练四 rank noncompact EP 的位置代理。源 case ID、速度指标、文件 SHA-256 和限制标签
均保存在 `training_operator_amdahl.json`。

| 模型 | S | 算子数 | 算子均值 | 系统 full-train | 算子/系统 |
|---|---:|---:|---:|---:|---:|
| LLaMA-2-7B | 2304 | 4 | 3.8757× | 1.0869× | 3.57 |
| LLaMA-2-7B | 36864 | 4 | 4.0694× | 1.0558× | 3.85 |
| GPT-3-175B | 2304 | 4 | 3.7535× | 1.0725× | 3.50 |
| GPT-3-175B | 36864 | 4 | 3.8951× | 1.0506× | 3.71 |
| LLaMA-3-8B | 2304 | 4 | 3.9390× | 1.0784× | 3.65 |
| LLaMA-3-8B | 36864 | 4 | 4.1334× | 1.0541× | 3.92 |
| LLaMA-3.1-405B | 2304 | 4 | 3.7858× | 1.0700× | 3.54 |
| LLaMA-3.1-405B | 36864 | 4 | 3.9229× | 1.0494× | 3.74 |
| Mixtral-8x7B | 2304 | 6 | 3.2436× | 1.0762× | 3.01 |
| Mixtral-8x7B | 36864 | 6 | 3.3478× | 1.0524× | 3.18 |
| DeepSeek-V3 | 2304 | 4 | 2.3360× | 1.0158× | 2.30 |
| DeepSeek-V3 | 36864 | 4 | 2.6201× | 1.0629× | 2.47 |

图中紫红线是系统 full-train speedup，紫色上方折线是算子均值。所有 12 点都显示明显
Amdahl 稀释，但这个均值是按用户口径计算的描述性算术平均，不是按 baseline 时间占比
加权的 Amdahl 预测。尤其 MoE 的 TP 子算子和 EP 复合算子可能嵌套，不能从“算子/系统”
比值反推出可优化时间比例；严谨的 Amdahl 拟合仍需逐算子 baseline 时间占比。

## Prefill 单算子与系统对照

Prefill 图使用纯 prefill 阶段延迟，不把 KV handoff/wait 混入系统 speedup；原 P/D TTFT
breakdown 仍单独保留。Dense 使用 attention/MLP × AG_GEMM/GEMM_RS，Mixtral 再加入
DISPATCH_GEMM/GEMM_COMBINE，DeepSeek 只纳入有 exp1 锚点的 routed-MLP 与 MoE 项，
MLA attention 缺失作为限制标签保留。exp1-1 取精确 2x3 mesh，以 2048/32768 分别代理
2304/36864；exp1-2 取 H128、compact 和精确目标序列，作为连续 2x3 prefill instance 的
四 rank 拓扑代理。

| 模型 | S | 算子数 | 算子均值 | 系统 prefill | 算子/系统 |
|---|---:|---:|---:|---:|---:|
| LLaMA-2-7B | 2304 | 4 | 3.8111× | 1.4160× | 2.69 |
| LLaMA-2-7B | 36864 | 4 | 3.9997× | 1.0491× | 3.81 |
| GPT-3-175B | 2304 | 4 | 3.7530× | 1.4317× | 2.62 |
| GPT-3-175B | 36864 | 4 | 3.8828× | 1.0502× | 3.70 |
| LLaMA-3-8B | 2304 | 4 | 3.8577× | 1.4175× | 2.72 |
| LLaMA-3-8B | 36864 | 4 | 4.0476× | 1.0492× | 3.86 |
| LLaMA-3.1-405B | 2304 | 4 | 3.7502× | 1.4339× | 2.62 |
| LLaMA-3.1-405B | 36864 | 4 | 3.8782× | 1.0504× | 3.69 |
| Mixtral-8x7B | 2304 | 6 | 3.1894× | 1.1554× | 2.76 |
| Mixtral-8x7B | 36864 | 6 | 3.2907× | 1.1155× | 2.95 |
| DeepSeek-V3 | 2304 | 4 | 2.7101× | 1.0290× | 2.63 |
| DeepSeek-V3 | 36864 | 4 | 3.0081× | 1.2111× | 2.48 |

算子均值仍是描述性未加权算术平均，不是按 baseline 时间份额加权的 Amdahl 预测；特别是
MoE 的 TP 子算子与 EP 复合项存在嵌套风险。图中系统折线为
`prefill_cycles_base/prefill_cycles_overlap`，柱子则比较两种状态的归一化绝对延迟。

## Prefill + Decode 联合请求评价

联合主口径冻结输入 S=2304、输出 G=512，并使用
`T_request(B,G) = T_TTFT + G × T_decode_step(B)`。它直接关联分项记录的 result digest，
没有重新拟合性能参数。

| 模型 | B | speedup | overlap 请求时延 (s) | TTFT 占比 | decode 占比 |
|---|---:|---:|---:|---:|---:|
| LLaMA-2-7B | 64 | 1.0000 | 10008.1659 | 0.0048% | 99.9952% |
| LLaMA-2-7B | 512 | 1.0003 | 79281.0302 | 0.0006% | 99.9994% |
| GPT-3-175B | 64 | 1.0001 | 91986.0678 | 0.0125% | 99.9875% |
| GPT-3-175B | 512 | 1.0006 | 715441.7993 | 0.0016% | 99.9984% |
| LLaMA-3-8B | 64 | 1.0001 | 2594.5195 | 0.0193% | 99.9807% |
| LLaMA-3-8B | 512 | 1.0013 | 19913.6978 | 0.0025% | 99.9975% |
| LLaMA-3.1-405B | 64 | 1.0007 | 16463.5741 | 0.1575% | 99.8425% |
| LLaMA-3.1-405B | 512 | 1.0098 | 84672.9003 | 0.0306% | 99.9694% |
| Mixtral-8x7B | 64 | 1.0001 | 3246.8380 | 0.0848% | 99.9152% |
| Mixtral-8x7B | 512 | 1.0040 | 20566.0162 | 0.0134% | 99.9866% |
| DeepSeek-V3 | 64 | 1.0001 | 12659.4878 | 0.3147% | 99.6853% |
| DeepSeek-V3 | 512 | 1.0504 | 21948.4483 | 0.1815% | 99.8185% |

G=512 下 decode 占 overlap 请求时间的 99.6853%–99.9994%，所以联合 speedup 接近 decode，
不等于 prefill 优化失效。该组合把 KV=36864 附近的 local TPOT 常数外推 512 次，未逐 token
重放 KV 增长；联合绝对时延和分项一样仍是容量不可行、目标硬件未闭合的解析投影。

## Two-token local decode 主结果

| 模型 | B | speedup | overlap step (s) | system tokens/s | resident (GiB) |
|---|---:|---:|---:|---:|---:|
| LLaMA-2-7B | 64 | 1.0000 | 19.5463 | 6.549 | 2321.05 |
| LLaMA-2-7B | 512 | 1.0003 | 154.8448 | 6.613 | 18449.05 |
| GPT-3-175B | 64 | 1.0000 | 179.6378 | 0.713 | 21101.73 |
| GPT-3-175B | 512 | 1.0005 | 1397.3247 | 0.733 | 166253.73 |
| LLaMA-3-8B | 64 | 1.0000 | 5.0664 | 25.264 | 592.08 |
| LLaMA-3-8B | 512 | 1.0013 | 38.8930 | 26.329 | 4624.08 |
| LLaMA-3.1-405B | 64 | 1.0000 | 32.1048 | 3.987 | 3028.39 |
| LLaMA-3.1-405B | 512 | 1.0097 | 165.3261 | 6.194 | 18904.39 |
| Mixtral-8x7B | 64 | 1.0000 | 6.3361 | 20.202 | 664.12 |
| Mixtral-8x7B | 512 | 1.0040 | 40.1626 | 25.496 | 4696.12 |
| DeepSeek-V3 | 64 | 1.0000 | 24.6478 | 5.193 | 1559.30 |
| DeepSeek-V3 | 512 | 1.0504 | 42.7903 | 23.931 | 3720.99 |

`system tokens/s = 2×B×500 MHz/decode_step_cycles`。表中的 resident 使用一份全局只读
FP16 weight 加 4P/2D 的 resident KV；即便是该较乐观权重模式也全部不可行。

B64 只有一个窗口，baseline 与 overlap 的依赖图没有可利用的 chunk 独立性，因此六个模型
都为 1.0000×。B512 有四个窗口，但在长 KV HBM service、显式 KV append 和固定启动/逐 hop
延迟下，收益只有 0.03%–5.04%。B64 dense tensor 的 M/128 利用率为 0.5；Mixtral/DeepSeek
因 expert M 更小，FLOP 加权利用率最低分别约 0.199/0.086。B512 必须按每窗口计算：Mixtral
expert M=32、加权利用率约 0.397；DeepSeek expert M=4、加权利用率约 0.172。旧版约 1.48×
主要来自把 KV-cache bytes 错算进 reducer、为 B64 固定构造 8 个窗口，以及允许 collective
在 HBM 数据 ready 前启动；这些路径均已移除。

## Prefill/PD TTFT

| 模型 | S | speedup | overlap TTFT (s) | resident (GiB) | capacity |
|---|---:|---:|---:|---:|---|
| LLaMA-2-7B | 2304 | 1.3998 | 0.4844 | 17.05 | feasible |
| LLaMA-2-7B | 36864 | 1.0475 | 9.2866 | 84.55 | projection |
| GPT-3-175B | 2304 | 1.4254 | 11.5315 | 365.73 | projection |
| GPT-3-175B | 36864 | 1.0492 | 144.9662 | 973.23 | projection |
| LLaMA-3-8B | 2304 | 1.4136 | 0.5015 | 16.08 | feasible |
| LLaMA-3-8B | 36864 | 1.0488 | 9.3273 | 32.96 | feasible |
| LLaMA-3.1-405B | 2304 | 1.4336 | 25.9313 | 760.39 | projection |
| LLaMA-3.1-405B | 36864 | 1.0503 | 300.8429 | 826.84 | projection |
| Mixtral-8x7B | 2304 | 1.1551 | 2.7528 | 88.12 | projection |
| Mixtral-8x7B | 36864 | 1.1148 | 13.3615 | 104.99 | projection |
| DeepSeek-V3 | 2304 | 1.0290 | 39.8360 | 1250.49 | projection |
| DeepSeek-V3 | 36864 | 1.2110 | 92.6753 | 1259.53 | projection |

该表分别审计两档 prefill 的一份 shared weight 与四个 P instance KV，不等于主 decode
resident 状态。图中按模型并列长短序列，并将 prefill、KV handoff 与 ready-wait 分开展示。

## MoE sensitivity

λ 从 1.0 增至 1.5 时，逻辑 FLOPs、bytes 和 HBM work 保持不变，只增加最忙
expert/route 的 runtime service。训练按 full-train 主状态统计，latency 变化：

- Mixtral training：S2304 +2.32%，S36864 +34.07%；
- DeepSeek training：S2304 +0.05%，S36864 +23.11%；
- Mixtral decode：B64 +0.42%，B512 +0.07%；
- DeepSeek decode：B64 +1.44%，B512 +0.83%。

B64 的 expert service 会随 λ 增长，但因只有一个窗口仍无 overlap speedup；B512 仍主要受
HBM service 支配，λ 对绝对 latency 的影响较小，不能据此宣称 skew 改善性能。

DeepSeek shared expert 单独 opt-in 后，训练 full-train latency 相对 routed-only 增加：

- training：S2304 +1.44%，S36864 +1.60%；
- decode：B64 +0.33%，B512 +0.19%。

## 可解释边界

结果可用于比较同一解析模型下 base/overlap 的趋势、定位容量门禁和判断需补哪些 anchor。
它不能证明目标 1 TB/s D2D 下的绝对周期，也不能证明 64 GiB 上可运行不可行 case。
DeepSeek MLA、decode shape efficiency、KV append、vector rate、代表性 collective route、
die-level local NoC、HBM ingress 以及目标配置的通信/HBM 固定延迟仍保留显式 limitation
tag；两 token local interval 未验证稳态收敛。误差区间为模型区间，不是统计置信区间。
