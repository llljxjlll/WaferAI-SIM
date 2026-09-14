# exp2-1 开发报告

## 实现范围

实验严格使用六个模型和两档训练/推理条件，不保留原 plan 的“八模型”笔误。实现分为：

- `model_manifests.py` 与 `manifests/`：冻结六个模型结构、参数分解、来源快照与 digest；
- `capacity_model.py`：64 GiB byte-exact resident-state 审计；
- `hardware_binding.py` 与 `configs/`：6×6 wafer、16 cores/die、128 TFLOP/s/die、
  3 MiB SRAM/core、四个 16 GiB 边缘 HBM stack 的目标绑定；
- `e2e_replay.py` 与 `placements.py`：资源显式 action DAG、训练三状态/推理双状态回放；
- `run_experiment.py`：主矩阵、请求级联合指标、MoE λ 和 shared-expert sensitivity；
- `build_operator_amdahl.py`：按训练 case 关联 exp1-1/exp1-2 源算子、校验 mesh/seq/placement、
  生成未加权算子均值及完整来源 digest；
- `build_prefill_operator_amdahl.py`：按 prefill DAG 关联 2x3 exp1-1 与 compact exp1-2 算子；
- `plot_results.py`：无外部依赖的五张 SVG，并在训练与 prefill 图叠加算子均值折线；
- `tests/`：manifest、容量、硬件 gate、DAG、公式、digest、schema 与绘图回归。

## 周期精确复用

复用的 simulator workload 没有被线性放大成目标周期：

- Stage4 PDS：当前 build 双跑均为 6187 cycles，ProgramIO、64 个 logical/physical
  packets、byte conservation 和 drain 均通过；
- Stage3 prefill/decode：当前 build 双跑观察为 5805/5805 与 13081/13081 cycles，
  但 artifact/marker digest 与 reviewed golden 不一致，因此只保留结构先验；
- exp1-2 Flexible MoE train/inference：3566/3566 与 2727/2727 cycles；
- exp1-2 isolated GroupGEMM：primitive 6 cycles、makespan 70 cycles。

这些证据的文件 digest 被写入 `results/calibration_summary.json` 及每条结果记录。

## 目标硬件 gate

结构与配置算术可以表达：

- 6×6 dies、每 die 4×4 worker cores；
- dedicated dual-DTE control，不占用 16 个 workers；
- 每 core 8 TFLOP/s，即每 die 128 TFLOP/s；
- 四个 16 GiB HBM stack 位于 die 1/4/31/34，地址连续且不重叠；
- SRAM、NoC、DTE、HBM 的配置算术目标为 256 GB/s。

阻断项是 D2D：业务 payload 为 16 B、周期 2 ns，而 parser 拒绝大于 1 packet/cycle。
单 lane 可表达上限为 8 GB/s，距 1 TB/s 目标差 125×。因此
`simulator_unit_closure=false`、`publish_target_absolute_cycles=false`，目标直接
anchor 数为 0。

## 解析回放口径

时钟固定为目标 2 ns/cycle，即 500 MHz。资源包含 HBM stack/ingress、DTE channel、
directional XY D2D link、die-level local NoC、SRAM port、tensor、vector、reducer 和
control queue。

训练包含 base、forward-only、full-train 三态，action 的工作量、bytes、route、资源和
duration 完全相同，仅依赖边不同。forward-only 只放宽前向；full-train 还让下一层 dX
无需等待上一层 WGRAD/sync，并允许 WGRAD 与逐层 gradient collective 合法重叠。每层
sync 仍等待本层全部 WGRAD，AdamW 仍等待所有 layer×flow 的 sync。主训练指标采用
full-train，同时保留 forward-only 字段作为消融。12/12 主训练点均满足
full-train≤forward-only≤base，且 full-train 的 forward/backward/WGRAD phase 都严格
短于 base。推理仍采用 4P+2D 双状态；two-token local decode 与 TTFT 分开报告。
在此基础上额外输出 G=512 的 `TTFT + G×local_TPOT` 请求级派生指标；每条联合记录保存
prefill/decode 两条来源 digest。分项结果保持独立，联合项明确标记固定 KV local TPOT 外推
以及生成期 KV 增长未重放。

算子/系统 Amdahl 对照按模型分三类：Dense 取 exp1-1 Attention/MLP 的
AG_GEMM/GEMM_RS；Mixtral 再加入 exp1-2 DISPATCH_GEMM/GEMM_COMBINE；DeepSeek 因没有
MLA Attention anchor，只取 routed-expert MLP 两项及 exp1-2 两项。exp1-1 固定匹配
3x3 TP mesh，以 2048/32768 代理目标 2304/36864；exp1-2 固定 H128、noncompact 和精确
seq_len。每条输出保存 source case ID、源文件 SHA-256、匹配方式及限制标签。算术均值只作为
描述性对照，不伪装成按 baseline 时间份额加权的 Amdahl 预测；MoE 的 TP 子算子和 EP 复合
算子嵌套风险也显式保留。

Prefill 对照使用纯 prefill 阶段口径：系统 speedup 为 prefill base/overlap cycles，不混入
KV handoff 与 wait。Dense/Mixtral/DeepSeek 的算子集合按实际 DAG 分别为 4/6/4 项；
DeepSeek MLA attention 没有 exp1-1 锚点，因此显式排除并打标签。主矩阵与训练对齐为
S={2304,36864}。exp1-1 使用精确 2x3 mesh，以 2048/32768 代理 2304/36864；exp1-2 使用
H128/compact/精确目标序列，作为连续 2x3 P instance 的拓扑代理。原 prefill/PD breakdown
同步扩展为 12 个 case，请求级组合仍只选择 S=2304。


模型真实性修正包括：

- GPT-3 dense GELU 使用两矩阵和 4SHI FLOPs；SwiGLU 使用三矩阵和 6SHI；
- DeepSeek 前三层使用 dense I=18432，后 58 层使用 routed I=2048；
- DeepSeek 主路径只计算 routed top-8，shared expert 通过独立 sensitivity opt-in；
- MLA 使用 compressed latent+RoPE KV 宽度，始终为 `analytical_only_mla`；
- MoE 权重按实际触达专家数计算，训练 routed weights 按 EP=4 分片；
- λ 不改变 action logical work、bytes 或 HBM work，只增加最忙 expert/route 的 runtime
  service，且一致覆盖训练 forward/backward/WGRAD 与推理 prefill/decode。
- decode 使用两 token 自回归链和 per-instance commit interval，作为固定 KV 上下文附近的
  local TPOT；稳态收敛未验证并保留 limitation tag；
- B64/B512 的窗口数为 1/4；decode tensor 按每个窗口的 M/128 计空间利用率，MoE expert M
  另按 top-k/触达专家数计算；
- KV-cache read 只计入 HBM stream，activation reducer 不再吸收 KV bytes；每层新 KV 显式
  append 到四个 HBM stack，token commit 等待全部 layer append；
- collective 必须等待本窗口 HBM 数据 ready；DTE launch、逐 hop 与 HBM first-byte 固定延迟
  显式进入 action duration，并保持 `target_config_*_unvalidated` 限制。

## 验收

最终测试为 59/59 PASS，覆盖：

- 6 manifests、12 training、12 training operator Amdahl、12 prefill operator Amdahl、12 decode、
  12 prefill、12 request composite、24 λ、8 shared sensitivity；
- full-train/forward-only training tokens/s、decode 2×B throughput、TPOT、TTFT phase sum；
- exp1-1/exp1-2 mesh、seq_len、placement、源 SHA-256、算术均值及训练 result digest 绑定；
- `TTFT + 512×local_TPOT` 公式、分项 result digest 关联、占比和请求输出速率；
- 训练三态 same-work、调度单调性、前向/反向/WGRAD 严格加速和 AdamW 全梯度门禁；
- λ logical-work conservation；
- capacity status、MLA/source/limitation tags；
- current tool/config/result digest 一致性；
- 五张 SVG：training 为暖灰/橙色的相接宽柱、紧凑模型分组，以及参照 exp1-1 的紫红色
  系统 speedup 粗折线与白心圆点；算子均值使用相同格式的独立紫色上方折线，两者均不画
  uncertainty；prefill 主图采用紫/金柱、朱红系统折线与藏蓝算子折线，同样不画 uncertainty；
  其余推理图保留误差带，所有图保留 1.0× 线与组合纹理。
