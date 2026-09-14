# exp4：软硬件协同优化端到端实验开发方案

## 1. 目标与证据边界

本实验遍历 'wafer_scale_pareto_pruning/09-V1联合分组前沿与哨兵点.csv' 的383个硬件候选，
对训练、prefill、decode分别比较：

- 'naive'：不启用前面实验的inter-die和intra-die优化；
- 'sw_opt'：启用exp-2已定义的合法overlap、流水和映射优化；
- 'hw_opt_only'：在naive状态下搜索硬件；
- 'sw_hw_opt'：在sw_opt状态下搜索硬件。

方法固定为“少量周期精确锚点 + 资源显式解析回放 + 全候选外推”。本实验不为覆盖候选而
扩充仿真器的DTE、D2D或HBM实现。周期精确运行只验证action/lifecycle、固定延迟、小规模
资源竞争和重复性；候选间带宽、并行度、端口位置及HBM3差异由解析层表达。

完整候选结果统一标记为：

~~~text
cycle_anchor_calibrated_analytical_hardware_sweep
~~~

只有直接执行的短锚点可以标记 'cycle_accurate_direct'，不得把383个候选的完整模型结果
写成“周期精确仿真”。

## 2. 冻结口径

### 2.1 频率、算力和exp-2外部基准

计算、NoC、SRAM、DTE时钟统一为：

$$
f=500\ \mathrm{MHz},\qquad T_{cycle}=2\ \mathrm{ns}.
$$

单core矩阵峰值：

$$
P_{mat,core}=2N_{PE}f.
$$

所以4096 PE是4.096 TFLOP/s，8192 PE是8.192 TFLOP/s。原plan中“4096 PE对标
8 TFLOPS/core”来自旧的1 GHz扫描口径。

exp-2参考硬件 'H_exp2' 以其 'target_hardware.json' 和 'hardware_unit_closure.json'
为权威，关键口径为：

- 'exu_x=40, sa_cnt=5'，即8000 PE、500 MHz下8.0 TFLOP/s/core；
- 'dual_dte_dedicated'，按本实验语义对应 'n_ctrl=2'；
- 4x4 core/die、6x6 die、B=256 GB/s、K=3 MiB、B_s=256 GB/s、DTE channel=2；
- 4个16 GB HBM stack、exp-2解析目标256 GB/s/stack、D2D为1 TB/s/边/方向。

它作为sw_opt_only的外部控制组保留，不要求属于383个候选；不能把候选中的8192 PE或
HBM3/D2D配置改名为exp-2 baseline。

### 2.2 候选字段

每条候选解析：

~~~text
n_ctrl, router, B, K, B_s, N_PE, N, e_H, m, DTE_channel
~~~

并用die/wafer物理模型重新派生：

~~~text
t_hbm, d_d2d, HBM_stack_count, HBM_NoC_observable_GBs,
D2D_edge_one_dir_GBs, wafer_nx, wafer_ny, modules_per_wafer,
hbm_port_positions, d2d_port_positions
~~~

'DTE_channel' 必须满足：

$$
C_{DTE}=\left\lceil\frac{B}{128\ \mathrm{GB/s}}\right\rceil.
$$

不满足时记录 'candidate_semantic_mismatch'，禁止静默修正。

### 2.3 软件状态同工作量合同

复用exp-2的model manifests、placement语义、动作DAG、容量审计、资源命名和结果字段。
两个软件状态必须满足：

- action集合、FLOPs、HBM/SRAM/D2D/NoC bytes、route和runtime shape完全相同；
- 只允许依赖边、barrier、buffer slot和合法并发关系不同；
- naive对应exp-2的 'T_base_cycles'；
- 训练sw_opt对应 'T_full_train_overlap_cycles'；
- prefill/decode sw_opt对应 'T_overlap_cycles'；
- 不使用正收益clamp，软件优化变慢时保留小于1的speedup。

每条结果保存workload、model、action、software schedule及全部来源文件的digest。

## 3. 实验规模

### 3.1 主矩阵

完全复用exp-2的六个模型：

|类别|每模型配置|总数|
|---|---|---:|
|训练|'seq_len={2304,36864}, batch=1'|12|
|Prefill|'seq_len={2304,36864}, batch=1'|12|
|Decode|'seq_len=1, batch={64,512}, KV=36864'|12|

共36个primitive workload。两个软件状态下：

$$
383\times36\times2=\boxed{27,576}
$$

条主解析结果。

请求级推理沿用exp-2：

$$
T_{request}(B,G)=T_{prefill}(S=2304)+G\,T_{decode}(B),\qquad G=512.
$$

由primitive结果派生：

$$
383\times6\times2\text{ batches}\times2\text{ software states}
=\boxed{9,192}
$$

条请求级结果，不增加仿真或动作回放。长prefill 'S=36864'作为独立敏感性点。

### 3.2 可变wafer节点数

exp-2负载使用36个逻辑die rank，而候选的modules_per_wafer可变：

- 少于36个module：标记 'topology_capacity_infeasible'，不参加主Pareto；
- 至少36个module：确定性选择连通、边界最小、平均Manhattan距离最小的36节点子图；
- 训练分成4个连通的9-rank TP group，保持Dense的DP或MoE的EP语义；
- 推理分成6个连通的6-rank instance，4个P、2个D，最小化P到D handoff距离；
- 平局按全局module ID字典序裁决，保存mapping digest。

单副本延迟只使用36节点。另输出full-wafer throughput投影：

$$
R=\left\lfloor\frac{N_{module}}{36}\right\rfloor
$$

个互不通信的完整副本，系统吞吐为单副本吞吐乘 $R$；残余module不按比例折算。
单副本延迟和full-wafer吞吐必须分字段报告。

## 4. 四个阻塞项：只用解析类推解决

周期精确shadow配置只提供固定延迟、协议开销和竞争形状；目标候选的资源服务时间由本节
公式替换，不修改仿真器覆盖目标规格。

### 4.1 DTE多channel并非独立物理通路

**问题。** 当前多个DTE channel可能仍共享串行聚合总线，不能代表候选定义的
$C$条独立2048-bit物理通路。

**解析模型。** 每channel为128 GB/s、2048 bit@500 MHz；同一channel一次只执行一个方向，
不同channel并行。按action到达顺序采用earliest-finish list scheduling：

$$
T_{DTE,j}=\sum_{a\in j}\left(L_{launch,a}
+\left\lceil\frac{Q_a f}{128\times10^9}\right\rceil\right),\qquad
T_{DTE}=\max_j T_{DTE,j}.
$$

每个action还占用对应方向的SRAM和NoC入口，所以总吞吐受
$\min(C\times128,B,B_s)$约束，但不能用一个min替代资源日历。

**校准。** 只运行单channel和当前可闭合的双channel shadow motif，拟合launch、命令发射
和小消息尾部。通道并行度由结构式给出，不从当前串行实现测量。输出逐channel服务账本及
'dte_analytical_parallel_channels=true'。

### 4.2 D2D不能直接表达512 GB/s/边/方向

**问题。** 当前D2D wire的packet/cycle上限不能表示UCIe-A x64的512 GB/s；为每个逻辑
端口复制物理链路又会错误放大容量。

**解析模型。** 'd_d2d=d' 是每边逻辑NoC入口数，一条边只有一个共享物理PHY：

$$
C_{edge}^{\rightarrow}=\min(dB,512)\ \mathrm{GB/s}.
$$

复用周期精确/exp-2得到的逐有向边route和bytes。每条有向物理边建立共享资源日历：

$$
T_{e,cycles}=L_{hop,e}
+\left\lceil\frac{f\sum_f Q_{f,e}}{C_e^{\rightarrow}\times10^9}\right\rceil.
$$

通信时间由依赖最长路和最拥塞有向边共同决定，不使用四边带宽求和，也不使用
'bytes/(hop_count*bandwidth)'。

**校准。** 用1/2/多hop和小/大消息shadow motif拟合hop、packetization tail与route setup；
bulk rate始终由候选公式替换。输出逐有向边bytes、capacity、busy cycles和cut utilization。

### 4.3 单流只支持1/2/4路D2D stripe

**问题。** 候选存在 'd_d2d=9'，当前单流stripe枚举无法使用全部逻辑入口。

**解析模型。** 不扩充simulator stripe枚举。读取冻结的物理payload bytes $P_{payload}$，
对大小为 $Q$ 的flow使用：

$$
s=\min\left(d,\left\lceil\frac{Q}{P_{payload}}\right\rceil\right)
$$

条非空stripe。fragment按稳定的 '(flow_id, fragment_id)' 哈希后round-robin到clustered
端口，各stripe fragment数最多相差1。每条stripe按自己的入口NI和intra-die route计费，
但同一die边的所有stripe共享4.2节的一个物理PHY日历。

striping只改善入口和第一跳分布，不突破 $\min(dB,512)$。输出stripe数、逐端口bytes和：

$$
I_{stripe}=\frac{\max_p Q_p}{Q/s}.
$$

要求 $I_{stripe}\le1+P_{payload}/Q$。1/2/4路用现有路径校验，3/5/9路只验证字节守恒、
共享PHY容量和解析单调性。

### 4.4 HBM3及clustered非连续端口不能直接绑定

**问题。** 候选使用HBM3 16 GB、819.2 GB/s/stack、$\alpha_H=0.9$；HBM端口可能是
clustered非连续位置，当前resolver偏向既有memspec和连续attachment。

**解析模型。** 不增加HBM3 memspec或非连续attachment。解析层建立：

~~~text
stack_id, edge, capacity_bytes, peak_Bps, sustained_Bps,
port_ids[], port_positions[], address_ranges[], resource_calendar
~~~

总stack数 $M=e_Hm$。地址按exp-2已有的256 B stack interleave条带化：

$$
stack(addr)=\left\lfloor\frac{addr}{256}\right\rfloor\bmod M.
$$

单stack持续上限：

$$
C_{stack}=0.9\times819.2=737.28\ \mathrm{GB/s}.
$$

每条HBM边的NoC可观察上限：

$$
C_{H,edge}=\min(m\times737.28,t_HB)\ \mathrm{GB/s}.
$$

HBM action同时占用stack、HBM edge ingress和沿途NoC资源：

$$
T_{HBM,cycles}=\max\left(
\max_s\left\lceil\frac{Q_s f}{737.28\times10^9}\right\rceil,
\max_e\left\lceil\frac{Q_e f}{C_{H,e}\times10^9}\right\rceil,
\mathrm{NoC\ path\ service\ cycles}
\right)+L_{first-byte}.
$$

其中 $Q$ 的单位为byte，$f$ 为cycle/s，所有 $L$ 和 $T$ 均以cycle记录。

非连续端口直接使用物理扫描器的 'port_positions[]' 路由，不压缩为连续区间。

**校准。** 现有HBM/DRAMSys短运行只校准first-byte、burst tail和固定开销；HBM3 bulk rate、
stack并行和非连续端口由解析模型给出。容量逐stack审计，超出时标记
'capacity_infeasible_projection'。

### 4.5 SRAM总读/总写预算

这不是第五个blocker，而是四个模型共同遵守的资源合同。每core只有一个 $B_s$ 读资源和
一个独立的 $B_s$ 写资源；compute、LSU、DTE和NoC receive共享各方向预算。

~~~text
sram.read.core[k]  capacity=B_s
sram.write.core[k] capacity=B_s
~~~

采用work-conserving round-robin和256 B bank interleave；bank冲突系数由exp-2 SRAM motif
校准。不得给每个initiator各复制一份 $B_s$。

## 5. 资源显式解析回放

每个candidate至少建立：

~~~text
tensor.core, vector.core, reducer.core,
sram.read.core, sram.write.core,
dte.channel.core, noc.directed_link,
d2d.logical_port, d2d.directed_physical_edge,
hbm.stack, hbm.edge_ingress,
control.compute_issue, control.dte.issue
~~~

'n_ctrl=1' 时compute/DTE共享control issue，'n_ctrl=2' 时独立。router的四种档位只改变
相应collective action和固定服务项，不改变workload bytes。

计算action：

$$
T_{mat,cycles}=\left\lceil\frac{FLOPs_{runtime}}
{N_{active\ core}\times2N_{PE}\times\eta_{shape}}\right\rceil.
$$

$\eta_{shape}$ 由周期精确GEMM motif或exp3.2热力图插值，不能固定为1。调度器统一使用cycle：

$$
start(a)=\max\left(\max_{d\in deps(a)}finish(d),
\max_{r\in resources(a)}available(r)\right),
$$

$$
finish(a)=start(a)+duration(a).
$$

一个action使用多个资源时原子预留全部资源。理论下界：

$$
T_{lower}=\max(\mathrm{dependency\ longest\ path},
\max_r\mathrm{resource\ service}_r),
$$

并要求 $T_{result}\ge T_{lower}$。

复用exp-2的是action工作量和依赖语义，不直接复用绝对周期。硬件变化时逐项重算FLOPs、
shape效率、所有bytes资源、拓扑、端口、route、control和容量。禁止用全局
'old_time * old_peak/new_peak' 缩放完整E2E。

## 6. 少量周期精确校准

### 6.1 校准规模

按归一化特征：

~~~text
[N_PE, B, B_s, K, N, e_H*m, t_hbm, d_d2d, router, n_ctrl]
~~~

用maximin/farthest-point选择8个覆盖点，生成simulator可表达的shadow配置。保持shape、
拓扑和消息bytes，只裁剪无法表达的目标bulk rate。覆盖exp-2参考、compute/NoC/SRAM/HBM
高低压力点和 'router=both,n_ctrl=2' 角点。

每个shadow点运行6类短motif并双跑：

~~~text
isolated_gemm, sram_read_write, single_channel_dte,
d2d_1_2_multihop, hbm_read_write, collective_broadcast_reduce
~~~

$$
8\times6\times2=96.
$$

再取3个未参与拟合的shadow hardware，运行6个E2E signature window：

~~~text
dense_train_short, dense_prefill_long, dense_decode_B512,
gqa_decode_B64, moe_train_long, moe_decode_B512
~~~

两个软件状态、每项双跑：

$$
3\times6\times2\times2=72.
$$

总计168个短周期精确运行，占27,576条主sweep的0.61%。exp-2/Stage3/Stage4已有证据只作
结构证据，不能替代当前build的shadow双跑。

### 6.2 验收阈值

- 锚点双跑makespan、action count和byte counters完全一致；
- motif拟合MAPE中位数不超过10%，P95不超过20%；
- E2E signature留出误差不超过15%，且naive/sw_opt排序一致；
- 每个blocker至少有一个低压力点和一个饱和点；
- 超出锚点凸包的候选标记 'extrapolation_out_of_anchor_hull'；
- 失败时增加motif、解析项或不确定性范围，不扩充仿真器覆盖目标规格。

## 7. exp-2复用与sw_opt_only对齐

### 7.1 只读来源

~~~text
exps/exp2/exp2_1/manifests/models/*.json
exps/exp2/exp2_1/results/training_e2e.json
exps/exp2/exp2_1/results/inference_prefill_pd_breakdown.json
exps/exp2/exp2_1/results/inference_decode_e2e.json
exps/exp2/exp2_1/results/inference_request_e2e.json
exps/exp2/exp2_1/results/capacity_audit.json
exps/exp2/exp2_1/results/calibration_summary.json
~~~

exp4保存SHA-256和exp-2 'result_digest'，禁止写回exp-2结果。
exp-2当前发布状态是 'analytical_only_target_binding_unclosed'；导入时必须原样继承
calibration status和limitation tags。这里的“对齐”表示控制组数值、工作量和软件调度定义
完全对应，不表示把exp-2升级为目标硬件周期精确证据。

### 7.2 精确行连接

'sw_opt_only' 使用外部硬件 'H_exp2'，按以下key连接：

~~~text
training: (model_id, seq_len, batch_size=1)
prefill:  (model_id, seq_len, batch_size=1)
decode:   (model_id, batch_size, seq_len=1, kv_len=36864)
request:  (model_id, batch_size, prefill_seq=2304, output_tokens=512)
~~~

数值直接取：

~~~text
training speedup = T_base_cycles / T_full_train_overlap_cycles
prefill speedup  = T_base_cycles / T_overlap_cycles
decode speedup   = T_base_cycles / T_overlap_cycles
request speedup  = T_base_cycles / T_overlap_cycles
~~~

自动断言exp4导出的sw_opt_only与exp-2对应行相对误差不超过 $10^{-12}$，且
source_result_digest完全一致。该曲线不经过候选解析模型重算。

### 7.3 三种speedup的共同分母

图1以 'H_exp2 + naive' 为共同起点：

$$
S_{sw}=\frac{T(H_{exp2},naive)}{T(H_{exp2},sw\_opt)},
$$

$$
S_{hw}=\frac{T(H_{exp2},naive)}{\min_{h\in H}T(h,naive)},
$$

$$
S_{sw+hw}=\frac{T(H_{exp2},naive)}{\min_{h\in H}T(h,sw\_opt)}.
$$

'H_exp2' 是候选集外部控制组。容量或拓扑不可行候选不参与对应workload的最小值。

## 8. 输出、Pareto和绘图

主结果至少包含：

~~~text
candidate_id, candidate_digest, model_id, workload_id, software_state,
status, evidence_level, estimate_cycles, throughput, latency,
capacity_status, topology_status, mapping_digest,
compute_service_cycles, sram_read_cycles, sram_write_cycles,
dte_service_cycles, noc_service_cycles, d2d_service_cycles,
hbm_service_cycles, control_service_cycles, critical_resource,
theory_lower_cycles, attainment, four_blocker_assumptions,
calibration_digest, uncertainty_low, uncertainty_high, result_digest
~~~

逐资源账本另存。

图1中training/prefill/decode各自两个shape先在同一模型内取几何平均speedup，再画六模型
折线。三条线严格为sw_opt_only、hw_opt_only、sw_hw_opt。不可行投影使用不同标记。

图2对每模型和软件状态分别求硬件Pareto。训练分数取两个训练长度吞吐的几何平均；推理主
分数取 'S=2304,G=512' 下B64/B512请求吞吐的几何平均。每轴在同一模型内按有效候选最大值
归一化。平均曲线先逐模型归一化，再对六模型取几何平均。

同时输出训练极值、推理极值、几何平均最大均衡点、G_split、前沿点数、前沿张角，以及
naive与sw_opt下的argmax硬件是否切换。

## 9. 不确定性与稳健性

|参数|主值|敏感性|
|---|---:|---:|
|DTE channel持续效率|1.0|0.8, 0.9, 1.0|
|D2D PHY持续效率|1.0|0.8, 0.9, 1.0|
|HBM利用率 $\alpha_H$|0.9|0.8, 0.9, 0.95|
|HBM first-byte latency|校准值|0.5x, 1x, 2x|
|stripe端口不均衡|解析均匀值|1x, 1.1x, 1.25x|

主值在前沿、轻微扰动后消失的点标记 'fragile_pareto'。最优硬件发生切换时报告候选集合。

## 10. 建议代码结构

~~~text
exps/exp4/candidate_loader.py
exps/exp4/exp2_workload_adapter.py
exps/exp4/placement_mapper.py
exps/exp4/hardware_resources.py
exps/exp4/analytical_replay.py
exps/exp4/blocker_models.py
exps/exp4/run_calibration.py
exps/exp4/run_experiment.py
exps/exp4/build_pareto.py
exps/exp4/plot_results.py
exps/exp4/tests/
~~~

生成物进入 'results/'、'calibration/' 和 'figures/'，不得写回exp-2。

## 11. 开发阶段和门禁

### 阶段A：输入与语义闭合

载入383条候选，验证500 MHz口径和派生式；只读载入exp-2并验证digest；生成容量与
36-rank topology报告。

门禁：候选无重复；不一致有显式状态；36个primitive sw_opt_only可一一连接。

### 阶段B：解析资源模型

实现四个blocker、SRAM共享资源、可变topology route及naive/sw_opt依赖状态。

门禁：逐action FLOPs/bytes守恒；结果不低于理论下界；增加单一资源能力不使同一DAG变慢。

### 阶段C：周期精确校准

执行96个拟合motif和72个留出window并保存配置、日志、计数和SHA-256。

门禁：双跑一致并达到6.2节阈值；失败时只改解析模型、锚点或不确定性。

### 阶段D：全量sweep

生成27,576条primitive、9,192条请求级派生结果和敏感性结果，支持分片、断点续跑和缓存。

门禁：行数精确；无NaN/负周期；不可行点不进入可行Pareto；重复运行digest一致。

### 阶段E：报告

生成两张主图、最优配置、Pareto指标、argmax切换、瓶颈分解和稳健性附表。

门禁：图1 sw_opt_only与exp-2逐行一致；每个图点可回溯全部digest；区分direct anchor、
解析外推和不可行性能投影。

## 12. 最小自动化断言

~~~text
candidate_count == 383
primitive_workload_count == 36
software_state_count == 2
primary_result_count == 27576
request_derived_count == 9192
cycle_anchor_run_count == 168

forall candidate: f_Hz == 500_000_000
forall candidate: P_core_FLOPs == 2*N_PE*f_Hz
forall candidate: DTE_channel == ceil(B/128)
forall candidate: D2D_edge_GBs == min(d_d2d*B, 512)
forall candidate: HBM_stack_GBs == 819.2
forall candidate: HBM_stack_capacity_GB == 16

forall paired naive/sw_opt: same logical work and bytes
forall result: estimate_cycles >= theory_lower_cycles
forall resource: admitted_bytes == completed_bytes
forall sw_opt_only: relative_error_vs_exp2 <= 1e-12
forall sw_opt_only: source_result_digest == exp2.result_digest
forall pareto point: not dominated by any valid point in same group
~~~

若500 MHz物理面积模型重扫产生新CSV，流程不变，只更新候选digest和行数断言；新旧候选
不得混在同一条Pareto曲线中。
