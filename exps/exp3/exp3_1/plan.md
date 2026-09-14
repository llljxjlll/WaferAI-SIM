固定硬件配置
核层级：
|             | 含义                        | 取值     | 说明              |
| ----------- | --------------------------- | -------- | ----------------- |
| n_ctrl      | 控制核数                    | 1        |                   |
| r           | router 功能档位             | base     |                   |
| B           | 单 core 注入/单链路目标带宽 | 256 GB/s |                   |
| K           | 单 core SRAM 容量           | 3MB      |                   |
| B_s         | 单 core SRAM 聚合带宽       | 256 GB/s |                   |
| N_PE        | 矩阵 PE 数                  | 4096     | 对标8 TFLOPS/core |
| DTE_channel | DTE channel个数             | 2        |                   |

Die和Wafer层级：
|          | 含义                   | 取值 | 备注                   |
| -------- | ---------------------- | ---- | ---------------------- |
| core阵列 |                        | 4*4  |                        |
| e_H      | HBM 放置边数           | 2    |                        |
| m        | 每条 HBM 边的 stack 数 | 2    | 16GB/stack，因此是64GB |
| die阵列  |                        | 6*6  |                        |

设计思路
 实验①（软件消融，硬件不变）：全程用晶圆自己的 die 内硬件模型，只把 intra-die 优化开关关掉，测的是"inter-die 融合算子（跨 die 的 AG+GEMM / Dispatch+GEMM 等）单独拿出来、不借助 intra-die 优化加持时，自己能贡献多少收益"——回答"inter-die 这一层本身有没有用、有多少用"。
实验②（硬件替换，跨平台）：把 die 内的计算+通信行为整体换成一个等算力 GPU 的真实实测行为，inter-die 的 Swizzling 调度逻辑不变，测的是"如果把晶圆的 die 换成 GPU 节点，我们这套跨节点 Swizzling 编排算法是不是依然有效"——回答"inter-die 这层技术是不是晶圆专属的，还是一个通用的、可移植到任意硬件节点上的跨节点通信-计算重叠方案"。
这是攻击性完全不同的两个论证（前者是"归因"，后者是"可移植性/普适性"），所以设计上不能用同一套开关，但可以共享同一个 x 轴和同一套归一化方式，最后拼进一张图。

实验①具体：
固定：晶圆自己的 D2D 链路模型、DTE inter-die 通信代价模型不变。
- 两组对照： 
  - baseline：intra-die 优化关闭，但仍固定采用 16 核 `4×4×1` canonical 映射；本地计算、HBM 与本地 NoC 朴素顺序执行，inter-die 流式编排关闭
  - 处理组：保持完全相同的 16 核映射、本地实现和基础 HBM 访存，打开 inter-die 流式编排；两侧 HBM 工作量相同
- 扫描维度：不要只测一个 mesh size，固定扫描 D=6/9/36（2×3/3×3/6×6），这样才能和笔记 2707-2727 行已推导的 Dispatch+GEMM(EP-style/inter-die) 通信量随 D 衰减 O(1/√D) 的理论曲线对上。
- 报告指标：`native_inter_only=C00/C10`（固定 16 核、相同基础 HBM 成本下 inter-die 流式编排的单独提升）；另报告 `native_full=W00/W11`，即单 die 16 核 canonical baseline 对 Exp1 派生的 16 核 intra+inter 优化实现。两个轨道不交叉相除。

实验②具体：
关键是要把"等算力 GPU"这个替换做得站得住脚，否则审稿人会质疑标定方式，建议按这个流程走：
1. 先定义"等算力"的口径：是按峰值 FLOPS 对齐（晶圆单 die 的 N_PE×频率 换算成 TFLOPS，去匹配某型号 GPU 的峰值算力，或者按 SM 数/时钟降频折算出一个"部分 GPU"），还是按该具体算子 shape 下的实测达成算力对齐（更严谨，因为 GEMV/GEMM 在小 M 下达成率差异很大，直接用峰值对齐会失真）。建议用后者——用 Exp1.1 里同一批算子 shape，在真实 GPU（比如 A100/H100）上跑一遍 microbenchmark，拿到该 shape 下的实测延迟，作为"这个 die 换成 GPU 之后应该花多少时间"的输入，而不是理论换算值。这个标定方法本身要在论文里写清楚，这是这组实验最容易被挑战的地方（类似笔记规律六强调的"baseline 强弱必须交代清楚"）。
2. 替换方式：把 die 内的 T_comp（以及 die 内 core-to-core 的 intra 通信，如果 GPU 架构里没有对应概念，可以直接合并进 GPU 那个算子的整体实测延迟里，作为一个"黑箱节点耗时"）整体替换成第 1 步标定出来的 GPU 实测数字；inter-die 的 Swizzling 调度算法和 D2D 通信代价模型完全不变，只是它现在调度的对象从"晶圆的 die"变成了"等算力 GPU 节点"。
3. 两组对照：同样是 {inter-die 融合关闭} vs {inter-die 融合开启}，用刚才替换后的 GPU 化节点耗时模型，跑同一套 D 扫描。
4. 报告指标：与实验①完全一致的口径——inter-die 融合带来的 speedup（处理组/baseline），随 D 变化的曲线。

实验配置
预算 48 点：3 种 mesh 形状 × 16 种（算子×shape 组合）= 48。"inter-die 关/开"两个条件不算独立点——同一配置下测一对（关、开），画图时是同一 x 位置堆叠/并排的两根 bar，不计入点数。
mesh 点位（3 个，复用已有配置，不新增工程成本）
D=6（2×3）、D=9（3×3）、D=36（6×6）
- 舍弃 D=4（1×4）和 D=16/25（4×4/5×5）等点位，保留 D=6 的小规模锚点、D=9 的方形紧凑组和 D=36 的完整 wafer。6→9→36 总体覆盖 6 倍 D 跨度，相邻倍率分别为 1.5 倍和 4 倍，既能观察小规模局部变化，也能拉开 O(1/D) 与 O(1/√D) 的大规模差异。
- D=6、D=9 和 D=36 均复用已有 2×3、3×3、6×6 mesh 配置；其中 D=9 和 D=36 直接对应主实验（Exp 2.1）里的 TP=3×3 紧凑组和完整 6×6 wafer，测出来的达成率可以直接和主实验数字对照。
算子选择（2 个）

| 算子                | 类型                               | 理论衰减       |
| ----------------- | -------------------------------- | ---------- |
| **GEMM+RS**       | TP-style / in-axis resharding    | O(1) 平台期   |
| **Dispatch+GEMM** | EP-style / cross-axis resharding | O(1/√D) 衰减 |

16 个 shape 分配：8 + 8 对称切
两个算子各自天然存在 3 个二元维度，2³=8 刚好完整覆盖，不用再裁。
GEMM+RS（8 个）= 2 层类型 × 2 模型（规模极端值）× 2 seq_len
| 层类型              | 模型                                       | seq_len=2304 | seq_len=36864 |
| ---------------- | ---------------------------------------- | ------------ | ------------- |
| MLP down-proj    | LLaMA-2-7B（H=4096, I=11008，Dense MHA·小）  | shape1       | shape2        |
| MLP down-proj    | GPT-3-175B（H=12288, I=49152，Dense MHA·大） | shape3       | shape4        |
| Attention O-proj | LLaMA-2-7B（n_head=32, d_head=128）        | shape5       | shape6        |
| Attention O-proj | GPT-3-175B（n_head=96, d_head=128）        | shape7       | shape8        |
选 LLaMA-2-7B 和 GPT-3-175B 这一对，是因为它们是 Exp1.1 六模型表里 Dense MHA 的小/大两个极端（I/H 分别 2.69 和 4），直接复用已有参数，不用新定义模型。8 个 shape 同时覆盖了层类型（MLP vs Attention，验证"M 是否进 r"这条判据）和模型规模（小 vs 大，验证 O(1) 平台期是否跨架构规模都成立）两件事。

Dispatch+GEMM（8 个）= 2 个真实 MoE 模型 × 2 个 GEMM 阶段 × 2 seq_len
直接复用笔记 Exp1.1 模型表（line 2509-2517）里已经定义好的两档真实 MoE 模型，不再借用 TileLink 那套合成的 (E,topk)/H 正交表——H、I、E、topk 是绑在一起的真实架构参数，不能像之前那样拆成两个独立维度硬配对：
| 档位  | 模型                                 | H    | I         | E   | topk |
| --- | ---------------------------------- | ---- | --------- | --- | ---- |
| 粗粒度 | Mixtral-8×7B                       | 4096 | **14336** | 8   | 2    |
| 细粒度 | DeepSeek-V3（仅取 MoE-FFN 部分，不涉及 MLA） | 7168 | **2048**  | 256 | 8    |
第二个维度改用"GEMM 阶段"（gate/up-GEMM vs down-GEMM，和 MLP 拆 AG+GEMM/GEMM+RS 同一思路）：
| 模型                                     | GEMM 阶段                                    | seq_len=2304 | seq_len=36864 |
| -------------------------------------- | ------------------------------------------ | ------------ | ------------- |
| Mixtral-8×7B (H=4096, I=14336, topk=2) | Dispatch+up/gate-GEMM，shape=(M_rank, I, H) | shape1       | shape2        |
| Mixtral-8×7B                           | down-GEMM+Combine，shape=(M_rank, H, I)     | shape3       | shape4        |
| DeepSeek-V3 (H=7168, I=2048, topk=8)   | Dispatch+up/gate-GEMM                      | shape5       | shape6        |
| DeepSeek-V3                            | down-GEMM+Combine                          | shape7       | shape8        |
其中 M_rank = S × topk / D（D 取本实验的 mesh 扫描点 6/9/36），即 Dispatch+GEMM 的 shape 会随 D 联动变化——这一点和 GEMM+RS 不同（GEMM+RS 的 M 不随 D 变化，只有 N/K 按 TP 切分），需在实验记录表里单独注明，避免和 D 扫描的主变量混淆。gate/up 是两个独立 GEMM，同一 shape 测一次、耗时使用两次。


GPU 实测查找表口径（最小粗粒度版本）
- 三个 mesh 固定使用同一组全局 seq_len：S∈{2304,36864}，避免 D 扫描同时混入 workload scaling；两个值都能被 6、9、36 整除。
- GEMM shape 统一写为 (M,N,K)，表示 A[M,K]×B[K,N]→C[M,N]。测量对象是跨 die 分片后单个 GPU 实际执行的本地 GEMM，不包含 NCCL/D2D；D2D 通信仍使用原有链路与路由模型。
- 对 GEMM+RS，设经过语义 padding 后的全局 GEMM 为 (M,N,K)，最小查找表同时覆盖三种粒度：
  - 粗粒度 rank-local GEMM：(M,N,K/D)。
  - 1D Ring Swizzling 的最小 C=D 切分：每个 rank/chunk GEMM 为 (M,N/D,K/D)。若后续启用 C=2D/4D/8D，必须从 lowering 结果继续导出 (M,N/C,K/D)，不能用线性缩放代替实测。
  - 2D Row-Column Swizzling：对 D=Px×Py，rank-local GEMM 为 (M,N/Py,K/Px)；本实验固定 (Px,Py)=(2,3)/(3,3)/(6,6)。
- 对 Dispatch+GEMM，1D Ring 与 2D Row-Column 不是合法的个性化 all-to-all 算法，不能套用上述 N/K 切分。最小粗粒度近似使用等效 rank GEMM (S×topk/D,I,H)；回放 fused personalized-dispatch 时额外实测 source→expert chunk (S×topk/(D×E),I,H)。参数固定为 Mixtral `(H,I,E,topk)=(4096,14336,8,2)` 和 DeepSeek-V3 routed expert `(7168,2048,256,8)`。


结果展示
折线图只包含三条相对 speedup 曲线，不比较 wafer 与 GPU 的绝对时间：
```text
native_full              = W00 / W11  # 单 die 16 核 baseline 对 16 核 intra+inter 优化
native_inter_only        = C00 / C10  # 固定 16 核 4x4x1；只改变 inter-die 串行/流式编排
gpu_inter                = G00 / G10  # GPU baseline 与 optimized 都使用 GPU 本地计算
```
`W00/W11` 均为 16 核、但分别使用 canonical 与 Exp1 派生的 adaptive intra 实现；`C00/C10` 是保持 canonical intra 的受控 inter 消融轨道，二者均不混入第四个输出指标。GPU off/on 必须使用同一个 decomposition、lookup key 和计算时长，只改变 inter-die 调度。

每个算子族的 8 个 shape 必须逐一可见：一个 shape 对应一张以 `D=6/9/36` 为横轴、三条主指标为并列柱的图。只有在显式标为 `aggregate over 4 shapes` 的补充图中，才允许对同一 `S` 下的 4 个 shape 求均值；聚合图不得代替逐 shape 结果。

同时输出一张总览折线图：横轴嵌套为“算子类型 → mesh → shape”；每个 mesh 内的 8 个 shape 按实际 `padded_flops` 升序排列。`native_full`、`native_inter_only`、`gpu_inter` 分别用不同颜色的点和线表示，且线只连接同一 mesh 内的 8 个点，不能跨 mesh 或算子类型连接。



预期结果与Insights
D 衰减对比——本实验最核心的产出，直接补上笔记 2728 行指出的"全 survey 缺失的曲线"
- GEMM+RS：r 随 D 增大以 O(1/D) 塌陷，理论收益上限 (r+1)/max(r,1) 随 D 快速跌落，预期 D=36 时已经接近 1（几乎无收益空间）。
- Dispatch+GEMM：r 随 D 增大以 O(1/√D) 塌陷，衰减慢得多，预期 D=36 时理论上限依然明显高于 1。
- 预期核心现象：D=6 时两条曲线接近甚至可能 GEMM+RS 略高（in-axis 通信在小规模下开销更小）；D=36 时两者明显分化，Dispatch+GEMM 保留的收益应显著高于 GEMM+RS。这个"从接近到分化"的形状，是全篇论文里视觉冲击力最强的一张图。
② seq_len 不变性——统一验证一个此前从未被明确检验的理论边界
- 预期：GEMM+RS 的 4 组曲线（MLP×2seq_len、Attention-O-proj×2seq_len）应该全部重合，因为两种层类型本质都是线性 GEMM，M 在 r 里同样会约掉。如果实测确实全部重合，这就把"M cancellation"的适用范围从"仅 MLP"扩展验证到了"任何非 O(S²) 结构的线性 GEMM 类融合算子"，是一个干净、可以独立成文的理论贡献。
- 预期：Dispatch+GEMM 的 seq_len 两档也应该重合（EP-style 的 GroupGEMM 部分同样是线性 GEMM）。如果 Dispatch+GEMM 意外出现 seq_len 依赖，这反而是更有价值的发现——很可能是笔记里已经提到的"动态路由/专家负载不均"这个 MoE 特有"脏"因素开始在算子级显现，值得单独深挖，不是失败结果。
③ 实验①（关闭 intra-die 优化）预期
- 核心预期：即使 intra-die 优化关闭，inter-die 融合依然应带来正向收益——证明 inter-die 这层能力不依赖 intra-die 优化的存在，两者是可独立叠加的两层贡献。
- `C00/C10` 中 Dense 两侧均为固定 16 核 `4×4×1` canonical 映射和相同的基础计算/NoC/HBM 实现；远端 payload 还须经固定 4×4 core mesh 汇聚/分发至两个 256 GB/s DTE 端口，显式计入 core→port、port service、D2D fabric、port→core 的时间与拥塞。C00/C10 不包含中间 tile HBM 写回+重读的差分，唯一改变是该相同路径的串行或分段流式编排；MoE 的 C 轨道同样固定采用 Exp1.2 基线的 16-core grouped-GEMM 映射，并在 audit 中披露。
④ 实验②（GPU 等效替换）预期
- 预期两条曲线（晶圆原生 vs GPU 等效）纵向会有整体偏移（算力/带宽比不同导致基线 r 不同），但 "Dispatch+GEMM 比 GEMM+RS 在 D 增大时保留收益更久"这条结构性结论应该在两个平台上都成立。
- 这是整组消融里差异化价值最高的一条：如果验证成立，直接证明 inter-die Swizzling 的收益是由通信拓扑/数学结构决定的结构性效应，与 die 内部究竟是晶圆 PE 阵列还是 GPU SM 无关——呼应之前写入笔记的"跨硬件平台可移植性"论证。
⑤ 专家粒度对比（粗粒度 vs 细粒度 MoE）——留作开放问题，不强行预设方向
细粒度专家 DeepSeek-V3 routed expert（E=256, topk=8）相比粗粒度 Mixtral（E=8, topk=2）在 D 增大时的收益保留是更好还是更差，存在两个相反的机理同时起作用（topk 更大→每 token 通信目标更多→Dispatch 更重；但专家更分散→负载分摊到更大 mesh 上可能更均匀），方向不确定，建议如实报告实测结果并事后归因，不要在实验设计阶段预设结论。

可以给的insights
1. 这组消融不只是防御性的"证明我们方法有用"，本身就能产出两个独立可发表的发现：D 衰减曲线对比（填补 survey 空白）+ M-cancellation 边界的统一验证（扩展了此前只在 TP-style 上验证过的理论）。
2. 实验①②合起来构成一个完整的归因链：①证明 inter-die 收益不依赖 intra-die 优化（内部独立性），②证明 inter-die 收益不依赖晶圆硬件本身（外部可移植性）——两条证据同时成立，才能把"inter-die 编排是本文的核心贡献之一"这句话坐实，缺一个都不完整。
3. 若 Dispatch+GEMM 的 seq_len 不变性被打破，不要当作噪声处理，这可能是论文里唯一能实证观察到"MoE 动态路由破坏对称性假设"的地方（写作.md 1207 行提到的理论假设，目前只有文字推测，没有实证）。