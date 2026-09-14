> 评估GEMM+Collective算子的情况

# 固定硬件配置

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

# exp1-1

评估GEMM+RS/AG+GEMM算子的情况

## 参数选择

### 模型（决定矩阵大小中的N,K）
选六种模型：

|                   | 模型                      | H     | I     | n_head  | n_kv_head | d_head | I/H  | (QKV) | (O)   |
| ----------------- | ------------------------- | ----- | ----- | ------- | --------- | ------ | ---- | ----- | ----- |
| Dense MHA·小      | LLaMA-2-7B                | 4096  | 11008 | 32      | 32        | 128    | 2.69 | 12288 | 4096  |
| Dense MHA·大      | GPT-3-175B                | 12288 | 49152 | 96      | 96        | 128    | 4    | 36864 | 12288 |
| Dense GQA·小      | LLaMA-3-8B                | 4096  | 14336 | 32      | 8         | 128    | 3.5  | 6144  | 4096  |
| Dense GQA·大      | LLaMA-3.1-405B            | 16384 | 53248 | 128     | 8         | 128    | 3.25 | 18432 | 16384 |
| MoE·小（粗粒度）  | Mixtral-8×7B（单专家）    | 4096  | 14336 | 32      | 8         | 128    | 3.5  | 6144  | 4096  |
| MoE·大（细粒度）⚠️ | DeepSeek-V3（单路由专家） | 7168  | 2048  | — MLA — | —         | —      | 0.29 | n/a   | n/a   |

获取其在attention, MLP层产生的GEMM大小：
- MLP层

|                  | AG+GEMM（up/gate-proj）                                      | GEMM+RS（down-proj）                         |
| ---------------- | ------------------------------------------------------------ | -------------------------------------------- |
| M                | S（AllGather 补全的就是这一维）                              |                                              |
| K                | H（hidden_size，全量，不切）                                 | （intermediate_size 按 TP 切分后的本地份额） |
| N                | I/D（intermediate_size 按 TP 切分后的本地份额）              | （hidden_size，全量，不切）                  |
| 决定它的模型参数 | H = hidden_size（架构固定）；I = intermediate_size（架构固定，SwiGLU 约 I=2.67H，普通 MLP 约 I=4H） | 同左                                         |
| 决定M的参数      | batch_size、seq_len（部署/工作负载参数，非架构参数）         | 同左                                         |

- Attention层

|                  | AG+GEMM（QKV-proj）                                          | GEMM+RS（O-proj）                                            |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| M                | S                                                            | S                                                            |
| K                | H                                                            | (n_head+2n_kv_head)*d_head/D，若 MHA（n_kv_head=n_head）则等于 3n_head*d_head/D |
| N                | (n_head+2n_kv_head)*d_head/D                                 | H                                                            |
| 决定它的模型参数 | n_head、d_head（架构固定， n_head*d_head通常约等于 H，但 GQA/MQA 下 K,V 头数 n_kv_head < n_head，使 QKV 输出维小于 3H） | 同左                                                         |


由此可以得到AG+GEMM/GEMM+RS算子在六个模型的两层的算子计算量、通信量大小情况（但是Deepseek-v3因为采用MLA所以不能计算attn层的大小）

### Mesh阵列大小

小Mesh：1*4, 2*3, 3*3

大Mesh（占满整个wafer）：6*6

### seq_len选择（决定矩阵中的M）

对于小Mesh情况：

| 档位 |       | 理由                                                         |
| ---- | ----- | ------------------------------------------------------------ |
| 短   | 2048  | 对应 GPT-3/LLaMA-2 原生训练长度、典型交互式短对话场景；在大 （如 ）下每跳搬运量最小，最容易暴露 DTE 固定开销——预期这一档的 attainment rate 明显低于理论曲线，用来验证"小  会打折"这个此前建立的解释。 |
| 长   | 32768 | 对应当前主流长上下文部署档位（LLaMA-3.1/Qwen2 等原生或扩展支持到 32K+）； 已经进入带宽饱和区，DTE 开销可忽略，预期这一档的 attainment rate 应该紧贴理论曲线——作为"公式在理想条件下确实成立"的干净验证点，同时顺带压测 SRAM 分块在大 tile 下是否还稳定。 |

换算成6*6卡
短：2304
长：36864

对于大Mesh情况（验证强/弱扩展）：

| **档位**   | **值**     |
| ---------- | ---------- |
| **强扩展** | **36864**  |
| **弱扩展** | **147456** |

汇总：

|        | seq_len1 | seq_len2 |
| ------ | -------- | -------- |
| 小mesh | 2048     | 32768    |
| 大mesh | 36864    | 147456   |

### 汇总矩阵大小

外层到内层按(Attn/MLP层) -> 模型（按MHA小大/GQA小大/MoE小大）-> seq_len的顺序


## 理论上限计算

只与矩阵的MNK大小有关
T_theo = max(T_comm, T_comp)

Mesh阵列 D=Px*Py

| 算子    | 编排方式                      | T_comp（计算耗时）        | T_comm（通信耗时）                         | T_floor = max(T_comp, T_comm) |
| ------- | ----------------------------- | ------------------------- | ------------------------------------------ | ----------------------------- |
| AG+GEMM | 1D Ring Swizzling             | 2×M×(N/D)×K / R_die       | (D-1)/D × M×K×s / B_eff                    | max(左两项)                   |
| AG+GEMM | 2D Row-Column Swizzling       | 2×(M/Px)×(N/Py)×K / R_die | max[ M×K×s/(Px×B_eff) , K×N×s/(Py×B_eff) ] | max(左两项)                   |
| GEMM+RS | 1D Ring Swizzling             | 2×M×N×(K/D) / R_die       | (D-1)/D × M×N×s / B_eff                    | max(左两项)                   |
| GEMM+RS | 2D Row-Column Swizzling ⚠️类推 | 2×M×(N/Py)×(K/Px) / R_die | (Px-1)/Px × M×(N/Py)×s / B_eff             | max(左两项)                   |


对于每种算子，理论上限 = min(T_theo_1D, T_theo_2D)

## 结果展示

直方图：每个mesh大小+op对应一个直方图（8个）

每个直方图有22根bars：6模型*2layer(Attn/MLP)*2seq_len （DS-V3没有attn bar）

每根bar：理论上限时间、重叠优化时间、无重叠时间（集合通信时间+计算时间）分别取倒数，变成性能

