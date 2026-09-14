评估GEMM+Combine/Dispatch+GEMM算子的情况

## 参数选择

### 两个模型 
DS-V3
Mixtral-8*7B

### Mesh阵列情况
EP Mesh阵列形状：2*2

D2D 带宽与路由计算：
1. 每条有向 D2D link 的单向带宽为 1 TB/s；反方向具有独立的 1 TB/s；
2. 多条 256 GB/s NoC link 聚合连接 D2D，至少 4 条即可基本饱和一个方向；
3. 紧凑成员为 (0,0)(0,1)(1,0)(1,1)，瓶颈链路只有本 EP group 的 1 条流；
4. 非紧凑成员为 (0,0)(0,3)(3,0)(3,3)，最拥塞有向链路上最多有来自 3 个 EP group
   的 3 条流，因此单 group 的有效链路份额为 `1 TB/s / 3`；
5. 两者均按 XY route 逐有向链路累计本 group 流量，通信吞吐时间取
   `max(max_link_bytes/B_link_effective, bytes_per_die/B_inject)`。hop 只进入链路负载与
   packet/router latency，不把完整 payload 无条件乘 hop。

### Die 计算能力

- tensor GEMM：2000 TFLOP/s/die，即 125 TFLOP/s/core（16 cores）；
- vector：60 TFLOP/s/die，即 3.75 TFLOP/s/core；
- Dispatch SwiGLU vector ops：`4 × runtime_M × runtime_N`；
- Combine weighting/reduction vector ops：`rank_tokens × H × (2×top_k-1)`；
- Combine latency 同时受 vector throughput 和 local NoC contributor traffic 限制。

### Die 内 NoC

- 物理拓扑为 4×4 cores，每条有向 NoC link 为 256 GB/s；
- A/B broadcast、Split-K reduction 和 Combine contributor reduction 均按 XY route
  累计逐有向链路流量，时间取最忙链路 bytes/256 GB/s；
- baseline 使用固定 `4×4×1` 和 row-major `m-n-k` core mapping；optimized 枚举合法
  `PM×PN×PK=16` 及 m/n/k 物理维度排列，以最小化关键链路时间；
- baseline/optimized 的 tensor 与 local NoC 均对称应用各自 schedule efficiency。

### seq_len选择
对齐exp1-1的情况

| 档位 |       | 理由                                                         |
| ---- | ----- | ------------------------------------------------------------ |
| 短   | 2048  | 对应 GPT-3/LLaMA-2 原生训练长度、典型交互式短对话场景；在大 （如 ）下每跳搬运量最小，最容易暴露 DTE 固定开销——预期这一档的 attainment rate 明显低于理论曲线，用来验证"小  会打折"这个此前建立的解释。 |
| 长   | 32768 | 对应当前主流长上下文部署档位（LLaMA-3.1/Qwen2 等原生或扩展支持到 32K+）； 已经进入带宽饱和区，DTE 开销可忽略，预期这一档的 attainment rate 应该紧贴理论曲线——作为"公式在理想条件下确实成立"的干净验证点，同时顺带压测 SRAM 分块在大 tile 下是否还稳定。 |

短：2304
长：36864

## 理论上限计算

| S         | 总token数                |
| --------- | ------------------------ |
| k         | topk专家                 |
| H         | hidden_size              |
| I         | expert intermediate_size |
| D=P_x*P_y | EP mesh 大小             |
| R_die     | 单 die 算力              |
| B_link    | 每条有向 D2D link 单向带宽（1 TB/s） |

两个算子理论上限一样：
T_comp = 2*S*k*H*I/(D*R_die)
对 balanced all-to-all 进行 XY route，得到每条有向链路负载 L_e：
T_comm = max(max_e(L_e/B_link), bytes_per_die/B_inject)
T_floor = max(T_comp, T_comm)

## 结果展示

两个图对应2个op

（顺序由外向内）
每图8bars =2 mesh_configs * 2 models * 2 seq_len

mesh_configs只分紧凑/非紧凑情况，很像GPU通信不跨域/跨域场景

