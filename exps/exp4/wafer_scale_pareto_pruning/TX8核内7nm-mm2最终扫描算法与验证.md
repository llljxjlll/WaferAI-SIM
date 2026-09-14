# TX8-like Core 内 7 nm 面积扫描算法、输出规范与验证

## 1. 用途

本文档把 TX8-like core 面积模型写成可直接交给 agent 实现的离散扫描算法。程序必须完成四件事：

1. 枚举独立架构参数；
2. 对每个目标带宽选择满足约束的最小通信面积；
3. 依次检查容量、带宽、面积比例和物理几何约束；
4. 输出可行点、淘汰原因、逐级可行率和回归校验值。

所有面积单位为 $\mathrm{mm^2}@7\,\mathrm{nm}$；带宽单位为 GB/s；频率单位为 GHz；位宽单位为 bit。计算过程中保留双精度数值，只在展示时四舍五入。

---

## 2. 扫描模式

### 2.1 主模式：最小通信面积规范扫描

主模式把 $C\in\{1,\ldots,6\}$ 作为通信内层候选，对每个 $(B,r)$ 只保留：

$$
C^\star=\arg\min_C\left\{A_{comm}(B,C,r):CB_{ch}\ge B\right\}.
$$

由于通信面积对 $C$ 单调增加：

$$
C^\star=\left\lceil\frac{B}{B_{ch}}\right\rceil.
$$

主模式的外层原始空间为：

$$
2\times4\times6\times4\times4\times5=3840
$$

个配置，分别对应 $n_{ctrl}$、$r$、$B$、$K$、$B_S$ 和 $N_{PE}$。

### 2.2 回归模式：完整 channel 枚举

为验证实现与已有扫描口径一致，可把 $C$ 作为第七个外层维度：

$$
2\times6\times4\times6\times4\times4\times5=23040.
$$

回归模式保留所有满足 $CB_{ch}\ge B$ 的 $C$，包括 $C>C^\star$ 的超额并发配置。它用于复现旧结果，不是“指定带宽下最小通信面积”的最终 Pareto 输入。

---

## 3. 输入参数

### 3.1 离散搜索集合

```text
n_ctrl_set = [1, 2]
channel_set = [1, 2, 3, 4, 5, 6]
router_set = [base, broadcast, reduce, both]
B_set = [64, 128, 192, 256, 384, 512]
K_set = [1, 2, 3, 4]
B_S_set = [256, 512, 768, 1024]
N_PE_set = [1024, 4096, 8192, 12288, 16384]
```

### 3.2 固定校准值

```text
g_7 = 0.08748 mm^2/MGE
u_sc = 0.65
lambda_7 = g_7 / u_sc
delta_impl = 0.10
q_core = 1.0

f_N = 1.0 GHz
eta_N = 1.0
B_ch = 128 GB/s

f_S = 1.0 GHz
eta_S = 1.0
W_b_max = 512 bit

f_C = 1.0 GHz
a_PE_7 = 4.7005e-5 mm^2/PE
a_lane_7 = 0.004669 mm^2/lane
p_lane = 1.9777 GFLOP/s/lane/GHz
r_PV = 10
```

### 3.3 查表参数

router 功能倍率：

```text
m_R[base]      = 1.000000
m_R[broadcast] = 174.5 / 135 = 1.292593
m_R[reduce]    = 214.5 / 135 = 1.588889
m_R[both]      = 259.5 / 135 = 1.922222
```

SRAM 长短边比：

```text
q_S[1 MiB] = 2.001
q_S[2 MiB] = 2.152
q_S[3 MiB] = 2.404
q_S[4 MiB] = 2.609
```

### 3.4 最小 channel 数查表

在 $B_{ch}=128$ GB/s 下：

|$B$ / GB/s|64|128|192|256|384|512|
|---:|---:|---:|---:|---:|---:|---:|
|$C^\star$|1|1|2|2|3|4|

当前目标带宽上限为 $512$ GB/s，因此规范扫描不会选择 $C=5,6$。若需要把 $5/6$ channels 纳入最小面积主结果，应把 $B$ 扩展到 $640/768$ GB/s，或改变 $B_{ch}$。

---

## 4. 派生函数

### 4.1 通信函数

```text
b(B) = B / 256
W_link(B) = ceil(8 * B / (f_N * eta_N))
C_star(B) = ceil(B / B_ch)
```

```text
A_DTE(B,C)   = lambda_7 * (0.72 + 0.15*C + 0.48*b)
A_local(B,C) = lambda_7 * (0.60 + 1.20*C*b)
A_NI(B)      = lambda_7 * (0.60 + 0.60*b)
A_link(B)    = lambda_7 * (0.60*b)
A_R(B,r)     = lambda_7 * (0.672*b*m_R[r])
```

```text
A_comm(B,C,r) = A_DTE + A_local + A_NI + A_link + A_R
A_comm_min(B,r) = A_comm(B,C_star(B),r)
```

程序应保存五个通信子项，不能只保存 $A_{comm}$。

### 4.2 控制函数

```text
A_ctrl(n_ctrl) = 4.5 * n_ctrl * lambda_7
```

### 4.3 计算函数

```text
P_PE(N_PE) = 2 * N_PE * f_C
N_vec(N_PE) = ceil(P_PE / (r_PV * p_lane * f_C))
A_PE(N_PE) = N_PE * a_PE_7
A_vec(N_PE) = N_vec * a_lane_7
A_comp(N_PE) = A_PE + A_vec
```

当 $f_C$ 同时出现在 $P_{PE}$ 和分母中时可约掉，但实现中保留完整式更便于以后扫描频率。

### 4.4 SRAM 面积、bank 和尺寸函数

```text
A_SRAM_cap(K) = g_7 * (8/3) * K
A_SRAM_bw(B_S) = g_7 * B_S / 128
A_SRAM(K,B_S) = A_SRAM_cap + A_SRAM_bw

W_S_port(B_S) = ceil(8 * B_S / (f_S * eta_S))
n_b(B_S) = max(16, ceil(W_S_port / W_b_max))
bank_width = W_S_port / n_b

L_S = sqrt(A_SRAM * q_S[K])
T_S = sqrt(A_SRAM / q_S[K])
```

$L_S$、$T_S$ 是 SRAM banks 组成等效基准矩形时的长边和短边，仅用于 NVSim 形状校准；最终统一把相同 banks 重排为 U 型。

### 4.5 Core 面积和尺寸函数

```text
A_alloc = A_comm_min + A_ctrl + A_comp + A_SRAM
A_impl = delta_impl * A_alloc
A_core = A_alloc + A_impl

layout = U
W_core = sqrt(A_core)
L_core = sqrt(A_core)
q_core = 1.0
```

若运行回归模式，把 $A_{comm}^{min}$ 替换为当前枚举的 $A_{comm}(B,C,r)$。

U 型将部分 SRAM banks 从底边移到左右侧臂，面积与带宽模型不变，并使中央计算区、顶部通信带适配固定 $1{:}1$ core 外框。

---

## 5. 过滤条件与顺序

过滤顺序应先执行计算成本低、淘汰率高的条件，再执行面积和几何计算。

### 步骤 1：选择通信配置

主模式直接计算：

$$
C=C^\star(B).
$$

若 $C^\star>6$，记录 `REJECT_DTE_CHANNEL_LIMIT`。

回归模式枚举 $C$，若：

$$
CB_{ch}<B,
$$

记录 `REJECT_DTE_BW`。

### 步骤 2：检查 SRAM 基础带宽

若：

$$
B_S<B,
$$

记录 `REJECT_SRAM_BW`。

### 步骤 3：检查 SRAM 容量

若：

$$
K<1\ \mathrm{MiB},
$$

记录 `REJECT_SRAM_CAPACITY`。当前 $K$ 网格从 $1$ MiB 起，因此该过滤用于防止以后扩展输入时误收更小容量。

### 步骤 4：推导 SRAM bank 和宏形状

依次计算 $W_{S,port}$、$n_b$、bank width、$A_{SRAM}$、$q_S$、$L_S$ 和 $T_S$。

以下任一条件不满足则淘汰：

$$
n_b\ge16,
\qquad
\frac{W_{S,port}}{n_b}\le512\ \mathrm{bit},
\qquad
q_S\le3.
$$

拒绝码分别为 `REJECT_BANK_COUNT`、`REJECT_BANK_WIDTH` 和 `REJECT_SRAM_ASPECT`。

### 步骤 5：计算通信面积

分别计算 $A_{DTE}$、$A_{local}$、$A_{NI}$、$A_{link}$、$A_R$，再求 $A_{comm}$。主模式必须验证：

$$
A_{comm}(B,C^\star,r)
=\min_{C\in\mathcal C(B)}A_{comm}(B,C,r).
$$

若数值实现未满足，应报错而不是静默继续。

### 步骤 6：计算控制和计算面积

计算 $A_{ctrl}$、$P_{PE}$、$N_{vec}$、$A_{PE}$、$A_{vec}$ 和 $A_{comp}$。

先检查：

$$
0.5\le A_{comp}\le8.6\ \mathrm{mm^2}.
$$

注意：新增 $N_{PE}=1024$ 档的未舍入值为 $0.53370912\,\mathrm{mm^2}$，满足 $0.5\,\mathrm{mm^2}$ 下界。

再检查：

$$
\frac{A_{comp}}{A_{SRAM}}\le6.
$$

拒绝码为 `REJECT_COMPUTE_RANGE` 或 `REJECT_COMPUTE_SRAM_RATIO`。

### 步骤 7：计算 core 面积

计算 $A_{alloc}$、$A_{impl}$ 和 $A_{core}$，依次检查：

$$
\frac{A_{comm}}{A_{core}}\le0.25,
$$

$$
A_{comm}\le4.0375\ \mathrm{mm^2},
$$

$$
A_{core}\le13\ \mathrm{mm^2}.
$$

拒绝码分别为 `REJECT_COMM_SHARE`、`REJECT_COMM_AREA` 和 `REJECT_CORE_AREA`。

### 步骤 8：计算 core 几何

SRAM 统一采用 U 型，core 几何固定为：

$$
W_{core}=L_{core}=\sqrt{A_{core}},
\qquad
q_{core}=1,
\qquad
M_{SRAM}=\mathrm{U}.
$$

该步骤不淘汰面积可行点。所有点必须设置 `layout=U`；解析方形外框仍需等待 macro placement 和布线验证。

### 步骤 9：输出解析可行点

通过步骤 1–8 的点标记：

```text
modeled_feasible = true
compiler_verified = false
route_verified = false
timing_verified = false
```

只有目标 SRAM compiler、early global route 和目标 RTL 综合通过后，后三项才能改为 `true`。

---

## 6. 推荐伪代码

```text
results = []
reject_counters = map(default=0)

for n_ctrl in n_ctrl_set:
  for r in router_set:
    for B in B_set:

      candidate_channels = []
      for C in channel_set:
        if C * B_ch >= B:
          candidate_channels.append(C)

      if candidate_channels is empty:
        reject_counters[REJECT_DTE_CHANNEL_LIMIT] += 1
        continue

      # A_comm is monotonic in C for this model.
      C_star = min(candidate_channels)
      comm = evaluate_comm(B, C_star, r)

      # Optional assertion: explicitly compare all feasible C values.
      assert comm.total == min(evaluate_comm(B,C,r).total
                               for C in candidate_channels)

      for K in K_set:
        for B_S in B_S_set:
          if B_S < B:
            reject(REJECT_SRAM_BW)
            continue
          if K < 1:
            reject(REJECT_SRAM_CAPACITY)
            continue

          sram = evaluate_sram(K, B_S)
          if sram.n_b < 16:
            reject(REJECT_BANK_COUNT)
            continue
          if sram.bank_width > 512:
            reject(REJECT_BANK_WIDTH)
            continue
          if sram.q_S > 3:
            reject(REJECT_SRAM_ASPECT)
            continue

          for N_PE in N_PE_set:
            comp = evaluate_compute(N_PE)
            ctrl = evaluate_control(n_ctrl)

            if comp.area < 0.5 or comp.area > 8.6:
              reject(REJECT_COMPUTE_RANGE)
              continue
            if comp.area / sram.area > 6:
              reject(REJECT_COMPUTE_SRAM_RATIO)
              continue

            A_alloc = comm.total + ctrl.area + comp.area + sram.area
            A_core = (1 + delta_impl) * A_alloc

            if comm.total / A_core > 0.25:
              reject(REJECT_COMM_SHARE)
              continue
            if comm.total > 4.0375:
              reject(REJECT_COMM_AREA)
              continue
            if A_core > 13:
              reject(REJECT_CORE_AREA)
              continue

            layout = U
            W_core = sqrt(A_core)
            L_core = sqrt(A_core)
            q_core = 1.0

            results.append(full_record(...))

write_results(results)
write_stage_counts()
write_validation_summary()
```

可执行复现脚本：[13-core_scan.py](13-core_scan.py)。

---

## 7. 输出字段

每个保留点至少输出以下字段：

|类别|字段|
|---|---|
|输入配置|`n_ctrl, r, B, K, B_S, N_PE`|
|通信选择|`C_star, B_ch, W_link, m_R`|
|通信面积|`A_DTE, A_local, A_NI, A_link, A_R, A_comm`|
|控制|`A_ctrl`|
|计算|`N_vec, P_PE, A_PE, A_vec, A_comp`|
|SRAM|`W_S_port, n_b, bank_width, q_S, A_SRAM_cap, A_SRAM_bw, A_SRAM, L_S, T_S`|
|Core|`A_alloc, A_impl, A_core, layout, W_core, L_core, q_core`|
|状态|`modeled_feasible, compiler_verified, route_verified, timing_verified, fragile_flags`|

建议同时保存未通过点的第一个拒绝码，便于复核可行率变化。

---

## 8. 回归模式验证结果

完整枚举 $C=1$–$6$，并采用新容量、计算和方形布局约束，逐级结果为：

| 过滤阶段                        |    剩余点 | 占 23,040 点 | 相对上一阶段保留率 |
| --------------------------- | -----: | ---------: | --------: |
| 原始空间                        | 23,040 |    100.00% |         — |
| $CB_{ch}\ge B$              | 18,560 |     80.56% |    80.56% |
| $B_S\ge B$                  | 17,440 |     75.69% |    93.97% |
| $K\ge1$ MiB                 | 17,440 |     75.69% |   100.00% |
| bank 数、位宽和 $q_S$            | 17,440 |     75.69% |   100.00% |
| $0.5\le A_{comp}\le8.6$ mm² | 17,440 |     75.69% |   100.00% |
| $A_{comp}/A_{SRAM}\le6$     | 11,912 |     51.70% |    68.30% |
| $A_{comm}/A_{core}\le0.25$  |  8,892 |     38.59% |    74.65% |
| $A_{comm}\le4.0375$ mm²     |  8,892 |     38.59% |   100.00% |
| $A_{core}\le13$ mm²         |  8,564 | **37.17%** |    96.31% |
| U 型 SRAM、$q_{core}=1$       |  8,564 | **37.17%** |   100.00% |

回归断言为：

$$
\boxed{8564/23040=37.17\%},
\qquad
\boxed{N_U=8564,\ N_{rect}=0}.
$$

方形尺寸映射不改变面积可行率；全部 U 型点仍需验证侧臂 SRAM、中央逻辑和顶部通信带能否完成无重叠布局。

---

## 9. 主模式扫描结果

主模式对每个外层配置只保留 $C^\star$。逐级结果为：

|过滤阶段|剩余点|占 3,840 点|相对上一阶段保留率|
|---|---:|---:|---:|
|外层原始空间|3,840|100.00%|—|
|$B_S\ge B$|3,520|91.67%|91.67%|
|$K\ge1$ MiB|3,520|91.67%|100.00%|
|bank 数、位宽和 $q_S$|3,520|91.67%|100.00%|
|$0.5\le A_{comp}\le8.6$ mm²|3,520|91.67%|100.00%|
|$A_{comp}/A_{SRAM}\le6$|2,416|62.92%|68.64%|
|$A_{comm}/A_{core}\le0.25$|1,936|50.42%|80.13%|
|$A_{comm}\le4.0375$ mm²|1,936|50.42%|100.00%|
|$A_{core}\le13$ mm²|1,884|49.06%|97.31%|
|U 型 SRAM、$q_{core}=1$|1,884|**49.06%**|100.00%|

最终解析可行率为：

$$
\boxed{1884/3840=49.06\%}.
$$

容量下界下降和新增小计算档扩大了可行域；统一 U 型方形尺寸映射本身不淘汰面积可行点。

---

## 10. 主模式可行点分布

### 10.1 面积与尺寸范围

|量|最小值|最大值|
|---|---:|---:|
|$A_{comm}^{min}$|0.3981 mm²|2.4311 mm²|
|$A_{ctrl}$|0.6056 mm²|1.2113 mm²|
|$A_{comp}$|0.5337 mm²|8.5067 mm²|
|$A_{SRAM}$|0.4082 mm²|1.6330 mm²|
|$A_{core}$|2.1402 mm²|12.9468 mm²|
|$W_{core}=L_{core}$|1.4630 mm|3.5982 mm|
|最终 $q_{core}$|1.0000|1.0000|

### 10.2 离散参数分布

|参数|可行点数|
|---|---|
|$N_{PE}=1024/4096/8192/12288/16384$|404 / 558 / 558 / 320 / 44|
|$K=1/2/3/4$ MiB|300 / 413 / 513 / 658|
|$B=64/128/192/256/384/512$ GB/s|432 / 428 / 392 / 348 / 182 / 102|
|base/broadcast/reduce/both|483 / 476 / 467 / 458|
|U 型 SRAM|1884|

解释：

- $1$ MiB 和 $N_{PE}=1024$ 新增了小面积点，但计算/SRAM 比例与通信占比约束仍会删除失衡组合；
- 四种容量均统一采用 U 型，$L_S/T_S$ 只作为 bank 宏形状校准量；
- router 四档可行点数接近，说明 router 功能增量不是当前空间的首要淘汰来源；
- $512$ GB/s 有 $102$ 个点，主要限制来自局部总线面积、通信占比和总面积。

---

## 11. 代表性配置

### 11.1 最小可行点

```text
n_ctrl=1, r=base, B=64, C*=1,
K=1 MiB, B_S=256 GB/s, N_PE=1024, N_vec=104
```

|量|结果|
|---|---:|
|$A_{DTE}$|0.1332 mm²|
|$A_{local}$|0.1211 mm²|
|$A_{NI}$|0.1009 mm²|
|$A_{link}$|0.0202 mm²|
|$A_R$|0.0226 mm²|
|$A_{comm}$|0.3981 mm²|
|$A_{ctrl}$|0.6056 mm²|
|$A_{comp}$|0.5337 mm²|
|$A_{SRAM}$|0.4082 mm²|
|$A_{core}$|2.1402 mm²|
|SRAM 布局|U 型|
|$W_{core}\times L_{core}$|$1.4630\times1.4630$ mm|
|$q_{core}$|1.0000|

### 11.2 TX8-like 基准点

```text
n_ctrl=1, r=base, B=256, C*=2,
K=3 MiB, B_S=256 GB/s, N_PE=4096, N_vec=415
```

|量|结果|
|---|---:|
|$A_{DTE}$|0.2019 mm²|
|$A_{local}$|0.4038 mm²|
|$A_{NI}$|0.1615 mm²|
|$A_{link}$|0.0808 mm²|
|$A_R$|0.0904 mm²|
|$A_{comm}$|0.9383 mm²|
|$A_{ctrl}$|0.6056 mm²|
|$A_{comp}$|2.1302 mm²|
|$A_{SRAM}$|0.8748 mm²|
|$A_{core}$|5.0038 mm²|
|SRAM 布局|U 型|
|$W_{core}\times L_{core}$|$2.2369\times2.2369$ mm|
|$q_{core}$|1.0000|

在不含 $10\%$ 实现余量的 $A_{alloc}$ 中：

|类别|比例|FlooNoC 对照|
|---|---:|---:|
|计算+控制|60.1%|compute cores 约63%|
|SRAM|19.2%|SPM 约24%|
|通信|20.6%|NoC+xbar+DMA 约12.2%|

通信比例仍高于 FlooNoC，主要来自 TX8 的局部宽通路和较保守 DTE/NI 系数；但总量低于 $25\%$ guard，面积分布可作为合理的早期基准。

### 11.3 最小面积的 512 GB/s 可行点

```text
n_ctrl=1, r=base, B=512, C*=4,
K=2 MiB, B_S=1024 GB/s, N_PE=8192, N_vec=829
```

|量|结果|
|---|---:|
|$A_{comm}$|2.2643 mm²|
|$A_{ctrl}$|0.6056 mm²|
|$A_{comp}$|4.2557 mm²|
|$A_{SRAM}$|1.1664 mm²|
|$A_{core}$|9.1211 mm²|
|SRAM 布局|U 型|
|$W_{core}\times L_{core}$|$3.0201\times3.0201$ mm|
|$q_{core}$|1.0000|

该点解析可行，但必须优先检查 $4096$-bit NoC payload、$8192$-bit SRAM 聚合端口、顶部边界轨道和长线流水。

### 11.4 最大面积可行点

```text
n_ctrl=2, r=both, B=64, C*=1,
K=4 MiB, B_S=1024 GB/s, N_PE=16384, N_vec=1657
```

|量|结果|
|---|---:|
|$A_{comm}$|0.4190 mm²|
|$A_{ctrl}$|1.2113 mm²|
|$A_{comp}$|8.5067 mm²|
|$A_{SRAM}$|1.6330 mm²|
|$A_{core}$|12.9468 mm²|
|SRAM 布局|U 型|
|$W_{core}\times L_{core}$|$3.5982\times3.5982$ mm|
|$q_{core}$|1.0000|

该点接近 core 面积上限且采用 U 型布局，应标记为 `fragile_area=true`、`fragile_u_layout=true`。

---

## 12. 旧代表点的回归与新判定

以下两个点用于验证 agent 是否正确区分“旧完整 channel 扫描”和“新最小通信面积规范扫描”。

|旧配置|旧模型结果|新模型判定|
|---|---|---|
|$n_{ctrl}=1,C=6,r=both,B=512,K=4,B_S=1024,N_{PE}=12288$|$A_{core}=12.9109$ mm²，面积约束内|主模式拒绝：$C=6>C^\star=4$；回归模式采用 U 型方形 core|
|$n_{ctrl}=2,C=5,r=base,B=64,K=4,B_S=768,N_{PE}=16384$|$A_{core}=12.9979$ mm²，面积上边界|主模式拒绝：$C=5>C^\star=1$；回归模式采用 U 型方形 core|

程序在回归模式中应重现上述面积；在主模式中不应把它们写入最终可行集。

---

## 13. Fragile 标记和高精度复核顺序

解析可行不等于可签核。建议对距离边界不足 $10\%$ 的点增加标记：

```text
fragile_area       = A_core > 0.9 * 13
fragile_comm_share = A_comm/A_core > 0.9 * 0.25
fragile_balance    = A_comp/A_SRAM > 0.9 * 6
fragile_u_layout   = layout == U
fragile_high_bw    = B >= 384
fragile_wide_bank  = bank_width >= 0.9 * 512
```

复核顺序：

1. 用目标 7 nm SRAM compiler 校准 $A_{SRAM}$、bank 长宽和 $t_{cycle,S}\le1$ ns；
2. 对底边加左右侧臂的 U 型 $16$-bank SRAM、顶部通信带和中部计算块做 macro placement；
3. 对顶部 $W_{link}$ 边界 pins、可用高层金属 tracks 和 shielding 做 early global route；
4. 对长度超过约 $1$ mm 的 DTE—SRAM、PE—SRAM 和跨 tile link 插入/评估 repeater、pipeline；
5. 用 BF16/FP16 PE RTL、目标 vector RTL、DTE/NI/router RTL 替换跨节点面积缩放；
6. 最后运行 workload mapping、NoC 仿真、功耗和热约束，只从全部通过的点中提取 Pareto 前沿。

---

## 14. 自动化测试断言

agent 实现完成后至少执行以下断言：

```text
assert full_scan.raw_count == 23040
assert full_scan.area_feasible_count == 8564
assert full_scan.area_feasible_rate ~= 0.3717
assert full_scan.geometry_feasible_count == 8564
assert full_scan.rect_layout_count == 0
assert full_scan.u_layout_count == 8564

assert canonical_scan.raw_count == 3840
assert canonical_scan.pre_geometry_count == 1884
assert canonical_scan.final_count == 1884
assert canonical_scan.final_rate ~= 0.4906
assert canonical_scan.rect_layout_count == 0
assert canonical_scan.u_layout_count == 1884

assert C_star(64)  == 1
assert C_star(128) == 1
assert C_star(192) == 2
assert C_star(256) == 2
assert C_star(384) == 3
assert C_star(512) == 4

assert baseline.A_core ~= 5.0038
assert baseline.layout == U
assert baseline.W_core ~= 2.2369
assert baseline.L_core ~= 2.2369
assert baseline.q_core ~= 1.0000
```

面积断言建议容差为 $10^{-4}\,\mathrm{mm^2}$，尺寸断言建议容差为 $10^{-4}\,\mathrm{mm}$。

---

## 15. 参考资料

扫描中的定值与约束来源包括：TX8 architecture/specification and component-area materials；[ASAP7](https://github.com/The-OpenROAD-Project/asap7)；[FlooNoC](https://arxiv.org/abs/2409.17606)；[Collective-Capable NoC](https://arxiv.org/abs/2603.26438)；[NVSim](https://doi.org/10.1109/TCAD.2012.2185930)；[AraXL](https://arxiv.org/abs/2501.10301)；[Tempus Core](https://arxiv.org/abs/2412.19002)；[Tesla Dojo](https://doi.org/10.1109/MM.2023.3258906)；[Titan](https://doi.org/10.1145/3695053.3731016)；[Fovea](https://arxiv.org/abs/2608.03285)。

```bibtex
@misc{tx8_architecture, title={TX8 Architecture and Hardware Modeling Materials}, year={2026}, note={Project-provided architecture, bandwidth, and component-area references}}
@article{clark2016asap7, author={Clark, Lawrence T. and others}, title={ASAP7: A 7-nm FinFET Predictive Process Design Kit}, journaltitle={Microelectronics Journal}, volume={53}, pages={105--115}, year={2016}, doi={10.1016/j.mejo.2016.04.006}}
@article{fischer2025floonoc, author={Fischer, Tim and others}, title={FlooNoC: A 645-Gbps/link 0.15-pJ/B/hop Open-Source NoC with Wide Physical Links and End-to-End AXI4 Parallel Multi-Stream Support}, journaltitle={IEEE Transactions on Very Large Scale Integration Systems}, year={2025}, url={https://arxiv.org/abs/2409.17606}}
@inproceedings{colagrande2026collective, author={Colagrande, Luca and others}, title={A Lightweight High-Throughput Collective-Capable NoC for Large-Scale ML Accelerators}, booktitle={MLSys}, year={2026}, url={https://arxiv.org/abs/2603.26438}}
@article{dong2012nvsim, author={Dong, Xiangyu and others}, title={NVSim: A Circuit-Level Performance, Energy, and Area Model for Emerging Nonvolatile Memory}, journaltitle={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, volume={31}, number={7}, pages={994--1007}, year={2012}, doi={10.1109/TCAD.2012.2185930}}
@article{purayil2025araxl, author={Purayil, Navaneeth Kunhi and Perotti, Matteo and Fischer, Tim and Benini, Luca}, title={AraXL: A Physically Scalable, Ultra-Wide RISC-V Vector Processor Design for Fast and Efficient Computation on Long Vectors}, year={2025}, url={https://arxiv.org/abs/2501.10301}}
@article{vellaisamy2024tempus, author={Vellaisamy, Prabhu and others}, title={Tempus Core: Area-Power Efficient Temporal-Unary Convolution Core for Low-Precision Edge DLAs}, year={2024}, url={https://arxiv.org/abs/2412.19002}}
@article{talpes2023dojo, author={Talpes, Emil and others}, title={The Microarchitecture of DOJO, Tesla's Exa-Scale Computer}, journaltitle={IEEE Micro}, volume={43}, number={3}, pages={31--39}, year={2023}, doi={10.1109/MM.2023.3258906}}
@inproceedings{yu2025titan, author={Yu, Xingmao and Jiang, Dingcheng and Deng, Jinyi and Liu, Jingyao and Li, Chao and Yin, Shouyi and Hu, Yang}, title={Cramming a Data Center into One Cabinet, a Co-Exploration of Computing and Hardware Architecture of Waferscale Chip}, booktitle={ISCA}, pages={631--645}, year={2025}, doi={10.1145/3695053.3731016}}
@article{li2026fovea, author={Li, Jinxi and Wang, Huizheng and Deng, Jinyi and Hu, Yang and Yin, Shouyi}, title={Fovea: Physical-Implication-Aware Wafer-Scale DSE with Decision-Domain-Guided Cross-Fidelity Refinement}, year={2026}, url={https://arxiv.org/abs/2608.03285}}
```
