# TX8-like Die 内最终扫描参数、算法与验证

## 1. 用途与实现边界

本文是供自动化 Agent 实现扫描器的算法规范。扫描器接收上游可行 core 点，依次构造 $N\times N$ 计算 KGD、HBM/D2D 端口、计算 KGD+HBM 复合矩形和 $215\ \mathrm{mm}\times215\ \mathrm{mm}$ wafer 级二维 Mesh。

单位固定为：面积 $\mathrm{mm^2}$、长度 mm、带宽 GB/s、容量 GB/MiB、算力 GFLOP/s。所有除法产生数量时必须按本文指定的 `floor`/`ceil` 执行；浮点边界建议使用 $10^{-12}$ 相对容差。

扫描器不重复搜索 core 内部连续面积参数，而是调用核内 evaluator 或读取其可行记录。现有回归基准采用相同 evaluator 逐项重算，从而避免中间 CSV 精度误差。

## 2. 独立扫描参数

### 2.1 上游 core 离散集合

|变量|符号|集合|
|---|---|---|
|控制核数|$n_{ctrl}$|`[1, 2]`|
|router 档位|$r$|`[base, broadcast, reduce, both]`|
|单 core 注入/单链路带宽|$B$|`[64, 128, 192, 256, 384, 512]` GB/s|
|单 core SRAM 容量|$K$|`[1, 2, 3, 4]` MiB|
|单 core SRAM 聚合带宽|$B_S$|`[256, 512, 768, 1024]` GB/s|
|矩阵 PE 数|$N_{PE}$|`[1024, 4096, 8192, 12288, 16384]`|

原始 core 空间：

$$
2\times4\times6\times4\times4\times5=3840.
$$

上游 evaluator 返回至少以下字段：

```text
n_ctrl, router, B_link, K_MiB, B_S, N_PE,
A_core, core_side,
P_mat_core_GFLOPs, P_vec_core_GFLOPs
```

其中 `B_link` 对应 $B$，`core_side` 必须等于 $\sqrt{A_{core}}$。

### 2.2 Die、HBM 与端口离散集合

|变量|符号|集合|
|---|---|---|
|core mesh 阶数|$N$|`[4, 8, 16, 32]`|
|端口位置|$\pi$|`[clustered, uniform]`|
|HBM 放置边数|$e_H$|`[1, 2]`；2 表示两条相对边|
|每条 HBM 边的 stack 数|$m$|固定回归枚举 `[1, 2, 3, 4]`，再由 $m\le m_{max}$ 过滤|

每个 core 点展开：

$$
4\times2\times2\times4=64
$$

个 Die/HBM 配置。完整固定分母为：

$$
\boxed{3840\times64=245{,}760}.
$$

这一定义便于版本间回归。若未来把 $m$ 改成动态 `1..m_max`，必须同时报告动态分母，不能与 245,760 的可行率直接比较。

## 3. 固定常量

|代码名|符号|定值|说明|
|---|---|---:|---|
|`RHO_GAP`|$\rho_g$|$2/35$|FlooNoC 校准的 core 间距比例|
|`RETICLE_SHORT`|—|26 mm|曝光场短边|
|`RETICLE_LONG`|—|33 mm|曝光场长边|
|`RETICLE_AREA`|—|858 mm²|$26\times33$|
|`PHY_DEPTH`|$D_{PHY}$|1.540 mm|三类 PHY 深度最大值|
|`UNIFORM_ROUTING_RATIO`|$\eta_U$|0.03|uniform 额外回线长边比例|
|`MICROBUMP_PITCH`|$p_{\mu b}$|0.055 mm|统一 KGD microbump pitch|
|`HBM_BW`|$B_H$|819.2 GB/s|HBM3 单 stack 峰值|
|`HBM_ACTIVITY`|$\alpha_H$|0.90|HBM 目标利用系数|
|`HBM_CAPACITY_GB`|$C_H$|16 GB|单 stack 容量|
|`HBM_SHORELINE`|$L_H$|8.50 mm|单 HBM KGD 边长占用|
|`D2D_SHORELINE`|$L_D$|0.55 mm|每 HBM 边一个 D2D module 预留|
|`HBM_WIDTH`|$W_H$|8.13 mm|当前 HBM 抽象放置块宽；料号确定后替换|
|`HBM_HEIGHT`|$H_H$|4.92 mm|当前 HBM 抽象放置块高；料号确定后替换|
|`PACKAGE_GAP`|$s$|0.15 mm|HBM/计算 KGD 间距及外周留白|
|`OTHER_PORT_RATIO`|$r_O$|0.20|其他逻辑 Agent 预留|
|`D2D_PHY_PER_EDGE`|—|1|每边一个 x64 PHY module|
|`D2D_PHY_MODULE_DIR_BW`|$B_{D,module}^{\rightarrow}$|512 GB/s|x64@64 GT/s 单向上限|
|`WAFER_WIDTH`|$W_W$|215 mm|方形可用窗口宽|
|`WAFER_HEIGHT`|$H_W$|215 mm|方形可用窗口高|

来源优先级：TX8 用于 Agent 数比例；Dojo 用于 D2D 拓扑与带宽数量级；UCIe 用于 Dojo 未公开的 PHY/module 几何；JEDEC/SK hynix 用于 HBM3；FormFactor 用于 KGD HBM pad/shoreline；FlooNoC 用于 mesh 间距与二维路由；Nikon 用于曝光场。

## 4. 派生函数规范

### 4.1 `build_core(params)`

输入：$(n_{ctrl},r,B,K,B_S,N_{PE})$。

调用上游 core evaluator。若失败，返回失败原因；若成功，至少返回：

$$
S_{core}=\sqrt{A_{core}}.
$$

必须保留上游失败分类，以便区分 SRAM 带宽、计算/SRAM 比、通信占比和 core 面积等失败原因。

### 4.2 `build_compute_kgd(core, N, placement)`

$$
F_N=N+(N-1)\rho_g,
$$

$$
S_M=F_NS_{core}.
$$

若 `placement == uniform`：

$$
\Delta W_U=\eta_US_M;
$$

否则 $\Delta W_U=0$。

计算：

$$
W_C=S_M+D_{PHY}+\Delta W_U,
$$

$$
H_C=S_M,
$$

$$
A_C=W_CH_C.
$$

返回 `mesh_side, comp_die_w, comp_die_h, comp_die_area` 和光罩判定：

```text
min(W_C,H_C) <= 26
and max(W_C,H_C) <= 33
and A_C <= 858
```

### 4.3 `max_hbm_per_edge(W_C, N, B)`

边长上限：

$$
m_{max,IO}=\max\left[
0,
\left\lceil\frac{W_C-L_D+s}{L_H+s}\right\rceil-1
\right].
$$

NoC 截面与带宽上限：

$$
B_{bisec}=NB,
$$

$$
m_{max,BW}=\max\left[
0,
\left\lfloor\frac{B_{bisec}}{\alpha_HB_H}\right\rfloor
\right].
$$

合并：

$$
m_{max}=\min(m_{max,IO},m_{max,BW}).
$$

注意 shoreline 条件是严格小于。上式中的 `ceil(...)-1` 不得擅自改成 `floor(...)`，否则刚好相等的边界会被误判为可行。

### 4.4 `allocate_ports(N, B, e_H, m)`

HBM 端口：

$$
t_H^{raw}=\left\lceil\frac{m\alpha_HB_H}{B}\right\rceil,
$$

$$
t_H=\min(t_H^{raw},N-1).
$$

总端口：

$$
P_U=\lfloor(1-r_O)4N\rfloor.
$$

D2D 上限：

$$
d_{global}=\left\lfloor\frac{P_U-e_Ht_H}{4}\right\rfloor,
$$

$$
d_{edge}=N-t_H,
$$

$$
d=\min(d_{global},d_{edge}).
$$

若 $d<1$，返回不可行。不要用 `round` 代替 `ceil` 或 `floor`。

### 4.5 `port_positions(N, t_H, d, placement, is_hbm_edge)`

uniform：

```python
[floor(j*N/count) for j in range(count)]
```

clustered：

```python
left  = ceil(count/2)
right = floor(count/2)
list(range(left)) + list(range(N-right, N))
```

HBM 边不能独立生成两组位置，否则可能让 HBM 与 D2D 占用同一槽。正确顺序为：

```text
1. 用上述函数以 count=t_H 生成 HBM 位置 H。
2. remaining = sorted({0,...,N-1} - H)。
3. uniform：从 remaining 中取索引 floor(j*len(remaining)/d)，j=0..d-1。
4. clustered：从 remaining 的首尾分别取 ceil(d/2)、floor(d/2) 个。
5. 断言 H 与 HBM 边 D2D 位置的交集为空。
6. 非 HBM 边直接在完整 [0,N-1] 上以 count=d 生成 D2D 位置。
```

返回 `hbm_port_positions`、`d2d_positions_hbm_edge` 和 `d2d_positions_plain_edge`。每个数组必须严格递增、无重复、全部位于 `[0,N-1]`。

一个 D2D 边缘入口进入 mesh 后允许转到其他行/列，但不能绕开入口 NI、第一跳链路和截面。解析扫描不按“端口所在行专用”分配带宽，也不假设任意动态自适应绕行。

### 4.6 `build_composite(W_C, H_C, e_H, m)`

$$
W_{H,row}=mW_H+(m-1)s,
$$

$$
W_D=\max(W_{H,row},W_C)+2s.
$$

高度分支必须写为：

$$
H_D=
\begin{cases}
H_C+H_H+3s,&e_H=1,\\
H_C+2H_H+4s,&e_H=2.
\end{cases}
$$

$$
A_D=W_DH_D.
$$

禁止交换两个高度分支。$A_D$ 可以超过 $858\ \mathrm{mm^2}$，因为它包含多颗独立 die。

### 4.7 `performance(core, N, B, e_H, m, t_H, d)`

$$
N_{core}=N^2,
$$

$$
P_{mat,Die}=N^2P_{mat,core},
\qquad
P_{vec,Die}=N^2P_{vec,core}.
$$

$$
C_{H,total}=e_HmC_H,
$$

$$
B_{H,peak}=e_HmB_H,
$$

$$
B_{H,target}=e_Hm\alpha_HB_H,
$$

$$
B_{H,NoC}=e_H\min(m\alpha_HB_H,t_HB).
$$

$$
B_{D,edge}^{\rightarrow}=\min(dB,512),
$$

$$
B_{D,total}^{\rightarrow}=4B_{D,edge}^{\rightarrow}.
$$

必须同时输出 `HBM_peak_GBs` 与 `HBM_NoC_observable_GBs`；前者不能用于估计进入 NoC 的实际带宽。

### 4.8 `wafer(W_D, H_D, B_D_edge)`

$$
n_x=\left\lfloor\frac{215}{W_D}\right\rfloor,
\qquad
n_y=\left\lfloor\frac{215}{H_D}\right\rfloor,
$$

$$
N_{D/wafer}=n_xn_y,
$$

$$
E_W=(n_x-1)n_y+(n_y-1)n_x,
$$

$$
B_{W,x}^{\rightarrow}=n_yB_{D,edge}^{\rightarrow},
\qquad
B_{W,y}^{\rightarrow}=n_xB_{D,edge}^{\rightarrow}.
$$

若 $n_x<1$ 或 $n_y<1$，点不可行。

## 5. 推荐扫描顺序

必须按廉价且剪枝能力强的约束优先执行。

### 算法 1：主扫描

```text
输入：core 六个离散集合、N_SET、PLACEMENT_SET、HBM_EDGE_SET、HBM_M_SET、固定常量
输出：可行记录、分阶段计数、失败原因、代表点、范围统计

1. 初始化所有 stage counter、failure counter、按 N 统计表。

2. 枚举 3,840 个 core 参数组合：
   2.1 调用 build_core。
   2.2 若上游物理约束失败，记录 core failure reason，跳过。
   2.3 保存可行 core；断言 core_side^2 ≈ A_core。

3. 对每个可行 core 枚举 N=[4,8,16,32]：
   3.1 可选预剪枝：用该 N 和最小可行 core_side 判断是否必然超过光罩。
   3.2 枚举 placement=[clustered,uniform]。
   3.3 调用 build_compute_kgd。
   3.4 若计算 KGD 不满足 26×33 mm 或 858 mm²，记录 reticle failure，跳过。

4. 对每个计算 KGD 枚举 e_H=[1,2]：
   4.1 调用 max_hbm_per_edge，得到 m_max_IO、m_max_BW、m_max。
   4.2 若 m_max<1，记录 no-HBM failure，跳过全部 m。

5. 枚举 m=[1,2,3,4]：
   5.1 若 m>m_max，记录 HBM shoreline/bisection failure，跳过。
   5.2 显式复核 m*L_H+(m-1)*s+L_D < W_C。
   5.3 显式复核 N*B >= m*alpha_H*B_H。

6. 调用 allocate_ports：
   6.1 计算 t_H_raw，再截断为 t_H<=N-1。
   6.2 计算 d_global、d_edge 和 d。
   6.3 若 d<1，记录 port failure，跳过。
   6.4 断言 4*d+e_H*t_H <= floor((1-r_O)*4*N)。
   6.5 断言 HBM 边 d+t_H<=N。

7. 调用 port_positions：
   7.1 用 count=t_H 生成 HBM 位置。
   7.2 在 HBM 边的剩余槽位中生成 d 个 D2D 位置。
   7.3 在非 HBM 边的完整槽位中生成 d 个 D2D 位置。
   7.4 检查有序、无重复、边界合法性及 HBM/D2D 互斥。

8. 调用 build_composite：
   8.1 单边必须使用 H_C+H_H+3*s。
   8.2 双边必须使用 H_C+2*H_H+4*s。
   8.3 不对 A_D 施加 858 mm² 限制。

9. 调用 performance：
   9.1 输出 Die 峰值矩阵/vector 算力。
   9.2 分别输出 HBM peak、target、NoC-observable。
   9.3 输出每边和四边 D2D 单向带宽。

10. 调用 wafer：
    10.1 计算 floor(215/W_D)×floor(215/H_D)。
    10.2 若任一方向为 0，删除。
    10.3 输出 mesh 边数和两个中心 cut 带宽。

11. 保存完整记录，并更新各阶段、各 N、各 placement、各 e_H 的统计。

12. 扫描结束后：
    12.1 计算相对固定完整空间和相对 core 可行展开空间的可行率。
    12.2 输出所有连续量的 min/max。
    12.3 按确定性 tie-break 选择代表点。
    12.4 运行第 9 节全部回归断言。
```

### 算法 2：代表点选择

使用以下排序键，确保不同语言实现得到同一结果：

```text
minimum_module:
    min(module_area, -modules_per_wafer, comp_die_area, lexicographic_config)

maximum_wafer_count:
    max(modules_per_wafer, -module_area, -comp_die_area, reverse_lexicographic_config)

maximum_hbm:
    max(HBM_capacity_GB, HBM_NoC_observable_GBs, -module_area)

maximum_compute:
    max(P_mat_TFLOPs, -module_area)

reticle_edge:
    max(comp_die_area, -module_area)
```

TX8-like 点用精确标签过滤：

```text
n_ctrl=1, router=base, B=256, K=3, B_S=256, N_PE=4096,
N=8, placement=uniform, e_H=2, m=2
```

若同标签存在多个点，取 `module_area` 最小者。

## 6. 输出记录规范

每个可行记录至少输出：

|类别|字段|
|---|---|
|core 标识|`n_ctrl, router, B_link, K_MiB, B_S, N_PE, N_vec`|
|core 数值|`A_core, core_side, P_mat_core_GFLOPs, P_vec_core_GFLOPs`|
|阵列|`N, placement, factor, mesh_side`|
|计算 KGD|`comp_die_w, comp_die_h, comp_die_area`|
|HBM 上限|`B_bisec, m_max_IO, m_max_BW, m_max`|
|HBM 配置|`e_HBM, m_HBM_per_edge`|
|端口|`t_hbm_raw, t_hbm, d_d2d, hbm_port_positions, d2d_positions_hbm_edge, d2d_positions_plain_edge`|
|复合矩形|`module_w, module_h, module_area`|
|算力|`core_count, P_mat_TFLOPs, P_vec_TFLOPs`|
|HBM|`HBM_capacity_GB, HBM_peak_GBs, HBM_target_GBs, HBM_NoC_observable_GBs`|
|D2D|`D2D_edge_one_dir_GBs, D2D_four_edge_one_dir_GBs`|
|wafer|`wafer_nx, wafer_ny, modules_per_wafer, wafer_mesh_links`|
|wafer cut|`wafer_bisec_x_one_dir_GBs, wafer_bisec_y_one_dir_GBs`|

同时输出阶段计数，不能只输出最终可行率；否则无法判断模型修改影响了哪一层。

## 7. 当前固定空间扫描结果

当前基准执行完整 $245{,}760$ 点扫描，结果如下。

### 7.1 分阶段可行率

| 阶段                       |     通过点 |    阶段分母 |  条件通过率 |
| ------------------------ | ------: | ------: | -----: |
| 上游 core 物理约束             |   1,884 |   3,840 | 49.06% |
| core 可行点完全展开             | 120,576 | 245,760 | 49.06% |
| 计算 KGD 光罩约束              |  54,400 | 120,576 | 45.12% |
| HBM shoreline 与 NoC 截面带宽 |  11,862 |  54,400 | 21.81% |
| 端口约束                     |  11,862 |  11,862 |   100% |
| 可放入 $215\times215$ mm 窗口 |  11,862 |  11,862 |   100% |

相对完整固定空间：

$$
\boxed{R_{full}=\frac{11{,}862}{245{,}760}=4.83\%}.
$$

相对已通过 core 约束的展开空间：

$$
\boxed{R_{conditional}=\frac{11{,}862}{120{,}576}=9.84\%}.
$$

### 7.2 按 $N$ 的剪枝结果

|$N$|光罩约束通过率|光罩后 HBM 配置通过率|判断|
|---:|---:|---:|---|
|4|100.00%|13.56%|计算 KGD 均可曝光；小边长和低截面带宽常放不下一颗 HBM|
|8|79.83%|32.04%|当前扫描的主要可行区|
|16|0.64%|33.33%|仅最小 core 附近通过光罩|
|32|0%|—|最小 core 也超过光罩，当前常量下应预删除|

这里“HBM 配置通过率”的分母已经通过光罩，并包含 $e_H\in\{1,2\}$ 与固定 $m=1$--4 展开。

### 7.3 可行输出范围

|输出量|最小值|最大值|
|---|---:|---:|
|计算 KGD 长边 $W_C$|9.06 mm|28.31 mm|
|计算 KGD 短边 $H_C$|7.39 mm|25.99 mm|
|计算 KGD 面积 $A_C$|67.58 mm²|735.56 mm²|
|复合矩形宽度 $W_D$|9.36 mm|28.61 mm|
|复合矩形高度 $H_D$|12.76 mm|36.43 mm|
|复合矩形面积 $A_D$|120.54 mm²|1042.00 mm²|
|每 wafer 复合矩形数|35|352|
|wafer mesh 无向链路数|58|666|
|矩阵峰值算力|32.77 TFLOP/s|1572.86 TFLOP/s|
|向量峰值算力|3.29 TFLOP/s|157.33 TFLOP/s|
|HBM 容量|16 GB|96 GB|
|HBM NoC 可观察带宽|576 GB/s|4423.68 GB/s|
|每边单向 D2D 带宽|192 GB/s|512 GB/s|
|四边单向聚合 D2D 带宽|768 GB/s|2048 GB/s|

## 8. 代表性配置与回归值

### 8.1 最小复合矩形/最大 wafer 数

输入：

```text
n_ctrl=1, router=base, B=192 GB/s,
K=1 MiB, B_S=1024 GB/s, N_PE=1024,
N=4, placement=uniform, e_H=1, m=1
```

输出：

| 量             |                                               值 |
| ------------- | ----------------------------------------------: |
| $A_{core}$    |                                     3.13599 mm² |
| 计算 KGD        |          $9.14868\times7.38707$ mm，67.58188 mm² |
| 端口            |                         $t_H^{raw}=4,t_H=3,d=1$ |
| HBM/D2D 位置    | HBM 边：HBM `[0,1,2]`、D2D `[3]`；非 HBM 边 D2D `[0]` |
| 复合矩形          |        $9.44868\times12.75707$ mm，120.53739 mm² |
| wafer 阵列      |                                $22\times16=352$ |
| wafer mesh 边数 |                                             666 |
| 矩阵/vector 算力  |                        32.768 / 3.29089 TFLOP/s |
| HBM NoC 可观察带宽 |                                        576 GB/s |
| D2D           |                       192 GB/s/边/方向，四边 768 GB/s |

该点验证了 $t_H=N$ 时截断到 $N-1$，并仍保留 $d=1$。

### 8.2 TX8-like 基准点

输入：

```text
n_ctrl=1, router=base, B=256 GB/s,
K=3 MiB, B_S=256 GB/s, N_PE=4096,
N=8, placement=uniform, e_H=2, m=2
```

输出：

|量|值|
|---|---:|
|$A_{core}$|5.00381 mm²|
|计算 KGD|$20.89384\times18.79013$ mm，392.59803 mm²|
|$m_{max,IO},m_{max,BW}$|2，2|
|端口|$t_H=6,d=2$|
|HBM/D2D 位置|每条 HBM 边：HBM `[0,1,2,4,5,6]`、D2D `[3,7]`|
|复合矩形|$21.19384\times29.23013$ mm，619.49874 mm²|
|wafer 阵列|$10\times7=70$|
|wafer mesh 边数|123|
|矩阵/vector 算力|524.288 / 52.52771 TFLOP/s|
|HBM|64 GB；物理峰值 3276.8 GB/s；NoC 可观察 2949.12 GB/s|
|D2D|512 GB/s/边/方向；四边 2048 GB/s|
|wafer 两向 cut|3584 / 5120 GB/s|

该点同时卡在 HBM shoreline 和截面带宽的每边两颗上限；$dB=512$ GB/s 恰好达到单个 x64 PHY module 的上限。

### 8.3 最大 HBM 配置

输入：

```text
n_ctrl=1, router=both, B=384 GB/s,
K=2 MiB, B_S=768 GB/s, N_PE=8192,
N=8, placement=uniform, e_H=2, m=3
```

输出：

|量|值|
|---|---:|
|计算 KGD|$26.38830\times24.12456$ mm，636.60620 mm²|
|$m_{max,IO},m_{max,BW}$|3，4|
|端口|$t_H=6,d=2$；HBM 边 HBM `[0,1,2,4,5,6]`、D2D `[3,7]`|
|复合矩形|$26.68830\times34.56456$ mm，922.46942 mm²|
|HBM|6 stacks，96 GB，NoC 可观察 4423.68 GB/s|
|wafer 阵列|$8\times6=48$|
|矩阵/vector 算力|1048.576 / 104.92885 TFLOP/s|

该点由 shoreline 而非 NoC 截面限制 HBM 数。

### 8.4 最大计算配置

输入：

```text
n_ctrl=1, router=base, B=128 GB/s,
K=4 MiB, B_S=256 GB/s, N_PE=12288,
N=8, placement=clustered, e_H=1, m=1
```

输出：

|量|值|
|---|---:|
|计算 KGD|$27.39473\times25.85473$ mm，708.28312 mm²|
|端口|$t_H=6,d=2$；HBM 边 HBM `[0,1,2,5,6,7]`、D2D `[3,4]`|
|复合矩形|$27.69473\times31.22473$ mm，864.76021 mm²|
|wafer 阵列|$7\times6=42$|
|矩阵/vector 算力|1572.864 / 157.32999 TFLOP/s|
|HBM NoC 可观察带宽|737.28 GB/s|
|D2D|256 GB/s/边/方向|

### 8.5 最大计算 KGD 面积代表

输入：

```text
n_ctrl=2, router=reduce, B=384 GB/s,
K=4 MiB, B_S=1024 GB/s, N_PE=8192,
N=8, placement=uniform, e_H=1, m=1
```

输出：

|量|值|
|---|---:|
|计算 KGD|$28.30581\times25.98622$ mm，735.56085 mm²|
|端口|$t_H=2,d=5$；HBM 边 HBM `[0,4]`、D2D `[1,2,3,5,6]`|
|复合矩形|$28.60581\times31.35622$ mm，896.96989 mm²|
|wafer 阵列|$7\times6=42$|

该点接近当前扫描中的最大计算 KGD 面积，但仍同时满足短边、长边和 858 mm² 三项约束。

## 9. 自动化回归断言

实现完成后至少执行以下断言：

```text
1. raw_core_count == 3840
2. feasible_core_count == 1884
3. full_raw_count == 245760
4. core_feasible_expanded == 120576
5. compute_die_reticle_pass == 54400
6. hbm_m_feasible == 11862
7. port_feasible == 11862
8. wafer_fit == 11862
9. abs(full_feasible_ratio - 0.0482666015625) < 1e-12
10. abs(conditional_after_core_ratio - 0.09837778662420382) < 1e-12
11. every feasible point: min(W_C,H_C)<=26
12. every feasible point: max(W_C,H_C)<=33
13. every feasible point: W_C*H_C<=858
14. every feasible point: m*L_H+(m-1)*s+L_D < W_C
15. every feasible point: N*B >= m*alpha_H*B_H
16. every feasible point: 1<=m<=m_max
17. every feasible point: t_H<=N-1 and d>=1
18. every feasible point: 4*d+e_H*t_H<=floor(0.8*4*N)
19. every feasible point: d+t_H<=N on an HBM edge
20. every feasible point: HBM and D2D position sets are disjoint on an HBM edge
21. every feasible point: module_h(e_H=2)-module_h(e_H=1)=H_H+s
22. no constraint A_D<=858 exists
23. minimum module area ≈120.5373937905169 mm²
24. TX8-like module ≈21.193838260696243×29.230134233685675 mm
25. TX8-like modules_per_wafer == 70
```

若只改变 $\eta_U,\alpha_H,D_{PHY},L_H,L_D$ 等敏感性常量，应更新数值断言 5--10、23--25，但结构断言 11--22 必须保持。

## 10. 结果分析与约束复核

1. **当前主要 Die 级剪枝是光罩与 HBM 接口。** core 通过后，只有 45.12% 的展开点通过计算 KGD 光罩；其中又只有 21.81% 的固定 HBM 展开通过 shoreline 与截面带宽。
2. **$N=8$ 是主可行区。** 它在计算密度、光罩尺寸和 HBM 数之间最平衡。$N=4$ 的计算 KGD 虽全部可曝光，但常受 HBM shoreline/截面限制。
3. **$N=16$ 只应保留作边界探索。** 当前只有 0.64% 通过光罩，代表点高度依赖最小 core；任何 PHY depth、routing margin 或实现余量上调都可能清空该区域。
4. **$N=32$ 应预删除。** 在当前最小 core 和固定 PHY 条带下，光罩通过率为 0；继续枚举 HBM 与端口没有意义。
5. **端口约束当前没有继续删除 HBM 可行点。** 原因是 $t_H$ 被截断至 $N-1$，显式保证 $d\ge1$。这不意味着端口没有性能损失；HBM NoC 可观察带宽会低于目标值。
6. **uniform 的面积代价需要敏感性分析。** 3% 余量是早期估算。建议固定离散配置，分别用 2%、3%、5% 重扫并观察光罩边界点变化。
7. **D2D 端口可服务其他行/列，但入口带宽不复制。** 均匀放置改善潜在负载均衡；聚簇放置减少回线。实际持续带宽必须用 traffic simulation 或 P&R 校准。
8. **wafer 数是理想上界。** 35--352 未计供电、时钟、布线、散热、圆边和冗余，不能解释为制造良率或最终产品数量。

## 11. 推荐敏感性扫描

主离散空间回归通过后，按以下顺序单因素或小规模组合扫描：

|优先级|参数|建议值|观察量|
|---:|---|---|---|
|1|$\eta_U$|0.02、0.03、0.05|光罩通过率、$N=16$ 生存点|
|2|$\alpha_H$|0.80、0.90、0.95|$m_{max,BW}$、HBM 可行率|
|3|$D_{PHY}$|1.2、1.54、1.8 mm|计算 KGD 长边与面积|
|4|$L_H$|8.0、8.5、9.0 mm|$m_{max,IO}$|
|5|每边 D2D module 数|1、2|D2D 上限，同时按比例修改 $L_D$|
|6|$s$|0.10、0.15、0.25 mm|复合矩形和 wafer 装箱数|

禁止只增加 D2D module 带宽而不增加 shoreline；禁止只减小 HBM package 宽度而不重新校准 KGD shoreline。

## 12. 公开依据与 BibLaTeX

- [FlooNoC 物理实现](https://arxiv.org/abs/2409.17606)用于 core 间距比例与宽链路布局数量级。
- [FlooNoC 路由文档](https://pulp-platform.github.io/FlooNoC/floonoc/route_algos/)用于 XY/source/table-based 路由语义。
- [Tesla Dojo](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips_tesla_dojo_uarch.pdf)用于四边 D2D、二维路由和带宽数量级校验。
- [UCIe](https://www.uciexpress.org/specifications)用于 x64、64 GT/s PHY module 的替代建模。
- [JEDEC HBM3](https://www.jedec.org/standards-documents/docs/jesd238)与 [SK hynix HBM3](https://news.skhynix.com/en/meet-the-engineers-leading-the-worlds-first-mass-production-of-hbm3/)用于 HBM3 位宽和带宽。
- [FormFactor HBM2 KGD](https://www.formfactor.com/wp-content/uploads/SWTW_HBM2_KGD-June-2017-Final-3.pdf)用于 $55\ \mu$m 交错 pad 与 shoreline 校准。
- [Nikon](https://www.nikon.com/business/semi/lineup/)用于 $26\times33$ mm 曝光场。

```bibtex
@inproceedings{talpes2022dojo,
  author    = {Emil Talpes and Douglas Williams and Debjit Das Sarma and others},
  title     = {The Microarchitecture of Tesla's Exa-Scale Computer},
  booktitle = {2022 IEEE Hot Chips 34 Symposium},
  year      = {2022},
  doi       = {10.1109/HCS55958.2022.9895534},
  url       = {https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips_tesla_dojo_uarch.pdf}
}

@article{fischer2025floonoc,
  author  = {Tim Fischer and Michael Rogenmoser and Thomas Benz and Frank K. G{\"u}rkaynak and Luca Benini},
  title   = {{FlooNoC}: A 645-Gb/s/link 0.15-pJ/B/hop Open-Source {NoC}},
  journal = {IEEE Transactions on Very Large Scale Integration Systems},
  year    = {2025},
  doi     = {10.1109/TVLSI.2025.3527225},
  url     = {https://arxiv.org/abs/2409.17606}
}

@online{floonocRouting,
  author  = {{PULP Platform}},
  title   = {{FlooNoC} Routing Algorithms},
  url     = {https://pulp-platform.github.io/FlooNoC/floonoc/route_algos/},
  urldate = {2026-08-22}
}

@standard{ucieSpecification,
  author      = {{UCIe Consortium}},
  title       = {Universal Chiplet Interconnect Express Specification},
  institution = {UCIe Consortium},
  url         = {https://www.uciexpress.org/specifications}
}

@standard{jedecJESD238,
  author      = {{JEDEC Solid State Technology Association}},
  title       = {High Bandwidth Memory DRAM (HBM3)},
  number      = {JESD238},
  institution = {JEDEC},
  url         = {https://www.jedec.org/standards-documents/docs/jesd238}
}

@online{skhynix2022hbm3,
  author  = {{SK hynix}},
  title   = {Meet the Engineers Leading the World's First Mass-production of HBM3},
  year    = {2022},
  url     = {https://news.skhynix.com/en/meet-the-engineers-leading-the-worlds-first-mass-production-of-hbm3/}
}

@techreport{formfactor2017hbm2kgd,
  author      = {{FormFactor}},
  title       = {HBM2 KGD Test: Challenges and Solutions},
  institution = {FormFactor},
  year        = {2017},
  url         = {https://www.formfactor.com/wp-content/uploads/SWTW_HBM2_KGD-June-2017-Final-3.pdf}
}

@online{nikonLithography,
  author  = {{Nikon Corporation}},
  title   = {Semiconductor Lithography Systems Lineup},
  url     = {https://www.nikon.com/business/semi/lineup/},
  note    = {Maximum exposure field 26 mm by 33 mm}
}
```
