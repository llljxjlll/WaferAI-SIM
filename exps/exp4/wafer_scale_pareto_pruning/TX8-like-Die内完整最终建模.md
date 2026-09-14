# TX8-like Die 内完整最终建模

## 1. 目标、对象与单位

本文建立从可行单 core 到计算 KGD、HBM、D2D 和 wafer 级二维 Mesh 的解析模型。单 core 的通信、控制、计算、SRAM 面积与物理约束作为上游输入；本文不重复其内部面积推导，只使用每个可行 core 的面积、边长、算力和 NoC 链路带宽。

模型层级为：

```text
可行单 core
  → N×N core mesh
  → 加入边缘 PHY/端口条带的计算 KGD
  → 一边或两条相对边放置 HBM
  → 计算 KGD+HBM 复合矩形
  → 215 mm×215 mm 矩形窗口中的 wafer 级二维 Mesh
```

统一单位：面积为 $\mathrm{mm^2}$，长度为 $\mathrm{mm}$，带宽为 $\mathrm{GB/s}$，容量为 GB 或 MiB，算力为 GFLOP/s 或 TFLOP/s。除特别说明外，带宽均为单向有效数据带宽。

两个面积边界必须分开：

- 计算 KGD 是一颗独立 die，受 $26\ \mathrm{mm}\times33\ \mathrm{mm}$ 曝光场和 $858\ \mathrm{mm^2}$ 限制。
- 计算 KGD 与若干独立 HBM stack 的复合矩形只是封装/wafer 排布包围盒，不是一颗共同曝光的 die，因此不受 $858\ \mathrm{mm^2}$ 限制。

## 2. 上游 core 参数接口

一个通过核内物理约束的 core 点向本文提供：

| 符号                         | 含义                                              |
| -------------------------- | ----------------------------------------------- |
| $A_{core}$                 | 单 core 总面积                                      |
| $S_{core}=\sqrt{A_{core}}$ | 方形 core 的边长                                     |
| $B$                        | 单 core 目标注入带宽，同时作为一条 NoC 链路的带宽                  |
| $P_{mat,core}$             | 单 core 矩阵乘峰值算力                                  |
| $P_{vec,core}$             | 单 core 向量峰值算力                                   |
| $n_{ctrl},r,K,B_S,N_{PE}$  | 控制核数、router 档位、SRAM 容量、SRAM 带宽和矩阵 PE 数；用于标识上游配置 |

当前扫描使用 $B\in\{64,128,192,256,384,512\}\ \mathrm{GB/s}$，并只接收满足核内容量、带宽、面积和比例约束的 core 点。

## 3. 片外 Agent 类型与固定预留

### 3.1 资料来源

TX8 片外 Agent 拓扑给出 20 个逻辑 Agent：8 个 memory PHY/controller、8 个 D2D PHY/bridge、4 个 PCIe/I/O/accelerator 类 Agent，对应 $40\%:40\%:20\%$。Dojo 表明一个 I/O die 可以同时聚合 memory、PCIe 和 scale-out 功能，因此“逻辑 Agent 数”“PHY module 数”和“封装引脚数”不可互换。[Dojo 系统](https://doi.org/10.1109/HCS55958.2022.9895625)

### 3.2 关键推导

本文不固定 memory 与 D2D 的比例，而是由 HBM 数和端口约束推导；只固定其他 Agent 占总边缘端口槽的比例：

$$
r_O=\frac{4}{8+8+4}=0.20.
$$

### 3.3 关键参数式

若 $N\times N$ mesh 的每条边有 $N$ 个端口槽，则：

$$
P_E=N,\qquad P_T=4N,
$$

$$
P_U=\left\lfloor(1-r_O)P_T\right\rfloor.
$$

$P_E$ 是单边槽数，$P_T$ 是四边总槽数，$P_U$ 是扣除其他 Agent 后可供 HBM 和 D2D 使用的槽数。

### 3.4 定值依据与定值

$$
\boxed{r_O=0.20}
$$

这是 TX8-like 逻辑端口预算，不是面积或带宽比例。PCIe、主机接口和其他 accelerator 的具体协议在本阶段不独立扫描。

## 4. Core 阵列与基础计算 KGD

### 4.1 资料来源

FlooNoC 的已布局实现给出 core/tile 尺寸约 $1.5\ \mathrm{mm}\times0.75\ \mathrm{mm}$，对应 mesh 尺寸约 $6.2\ \mathrm{mm}\times6.3\ \mathrm{mm}$。取两个方向中较大的“间距/core 边长”作为统一保守比例。[FlooNoC](https://arxiv.org/abs/2409.17606)

### 4.2 关键推导

两个方向的比例为：

$$
\rho_x=\frac{6.3-8\times0.75}{7\times0.75},
\qquad
\rho_y=\frac{6.2-4\times1.5}{3\times1.5}.
$$

因此：

$$
\boxed{\rho_g=\max(\rho_x,\rho_y)=\frac{2}{35}\approx0.05714}.
$$

### 4.3 关键参数式

限制为 $N\times N$ 方形阵列。阵列方向的等效 core 数为：

$$
F_N=N+(N-1)\rho_g.
$$

只含 core 与 core 间距的正方形 mesh 边长为：

$$
\boxed{S_M=F_NS_{core}}.
$$

$N$ 是每行、每列的 core 数；$S_M$ 是基础 mesh 包围盒边长。

### 4.4 定值依据与定值

$$
N\in\{4,8,16,32\},\qquad \rho_g=2/35.
$$

## 5. KGD microbump 与 PHY 硬核条带

### 5.1 KGD microbump pitch

#### 资料来源

公开 HBM2 KGD 测试资料给出约 $55\ \mu\mathrm{m}$ 的交错 pad 阵列；UCIe advanced package 的公开范围覆盖 $25$--$55\ \mu\mathrm{m}$。本文统一描述计算 KGD 上的 microbump，不使用 HBM stack 内部或独立 I/O die 上的 pitch。[FormFactor HBM2 KGD](https://www.formfactor.com/wp-content/uploads/SWTW_HBM2_KGD-June-2017-Final-3.pdf)、[UCIe](https://www.uciexpress.org/specifications)

#### 关键参数式与定值

$$
\boxed{p_{\mu b}=55\ \mu\mathrm{m}=0.055\ \mathrm{mm}}.
$$

$p_{\mu b}$ 是计算 KGD 上相邻 microbump 中心距。它用于 shoreline 几何校准，不直接等同于 PHY 电路深度。

### 5.2 三类 PHY 深度

#### 资料来源

HBM footprint 与 UCIe reference PHY 给出三类早期物理深度：H mBM PHY $1.129$m、UCIe-A D2D PHY $1.043$ mm、其他 I/O/UCIe-S 类 PHY $1.540$ mm。Dojo 公开每边约 2 TB/s 和全 D1 576 个双向 SerDes，用于校验 D2D 的带宽与 lane 数量级。[Tesla Dojo](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips_tesla_dojo_uarch.pdf)

#### 关键推导

三类 PHY 可以沿边缘共享条带长度，但其深度不能简单相加；本阶段用最大深度形成保守包围盒：

$$
D_{PHY}=\max(D_H,D_D,D_O).
$$

#### 关键参数式与定值

$$
D_H=1.129,\quad D_D=1.043,\quad D_O=1.540\ \mathrm{mm},
$$

$$
\boxed{D_{PHY}=1.540\ \mathrm{mm}}.
$$

$D_H,D_D,D_O$ 分别是 HBM、D2D 和其他 I/O PHY 的参考深度；$D_{PHY}$ 是解析模型加入计算 KGD 长边的固定条带深度。

## 6. 端口分布与额外回线

### 6.1 资料来源

Fovea 强调 memory/D2D PHY 竞争有限的 die-edge access；OCP BoW 的 slice/bump 组织表明 PHY 与 bump 倾向聚簇。FlooNoC 表明宽链路 NoC 的物理实现需显式考虑长线和 floorplan。[Fovea](https://arxiv.org/abs/2608.03285)、[OCP BoW PHY 2.0](https://www.opencompute.org/chiplets/26/bunch-of-wires-bow-phy-specification-20)

### 6.2 关键推导

扫描两种逻辑端口位置：

- `clustered`：端口靠近两端聚簇，便于连接聚簇 PHY；不加额外长边余量。
- `uniform`：端口沿整边均匀分散，可降低局部注入热点，但连接聚簇 PHY 需要额外回线。

uniform 回线缺少公开同构 P&R 数据，因此只采用一阶长边比例余量：

$$
\Delta W_U=\eta_US_M.
$$

### 6.3 关键参数式

令 $\pi\in\{\mathrm{clustered},\mathrm{uniform}\}$，则计算 KGD 的长边、短边和面积为：

$$
\boxed{
W_C=S_M+D_{PHY}
+\mathbf 1_{\pi=\mathrm{uniform}}\eta_US_M},
$$

$$
\boxed{H_C=S_M},
$$

$$
\boxed{A_C=W_CH_C}.
$$

### 6.4 定值依据与定值

$$
\boxed{\eta_U=0.03},\qquad \text{敏感性范围 }0.02\text{--}0.05.
$$

$\eta_U$ 不是公开硅测量值。固定 $D_{PHY}$ 表示 PHY 宏深度，$\eta_U$ 只表示数字回线余量，二者不重复。

## 7. HBM3 单 stack 电气模型

### 7.1 资料来源

HBM3 使用 1024-bit 数据接口；SK hynix 报告 6.4 Gb/s/pin 和约 819 GB/s/stack。当前系统模型使用 16 GB/stack。[JEDEC JESD238](https://www.jedec.org/standards-documents/docs/jesd238)、[SK hynix HBM3](https://news.skhynix.com/en/meet-the-engineers-leading-the-worlds-first-mass-production-of-hbm3/)

### 7.2 关键推导

$$
B_H=\frac{w_HR_H}{8}
=\frac{1024\times6.4}{8}
=819.2\ \mathrm{GB/s}.
$$

### 7.3 关键参数式与定值

$$
\boxed{w_H=1024\ \mathrm{bit}},\quad
\boxed{R_H=6.4\ \mathrm{Gb/s/pin}},
$$

$$
\boxed{B_H=819.2\ \mathrm{GB/s}},\quad
\boxed{C_H=16\ \mathrm{GB}},\quad
\boxed{\alpha_H=0.90}.
$$

$\alpha_H$ 是目标持续带宽相对物理峰值的利用系数，属于研究假设，建议扫描 $0.8$--$0.95$。

## 8. HBM 与 D2D shoreline

### 8.1 HBM shoreline

#### 资料来源

HBM 使用交错 KGD bump 阵列。由 1024 根数据线、命令/地址/时钟/电源地冗余和逃逸空间估算单 stack 在计算 KGD 边缘占用的电气 shoreline。FormFactor 的 HBM KGD footprint 是主要几何校准，公开 HBM pad-placement 资料用于交叉检查。

#### 关键参数式与定值

$$
\boxed{L_H=8.50\ \mathrm{mm}}.
$$

$L_H$ 是一个 HBM stack 在计算 KGD 上的接口边长占用，不是 HBM package 宽度。

### 8.2 D2D shoreline 与物理带宽

#### 资料来源

Dojo 给出晶圆级 D2D 的目标带宽数量级，但没有公开可直接复现的 PHY 宏尺寸和 bump map；因此物理几何采用 UCIe-A x64 reference module，Dojo 只作带宽/lane 交叉校验。UCIe 3.0 支持 64 GT/s。[UCIe specifications](https://www.uciexpress.org/specifications)

#### 关键推导

一个 x64、64 GT/s module 的理想单向数据率为：

$$
B_{D,module}^{\rightarrow}=\frac{64\times64}{8}=512\ \mathrm{GB/s}.
$$

#### 关键参数式与定值

$$
\boxed{L_D=0.55\ \mathrm{mm}},\qquad
\boxed{B_{D,module}^{\rightarrow}=512\ \mathrm{GB/s}}.
$$

$L_D$ 是每条 HBM 边预留的一个 D2D module shoreline；$B_{D,module}^{\rightarrow}$ 是每边一个 module 的单向上限。若改为 Dojo 数量级的两个 module/边，应同时令 $L_D=1.10$ mm、物理上限为 $1024$ GB/s/边/方向，不能只改带宽。

## 9. HBM 放置数

### 9.1 资料来源

HBM 可放在计算 KGD 的一条边，或两条相对边。NVIDIA GH100 的 6 个 HBM stack 与多控制器实现说明多个 HBM 围绕大计算 die 放置是商业可行组织；具体一边/两边拓扑和 $0.15$ mm 间隔是本研究的封装级假设。[NVIDIA Hopper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

### 9.2 关键推导：边长限制

一条 HBM 边上的 $m$ 个 stack 及一个 D2D module 必须满足：

$$
mL_H+(m-1)s+L_D<W_C.
$$

严格不等式对应的最大整数为：

$$
m_{max,IO}=\max\left(
0,
\left\lceil\frac{W_C-L_D+s}{L_H+s}\right\rceil-1
\right).
$$

### 9.3 关键推导：NoC 截面带宽限制

$N\times N$ mesh 的单方向截面包含 $N$ 条链路：

$$
B_{bisec}=NB.
$$

HBM 目标流量必须满足：

$$
B_{bisec}\ge m\alpha_HB_H,
$$

所以：

$$
m_{max,BW}=\max\left(0,
\left\lfloor\frac{NB}{\alpha_HB_H}\right\rfloor
\right).
$$

### 9.4 关键参数式与定值

$$
\boxed{m_{max}=\min(m_{max,IO},m_{max,BW})},
$$

$$
\boxed{e_H\in\{1,2\}},\qquad
\boxed{m\in\{1,\ldots,m_{max}\}}.
$$

$e_H$ 是 HBM 放置边数，$e_H=2$ 时两边相对；$m$ 是每条 HBM 边的 stack 数。当前有限扫描先枚举 $m\in\{1,2,3,4\}$，再用 $m\le m_{max}$ 过滤。若 $m_{max}=0$，该点不可行。

## 10. HBM NoC 端口数

### 10.1 资料来源

HBM 的物理接口带宽与 NoC 逻辑端口带宽不必位宽相等；控制器/NI 可以完成协议和位宽转换。端口数应由带宽聚合推导，而不是由 1024-bit HBM 接口直接等同为 NoC 链路数。

### 10.2 关键推导

满足 HBM 目标带宽所需的原始端口数为：

$$
t_H^{raw}=\left\lceil\frac{m\alpha_HB_H}{B}\right\rceil.
$$

为保证 HBM 所在边至少保留一个 D2D 端口，当 HBM 会占满 $N$ 个槽时截断为 $N-1$：

$$
\boxed{t_H=\min(t_H^{raw},N-1)}.
$$

### 10.3 参数解释与带宽口径

$t_H^{raw}$ 是带宽等价需求，$t_H$ 是实际分配端口数。截断后不强制 $t_HB\ge m\alpha_HB_H$，因此必须区分 HBM 物理峰值、目标值和 NoC 可观察值。

## 11. D2D NoC 端口数

### 11.1 资料来源

TX8 的 D2D Agent 分布和 Dojo 四边 scale-out 结构支持“四边对称、每边至少一个 D2D 入口”的早期规则。FlooNoC 的 router/NI 结构说明一个边缘入口可在二维 mesh 中转向其他行/列，但入口 NI、第一跳链路和截面仍是共享瓶颈。[FlooNoC routing](https://pulp-platform.github.io/FlooNoC/floonoc/route_algos/)、[Dojo microarchitecture](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips_tesla_dojo_uarch.pdf)

### 11.2 关键推导

四边 D2D 端口数相等，记每边为 $d$。全局端口预算给出：

$$
d_{global}=\left\lfloor
\frac{P_U-e_Ht_H}{4}
\right\rfloor.
$$

HBM 所在边的局部槽数给出：

$$
d_{edge}=N-t_H.
$$

因此：

$$
\boxed{d=\min(d_{global},d_{edge})},
$$

并强制：

$$
\boxed{d\ge1}.
$$

### 11.3 路由语义

D2D PHY 固定连接某个边缘 router/NI，不能无成本跳到另一行；进入 NoC 后可通过二维路由到达其他行/列。基线采用静态确定性最短路；若边界到边界路径需要多次转向，应使用 source/table-based 路由并另计 hop、拥塞和死锁规避代价。本文不把一个 D2D 端口等价为每行一个独立端口。

端口 $p$ 的注入上限满足：

$$
B_{inj,p}\le
\min\left(B_{PHY,p},B_{NI,p},
\sum_{\ell\in out(p)}B_\ell\right).
$$

其中 $B_{inj,p}$ 是端口 $p$ 的注入带宽，$B_{PHY,p}$ 和 $B_{NI,p}$ 分别是该端口 PHY 与 NI 的单向上限，$out(p)$ 是入口 router 可用输出链路集合，$B_\ell$ 是链路 $\ell$ 的带宽。

## 12. HBM 与 D2D 端口位置

### 12.1 资料来源

均匀端口降低入口附近的流量集中；聚簇端口缩短 PHY 到 NoC 入口的物理回线。二者均不改变二维 mesh 对内部 core 的基本可达性，但会改变热点、平均 hop 和持续带宽。

### 12.2 关键参数式

对一条有 $x$ 个同类端口的边，uniform 位置集合为：

$$
\boxed{
\mathcal U(N,x)=
\left\{
\left\lfloor\frac{jN}{x}\right\rfloor
\mid j=0,\ldots,x-1
\right\}}.
$$

clustered 位置集合为：

$$
x_L=\left\lceil\frac{x}{2}\right\rceil,
\qquad
x_R=\left\lfloor\frac{x}{2}\right\rfloor,
$$

$$
\boxed{
\mathcal C(N,x)=
\{0,\ldots,x_L-1\}
\cup
\{N-x_R,\ldots,N-1\}}.
$$

先用 $x=t_H$ 生成 HBM 所在边的 HBM 集合 $\mathcal H$。令剩余槽位的有序集合为：

$$
\mathcal R=\{0,\ldots,N-1\}\setminus\mathcal H.
$$

HBM 所在边的 D2D 端口必须从 $\mathcal R$ 中选择，不能与 $\mathcal H$ 重叠。若 $\mathcal R=[r_0,\ldots,r_{q-1}]$，uniform 选择为：

$$
\boxed{
\mathcal D_{H,U}=
\left\{r_{\lfloor jq/d\rfloor}\mid j=0,\ldots,d-1\right\}};
$$

clustered 则从 $\mathcal R$ 的首尾各取 $\lceil d/2\rceil$ 和 $\lfloor d/2\rfloor$ 个。没有 HBM 的边直接以 $x=d$ 使用 $\mathcal U(N,d)$ 或 $\mathcal C(N,d)$。

由此：

$$
\boxed{\mathcal H\cap\mathcal D_H=\varnothing}.
$$

正式 floorplan 仍需检查边缘 router radix；当前模型把一个槽位限制为一种片外 Agent，保持与 $t_H+d\le N$ 的端口预算一致。

## 13. 计算 KGD 与 HBM 的复合矩形

### 13.1 资料来源

系统级几何扫描使用单颗 HBM 的 $8.13\ \mathrm{mm}\times4.92\ \mathrm{mm}$ 抽象放置块。该数值与 $s=0.15$ mm 均是 TX8-like 设计空间定值，不是 JESD238 强制的商品 package 外形；因此只用于保持当前扫描基线。正式选定 HBM3 料号后，应以供应商 mechanical outline 替换 $W_H,H_H$ 并重扫。HBM 间、HBM 与计算 KGD 间、复合矩形四周留白均取 $s$。

### 13.2 宽度推导

一排 $m$ 颗 HBM 的物理宽度为：

$$
W_{H,row}=mW_H+(m-1)s.
$$

复合矩形宽度需要包围 HBM 排和计算 KGD，并在左右各留 $s$：

$$
\boxed{
W_D=\max(W_{H,row},W_C)+2s}.
$$

### 13.3 高度推导

单边放置一排 HBM：

$$
\boxed{H_D=H_C+H_H+3s,\qquad e_H=1}.
$$

双边相对放置两排 HBM：

$$
\boxed{H_D=H_C+2H_H+4s,\qquad e_H=2}.
$$

复合矩形面积为：

$$
\boxed{A_D=W_DH_D}.
$$

### 13.4 定值依据与定值

$$
\boxed{W_H=8.13\ \mathrm{mm}},\quad
\boxed{H_H=4.92\ \mathrm{mm}},\quad
\boxed{s=0.15\ \mathrm{mm}}.
$$

$W_H,H_H$ 是当前 HBM 抽象放置块；它们与由公开 pad footprint 校准的电气 shoreline $L_H=8.50$ mm 是不同量。

## 14. 单个复合矩形的性能

### 14.1 计算算力

core 数为：

$$
N_{core}=N^2.
$$

不计利用率和同步损失的峰值算力为：

$$
\boxed{P_{mat,Die}=N^2P_{mat,core}},
$$

$$
\boxed{P_{vec,Die}=N^2P_{vec,core}}.
$$

### 14.2 HBM 容量与带宽

总 stack 数、容量、物理峰值和目标带宽为：

$$
M_H=e_Hm,
$$

$$
\boxed{C_{H,total}=e_HmC_H},
$$

$$
\boxed{B_{H,peak}=e_HmB_H},
$$

$$
B_{H,target}=e_Hm\alpha_HB_H.
$$

考虑 HBM 端口截断后的 NoC 可观察上限：

$$
\boxed{
B_{H,NoC}=e_H\min(m\alpha_HB_H,t_HB)}.
$$

性能模型应使用 $B_{H,NoC}$，器件选型使用 $B_{H,peak}$。

### 14.3 D2D 带宽

每边有 $d$ 个逻辑 NoC 端口，但当前只配置一个 x64 PHY module，因此：

$$
\boxed{
B_{D,edge}^{\rightarrow}
=\min(dB,B_{D,module}^{\rightarrow})}.
$$

四边同时向外的理想聚合值为：

$$
\boxed{B_{D,total}^{\rightarrow}=4B_{D,edge}^{\rightarrow}}.
$$

该式是端口与 PHY 上限，不保证任意 workload 均达到；实际值还受第一跳拥塞、路由和 wafer 级 traffic cut 限制。

## 15. Wafer 级二维 Mesh

### 15.1 资料来源与边界

本文把 $215\ \mathrm{mm}\times215\ \mathrm{mm}$ 定义为可用的方形排布窗口，每个 $W_D\times H_D$ 复合矩形作为一个二维 Mesh 节点。该尺寸是系统建模假设，不是 300 mm 圆晶圆直径。

### 15.2 关键参数式

$$
n_x=\left\lfloor\frac{W_W}{W_D}\right\rfloor,
\qquad
n_y=\left\lfloor\frac{H_W}{H_D}\right\rfloor,
$$

$$
\boxed{N_{D/wafer}=n_xn_y},
$$

$$
\boxed{E_W=(n_x-1)n_y+(n_y-1)n_x}.
$$

$n_x,n_y$ 是两个方向的复合矩形数，$N_{D/wafer}$ 是总节点数，$E_W$ 是无向相邻 D2D 链路数。

穿过两个中心 cut 的单向带宽为：

$$
\boxed{B_{W,x}^{\rightarrow}=n_yB_{D,edge}^{\rightarrow}},
$$

$$
\boxed{B_{W,y}^{\rightarrow}=n_xB_{D,edge}^{\rightarrow}}.
$$

### 15.3 定值

$$
\boxed{W_W=H_W=215\ \mathrm{mm}}.
$$

该装箱模型暂不计供电、时钟树、布线通道、散热、划片道、圆边损失、良率和冗余，因此 $N_{D/wafer}$ 是理想矩形装箱上界。

## 16. 物理可行性约束

### 16.1 上游 core 可行性

**资料来源。** 单 core 必须先通过 7 nm 通信、控制、计算和 SRAM 模型约束。

**关键分析。** Die 层不重新组合被核内模型删除的点，以避免把不可实现 core 放大成阵列。

**约束。** 输入记录必须带 `core_feasible=True`，并提供有限正值 $A_{core},B,P_{mat,core},P_{vec,core}$。

### 16.2 光罩尺寸与面积

**资料来源。** Nikon 光刻系统公开最大曝光场 $26\ \mathrm{mm}\times33\ \mathrm{mm}$。[Nikon](https://www.nikon.com/business/semi/lineup/)

**关键分析。** 约束单颗计算 KGD；允许旋转，因此短边、长边分别比较。

**约束。** 

$$
\boxed{\min(W_C,H_C)\le26},
$$

$$
\boxed{\max(W_C,H_C)\le33},
$$

$$
\boxed{A_C=W_CH_C\le858}.
$$

### 16.3 HBM 边长

**资料来源。** 由 KGD microbump、交错阵列、逃逸空间以及一个 D2D module 的共边预留形成 shoreline 模型。

**关键分析。** 物理 package 宽度不能代替 KGD 电气 shoreline。

**约束。** 

$$
\boxed{mL_H+(m-1)s+L_D<W_C}.
$$

### 16.4 HBM–NoC 截面带宽

**资料来源。** 二维 mesh 每个中心截面有 $N$ 条同带宽链路；HBM 持续目标取峰值的 $\alpha_H$。

**关键分析。** 这是整条 HBM 边的注入上限，不要求 HBM PHY 位宽与单条 NoC 链路位宽相等。

**约束。** 

$$
\boxed{NB\ge m\alpha_HB_H}.
$$

### 16.5 HBM 至少一颗

**资料来源。** 当前研究对象要求外部高带宽存储。

**关键分析。** 若一颗 HBM 都不能同时满足 shoreline 与截面带宽，该 $N,B,W_C$ 点没有可扫描的 HBM 配置。

**约束。** 

$$
\boxed{m_{max}\ge1,\qquad1\le m\le m_{max}}.
$$

### 16.6 D2D 端口保留

**资料来源。** TX8/Dojo 均需要跨 die 连接；本文 wafer 使用二维 Mesh。

**关键分析。** 四边 D2D 数相等且每边至少一个；HBM 端口必要时截断为 $N-1$。

**约束。** 

$$
\boxed{t_H\le N-1,\qquad d\ge1}.
$$

### 16.7 全局与局部端口预算

**资料来源。** TX8 的 20% other Agent 固定预留和每边 $N$ 个槽的早期抽象。

**关键分析。** 同时满足全芯片总量和 HBM 所在边局部容量。

**约束。** 

$$
\boxed{4d+e_Ht_H\le\lfloor(1-r_O)4N\rfloor},
$$

$$
\boxed{d+t_H\le N\quad\text{(HBM 所在边)}}.
$$

### 16.8 D2D 入口与二维路由

**资料来源。** Dojo 公开“目标 D1 内简单二维路由”和每 D1 可编程路由表；FlooNoC 支持 XY、source-based 和 table-based 静态路由。

**关键分析。** 固定边缘入口可以服务其他行/列，但不能绕开自身 NI 和第一跳瓶颈；纯 XY 也不保证所有边界到边界的多转向路径。

**约束。** 每个 D2D 端口必须连接合法边缘 router/NI；路由表必须保证所有目标 core 可达且无死锁。解析扫描只验证容量，不宣称拥塞自适应。

### 16.9 Wafer 窗口可放置

**资料来源。** 本研究的 $215\ \mathrm{mm}\times215\ \mathrm{mm}$ 可用窗口。

**关键分析。** 至少放置一个复合矩形才是有效系统点。

**约束。** 

$$
\boxed{n_x\ge1,\qquad n_y\ge1}.
$$

## 17. 最终汇总

依次计算：

$$
S_{core}=\sqrt{A_{core}},
\qquad
S_M=[N+(N-1)\rho_g]S_{core},
$$

$$
W_C=S_M+D_{PHY}
+\mathbf1_{\pi=\mathrm{uniform}}\eta_US_M,
\qquad H_C=S_M,
$$

$$
m_{max}=\min\left[
\max\left(0,\left\lceil\frac{W_C-L_D+s}{L_H+s}\right\rceil-1\right),
\max\left(0,\left\lfloor\frac{NB}{\alpha_HB_H}\right\rfloor\right)
\right],
$$

$$
t_H=\min\left(
\left\lceil\frac{m\alpha_HB_H}{B}\right\rceil,
N-1
\right),
$$

$$
d=\min\left[
\left\lfloor\frac{\lfloor(1-r_O)4N\rfloor-e_Ht_H}{4}\right\rfloor,
N-t_H
\right],
$$

$$
W_D=\max[mW_H+(m-1)s,W_C]+2s,
$$

$$
H_D=
\begin{cases}
H_C+H_H+3s,&e_H=1,\\
H_C+2H_H+4s,&e_H=2,
\end{cases}
$$

$$
A_D=W_DH_D,qquad
N_{D/wafer}=
\left\lfloor\frac{215}{W_D}\right\rfloor
\left\lfloor\frac{215}{H_D}\right\rfloor.
$$

全部硬约束为：

$$
\boxed{
\begin{aligned}
&\text{上游 core 可行},\\
&N\in\{4,8,16,32\},\quad e_H\in\{1,2\},\\
&\min(W_C,H_C)\le26,\quad\max(W_C,H_C)\le33,\\
&W_CH_C\le858,\\
&mL_H+(m-1)s+L_D<W_C,\\
&NB\ge m\alpha_HB_H,\\
&1\le m\le m_{max},\\
&t_H\le N-1,\quad d\ge1,\\
&4d+e_Ht_H\le\lfloor(1-r_O)4N\rfloor,\\
&d+t_H\le N\quad\text{(HBM 边)},\\
&n_x\ge1,\quad n_y\ge1.
\end{aligned}}
$$

## 18. 参数严格分类

### 18.1 独立搜索参数

|符号|含义|集合/范围|层级|
|---|---|---|---|
|$n_{ctrl}$|单 core 控制核数|$\{1,2\}$|上游 core|
|$r$|router 功能档位|base、broadcast、reduce、both|上游 core|
|$B$|单 core 注入/单链路带宽|$\{64,128,192,256,384,512\}$ GB/s|上游 core/Die|
|$K$|单 core SRAM 容量|$\{1,2,3,4\}$ MiB|上游 core|
|$B_S$|单 core SRAM 聚合带宽|$\{256,512,768,1024\}$ GB/s|上游 core|
|$N_{PE}$|单 core 矩阵 PE 数|$\{1024,4096,8192,12288,16384\}$|上游 core|
|$N$|每行/列 core 数|$\{4,8,16,32\}$|计算 KGD|
|$\pi$|端口位置模式|clustered、uniform|计算 KGD|
|$e_H$|HBM 放置边数|$\{1,2\}$，2 为相对边|封装|
|$m$|每条 HBM 边的 stack 数|实现时先扫 $\{1,2,3,4\}$，再约束 $m\le m_{max}$|封装|

除本表十项外，不允许把任何量作为自由扫描维度。

### 18.2 固定校准或依赖推导参数

|符号|类型|含义|定值或依赖关系|
|---|---|---|---|
|$A_{core}$|上游推导|单 core 面积|由核内模型计算并先过滤|
|$S_{core}$|依赖推导|方形 core 边长|$\sqrt{A_{core}}$|
|$P_{mat,core},P_{vec,core}$|上游推导|单 core 两类峰值算力|由核内 $N_{PE}$ 和 vector 比例推导|
|$\rho_g$|固定校准|core 间距比例|$2/35$|
|$F_N$|依赖推导|阵列等效 core 数|$N+(N-1)\rho_g$|
|$S_M$|依赖推导|基础 mesh 边长|$F_NS_{core}$|
|$p_{\mu b}$|固定校准|KGD microbump pitch|0.055 mm|
|$D_H,D_D,D_O$|固定校准|三类 PHY 深度|1.129、1.043、1.540 mm|
|$D_{PHY}$|依赖推导|保守条带深度|$\max(D_H,D_D,D_O)=1.540$ mm|
|$\eta_U$|固定估算|uniform 回线比例|0.03；敏感性 0.02--0.05|
|$W_C,H_C,A_C$|依赖推导|计算 KGD 长、短边和面积|第 6 节公式|
|$r_O$|固定校准|其他 Agent 槽比例|0.20|
|$P_E,P_T,P_U$|依赖推导|单边、总计、可用槽数|$N,4N,\lfloor(1-r_O)4N\rfloor$|
|$w_H,R_H$|固定标准/产品值|HBM3 位宽与 pin 速率|1024 bit、6.4 Gb/s/pin|
|$B_H$|依赖推导|单 stack 峰值带宽|$w_HR_H/8=819.2$ GB/s|
|$C_H$|固定系统值|单 stack 容量|16 GB|
|$\alpha_H$|固定假设|HBM 目标利用系数|0.90；敏感性 0.8--0.95|
|$L_H$|固定校准|单 HBM KGD shoreline|8.50 mm|
|$L_D$|固定校准|每边一个 D2D module shoreline|0.55 mm|
|$B_{D,module}^{\rightarrow}$|固定/依赖推导|x64@64 GT/s 单向上限|512 GB/s|
|$B_{bisec}$|依赖推导|计算 KGD 单方向截面带宽|$NB$|
|$m_{max,IO}$|依赖推导|边长允许的每边 HBM 数|第 9.2 节公式|
|$m_{max,BW}$|依赖推导|截面允许的每边 HBM 数|第 9.3 节公式|
|$m_{max}$|依赖推导|最终每边 HBM 上限|两者最小值|
|$t_H^{raw},t_H$|依赖推导|HBM 原始需求和实际端口数|第 10 节公式|
|$d_{global},d_{edge},d$|依赖推导|D2D 两个上限及实际每边端口数|第 11 节公式|
|$p,\ell,out(p)$|局部索引/集合|D2D 入口、输出链路及其集合|仅用于入口瓶颈式|
|$B_{inj,p},B_{PHY,p},B_{NI,p},B_\ell$|依赖上限|入口、PHY、NI 和链路带宽|满足第 11.3 节最小值约束|
|$x,j,x_L,x_R$|局部索引|同类端口数、枚举索引和两端聚簇数|仅用于第 12 节位置生成|
|$\mathcal U,\mathcal C$|依赖推导|uniform/clustered 基础位置集合|第 12 节公式|
|$\mathcal H,\mathcal R,\mathcal D_H$|依赖推导|HBM 位置、剩余槽位和 HBM 边 D2D 位置|互斥分配，见第 12.2 节|
|$W_H,H_H$|固定设计值|HBM 抽象放置块|8.13、4.92 mm；料号确定后替换|
|$s$|固定系统值|封装间距/外周留白|0.15 mm|
|$W_{H,row}$|依赖推导|一排 HBM 宽度|$mW_H+(m-1)s$|
|$W_D,H_D,A_D$|依赖推导|复合矩形尺寸和面积|第 13 节公式|
|$N_{core}$|依赖推导|计算 KGD core 数|$N^2$|
|$P_{mat,Die},P_{vec,Die}$|依赖推导|Die 两类峰值算力|$N^2$ 乘单 core 值|
|$M_H,C_{H,total}$|依赖推导|HBM 总数与容量|$e_Hm,e_HmC_H$|
|$B_{H,peak},B_{H,target},B_{H,NoC}$|依赖推导|三种 HBM 带宽口径|第 14.2 节公式|
|$B_{D,edge}^{\rightarrow},B_{D,total}^{\rightarrow}$|依赖推导|D2D 单边和四边单向带宽|第 14.3 节公式|
|$W_W,H_W$|固定系统值|wafer 可用矩形窗口|215、215 mm|
|$n_x,n_y,N_{D/wafer}$|依赖推导|wafer 两向数量与总数|第 15 节公式|
|$E_W$|依赖推导|wafer mesh 无向边数|$(n_x-1)n_y+(n_y-1)n_x$|
|$B_{W,x}^{\rightarrow},B_{W,y}^{\rightarrow}$|依赖推导|wafer 两个 cut 的单向带宽|第 15.2 节公式|

## 19. 参考资料与链接

1. [Tesla Dojo microarchitecture, Hot Chips 34](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips_tesla_dojo_uarch.pdf)
2. [Dojo system scaling, Hot Chips 34](https://doi.org/10.1109/HCS55958.2022.9895625)
3. [FlooNoC, IEEE TVLSI](https://arxiv.org/abs/2409.17606)
4. [FlooNoC routing algorithms](https://pulp-platform.github.io/FlooNoC/floonoc/route_algos/)
5. [UCIe Consortium specifications](https://www.uciexpress.org/specifications)
6. [OCP BoW PHY Specification 2.0](https://www.opencompute.org/chiplets/26/bunch-of-wires-bow-phy-specification-20)
7. [JEDEC HBM3 JESD238](https://www.jedec.org/standards-documents/docs/jesd238)
8. [SK hynix HBM3 6.4 Gb/s/pin and 819 GB/s](https://news.skhynix.com/en/meet-the-engineers-leading-the-worlds-first-mass-production-of-hbm3/)
9. [FormFactor HBM2 KGD footprint](https://www.formfactor.com/wp-content/uploads/SWTW_HBM2_KGD-June-2017-Final-3.pdf)
10. [NVIDIA Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
11. [Nikon semiconductor lithography systems](https://www.nikon.com/business/semi/lineup/)
12. [Fovea wafer-scale DSE](https://arxiv.org/abs/2608.03285)

## BibLaTeX

```bibtex
@misc{tx8_offchip_agent_topology,
  title        = {{TX8} Off-Chip Agent Topology},
  year         = {2026},
  howpublished = {Project-provided TX8 architecture material},
  note         = {Twenty logical agents: eight memory, eight D2D, and four PCIe/I/O/accelerator-class agents}
}

@inproceedings{talpes2022dojo,
  author    = {Emil Talpes and Douglas Williams and Debjit Das Sarma and others},
  title     = {The Microarchitecture of Tesla's Exa-Scale Computer},
  booktitle = {2022 IEEE Hot Chips 34 Symposium},
  year      = {2022},
  doi       = {10.1109/HCS55958.2022.9895534},
  url       = {https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips_tesla_dojo_uarch.pdf}
}

@inproceedings{chang2022dojosystem,
  author    = {Bill Chang and Rajiv Kurian and Doug Williams and Eric Quinnell},
  title     = {{DOJO}: Super-Compute System Scaling for {ML} Training},
  booktitle = {2022 IEEE Hot Chips 34 Symposium},
  year      = {2022},
  doi       = {10.1109/HCS55958.2022.9895625},
  url       = {https://doi.org/10.1109/HCS55958.2022.9895625}
}

@article{fischer2025floonoc,
  author  = {Tim Fischer and Michael Rogenmoser and Thomas Benz and Frank K. G{\"u}rkaynak and Luca Benini},
  title   = {{FlooNoC}: A 645-Gb/s/link 0.15-pJ/B/hop Open-Source {NoC} With Wide Physical Links and End-to-End {AXI4} Parallel Multistream Support},
  journal = {IEEE Transactions on Very Large Scale Integration Systems},
  year    = {2025},
  volume  = {33},
  number  = {4},
  pages   = {1094--1107},
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

@standard{ocpBowPhy20,
  author      = {{Open Compute Project}},
  title       = {Bunch of Wires PHY Specification 2.0},
  institution = {Open Compute Project},
  url         = {https://www.opencompute.org/chiplets/26/bunch-of-wires-bow-phy-specification-20}
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
  url     = {https://news.skhynix.com/en/meet-the-engineers-leading-the-worlds-first-mass-production-of-hbm3/},
  urldate = {2026-08-22}
}

@techreport{formfactor2017hbm2kgd,
  author      = {{FormFactor}},
  title       = {HBM2 KGD Test: Challenges and Solutions},
  institution = {FormFactor},
  year        = {2017},
  url         = {https://www.formfactor.com/wp-content/uploads/SWTW_HBM2_KGD-June-2017-Final-3.pdf}
}

@online{nvidia2022hopper,
  author  = {{NVIDIA}},
  title   = {NVIDIA Hopper Architecture In-Depth},
  year    = {2022},
  url     = {https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/}
}

@online{nikonLithography,
  author  = {{Nikon Corporation}},
  title   = {Semiconductor Lithography Systems Lineup},
  url     = {https://www.nikon.com/business/semi/lineup/},
  urldate = {2026-08-22},
  note    = {Maximum exposure field 26 mm by 33 mm}
}

@article{li2026fovea,
  author  = {Jinxi Li and Huizheng Wang and Jinyi Deng and Yang Hu and Shouyi Yin},
  title   = {Fovea: Physical-Implication-Aware Wafer-Scale DSE with Decision-Domain-Guided Cross-Fidelity Refinement},
  year    = {2026},
  eprint  = {2608.03285},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2608.03285}
}
```
