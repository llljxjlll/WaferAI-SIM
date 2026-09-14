# TX8-like Core 内 7 nm 面积、带宽与物理布局最终模型

## 1. 模型目标与适用范围

本模型用于 TX8-like 单计算核 core 的早期设计空间探索，统一输出为 $\mathrm{mm^2}@7\,\mathrm{nm}$。一个 core 包含：

- $1$ 或 $2$ 个控制核；
- 矩阵 PE 阵列和 Vector Unit；
- $16$-bank SRAM/SPM；
- DTE、局部总线/端口 MUX、Network Interface、NoC router 和链路缓冲。

模型先分别计算每个部件，再加入实现余量和物理布局约束。它适合筛选候选配置，不替代目标 7 nm PDK 下的 RTL 综合、SRAM compiler、宏摆放和 P&R。

本模型采用单计算核、$1{:}1$ 方形 core 布局：SRAM banks 统一重排为底边加左右侧臂的 U 型。NoC router、NI、DTE 和边界链路缓冲位于顶部边缘；PE 阵列位于中央；Vector Unit 和控制核靠近计算阵列。单计算核不需要连接多个计算核的全局 crossbar，但仍需要面向 PE、Vector、控制核、DTE 和 SRAM banks 的局部总线或端口 MUX。

---

## 2. 统一面积单位

### 2.1 资料来源

ASAP7 RVT 标准单元库中 NAND2x1 的尺寸为 $0.324\,\mu\mathrm{m}\times0.270\,\mu\mathrm{m}$。因此，一个 million gate equivalent（MGE）的原始 NAND2 等效面积为：

$$
g_7=0.324\times0.270=0.08748\ \mathrm{mm^2/MGE}.
$$

标准单元在 P&R 中不能达到 $100\%$ 利用率。设 $u_{sc}$ 为标准单元利用率，则 placed logic 的换算系数为：

$$
\lambda_7=\frac{g_7}{u_{sc}}.
$$

### 2.2 参数定值

$$
u_{sc}=0.65,
\qquad
\lambda_7=\frac{0.08748}{0.65}=0.134585\ \mathrm{mm^2/MGE}.
$$

$g_7$ 用于 SRAM 等效宏面积；$\lambda_7$ 用于通信和控制标准单元面积。$u_{sc}$ 的敏感性范围取 $0.55$–$0.75$。

---

## 3. 通信面积

### 3.1 共同带宽关系

目标注入带宽记为 $B$，单位为 GB/s。NoC 频率为 $f_N$ GHz，有效传输率为 $\eta_N$，则链路 payload 位宽为：

$$
W_{link}=\left\lceil\frac{8B}{f_N\eta_N}\right\rceil\ \mathrm{bit}.
$$

定义归一化带宽：

$$
b=\frac{B}{256}.
$$

名义值 $f_N=1$ GHz、$\eta_N=1$ 时，$W_{link}=8B$ bit。模型要求：

$$
BW_{DTE}=BW_{local}=BW_{NI}=BW_{link}=B.
$$

TX8 的宽数据路径、FlooNoC 的 512-bit 实现以及 ORION/NoC 位宽模型共同支持“一阶面积随位宽近似线性增长”的处理。固定控制逻辑保留常数项，宽 datapath、buffer 和 MUX 进入带宽项。

### 3.2 DTE

#### 资料来源

TX8 的 DTE 含全局描述符/调度逻辑和多个独立 channel backend。FlooNoC 的 DMA 结果表明，增加 stream/channel 主要增加 backend、FIFO 和局部互连，而不会把 NI/router 整体按 channel 数复制。

#### 关键推导

设 $C$ 为 DTE channel 数，单 channel 峰值带宽为 $B_{ch}$。DTE 的面积拆成全局控制、每 channel 控制和宽数据路径：

$$
\boxed{
A_{DTE}=\lambda_7\left(0.72+0.15C+0.48b\right)
}.
$$

- $0.72$ MGE：DTE 全局控制、descriptor 和完成逻辑；
- $0.15C$ MGE：$C$ 个 channel 的独立控制/FIFO 增量；
- $0.48b$ MGE：随目标带宽扩展的共享宽数据路径。

#### 定值

$$
B_{ch}=128\ \mathrm{GB/s},
\qquad
C\in\{1,2,3,4,5,6\}.
$$

$B_{ch}$ 是 TX8 早期建模锚点，后续应由 DTE RTL 的持续吞吐综合结果替换。

### 3.3 局部总线与 SRAM 端口 MUX

#### 资料来源

TX8 的 L1SPM 同时服务 PE、Vector、控制核、LSU/DTE 和 NoC；FlooNoC 中本地 AXI xbar 为 $191$ kGE，与完整 NoC 的 $196$ kGE 同量级。即使只有一个计算核，也不能删除多客户端到多 bank 的本地选择网络。

#### 关键推导

channel 数增加会增加局部端口或仲裁请求，宽度增加会线性增加 MUX datapath：

$$
\boxed{
A_{local}=\lambda_7\left(0.60+1.20Cb\right)
}.
$$

- $0.60$ MGE：仲裁、地址译码和固定端口控制；
- $1.20Cb$ MGE：随 channel 数和宽度共同增长的数据选择网络。

这里的 $A_{local}$ 是局部总线/端口 MUX，不是连接多个计算核的全局 crossbar。

### 3.4 Network Interface

#### 资料来源

FlooNoC 将 AXI 与 NoC 协议通过 RoB-less NI 解耦；NI 含 header、route metadata、pack/unpack、FIFO 和 valid-ready 控制。其固定控制不随数据位宽同比增长。

#### 关键推导

$$
\boxed{
A_{NI}=\lambda_7\left(0.60+0.60b\right)
}.
$$

$0.60$ MGE 是协议与状态机固定项；$0.60b$ MGE 是 pack/unpack、FIFO 和数据通路项。模型假设不使用大容量 reorder buffer；若加入 RoB，应单独增加 $A_{RoB}$。

### 3.5 链路端点、流水寄存器与缓冲

#### 资料来源

FlooNoC 指出，超过约 $1$ mm 的宽链路需要输出缓冲以满足时序；物理链路成本还受线宽、线距、长度、shielding 和 repeater 影响。早期模型先计入端点逻辑，P&R 阶段再检查金属轨道和长线 RC。

#### 关键推导

$$
\boxed{
A_{link}=\lambda_7\left(0.60b\right)
}.
$$

该项包含链路 pipeline register、elastic/skid buffer 和端点驱动的等效逻辑面积，不把整条高层金属走线面积重复加到逻辑面积中。

### 3.6 Router 基础面积

#### 资料来源

FlooNoC 在 12 nm、512-bit 物理链路下报告 router 面积 $168$ kGE、NI 面积 $28$ kGE。门等效数量主要用于跨工艺逻辑复杂度校准，再通过 ASAP7 的 $\lambda_7$ 转为 7 nm placed area。

#### 关键推导

对固定 radix、固定 flit depth 的 router，一阶近似按位宽线性缩放：

$$
G_R^{base}=0.168\frac{W_{link}}{512}\ \mathrm{MGE}.
$$

在 $f_N=1$ GHz、$\eta_N=1$ 下：

$$
G_R^{base}=0.672b.
$$

### 3.7 Router 四档功能增量

#### 资料来源

TX8 集合通信建模把基础、broadcast、reduce、broadcast+reduce router 分别估为 $135$、$174.5$、$214.5$、$259.5$ kGE。基础 router 的绝对值由 FlooNoC 的 $168$ kGE 修正；四档之间保留 TX8 集合通信模型的相对倍率。Colagrande 等人的 7 nm collective-capable NoC 作为次要敏感性来源，其 full collective router 增量较小，说明四档主模型偏保守。

| 档位 $r$    | 功能                     |   TX8 参考值 | 倍率 $m_R(r)$ |
| --------- | ---------------------- | --------: | ----------: |
| base      | 普通路由                   |   135 kGE |       1.000 |
| broadcast | VCT + FANOUT broadcast | 174.5 kGE |       1.293 |
| reduce    | VCT-guided DCA reduce  | 214.5 kGE |       1.589 |
| both      | broadcast + DCA reduce | 259.5 kGE |       1.922 |

router 面积为：

$$
\boxed{
A_R=\lambda_7\left(0.672b\,m_R(r)\right)
}.
$$

DCA 复用邻近 Vector/FPU 运算能力，因此不在 router 内重复计入一套 reduction ALU。

### 3.8 支持指定带宽的最小通信面积

满足目标带宽的 channel 集合为：

$$
\mathcal C(B)=\left\{C\in\{1,\ldots,6\}:CB_{ch}\ge B\right\}.
$$

由于 $A_{DTE}$ 和 $A_{local}$ 对 $C$ 单调增加，最小面积 channel 数为：

$$
\boxed{
C^\star(B)=\left\lceil\frac{B}{B_{ch}}\right\rceil
},
\qquad C^\star\le6.
$$

指定 $(B,r)$ 后，通信面积按以下各项分别计算并求和：

$$
\boxed{
A_{comm}^{min}(B,r)=
A_{DTE}(B,C^\star)+A_{local}(B,C^\star)+A_{NI}(B)+A_{link}(B)+A_R(B,r)
}.
$$

若研究超额并发能力，可保留 $C>C^\star$ 的配置，但它不属于“指定带宽下最小通信面积”的主结果。

---

## 4. 控制核面积

### 4.1 资料来源

TX8 控制块的 logic-only 面积锚点为每核 $4.5$ MGE；带 cache 的估计不用于本模型，避免与 SRAM 项重复。

### 4.2 关键参数式

设控制核数量为 $n_{ctrl}$：

$$
\boxed{
A_{ctrl}=4.5n_{ctrl}\lambda_7
}.
$$

### 4.3 定值

$$
n_{ctrl}\in\{1,2\}.
$$

当 $n_{ctrl}=2$ 时，可将第二个控制核专用于 DTE/通信编排；面积模型不假设两个控制核共享前端。

---

## 5. 矩阵 PE 面积

### 5.1 资料来源

Tempus Core 给出的 45 nm P&R 对照中，$16\times4$ 的 NVDLA binary CMAC 面积约为 $0.0361\,\mathrm{mm^2}$。本模型把其中 $64$ 个 MAC-equivalent 单元均摊，并采用 $45$ nm 到 $7$ nm 的名义面积缩放 $s_{45\to7}=12$。该缩放结合公开工艺缩放趋势，仅用于早期估算。

### 5.2 关键推导

单 PE 在 45 nm 下的面积为：

$$
a_{PE,45}=\frac{0.0361}{64}=5.6406\times10^{-4}\ \mathrm{mm^2}.
$$

投影到 7 nm：

$$
a_{PE,7}=\frac{a_{PE,45}}{s_{45\to7}}
=4.7005\times10^{-5}\ \mathrm{mm^2/PE}.
$$

因此：

$$
\boxed{
A_{PE}=N_{PE}a_{PE,7}
}.
$$

设计算频率为 $f_C$ GHz，每个 PE 每周期完成一个 MAC、按 $2$ FLOP/MAC 计，则矩阵峰值算力为：

$$
\boxed{
P_{PE}=2N_{PE}f_C\ \mathrm{GFLOP/s}
}.
$$

### 5.3 定值

$$
N_{PE}\in\{1024,4096,8192,12288,16384\},
\qquad f_C=1\ \mathrm{GHz}.
$$

$s_{45\to7}$ 的敏感性范围取 $10$–$15$。Tempus/NVDLA 数据是低精度 MAC 锚点；正式 BF16/FP16 结果必须由目标 PE RTL 替换。

---

## 6. Vector Unit 面积

### 6.1 资料来源

AraXL 在 22 nm 的 16-lane floorplan 为 $1.015\,\mathrm{mm}\times0.736\,\mathrm{mm}$，并报告 $1.4$ GHz 下约 $44.3$ GFLOP/s。模型采用 $22$ nm 到 $7$ nm 的名义面积缩放 $s_{22\to7}=10$。

### 6.2 关键推导

单 vector lane 的 7 nm 等效面积为：

$$
a_{lane,7}=\frac{1.015\times0.736}{16s_{22\to7}}
=0.004669\ \mathrm{mm^2/lane}.
$$

单 lane、每 GHz 峰值为：

$$
p_{lane}=\frac{44.3}{16\times1.4}
=1.9777\ \mathrm{GFLOP/s/lane/GHz}.
$$

设矩阵峰值与 vector 峰值的目标比例为 $r_{PV}$，则 vector lane 数由矩阵规模推导：

$$
\boxed{
N_{vec}=\left\lceil
\frac{P_{PE}}{r_{PV}p_{lane}f_C}
\right\rceil
}.
$$

Vector Unit 面积为：

$$
\boxed{
A_{vec}=N_{vec}a_{lane,7}
}.
$$

### 6.3 定值

$$
r_{PV}=10,
\qquad s_{22\to7}=10.
$$

$r_{PV}$ 的敏感性范围取 $8$–$12$，$s_{22\to7}$ 的敏感性范围取 $8$–$12$。AraXL 是 64-bit vector/FPU 锚点，对低精度 LLM vector 运算偏保守。

### 6.4 五档计算结果

| $N_{PE}$ | $N_{vec}$ | $A_{PE}$ / mm² | $A_{vec}$ / mm² | $A_{comp}=A_{PE}+A_{vec}$ / mm² | $P_{PE}@1$ GHz |
| -------: | --------: | -------------: | --------------: | ------------------------------: | -------------: |
|    1,024 |       104 |         0.0481 |          0.4856 |                          0.5337 |  2.048 TFLOP/s |
|    4,096 |       415 |         0.1925 |          1.9376 |                          2.1302 |  8.192 TFLOP/s |
|    8,192 |       829 |         0.3851 |          3.8706 |                          4.2557 | 16.384 TFLOP/s |
|   12,288 |     1,243 |         0.5776 |          5.8036 |                          6.3812 | 24.576 TFLOP/s |
|   16,384 |     1,657 |         0.7701 |          7.7366 |                          8.5067 | 32.768 TFLOP/s |

---

## 7. SRAM 面积

### 7.1 面积拆分与资料来源

SRAM 由 bitcell/局部阵列、bank 外围和顶层多 bank 聚合网络组成。概念式为：

$$
A_{SRAM}=8a_{cell}K_B+a_{bank}n_b+A_{top}.
$$

- $K_B$：SRAM 容量，byte；
- $a_{cell}$：含局部阵列损耗的等效单 bit 面积，$\mathrm{mm^2/bit}$；
- $a_{bank}$：单 bank decoder、precharge、sense amplifier 和内部 MUX 面积；
- $n_b$：物理 bank 数；
- $A_{top}$：顶层 bank selector、聚合 MUX、仲裁、宽总线与端口网络面积。

TX8 对 $3$ MiB、$256$ GB/s L1SPM 的名义锚点为 $10$ MGE。NVSim 用于确认容量、bank 外围和端口宽度增加时的趋势及宏块横纵比；由于其工作模型为 32 nm，不直接决定 7 nm 总面积。

### 7.2 容量项

将 TX8 锚点中的 $8$ MGE 分配给容量、阵列和 bank 内外围：

$$
\boxed{
A_{SRAM,cap}=g_7\frac{8}{3}K
}.
$$

$K$ 的单位为 MiB，因此 $K=3$ 时容量项为 $8g_7$。

### 7.3 带宽与顶层外围项

将锚点中的 $2$ MGE 分配给 $256$ GB/s 的顶层端口、MUX、仲裁和宽总线：

$$
\boxed{
A_{SRAM,bw}=g_7\frac{B_S}{128}
}.
$$

$B_S$ 是 SRAM 聚合峰值带宽，单位 GB/s。该项吸收 $A_{top}$ 的一阶增长，不表示 bitcell 面积随带宽改变。

### 7.4 SRAM 总宏面积

先分别计算容量项和带宽外围项，再相加：

$$
\boxed{
A_{SRAM}=A_{SRAM,cap}+A_{SRAM,bw}
=g_7\left(\frac{8}{3}K+\frac{B_S}{128}\right)
}.
$$

锚点回代：

$$
A_{SRAM}(3,256)=10g_7=0.8748\ \mathrm{mm^2}.
$$

### 7.5 聚合位宽和 bank 数

设 SRAM 工作频率为 $f_S$ GHz、有效传输率为 $\eta_S$：

$$
W_{S,port}=\left\lceil\frac{8B_S}{f_S\eta_S}\right\rceil\ \mathrm{bit}.
$$

设单 bank 最大数据位宽为 $W_b^{max}$，则：

$$
\boxed{
n_b=\max\left(16,\left\lceil\frac{W_{S,port}}{W_b^{max}}\right\rceil\right)
}.
$$

名义值 $f_S=1$ GHz、$\eta_S=1$、$W_b^{max}=512$ bit。在 $B_S=256$–$1024$ GB/s 范围内，$W_{S,port}=2048$–$8192$ bit，故 $n_b=16$，单 bank 位宽为 $128$–$512$ bit。

---

## 8. SRAM 长宽与 Core 尺寸

### 8.1 SRAM 横纵比来源

NVSim 对 $16$ banks 的规则 $4\times4$ 同向排列给出以下长短边比。NVSim 只提供形状，面积仍由第 7 节模型决定。

|总容量 $K$|每 bank 容量|SRAM 长短边比 $q_S=L_S/T_S$|
|---:|---:|---:|
|1 MiB|64 KiB|2.001|
|2 MiB|128 KiB|2.152|
|3 MiB|192 KiB|2.404|
|4 MiB|256 KiB|2.609|

其中 $L_S$ 为 SRAM 长边，$T_S$ 为 SRAM 短边。

### 8.2 SRAM 长宽计算

$$
\boxed{
L_S=\sqrt{A_{SRAM}q_S},
\qquad
T_S=\sqrt{\frac{A_{SRAM}}{q_S}}
}.
$$

上述 $L_S\times T_S$ 是 SRAM banks 规则拼接后的等效基准矩形，仅用于保留 NVSim 校准的宏形状信息。Core 内统一把相同 banks 重排为底边加左右侧臂的 U 型；U 型只改变 bank 组合形状，不改变 $A_{SRAM}$、bank 数和聚合带宽。

### 8.3 Core 尺寸关系

定义 $A_{alloc}$ 为通信、控制、计算和 SRAM 的可分配面积。实现余量为：

$$
A_{impl}=\delta_{impl}A_{alloc}.
$$

core 总面积为：

$$
A_{core}=A_{alloc}+A_{impl}.
$$

Core 外形固定为 $1{:}1$，因此两条边均由总面积直接确定：

$$
\boxed{
W_{core}=L_{core}=\sqrt{A_{core}},
\qquad q_{core}=1,
\qquad M_{SRAM}=\mathrm{U}}
$$

其中 $M_{SRAM}$ 是 SRAM 布局模式。Die 的 core 节距不再受 $L_S$ 单独锁定；U 型底边和侧臂围绕中央计算/控制区排布，顶部仍保留连续通信带。该关系给出解析外框，仍需 macro placement 验证侧臂厚度、中央净空和顶部通信带无重叠。

### 8.4 布局规律及原因

1. **统一采用 U 型 SRAM。** SRAM banks 沿底边和左右侧臂分布，为中央计算区和顶部通信带留出规则空间；等效矩形 $L_S\times T_S$ 只用于宏形状校准。
2. **局部总线贴近 SRAM 上边缘。** PE、Vector、控制核和 DTE 到 banks 的多客户端选择在局部完成，避免横跨整个 core 的全局 crossbar。
3. **PE 阵列位于中央。** 计算是最大逻辑块，中央放置可均衡到 Vector、控制和 SRAM 的距离。
4. **Vector 和控制核放在 PE 阵列侧边。** Vector 可同时服务逐元素计算和 DCA reduce；控制核靠近 DTE 可缩短调度路径。
5. **router、NI、DTE 和 link buffer 靠顶部边缘。** 顶部直接面向相邻 tile，缩短跨 tile link，并与底部 SRAM 端口拥塞区分离。
6. **宽全局链路优先走高层金属。** SRAM 宏占用低/中层资源；高层金属可跨越宏块，但仍需检查 track、shielding 和 repeater。
7. **规则化边界优于集中式长互连。** FlooNoC、TeraNoC 和 Dojo 都表明，规则 mesh 和短局部路径比超大全局 xbar 更容易获得高利用率和时序收敛。
8. **通信门数与物理影响分开判断。** router 门数可能只有几个百分点，但边界 pin、宽链路和长线缓冲仍可能扩大实际 footprint。
9. **router 标准单元可在顶部通信带内 flatten。** 不把 router 强制做成孤立硬宏，可利用 NI、DTE 和边界缓冲之间的零散空间，并分散长线寄存器。
10. **相邻 core 按方形边长规则拼接。** 两个方向都以 $\sqrt{A_{core}}$ 为 core 节距基准；U 型 SRAM 仍保持直线形顶部通信边界，便于 mesh abutment、时钟/电源复制及跨 core 链路对齐。

---

## 9. 物理可行性约束

### 9.1 DTE 带宽约束

**来源。** TX8 DTE channel 是独立数据通路；并发带宽可按 channel 峰值相加。

**分析。** 目标注入带宽不能超过所有 channel 的持续峰值：

$$
\boxed{CB_{ch}\ge B}.
$$

主扫描取最小满足值 $C=C^\star(B)$。

### 9.2 SRAM 带宽约束

**来源。** TX8 数据流要求 SRAM 同时服务计算和通信；若 $B_S<B$，DTE/NoC 的目标注入带宽无法持续获得本地数据。

**分析。** 最低约束为：

$$
\boxed{B_S\ge B}.
$$

实际实现还应为 PE 读写保留余量；本式只是不会被通信端口立即限速的必要条件。

### 9.3 SRAM 容量下界

**来源。** TX8 基线 L1SPM 为 $3$ MiB；TPU、Dojo、FlooNoC 和大模型 tile 均采用分布式片上存储容纳权重/激活分块和通信缓冲。为覆盖更细粒度的 $16\times16$ mesh，本次把原扫描中已有的 $1$ MiB 档纳入主可行域。

**分析。** 主可行域采用较高下界：
$$
\boxed{K\ge1\ \mathrm{MiB}}.
$$

$1$ MiB 是本模型的容量下界；是否足以容纳具体大模型 tile 的工作集仍须由 workload mapping 验证。

### 9.4 SRAM bank 数和单 bank 位宽

**来源。** TX8 采用 $16$ banks；NVSim/CACTI 表明，过宽单 bank 端口会增加 decoder、MUX、bitline 和时序负担。

**分析。** 
$$
\boxed{n_b\ge16},
\qquad
\boxed{\frac{W_{S,port}}{n_b}\le W_b^{max}=512\ \mathrm{bit}}.
$$

### 9.5 SRAM 宏形状和时序

**来源。** NVSim 提供早期长宽趋势；正式可实现性必须由目标 SRAM compiler 验证。

**分析。** 

$$
\boxed{q_S\le3},
\qquad
\boxed{t_{cycle,S}\le1\ \mathrm{ns}}.
$$

$t_{cycle,S}$ 是 SRAM 周期；32 nm NVSim 不能证明 7 nm 的 $1$ ns 时序，因此该约束在解析扫描中标记为“待 compiler 验证”。

### 9.6 计算规模下界和上界

**来源。** 原四档 PE 对应 $8.192$–$32.768$ TFLOP/s@1 GHz；新增 $N_{PE}=1024$ 档对应约 $2.048$ TFLOP/s，用于探索更细粒度 mesh。上界仍受面积和功耗约束。

**分析。** 

$$
\boxed{0.5\le A_{comp}\le8.6\ \mathrm{mm^2}}.
$$

新增最低档的未舍入面积为 $0.53370912\,\mathrm{mm^2}$，满足 $0.5\,\mathrm{mm^2}$ 下界。

### 9.7 计算与 SRAM 平衡约束

**来源。** FlooNoC tile 的计算约 $63\%$、SPM 约 $24\%$；Dojo、TPU 和 Titan/Fovea 都将计算与片上存储作为耦合设计变量。跨架构比例不能直接照搬，因此采用宽松工程 guard。

**分析。** 删除“最大计算、最小 SRAM”的明显失衡点：

$$
\boxed{\frac{A_{comp}}{A_{SRAM}}\le6}.
$$

该值不是通用物理常数，应在 workload mapping 后用算术强度和 tile working set 替换。

### 9.8 通信面积约束

**来源。** FlooNoC 中 NoC+AXI xbar+DMA 约占 tile $12.2\%$；TX8 含更宽的本地路径和更保守的 DTE，因此采用更宽松上限。

**分析。** 

$$
\boxed{\frac{A_{comm}^{min}}{A_{core}}\le0.25},
\qquad
\boxed{A_{comm}^{min}\le30\lambda_7=4.0375\ \mathrm{mm^2}}.
$$

面积满足不代表边界一定可布线；$B=384/512$ GB/s 的点仍需检查 pin density、track 和 repeater。

### 9.9 Core 总面积约束

**来源。** Titan 和 Fovea 强调 die/core 面积、边界访问、同构 tiling 和物理不可布局区必须联合检查。$13\,\mathrm{mm^2}$ 是当前 TX8-like die 预算下的 early-floorplan guard。

**分析。** 

$$
\boxed{A_{core}\le13\ \mathrm{mm^2}}.
$$

该上限应在 die 层根据 core 数、HBM/D2D shoreline、PHY、供电、散热和良率重新计算。

### 9.10 Core 长宽比约束

**来源。** FlooNoC 的 compute tile 约为 $0.75\,\mathrm{mm}\times1.5\,\mathrm{mm}$，证明规则外形有利于宏块排列和边界对接；Fovea 将长宽比作为 modeled-feasible 条件。本次为使两个 mesh 方向具有相同节距，进一步固定为方形 core。

**分析。** Core 两边均由总面积平方根确定，SRAM 统一采用 U 型：

$$
\boxed{q_{core}=1},
\qquad
\boxed{W_{core}=L_{core}=\sqrt{A_{core}}}.
$$

U 型把 SRAM banks 分布到底边和两侧，在总面积不变时适配方形外框。该解析变换只保证外形比例，仍需 macro placement 检查 U 型侧臂、中央计算区和顶部通信带是否无重叠且可布线。

### 9.11 长线、边界和路由约束

**来源。** FlooNoC 对超过约 $1$ mm 的链路加入缓冲；TeraNoC 显示逻辑面积很小的互连也可能因拥塞显著扩大 footprint。

**分析。** 以下项目不在解析面积中硬编码数值，但必须作为后续签核条件：

$$
\boxed{
W_{edge,used}\le W_{edge,max},
\qquad
t_{wire}+t_{logic}\le N_{pipe}T_{clk}
}.
$$

$W_{edge,used}$ 是顶部边缘实际占用的金属轨道，$W_{edge,max}$ 是可分配轨道预算；$t_{wire}$、$t_{logic}$、$T_{clk}$ 和 $N_{pipe}$ 分别是线延迟、逻辑延迟、时钟周期和流水级数。

---

## 10. 参数严格分类

### 10.1 独立搜索参数

| 符号         | 含义                     | 单位   | 扫描值                          | 角色                      |
| ---------- | ---------------------- | ---- | ---------------------------- | ----------------------- |
| $n_{ctrl}$ | 控制核数                   | 个    | $1,2$                        | 外层搜索                    |
| $r$        | router 功能档位            | —    | base、broadcast、reduce、both   | 外层搜索                    |
| $B$        | 目标注入带宽                 | GB/s | $64,128,192,256,384,512$     | 外层搜索                    |
| $C$        | DTE channel 候选数        | 个    | $1,2,3,4,5,6$                | 通信内层搜索；主结果只保留 $C^\star$ |
| $K$        | SRAM 容量                | MiB  | $1,2,3,4$                    | 外层搜索；主可行域要求 $K\ge1$     |
| $B_S$      | SRAM 聚合带宽              | GB/s | $256,512,768,1024$           | 外层搜索                    |
| $N_{PE}$   | 矩阵 MAC-equivalent PE 数 | 个    | $1024,4096,8192,12288,16384$ | 外层搜索                    |

### 10.2 非独立参数：依赖推导或固定校准

|符号|含义|定值或依赖关系|类别|
|---|---|---|---|
|$g_7$|ASAP7 raw GE 面积系数|$0.08748\,\mathrm{mm^2/MGE}$|固定校准|
|$u_{sc}$|标准单元利用率|$0.65$|固定校准|
|$\lambda_7$|placed GE 面积系数|$g_7/u_{sc}$|依赖推导|
|$\delta_{impl}$|时钟、电源、halo 和余留布线比例|$0.10$|固定校准|
|$f_N$|NoC 频率|$1$ GHz|固定校准|
|$\eta_N$|NoC 有效传输率|$1.0$|固定校准|
|$b$|归一化目标带宽|$B/256$|依赖推导|
|$W_{link}$|NoC payload 位宽|$\lceil8B/(f_N\eta_N)\rceil$|依赖推导|
|$B_{ch}$|单 DTE channel 峰值|$128$ GB/s|TX8 固定校准|
|$C^\star$|最小 DTE channel 数|$\lceil B/B_{ch}\rceil$|依赖推导|
|$m_R$|router 功能倍率|由 $r$ 查表|依赖推导|
|$A_{DTE}$|DTE 面积|$B,C^\star,u_{sc}$|依赖推导|
|$A_{local}$|局部总线/MUX 面积|$B,C^\star,u_{sc}$|依赖推导|
|$A_{NI}$|NI 面积|$B,u_{sc}$|依赖推导|
|$A_{link}$|链路端点与缓冲面积|$B,u_{sc}$|依赖推导|
|$A_R$|router 面积|$B,r,u_{sc}$|依赖推导|
|$A_{comm}^{min}$|最小通信面积|五项通信面积之和|依赖推导|
|$A_{ctrl}$|控制核面积|$n_{ctrl},u_{sc}$|依赖推导|
|$s_{45\to7}$|PE 面积缩放|$12$|固定校准|
|$a_{PE,7}$|单 PE 面积|$4.7005\times10^{-5}\,\mathrm{mm^2}$|固定校准|
|$f_C$|计算频率|$1$ GHz|固定校准|
|$P_{PE}$|矩阵峰值算力|$2N_{PE}f_C$|依赖推导|
|$s_{22\to7}$|Vector 面积缩放|$10$|固定校准|
|$a_{lane,7}$|单 vector lane 面积|$0.004669\,\mathrm{mm^2}$|固定校准|
|$p_{lane}$|单 lane 每 GHz 峰值|$1.9777$ GFLOP/s|固定校准|
|$r_{PV}$|矩阵/vector 峰值比|$10$|固定校准|
|$N_{vec}$|vector lane 数|由 $P_{PE},r_{PV},p_{lane},f_C$ 推导|依赖推导|
|$A_{PE}$|PE 阵列面积|$N_{PE}a_{PE,7}$|依赖推导|
|$A_{vec}$|Vector Unit 面积|$N_{vec}a_{lane,7}$|依赖推导|
|$A_{comp}$|总计算面积|$A_{PE}+A_{vec}$|依赖推导|
|$f_S$|SRAM 频率|$1$ GHz|固定校准|
|$\eta_S$|SRAM 有效传输率|$1.0$|固定校准|
|$W_b^{max}$|单 bank 最大位宽|$512$ bit|固定校准|
|$W_{S,port}$|SRAM 聚合端口位宽|$\lceil8B_S/(f_S\eta_S)\rceil$|依赖推导|
|$n_b$|SRAM bank 数|$\max(16,\lceil W_{S,port}/W_b^{max}\rceil)$|依赖推导|
|$A_{SRAM,cap}$|SRAM 容量面积项|$g_7(8K/3)$|依赖推导|
|$A_{SRAM,bw}$|SRAM 带宽/外围面积项|$g_7B_S/128$|依赖推导|
|$A_{SRAM}$|SRAM 宏总面积|容量项与带宽项之和|依赖推导|
|$q_S$|SRAM 长短边比|由 $K$ 查 NVSim 校准表|依赖推导|
|$L_S,T_S$|SRAM 长边、短边|由 $A_{SRAM},q_S$ 推导|依赖推导|
|$A_{alloc}$|可分配面积|通信、控制、计算和 SRAM 之和|依赖推导|
|$A_{impl}$|实现余量面积|$\delta_{impl}A_{alloc}$|依赖推导|
|$A_{core}$|core 总面积|$A_{alloc}+A_{impl}$|依赖推导|
|$M_{SRAM}$|SRAM 布局模式|统一固定为 U 型|固定布局|
|$W_{core}$|core 宽度|$\sqrt{A_{core}}$|依赖推导|
|$L_{core}$|core 长度|$\sqrt{A_{core}}$|依赖推导|
|$q_{core}$|core 长宽比|$L_{core}/W_{core}=1$|依赖推导|

---

## 11. 最终汇总式与约束

先计算各独立部件：

$$
\begin{aligned}
A_{comm}^{min}&=A_{DTE}+A_{local}+A_{NI}+A_{link}+A_R,\\
A_{comp}&=A_{PE}+A_{vec},\\
A_{SRAM}&=A_{SRAM,cap}+A_{SRAM,bw}.
\end{aligned}
$$

再计算可分配面积、实现余量和总面积：

$$
\boxed{
A_{alloc}=A_{comm}^{min}+A_{ctrl}+A_{comp}+A_{SRAM}
},
$$

$$
\boxed{
A_{core}=(1+\delta_{impl})A_{alloc}
}.
$$

最后统一采用 U 型 SRAM，并按方形外框计算 core 尺寸：

$$
\boxed{
W_{core}=L_{core}=\sqrt{A_{core}},
\qquad q_{core}=1,
\qquad M_{SRAM}=\mathrm{U}}
$$

代入名义定值后：

$$
\boxed{
\begin{aligned}
A_{core}=1.10\Bigg[&\frac{0.08748}{0.65}
\left(1.92+0.15C^\star+\frac{B}{256}
\left(1.68+1.20C^\star+0.672m_R\right)+4.5n_{ctrl}\right)\\
&+0.08748\left(\frac{8}{3}K+\frac{B_S}{128}\right)
+4.7005\times10^{-5}N_{PE}+0.004669N_{vec}\Bigg],\\
C^\star=&\left\lceil\frac{B}{128}\right\rceil,\\
N_{vec}=&\left\lceil\frac{2N_{PE}}{10\times1.9777}\right\rceil.
\end{aligned}
}
$$

物理约束汇总为：

$$
\boxed{
\begin{gathered}
C^\star\le6,\qquad C^\star B_{ch}\ge B,\qquad B_S\ge B,\\
K\ge1\ \mathrm{MiB},\qquad n_b\ge16,\qquad
W_{S,port}/n_b\le512\ \mathrm{bit},\\
q_S\le3,\qquad t_{cycle,S}\le1\ \mathrm{ns},\\
0.5\le A_{comp}\le8.6\ \mathrm{mm^2},\qquad
A_{comp}/A_{SRAM}\le6,\\
A_{comm}^{min}/A_{core}\le0.25,\qquad
A_{comm}^{min}\le4.0375\ \mathrm{mm^2},\\
A_{core}\le13\ \mathrm{mm^2},\qquad
q_{core}=1,\qquad M_{SRAM}=\mathrm{U},\\
W_{edge,used}\le W_{edge,max},\qquad
t_{wire}+t_{logic}\le N_{pipe}T_{clk}.
\end{gathered}
}
$$

其中最后一行必须由 early global route 和时序分析验证，不应仅凭解析面积判为物理可行。

---

## 12. 参考资料与链接

1. TX8 architecture/specification materials：核内模块、$3$ MiB/$16$-bank L1SPM、控制核、DTE、局部宽总线和 NoC 关系。
2. TX8 component area study：控制块、DTE、bus/TMNoC、SRAM、NI 和 router 的 kGE 锚点。
3. TX8 collective-router study：$135/174.5/214.5/259.5$ kGE 四档 router 及 VCT、FANOUT、DCA 组成。
4. Clark 等，[ASAP7](https://github.com/The-OpenROAD-Project/asap7)及 [RVT LEF](https://raw.githubusercontent.com/The-OpenROAD-Project/asap7sc7p5t_28/master/LEF/asap7sc7p5t_28_R_1x_220121a.lef)。
5. Fischer 等，[FlooNoC](https://arxiv.org/abs/2409.17606)：512-bit NoC、router/NI/xbar/DMA 面积和 floorplan。
6. Colagrande 等，[Collective-Capable NoC](https://arxiv.org/abs/2603.26438)：7 nm collective router、DCA 和 tile 级面积增量。
7. Dong 等，[NVSim](https://github.com/SEAL-UCSB/NVSim)，[DOI](https://doi.org/10.1109/TCAD.2012.2185930)：bank、外围、端口和长宽模型。
8. Purayil 等，[AraXL](https://arxiv.org/abs/2501.10301)及[开源实现](https://github.com/pulp-platform/AraXL)：22 nm vector floorplan 和性能。
9. Vellaisamy 等，[Tempus Core](https://arxiv.org/abs/2412.19002)：45 nm NVDLA/Tempus P&R 面积。
10. Villa 等，[Scaling the Power Wall](https://research.nvidia.com/publication/2014-11_scaling-power-wall-path-exascale)：跨节点面积缩放趋势。
11. Enright Jerger 等，[Virtual Circuit Tree Multicasting](https://doi.org/10.1109/ISCA.2008.12)。
12. Balasubramonian 等，[CACTI 7](https://users.cs.utah.edu/~rajeev/cacti7/)及[代码](https://github.com/HewlettPackard/cacti)。
13. Kahng 等，[ORION 2.0](https://escholarship.org/uc/item/5jd3c1gv)。
14. Talpes 等，[Tesla Dojo D1](https://doi.org/10.1109/MM.2023.3258906)：规则 node mesh、分布式 SRAM 和边缘通信。
15. Yu 等，[Titan / Cramming a Data Center into One Cabinet](https://doi.org/10.1145/3695053.3731016)：晶圆级计算、存储与物理面积联合探索。
16. Li 等，[Fovea](https://arxiv.org/abs/2608.03285)：物理含义约束、长宽比、边界访问和 modeled-feasible 空间。

## BibLaTeX

```bibtex
@misc{tx8_architecture, title={TX8 Architecture and Hardware Modeling Materials}, year={2026}, note={Project-provided architecture, bandwidth, and component-area references}}
@article{clark2016asap7, author={Clark, Lawrence T. and others}, title={ASAP7: A 7-nm FinFET Predictive Process Design Kit}, journaltitle={Microelectronics Journal}, volume={53}, pages={105--115}, year={2016}, doi={10.1016/j.mejo.2016.04.006}}
@article{fischer2025floonoc, author={Fischer, Tim and others}, title={FlooNoC: A 645-Gbps/link 0.15-pJ/B/hop Open-Source NoC with Wide Physical Links and End-to-End AXI4 Parallel Multi-Stream Support}, journaltitle={IEEE Transactions on Very Large Scale Integration Systems}, year={2025}, url={https://arxiv.org/abs/2409.17606}}
@inproceedings{colagrande2026collective, author={Colagrande, Luca and others}, title={A Lightweight High-Throughput Collective-Capable NoC for Large-Scale ML Accelerators}, booktitle={MLSys}, year={2026}, url={https://arxiv.org/abs/2603.26438}}
@article{dong2012nvsim, author={Dong, Xiangyu and others}, title={NVSim: A Circuit-Level Performance, Energy, and Area Model for Emerging Nonvolatile Memory}, journaltitle={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, volume={31}, number={7}, pages={994--1007}, year={2012}, doi={10.1109/TCAD.2012.2185930}}
@article{purayil2025araxl, author={Purayil, Navaneeth Kunhi and Perotti, Matteo and Fischer, Tim and Benini, Luca}, title={AraXL: A Physically Scalable, Ultra-Wide RISC-V Vector Processor Design for Fast and Efficient Computation on Long Vectors}, year={2025}, url={https://arxiv.org/abs/2501.10301}}
@article{vellaisamy2024tempus, author={Vellaisamy, Prabhu and others}, title={Tempus Core: Area-Power Efficient Temporal-Unary Convolution Core for Low-Precision Edge DLAs}, year={2024}, url={https://arxiv.org/abs/2412.19002}}
@inproceedings{villa2014scaling, author={Villa, Oreste and others}, title={Scaling the Power Wall: A Path to Exascale}, booktitle={SC}, year={2014}, url={https://research.nvidia.com/publication/2014-11_scaling-power-wall-path-exascale}}
@inproceedings{jerger2008vctm, author={Enright Jerger, Natalie and Peh, Li-Shiuan and Lipasti, Mikko H.}, title={Virtual Circuit Tree Multicasting}, booktitle={ISCA}, year={2008}, doi={10.1109/ISCA.2008.12}}
@article{balasubramonian2017cacti7, author={Balasubramonian, Rajeev and others}, title={CACTI 7}, journaltitle={ACM Transactions on Architecture and Code Optimization}, year={2017}, url={https://users.cs.utah.edu/~rajeev/cacti7/}}
@inproceedings{kahng2009orion2, author={Kahng, Andrew B. and others}, title={ORION 2.0: A Fast and Accurate NoC Power and Area Model for Early-Stage Design Space Exploration}, booktitle={DATE}, year={2009}, url={https://escholarship.org/uc/item/5jd3c1gv}}
@article{talpes2023dojo, author={Talpes, Emil and others}, title={The Microarchitecture of DOJO, Tesla's Exa-Scale Computer}, journaltitle={IEEE Micro}, volume={43}, number={3}, pages={31--39}, year={2023}, doi={10.1109/MM.2023.3258906}}
@inproceedings{yu2025titan, author={Yu, Xingmao and Jiang, Dingcheng and Deng, Jinyi and Liu, Jingyao and Li, Chao and Yin, Shouyi and Hu, Yang}, title={Cramming a Data Center into One Cabinet, a Co-Exploration of Computing and Hardware Architecture of Waferscale Chip}, booktitle={ISCA}, pages={631--645}, year={2025}, doi={10.1145/3695053.3731016}}
@article{li2026fovea, author={Li, Jinxi and Wang, Huizheng and Deng, Jinyi and Hu, Yang and Yin, Shouyi}, title={Fovea: Physical-Implication-Aware Wafer-Scale DSE with Decision-Domain-Guided Cross-Fidelity Refinement}, year={2026}, url={https://arxiv.org/abs/2608.03285}}
```
