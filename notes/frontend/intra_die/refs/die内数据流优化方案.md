
# die 内数据流编排完整方案

> 本文是 [[写作]] §5 的工程化版本。§5 讲的是"为什么这样做"，本文讲"怎么做"。
> 面向的执行者是一个能调用仿真器、能读写 IR 的 agent。

---

## 0. 一句话概括

把 §4 投影到单 die 的一张**带类型的计算通信图**，改写成**每个核一条指令流**。中间经过三步下降：

```
IR-2 (die 级动作图)
   │  ① 算子细化 + 任务划分        ← 逻辑结构，5.2 外层
   ↓
Core Group 划分 (𝒞, φ) + 核级节点集 V
   │  ② 参数与时序                  ← 5.2 内层
   ↓
完整的核级 DAG
   │  ③ 落位 + 路由 + 注入时序      ← 5.3
   ↓
Placed DAG → Simulator → makespan
```

三步之间**信息单向流动**，没有迭代闭环。最终择优发生在第 ③ 步之后。

---

## 1. 输入与输出

### 输入

| 项 | 来源 | 说明 |
|---|---|---|
| IR-2 动作图 $G=(V,E)$ | §4 跨 die 编排的投影 | 节点带**类型**（COMP / SEND / RECV / REDUCE / LOAD / STORE），边表依赖 |
| 跨 die 轮转步数 | §4 已定 | = die 数，本节只在其内部再细分 |
| 硬件参数表 | 见 §7 | 须先标定 |
| 模板库 | 见 §3.1 | 参数化展开规则，非现成子图 |

### 输出

| 项 | 内容 |
|---|---|
| $(\mathcal C,\varphi)$ | CG 的构成与节点归属 |
| 核级 DAG | 每个核的节点序列、缓冲区布局、同步点 |
| $\pi$ | 逻辑核 → 物理核的双射 |
| 路由与注入时序表 | 每条流的 XY 路径与发起时刻 |

---

## 2. 决策空间

**两类主动决策 + 一类被动后果。** 补全不是决策，不占空间维度，但决定前两类行不行得通。

### 2.1 算子细化 —— 每个节点在 CG 上怎么摊开

节点替换：$\text{template}:(\text{节点类型},\ |C_k|,\ \text{参数})\longmapsto\text{核级子图}$

**GEMM 按切分维度分类：**

- **非归约维 $m$、$n$**：不破坏归约完整性，无须核间求和，但未被切开的操作数为各核共享
  - 关键：若各核各持完整副本，其**驻留量不随核数下降**（唯一不被摊薄的一项），超出 SRAM 后各核须从片外重复读取，片外访存量放大至核数倍 → 故须轮转
  - 只切一维 → 1D-TP，环状 swizzling，**要求 CG 能构成环路**
  - $m,n$ 同切 → 2D-TP，SUMMA / 脉动 / 嵌套，**要求规则二维子网格**
- **归约维 $k$**：可与上者组合。每核只算部分和，须补一次求和。**唯一会向图中引入新节点的切分**，该归约的承担者与执行时机可调配（同 CG 内 / 拆两级交由不同 CG）

**跨 die P2P**：拆 SEND / RECV，可分属不同 CG；模式取 push 或 pull。

> **push/pull 规则**：通信处于生产者位置时用 push，处于消费者位置时用 pull——总由**非关键路径的一侧**发起搬运，使发起的核开销与延迟被另一侧的计算掩盖。
> - comm-as-producer（如 AG-GEMM）：关键路径在接收侧，其核应尽数投入计算 → 发送侧发起 → **push**
> - comm-as-consumer（如 GEMM-RS）：关键路径在发送侧 → 接收侧发起 → **pull**；且 pull 的请求可提前挂起，往返被发送侧计算掩盖
> - **例外**：push 要求接收侧预备缓冲区；接收侧 SRAM 不足时纵使通信是生产者也须退回 pull
> - ⚠️ 前提是硬件同时支持远程写与远程读，**须核实**

**push/pull 的一个副作用**：它决定 SEND/RECV 这对节点中哪个是实的（占核）、哪个是虚的（仅一个到达事件），从而直接改变两侧 die 的核数分配。

### 2.2 任务划分 —— 哪些核成组、每组干什么

只确定 CG 个数 $K$、各 CG 核数 $\{|C_k|\}$、节点到 CG 的映射 $\varphi$。
**既不改变节点集合，也不涉及各 CG 在核阵列上的落位。**

### 2.3 被动后果（补全）—— 前两类决策强制推出，零自由度

| 后果 | 触发条件 | 内容 |
|---|---|---|
| **CG 间通信** | $\varphi$ 割断一条边 | 由**语义**与**两 CG 核数配比**唯一确定收发原语（见下表） |
| **归约节点** | 切了 $k$ 维 | 同 CG 内则组内完成；分处不同 CG 则与上者合并为一次带归约的 CG 间通信 |
| **片外中转** | 每核驻留量超单核 SRAM | 该次通信的起点或终点改到 HBM，有效带宽转由片外访存带宽决定 |

**NoC 收发两端原语的 3×3 组合**（九格恰好覆盖九个原语，无空缺无重复）：

| 发送端 TX ＼ 接收端 RX | 直接写入 (Unicast) | 按偏移拼接 (Gather) | 就地累加 (Reduce) |
|---|---|---|---|
| **整份单播 Unicast** | P2P | Gather | Reduce |
| **切开分发 Scatter** | Scatter | AllToAll | ReduceScatter |
| **整份多播 Broadcast** | Broadcast | AllGather | AllReduce |

> 两端模式一定，格子唯一确定——**从割边到原语的整条链上，编译器没有任何一处需要作出选择**。
> ⚠️ 须核实硬件是否真按这两组三档划分。

**一处须拍板**：集合原语的**算法实现**（环状 vs 树状 AllReduce）显然可选且决定形状要求。建议划法：**类型由补全唯一确定，算法实现归入算子细化**。

---

## 3. 模板库

### 3.1 GEMM 模板

| 模板 | 切分维 | 组内数据流 | 归约 | 形状要求 |
|---|---|---|---|---|
| 只切 $k$ | $k$ | 无 | 是 | 任意（**填缝料**） |
| 1D-TP Ring Swizzling | $m$ 或 $n$ | 环状 swizzling | 否 | 环路 / 哈密顿环 |
| SUMMA | $m{+}n$ | 行列广播 | 否 | 二维子网格 |
| 脉动 (Systolic) | $m{+}n$ | 近邻推进 | 否 | 二维子网格 |
| 脉动/SUMMA 嵌套 | $m{+}n$ | 两级嵌套 | 否 | 二维子网格 |
| Split-K SUMMA | $m{+}n{+}k$ | 行列广播 | 是 | 二维子网格 |
| P2P 拆分 | — | 无 | 否 | 任意 |
| **复制式非归约维切分** | $m$ 或 $n$ | **无** | 否 | **任意** |

最后一条很重要：它使"轮不轮转"本身成为一个由 SRAM 余量裁定的决策，而非无条件必须轮转。

**库里没有的三样**：参数（$A\times B\times C$ 分解、tile 形状）、补全、**可行性**。

**审稿人问题预案**："方法是否受限于库的完备性？"——库不必完备，任何库外实现总可退化为"只切 $k$ + 一次归约"（任意核数、任意形状均可用），故空间恒非空，方法只失最优性不失可行性。

### 3.2 展开约定

同一 CG 内的核共享同一子图、同一套缓冲区布局、同一同步域，指令流由**一份参数化模板按核索引批量展开**，不逐核定制。

---

## 4. 目标函数与代价评估

### 4.1 目标函数

设单位操作被重复 $R$ 次（= 跨 die 轮数 × 两级分裂份数）：

$$T=R\cdot\max_{k}\Big(\max\big(T_k^{\text{comp}},T_k^{\text{comm}}\big)+T_k^{\text{sync}}\Big)+T_{\text{fill}}+T_{\text{drain}}$$

- 内层 $\max$：CG 内计算通信交叠（与 §4 的 $t_{\text{phase}}=\max(t_{\text{comp}},t_{\text{comm}})$ 同构，下沉一级）
- 外层 $\max_k$：各 CG 并发，慢者定局
- $R$：份数越大 fill/drain 摊得越薄，但 $T^{\text{sync}}$ 被放大 $R$ 倍

**被动后果全部自动进入该式，无须另设惩罚项。** phase 化启用时按 phase 分别求值再相加，计入 $t_{\text{shift}}$。

**只有这一个目标函数。** 不要引入 $\text{Score}=\sum w_ip_i$ 之类的加权和——超参无处可辩，且与最终评估不同量纲。原先那些"软指标"（计算利用率、通信掩盖率、同步开销占比）是 $T$ 的分项，留作诊断输出。

### 4.2 三级保真度

| 级别 | 用途 | 调用量级 | 单次代价 | 实现 |
|---|---|---|---|---|
| **① 增量解析** | R1/R2 剪枝、内层贪心 | $10^7$ | $O(1)$ | 累加器 |
| **② 完整解析** | 外层候选排序 | $10^4$ | $O(\|V\|)$ | 全式求值 |
| **③ 事件级模拟** | top-$k$ 精评 | $10^1$ | 高 | 三部件时间线 |

**① 的实现**：每个 CG 维护四个累加器 —— $\Sigma T^{\text{comp}}$、$\Sigma T^{\text{comm}}$、$\Sigma$工作集、$\Sigma$缓冲区。每放入一个节点即累加：

$$T^{\text{comp}}_v=\frac{2\,m_vn_vk_v}{\eta_{\text{PE}}\,\eta_k\,P_{\text{peak}}\,|C_k|},\qquad T^{\text{comm}}_v=T_{\text{setup}}+T_{\text{lat}}+\frac{V_v}{B_{\text{eff}}}$$

$\eta_{\text{PE}},\eta_k$ 由 tile 形状查表；$T_{\text{setup}},T_{\text{lat}},B_{\text{eff}}$ 按**通路类型**查表（四类，见 §7）。

**累加器只增不减**，故约束一旦违反其任何后继必违反，剪整棵子树安全。这是"增量可判定"的全部实现。

**③ 抓什么**：DTE 通道与描述符 FIFO 竞争、SPM bank 冲突、控制核发射串行、跨 CG 同步的实际等待。这些在 ①② 中按零竞争处理。

### 4.3 一处必须记住的乐观近似

①②③ 中 $B_{\text{eff}}$ 均取**零负载值**，链路拥塞要到落位之后才能评估。故 §5.2 阶段的所有估值都是乐观的——这保证不误剪，但也意味着**最终排序必须放到落位之后**。

---

## 5. 搜索流程

### 5.1 分层依据

> 决策之间的依赖是否单向，决定它们能否分层：互为前提者同层，单向者分层。

- **外层**：算子细化 + 任务划分（互为前提，必须联合）
- **内层**：核数、分裂份数、时序（归属定后才有定义，且不反过来改变归属）

外层用**最乐观的占位值**代入内层参数求值 → 保证不误剪 → 分层不损失可行域。

### 5.2 外层

```python
def outer_search(G, template_lib, hw):
    # Step 1: 细化候选集（只筛掉与划分无关的不可行取值）
    refine_cands = []
    for combo in enumerate_refinements(G, template_lib):
        if violates_single_core_hw_limits(combo, hw):   # 见下 6.1
            continue
        refine_cands.append(combo)                      # 整体保留，不敲定

    results = []
    for r in refine_cands:
        V = apply_refinement(G, r)                      # 节点替换
        order = bfs_order(V, anchor=find_gemm(V))       # BFS 定序
        results += dfs_partition(V, order, placeholder_cores(hw))
    return topk(results, key=lambda c: full_analytic_cost(c))


def dfs_partition(V, order, cores, depth=0, state=None):
    if depth == len(order):
        return [finalize(state)]
    out = []
    for cg in candidate_cgs(state, order[depth]):
        s2 = place(state, order[depth], cg)             # O(1) 累加器更新
        if R1_violated(s2) or R2_violated(s2) or R3_violated(s2):
            continue                                    # 剪整棵子树
        out += dfs_partition(V, order, cores, depth+1, s2)
    return out
```

**结构性收缩三条**（缩空间，不查约束）：

1. **GEMM 锚定 CG₀** —— 破除组标号对称性（$\div K!$）
2. **同类型节点同 CG** —— 有效 $|V|$ 从十几压到类型数 6–8（$B(15)\to B(7)$，**$1.6\times10^6$ 倍，削减的主力**）
3. **核数两阶段** —— 外层用占位核数，结构定了再搜实值

**定序 BFS、搜索 DFS 的理由**：
- BFS 定序 → 离 GEMM 越近约束越紧，强约束节点先决定，下游分支被大量削减
- DFS 搜索 → branch-and-bound，内存 $O(\text{深度})$，放一个节点即可增量剪枝；BFS 搜索的前沿按 Bell 数爆炸且须到最后一层才有第一个完整划分

### 5.3 内层

**(a) 图变换**（不增删依赖，只改时间位置与粒度；不改 CG 构成与归属）

| 变换 | 作用 | 前提 |
|---|---|---|
| **节点提前 hoist** | 异步 recv/load 移到最早合法点，延时被其间计算掩盖 | 可异步发起、有空 buffer slot、无 WAR 冲突 |
| **节点推后 sink** | 搬运推迟到临用前，缩短 SRAM 存活窗口 | 仍在消费者之前，晚发不致 stall |
| **节点重排** | ① 无依赖 recv 换序，先到先算 ② 部分和边到边累加，省缓冲 | ① 无数据依赖 ② 满足交换/结合律（⚠️ 浮点舍入会变） |
| **节点分裂** | 切成若干步，与相邻节点交叠 | 各段继承原节点归属 |
| **跨 CG send 错峰** | 摊平 $\text{load}(e,t)$ 峰值 | 只调顺序不改内容与路由；**收益须待落位才可评估，取值移交 §5.3** |

> **hoist 与 sink 在同一块存储上反向拉扯**——凡为掩盖延时而提前的，都以更长驻留为代价——故不存在可贯彻到底的方向，必须作为一对取值联合确定。

**(b) 参数搜索**

**核数**：目标是使各 CG 完工时间尽量接近（因目标函数是 $\max_k T_k$）。依据是三类 CG 的标度关系不同：

| CG 类型 | $T_k(N_k)$ 行为 |
|---|---|
| 计算为主 | 近似 $\propto 1/N_k$，直到 tile 碎化使 $\eta_{\text{PE}}$ 下降 |
| 跨 die 收发 | 链路带宽打满后不再下降（单边单向 $1\,\text{TBps}/256\,\text{GBps}\approx4$ 核即饱和），此后增核空耗 |
| 片外访存 | 同理受访存带宽所限 |

```python
def balance_cores(cgs):
    alloc = proportional_init(cgs)                  # 按工作量比例
    while True:
        src = argmin(alloc, key=completion_time)    # 最早完工
        dst = argmax(alloc, key=completion_time)    # 最晚完工
        trial = move_one_core(alloc, src, dst)
        if makespan(trial) < makespan(alloc):       # 严格下降才接受
            alloc = trial
        else:
            return alloc                            # 必然终止
```

终止性：每次接受的移动严格减小目标函数，核数分拆有限。分拆总数不过数百，必要时可直接枚举验证全局最优。

**分裂份数**：决定一个节点拆成几步。与外层切分分属两维——外层定**空间上**摊给多少核，份数定**时间上**分成多少步。受两端夹持：

$$V_{\text{chunk}}\ge V_{\text{half}}=B_{\text{DTE}}\cdot T_{\text{su}}\quad\text{(下界:打满带宽)},\qquad R\cdot T_{\text{bar}}\le T_{\text{comp,all}}\quad\text{(上界:同步开销)}$$

区间很窄，直接穷举。

### 5.4 落位（§5.3 Physical Mapper）

**输入**：外层输出的 top-$k$ 候选，各携带 $(K,\{|C_k|\},\varphi)$、形状签名、端口方向需求、CG 间通信矩阵与已匹配的原语。

**代价函数（第一级 · 加权跳数）**

$$\text{Cost}(\pi)=\sum_{i\neq j}\text{bytes}(i,j)\cdot\text{seam}(i,j,\pi)+\sum_k \text{bytes}_{\text{intra}}(k)\cdot\text{hop}_{\text{intra}}(k,\pi)+\sum_k \text{bytes}_{\text{io}}(k)\cdot\text{hop}(k,\text{port},\pi)$$

第一项的权重直接取自 CG 间通信矩阵。**不引入 relay group 概念**（试用后弃置）。

**代价函数（第二级 · 链路负载）** —— 用于 top-$k$ 精评：

1. 按 XY 路由把每条流 $f$ 映射到 $\text{path}(f)$
2. $\text{load}(e)=\sum_{f\ni e}B_{\text{demand}}(f)$
3. $\rho(e)=\text{load}(e)/\text{cap}(e)$
4. $T_{\text{queue}}(e)\approx T_0(e)\cdot\dfrac{\rho(e)}{1-\rho(e)}$
5. 在 $\rho\ge1$ 的瓶颈集合上跑 max-min water-filling 解每条流的 $B_{\text{eff}}$

> 集合通信的拥塞**结构化且可预先算出**——扇入扇出度由已匹配的原语给定，流集合与带宽需求编译期完全已知，$\rho(e)$ 无须仿真。

**构造流程**

```
Step 0  可行性筛：按形状签名判定能否嵌入
        · 哈密顿环 / 规则二维子网格 = 硬要求
        · 只切 k 的 CG 形状任意 → 填缝料
        · 不通过 → 淘汰该候选，退回次优
Step 1  定序：按流量由大到小（流量是 hop 的系数，大的先占贴合位）
Step 2  向外扩张放置：
        · 有端口方向要求的 CG 先落边缘（自由度最小）
        · 其余按接缝流量依次贴合扩张，每块可旋转翻转
        · 形状任意的 CG 最后填空隙
Step 3  局部优化：交换 + 滑移，模拟退火
Step 4  精评：对 top-k 布局跑第二级，得修正后 B_eff
```

**路由与注入时序**

默认 XY 维序路由（无死锁、零配置）。其下编译器只剩三种手段：

1. **布局** —— 改 $(\text{src},\text{dst})$，**唯一**能改变空间路径的手段
2. **相位调度** —— 路径不可改，注入时刻可改，摊平 $\text{load}(e,t)$ 峰值（内层移交的 send 错峰在此定值）
3. **集合算法分解** —— 环/树/递归折半落到不同链路集合

可编程源路由仅作逃生口，须单独死锁验证。

### 5.5 最终择优

**发生在落位之后，不在划分之后。** §5.2 输出 top-$k$ 时 $B_{\text{eff}}$ 是零负载的乐观值；落位给出修正值，最终比较在此完成。信息单向流动，落位从不回改划分，只是重新排序并淘汰不可嵌入者。

---

## 6. 约束与剪枝规则

### 6.1 单核硬件约束（与划分无关，可在展开划分之前先行施用）

| 约束 | 形式 |
|---|---|
| 阵列填充率 | $\eta_{\text{PE}}=\dfrac{m_tn_t}{\lceil m_t/a_h\rceil a_h\cdot\lceil n_t/a_w\rceil a_w}\ge\eta_{\text{PE}}^{\min}$ |
| 归约维深度 | $\eta_k=\dfrac{k_t}{k_t+L}$，即 $k_t\ge\dfrac{L\,\eta^{\min}}{1-\eta^{\min}}$ |
| SRAM 容量 | 操作数放得下 |
| SRAM 带宽 | $B_{\text{comp}}+B_{\text{dte}}\le B_{\text{SRAM}}$，其中 $B_{\text{comp}}\approx(a_h+a_w)Fb$ |
| DTE 半带宽点 | $V_{\text{chunk}}\ge V_{\text{half}}=B_{\text{DTE}}T_{\text{su}}$ |
| DTE block 数 | $D_{\text{req}}\le D$ |

### 6.2 三条剪枝规则（构造中增量求值）

| | 层面 | 内容 | 增量性来源 |
|---|---|---|---|
| **R1** | 硬件资源 | $\dfrac{W_{\text{GEMM}}(C_k)}{\|C_k\|}+\sum_{j\in\text{并发通信}(C_k)}b_j\le S_{\text{core}}$ | 累加器单调递增 |
| **R2** | 性能 | $\min_k T_k/\max_k T_k>R_{\min}$（**跨 CG** 完工时间平衡） | 占位核数下可估 |
| **R3** | 数据流语义 | 与 GEMM 紧耦合的那侧通信须与 GEMM 同 CG | 由节点类型直接读出，$O(1)$ |

**R2 必须用跨 CG 版本，不能用单 CG 内的 $\min(T_{\text{comm}},T_{\text{comp}})/\max(\cdot)$。** 后者会把"一个 CG 纯算、一个 CG 纯传"这类解耦方案全部剪掉——而那正是本工作要找的方案。目标函数是 $\max_kT_k$，闲置本身不是罪，拖长 $\max_kT_k$ 才是。

**R3 的具体形态**：GEMM 为消费者时紧耦合侧是 RECV（剪掉 SEND+GEMM 同组、RECV 孤立的划分）；GEMM 为生产者时紧耦合侧是 SEND。**与 push/pull 规则同源**——同一个生产者-消费者关系，一处定实现方向，一处定归属。

---

## 7. 须先标定的硬件参数

### 7.1 已有典型值

| 项 | 值 |
|---|---|
| GEMM 算力 | 1000 TFLOPS |
| 单 router 延时 | 5 ns |
| SRAM 访问 | 50 ns |
| HBM 访问 | 500 ns |
| C2C 端口 | 1000 ns |
| NoC 带宽 | 10 TB/s |
| SRAM 带宽 | 20 TB/s |
| HBM3 带宽 | 3.35 TB/s |
| D2D link | 10 TB/s |
| 控制核 issue | 触发 30 ns / 配置 200 ns / 同步 40 ns |
| DTE 启动 | 10 ns |
| 计算单元启动 | 100 ns |
| Reduce RX 增量延时 | 100 ns |
| 单核 SPM | 3 MB（约 16 × 192 KB bank） |

### 7.2 DTE 四类通路（$T^{\text{comm}}$ 查表的依据）

| 通路 | $T_{\text{setup}}$ | $T_{\text{lat}}$ | $B_{\min}$ |
|---|---|---|---|
| 单 die SPM2SPM | 200–500 ns | 4–16 ns | NoC link 带宽 |
| 单 die HBM2SPM | 200–500 ns | 105–115 ns（HBM 首字节 ~100 ns 主导） | 单 tile 取数时 NoC≈HBM；多 tile 时 HBM 成瓶颈 |
| 跨 die SPM2SPM | 300–800 ns | 30–200 ns | $\min(B_{\text{SPM}},B_{\text{NoC}},B_{\text{D2D}}/N)$ |
| 跨 die HBM2HBM | 300–800 ns | 100–300 ns | $\min$(HBM读, NoC, C2C, HBM写) |

**跨 die 带宽公式**：
$$B_{\text{d2d}}(\text{方向})=\min\Big(\textstyle\sum_{\text{朝该方向的 tile}}N\cdot 1024b\cdot f,\ \text{NoC 朝该 die 边的截面带宽},\ \text{C2C SerDes 总宽度}\Big)$$
$$\text{单核实际带宽}=\min(B_{\text{noc}},\ B_{\text{d2d,dir}}/N)$$

**所有核发往同一方向会造成带宽下降** —— 这是跨 die 2D TP 均摊四方向流量的价值所在。

### 7.3 DTE 占用（决定并行性）

- **TX 侧**：$T_{\text{TX,busy}}=T_{\text{setup}}+\text{size}/B_{\text{TX,min}}$，**在 NoC 传输途中即释放**，不等 ACK；提前释放约 $T_{\text{NoC,traversal}}$
- **RX 侧**：从首字节到达开始工作，最后一字节写完才释放，**持续占用接近整个 $T_{\text{total}}$**
- 多通道 DTE 每通道独立计时，仿真器须按"每通道一条 timeline"建模，不可当单一资源

### 7.4 须核实清单

1. 硬件是否同时支持远程写与远程读（决定 push/pull 规则能否成立），查 DTE 四类通路延时表
2. NoC 集合通信是否真按 TX/RX 两组三档划分（3×3 表的前提）
3. 路由是否为 XY 维序，是否支持源路由/可编程路由
4. $\eta_{\text{PE}}$、$\eta_k$ 的查表数据
5. 片外与片上带宽的实际比值（正文写"约低一个数量级"，须核实）
6. $k$（分裂份数）的可取值集合
7. IR-2 的实际 $|V|$ 与节点类型数

---

## 8. 与仿真器的接口

### 8.1 仿真器职责

> **只评估，不优化。** 输入 Placed DAG，输出 makespan + critical path + per-resource utilization + bottleneck class。

三个部件并行建模：

| 部件 | 建模要点 |
|---|---|
| **控制核** | 三态：执行态（$T_{\text{issue}}$ 分触发/配置/同步三类）、显式同步态（wait 阻塞）、隐式同步态（FIFO 满时 issue 也等）。双核建模时须补两核间通信开销 |
| **DTE** | $T_{\text{total}}=T_{\text{setup}}+T_{\text{lat}}+\text{size}/B_{\min}$；TX/RX 占用不对称（见 7.3）；每通道独立 timeline |
| **计算单元** | $T_{\text{op}}=T_{\text{issue}}+T_{\text{setup}}+\max(T_{\text{data}},T_{\text{exec}})+T_{\text{drain}}$（读与算 overlap，故取 max 非加法）。竞争点：SPM bank、NCC InstFIFO、RAM_ACC |

**拥塞感知的改进**（把静态分析接进仿真器）：
- 每个传输占用其 path 上所有链路的 load
- 每个仿真时间窗重算各链路 $\rho$ 与 water-filling $B_{\text{eff}}$
- $T_{\text{exec}}=H\cdot t_r+\text{size}/B_{\text{eff}}$（+ 队列项）
- 反馈 binding 链路 + congestion_factor

### 8.2 agent 调用协议

| 阶段 | 调用什么 | 频次 |
|---|---|---|
| 外层剪枝 | 解析累加器（本地） | $10^7$，不碰仿真器 |
| 外层排序 | 解析全式（本地） | $10^4$，不碰仿真器 |
| 落位构造 | 加权跳数（本地） | $10^4$–$10^5$，不碰仿真器 |
| 落位精评 | 链路负载模型（本地） | $10^1$–$10^2$ |
| 最终评估 | **仿真器** | $10^1$ |

> **关键纪律**：仿真器的调用预算是 $10^1$ 量级。任何把仿真器塞进搜索内循环的做法都会让整个流程跑不动。若发现解析模型与仿真器偏差过大，正确的做法是**修解析模型的查表参数**，而不是提高仿真器的调用频次。

### 8.3 反馈的用法（工具口径 vs 论文口径）

⚠️ **两者可以不同，不要混淆：**

- **论文口径**：编译期一次性构造搜索，无运行时反馈、无迭代回环。这是必须守住的方法论声明。
- **工具口径**：agent 做**探索与标定**时，用仿真器的 bottleneck class 定向回退是完全合理的，可大幅加快找到好解。

bottleneck → 定向回退表（仅供工具阶段使用）：

| Bottleneck | 回退到 |
|---|---|
| Compute-bound | 调整 CG 核数 + GEMM 维度切分 |
| Comm-bound | push/pull 重选 + HBM/SPM 重选 + 更细分裂 |
| Sync-bound | 份数减小 + 插双缓冲 + hoist + 清理冗余 sync |
| SPM-overflow | 减少 CG 内核数 + spill + 部分张量改 reload |
| Bank-conflict | buffer 区域调整 + 重新 banking |
| NoC congestion | send 错峰 + 重新落位 |
| 某 CG 早完干等 | 启用 phase 化 / 重选 phase 边界 |
| 跨 CG 同步阻塞 | 合并强耦合 CG + 重选 sync 类型 |

**用法**：先在工具阶段用反馈环摸清各类负载的最优形态，把结论固化成剪枝规则与模板库的默认参数，**最终产品流程是单趟的**。论文报告的是收敛后的单趟流程。

---

## 9. 搜索规模（用于校验实现是否走对）

以 $n=15$ 节点、$P=16$ 核、类型数 $t=7$、$K\approx4$ 计：

| 阶段 | 量级 |
|---|---|
| 朴素联合空间 | $10^{17}$ |
| 同类型节点同组后 | $10^{5}$ |
| 锚定 GEMM 破对称后 | $10^{4}$ |
| 三条剪枝后实际展开 | $10^{3}$–$10^{4}$ |
| 每候选的内层求值 | $10^{3}$ |
| **总求值次数** | $\sim10^{7}$ |
| 送入仿真器 | $10^{1}$ |

**削减的主力是「同类型节点同组」（六个数量级），剪枝只贡献约一个数量级。** 实现时若发现搜索跑不动，先检查这条收缩是否真的生效，而不是去加剪枝规则。

---

## 10. 分阶段落地建议

| 阶段 | 目标 | 验收 |
|---|---|---|
| **P0** | 标定硬件参数表（§7.4 七项） | 解析 $T^{\text{comp}}/T^{\text{comm}}$ 与仿真器偏差 <20% |
| **P1** | 模板库 + 细化展开 + 单核硬件约束筛 | 给定 $(m,n,k)$ 与核数能输出合法核级子图 |
| **P2** | 累加器 + R1/R2/R3 + BFS 定序 + DFS 搜索 | 搜索规模落在 §9 的量级 |
| **P3** | 内层：核数贪心 + 份数穷举 + hoist/sink | 内层求值 $10^3$ 以内 |
| **P4** | 落位：形状可行性 + 贴合构造 + 退火 | 加权跳数较随机布局显著下降 |
| **P5** | 链路负载模型 + 仿真器精评 | 复现"贴合 vs 随机布局"的带宽差异 |

**P5 的对照实验是论文 §5.3 单独成节的全部理由**（同一划分候选，贴合布局 vs 随机布局，比较最堵链路 $\rho$ 与端到端 makespan），须优先安排。

---

## 附：未决事项汇总

1. **R2 重定义**（跨 CG 完工时间平衡）须正式拍板，并回改 [[写作]] §5.1 候选二"其代价是资源的闲置"的措辞
2. **集合原语的算法选择**（环 vs 树）归入细化还是补全
3. **$k_1$ 与 $k_2$ 是否独立**（若 CG 内一轮须与跨 die 一份对齐则存在整除关系）
4. **push/pull 的硬件前提**（远程读是否支持）
5. **3×3 TX/RX 表与硬件实际划分是否一致**
6. **phase 化**（kernel shift）作为最外层决策尚未写入方案，须补：先定 phase 数（1 或 2），每 phase 内跑一次上述流程，$t_{\text{shift}}$ 计入目标
7. **SPM banking 与 HBM 数据放置**本方案未展开，属 Implementation
s