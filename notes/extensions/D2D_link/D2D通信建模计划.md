# D2D通信建模计划

> **实现状态（2026-07-23）**：V0～V3 已完成；V3 已落地生产周期精确的有限
> SAF/inflight/RX/CTRL 流水线、端口/链路限速、DATA/CTRL 信用、whole-flow SAF 全路径原子预留，
> 并通过瓶颈、混合拥塞、背压与最小缓冲安全门禁。冻结标识为 `d2d-v3-baseline`。
> V4 Behavioral 校准与 V5 多端口/striping 尚未实现。详细证据见 `D2D_link_test.md`、
> `log/V3_development.md` 和 `llm/test/d2d_link/README.md`。

# 主要思路

在仿真器现在的core和chip层之间插入die层



## 加入Die层的概念

### 编址

**先拆解 `GRID_SIZE` 的三重超载（地基，必须最先做）**

现状：`GRID_SIZE` 在代码里同时承担三种语义，直接复用/改名都会中招：

| 现用法 | 真实语义 | 证据 |
| --- | --- | --- |
| 数组维度 | “全系统核数” | `workerCores/routers/channel/cache/dram` 全按 `[GRID_SIZE]` 分配，共 ~97 处（`monitor.cpp:68`、`router.cpp:8`） |
| 目的地址 | HOST endpoint | `des_ == GRID_SIZE`（`router_utils.cpp:36`、`router.cpp:344`） |
| 源地址 | “来自 HOST” | `m.source_ = GRID_SIZE`（`config_helper_base.cpp:70`） |

拆成一组语义明确的量：

```Bash
CORES_PER_DIE = GRID_X * GRID_Y          // 每 die 核数（替代 GRID_SIZE 的“每 die 核数”义）
DIE_X, DIE_Y                             // die 级 mesh 尺寸
DIE_COUNT     = DIE_X * DIE_Y
TOTAL_CORES   = CORES_PER_DIE * DIE_COUNT // ★核级数组一律用它（方案 A：扁平全局数组）

// 编址：全局核 id ↔ (die_id, local_id)
global_id = die_id * CORES_PER_DIE + local_id
die_id    = die_y * DIE_X + die_x
local_id  = global_id % CORES_PER_DIE
die_id    = global_id / CORES_PER_DIE

// endpoint 地址空间：TOTAL_CORES 之上保留一段给非核端点，HOST_ENDPOINT_ID 是其中之一
// （host + 各 mem 控制器），而非孤立魔法数
HOST_ENDPOINT_ID = TOTAL_CORES           // 保留区段起点，替代所有 == GRID_SIZE 的判定
```

**★方案 A（扁平全局数组）** —— 本计划采用：所有 die 的 router/worker/channel 塞进**同一个** `[TOTAL_CORES]` 大数组，按 `global_id` 索引，die 只是逻辑分区（router 带 `my_die` 成员区分）。D2D link = 连 `router[边缘 tile 的 global_id]` ↔ `router[邻 die 入口 tile 的 global_id]`。改动最小，与后文「router 加 my_die 成员」一致。（备选方案 B：die 做成 SC_MODULE 实例化 DIE_COUNT 次、每 die 自带一套数组——更干净但要大改 monitor 实例化结构，本计划不采用。）

**`GRID_SIZE` 拆解 checklist（改名扫描时逐类处理，勿机械替换）：**

| 现出现处 | 换成 | 注意 |
| --- | --- | --- |
| 数组分配 `new X[GRID_SIZE]`（monitor / router / cache / dram 等 ~97 处） | `TOTAL_CORES` | 扁平全局数组 |
| `des_ == GRID_SIZE` / `source_ == GRID_SIZE` / DONE 发往 `GRID_SIZE`（`logic.cpp:116`） | `HOST_ENDPOINT_ID` | 保留 endpoint 区段 |
| **`recv_tag < GRID_SIZE`（`config_helper_core.cpp:59`）** | **`< TOTAL_CORES`（不是数组维度！见下方「tag 语义」）** | ⚠ 这里的 `GRID_SIZE` 是「核 id tag / 显式 tag」的分界，是它的**第四种语义** |
| 逻辑↔物理映射 `o2r/r2o[GRID_SIZE]`（`config_helper_core.cpp:31`） | 视映射是 per-die 还是全局 → `CORES_PER_DIE` 或 `TOTAL_CORES` | 按映射语义定 |

> 验收：拆完后，**单 die（DIE_COUNT=1）配置逐位回归不变**——`TOTAL_CORES==CORES_PER_DIE==旧 GRID_SIZE`、`HOST_ENDPOINT_ID==旧 GRID_SIZE`，所有既有测试结果不变。

**tag 语义（`GRID_SIZE` 的第四种超载，最隐蔽）：**

`recv_tag` 是每个 workload `worklist` 项的**会合键**——发送方 `cast.tag` 必须 == 接收方 `recv_tag` 才能接上，且 router **按 tag 给输出链路上锁**（`router.cpp:344`，拥塞串行化就是 per-tag）。

默认约定 **tag == 接收核的 id**（`config_utils.cpp:242-247`、`config_helper_core.cpp:59-66`：`recv_tag = o2r[oid]`、`cast.tag = o2r[cast.tag]`），所以 **tag 空间与 core-id 空间在 `[0, GRID_SIZE)` 重叠**。那行 `< GRID_SIZE` 的真实作用是**区分**：

- `tag < GRID_SIZE` → 认定是裸核 id → 跟着 core placement 置换 `o2r` 一起重映射（核被打乱放置时 tag 跟着核走）；
- `tag ≥ GRID_SIZE` → 认定是显式/自定义 tag（如 PD 模式 `send_tag = core_id + tp_size*send_dest`，会超过 GRID_SIZE）→ 原样保留，不重映射。

**多 die 风险（landmine 本体）**：跨 die 若仍用「接收核**局部** id」当 tag，则**不同 die 上同 local-id 的核 tag 相同**（都 =5）→ router 的 per-tag `output_lock` 会把两条不相关的流**错误合并/串行**，甚至重组错乱。

**结论**：

1. 该 guard 改为 `< TOTAL_CORES`，`o2r` 若为全局置换则 `[TOTAL_CORES]`；
2. **跨 die tag 必须全局唯一** —— 用「全局接收核 id」（≤ `TOTAL_CORES`）或保留区显式 tag（`≥ TOTAL_CORES`）；
3. `M_D_TAG_ID = 16` 位（`macros.h:103`），tag 空间同受 **65535 规模上限**约束（与 `des_/source_` 一致）。



**两级XY路由：**

`GetNextHop` 增加前置判断：

- `des` 与 `pos` 同 die → 走今天的片内 XY（原逻辑不动）。

- **跨 die**

    - 先不看目标核，只看目标 die。用 die 级 XY 算出"下一跳该往哪条 die 边走"（东/西/南/北），再把包路由到本 die 那条边上指派的边缘端口 tile。出端口 → 过 D2D 链路 → 落到邻 die 对应边的入端口 → 邻 die 重新进入 `GetNextHop`：如果这时到了目的 die 就转片内 XY，否则继续往下一条 die 边走。

    - 关键点：跨 die 包在每一跳 die 上都会重新在该 die 的 NoC 里走一段真实片内路由（从入端口 tile 到出端口 tile，或到目的核）。这正是"跨 die 传输诱发片上流量"的来源——一条穿 3 块 die 的流，会在中间那块 die 的 mesh 上和它自己的本地流量抢链路。



（把现有 `des==GRID_SIZE→WEST margin` 的 special case 一般化。单 die 时行为与现在完全一致（回归安全）。）



### 跨die传输原语

> ⚠ 「加 `SEND_DIE` 类型」已被文末**第二轮修正 #5** 覆盖：跨 die 是寻址属性、不是新协议阶段。**不加 `SEND_DIE` 枚举**；REQ/ACK/DATA 阶段不变，`des_id` 改全局 endpoint，router 靠 `des_die!=my_die` 自动判跨 die。

`Send_prim` 加一个 `SEND_DIE` 类型（\+全局寻址）配对 `Recv`，执行分三段、每段都产生真实流量：

1. 源 die：源核 → 边缘端口（真实 NoC 跳 \+ 拥塞）

2. D2D 链路：加 SerDes 延迟 \+ 占用端口带宽（两层拥塞在此汇合）

3. 目的 die：入口端口 → 目的核（目的 mesh 上的真实 NoC 跳 \+ 拥塞）



## 仿真器扩展\-端口

### 端口放置

现在`IsMarginCore(id) = (id % GRID_X == 0)` 只认西边，且 HOST 出口硬编死在西边（router\_utils\.cpp、router\.cpp:344）。



- HOST 现在是"西边 margin 的特例",建议把它降格成**端口的一种目的类型**（`PORT_HOST` vs `PORT_D2D`），统一走端口出口路径。这样 host 注入/回收和 die 间传输共用一套缓冲\+lock 代码。



**端口指派功能**

端口编号：

X\*Y的2D Mesh，一共有2\*\(X\+Y\)个端口

> ⚠ N/S 写反，已被文末**第二轮修正 #2** 覆盖：正确为 `S=row 0`、`N=row(Y-1)`、`W=col 0`、`E=col(X-1)`（`NORTH=y+`）。

- 北边:第 0 行 X 个 tile 的 N 侧 → X 个端口

- 南边:最后一行 X 个 tile 的 S 侧 → X 个

- 西/东边:第 0 列 / 最后一列各 Y 个 tile 的 W/E 侧 → 各 Y 个

- 角 tile 贡献两个端口\(如左上角有 N 和 W 两个朝外面\),合计正好 2\*\(X\+Y\)

天然身份是 `(tile_id, side)`，配置里用`{side, idx}`\(side 上从固定端起数的第 idx 个\)当 key，内部再映射成一个线性 `port_id`

---

配置形式：

*不要手写全部 2\(X\+Y\) 个端口*\*\(X 一变就废,且易错\)。用分层写法,这也正好贴合你们现有 `cores[0]` 模板\+按 id 覆盖的惯例:

```Plain Text
"die_ports": {
  // 1) 整条边的默认 role(最省事的一层)
  "edges": {
    "N": { "role": "c2c", "dir": "N" },   // 北边整条默认接北向 C2C
    "S": { "role": "c2c", "dir": "S" },
    "E": { "role": "mem" },               // 东边整条挂内存控制器
    "W": { "role": "mem" }
  },
  // 2) 逐端口覆盖(需要混合时才写)
  "overrides": [
    { "side": "N", "idx": [0,1], "role": "mem" },          // 北边前两个改成内存
    { "side": "N", "idx": [6,7], "role": "c2c", "dir": "E" } // 北边最后两个反而服务东向 C2C
  ],
  // 3) C2C 链路物理参数(端口级,可被 override 单独覆盖)
  "c2c": { "bw_per_cycle": 8, "latency": 20, "buffer_depth": 4 }
}
```

`role` 是个 tagged union:`mem`\(内存控制器\) 或 `{c2c, dir}`\(通往 dir 方向邻 die\)。`dir` 独立于 `side`,这就实现了你要的"手动指定哪些端口接哪个方向 C2C"。整边默认覆盖 90% 情况,overrides 处理边角混合。

---

内部成表：

配置解析后固化成两个查找结构,运行时零分支开销:

1. `port_table[port_id] → {tile_id, side, role, dir, bw, latency, buffer}`——端口物理\+功能属性,router 的边缘出链绑到它。

2. `port_for[core_id][dir] → port_id`——预计算"从任一核,要往 dir 方向出 die,该去哪个 C2C 端口 tile"。这是端口选择逻辑的核心。当某方向有多个 C2C 端口时,用一个可插拔策略填这张表:

    - `nearest`\(推荐默认\):选 Manhattan 距离最近的同向端口 → 不同源核天然分散到不同端口,减少端口争用;

    - `round_robin` / `hash(tag)`:按流散列,当负载不均时用;

    - 这正好当拥塞对照变量\(和 noc\_congestion 里"共享 vs 不共享"同构\)。

---

端口选择策略：

1. 铁律:per\-flow pinning\(一条流决定一次、钉死到底\)

    选择必须**在流启动时\(SEND\_REQ\)算一次**,存进该流的 context,之后这条流的每个 DATA 包都复用同一个出口端口。**绝不允许逐包重选**。原因:

    - 出口端口 = 邻 die 上一个**固定的**入口点\(C2C 是点对点物理链路\)。逐包换端口 → 包从邻 die 不同位置注入 → 到达目的核**乱序**,而目的端的 tag\-lock 重组\(router\.cpp:360 按 `seq_id==1` 上锁、`is_end` 解锁\)直接崩。

    - 这和现有 router 的 tag\-lock 语义天然一致:锁本来就是 per\-flow 的,端口选择也 per\-flow,粒度对齐。

    所以下面无论静态还是动态策略,输出都是"给定 `(src_core, 流)` → 一个端口",而不是"给定包"

2. 策略做成可插拔policy

    `port_for` 这张表的填法抽象成一个策略接口,配置里一个字段切换,也直接当拥塞对照变量:

    ```Plain Text
    "die_ports": {
      "select_policy": "banded_nearest",   // banded_nearest | tag_hash | hybrid | dynamic
      ...
    }
    ```

    分三层,复杂度递增,默认只用 L0。

    **L0\(默认\)· 分带就近 banded\-nearest —— 同时要局部性和均衡**

    "字面最近"的毛病:流量倾斜时会把一堆核挤到同一个最近端口。正确做法是**按垂直轴做均衡分带**:

    以东向为例,某方向的候选端口 `ports_E = {role=c2c, dir=E}`,它们只在 y 轴上分布。设排序后 y 坐标 `y_0<y_1<…<y_{k-1}`:

    1. 把 mesh 的 Y 行划成 **k 个连续、行数尽量相等的带**\(多出的行按固定规则给低编号带 → 完全确定、可复现\);

    2. 带 j 绑定端口 `y_j`;

    3. `port_for[core][E]` = 包含 `core.y` 的那个带的端口。

    为什么这个比"字面最近"好:XY 路由下核去东边端口先东后 y,额外跳数 = `|core.y − y_port|`,分带把它**上界锁在半个带宽**,既短\(局部性\)又**每端口负载 = 行数/k 均衡**。北/南向对称地按 x 分带。这一层零运行时开销、纯静态预计算、完全可复现——**符合你们配置驱动的哲学,应作默认**。

    > ⚠ 见文末**第三轮修正 次-1/次-2/次-3**:「均衡」是**每端口源核数相等(结构均衡)非流量均衡**(倾斜用 tag_hash/dynamic);「跳数≤半带宽」仅在端口沿轴大致均布时成立(聚集时用 nearest/Voronoi 变体);`bandOf` 须 `assert k≤axisLen`(MVP side==dir 下天然成立)。

    > 特例甜点:若东边整条 Y 个 tile 都是 c2c:E\(k=Y\),分带退化成"每行一个端口",端口选择本身零争用,争用只来自同一行内多个核——最干净的基线。
    >
    >

    **L1\(可叠加\)· tag 散列 —— 解决"同源多流全挤一个端口"**

    分带是**按空间**分的:同一个核\(或同一带内的核\)同时发多条同方向流时,L0 会把它们全塞进那一个端口 → 串行化。叠加一层:当 `|ports_E|>1`,用 `ports_E[ hash(tag) % k ]` 把**一个核自己的多条流**打散到多个同向端口。代价是牺牲局部性\(可能绕远\)。`hybrid` = 先分带定候选子集、再在子集内按 tag 散列,取折中。

    **L2\(可选/进阶\)· 动态最少负载 —— 捕获运行时倾斜**

    注入时\(仍是流启动那一刻,遵守铁律\)在 `ports_E` 里挑**出口队列最短/近期占用最低**的那个,钉住。能捕获静态策略看不到的运行时热点。代价:非确定、难复现,且必须严格 per\-flow 钉死。**只在实验显示静态策略出现端口热点时才开**,不做默认。

3. 双端对称性

    出口端口 p 唯一对应邻 die 的入口点 p′,所以选端口同时决定了目的 mesh 的再注入位置,进而影响目的端跳数和争用。

    **做法：**只按源端 banded\-nearest 选出口,邻 die 从 p′ 用普通 XY 路由到目的核,目的端代价自然落下,不额外优化。

4. 配置校验

    - 解析后检查:die\-mesh 里每个存在的邻居方向 D,`|ports_D| ≥ 1`,否则那个方向的邻 die **不可达** → 直接报错,别留到运行时死锁。

    - `select_policy=dynamic` 时警告"结果不可复现",并强制 per\-flow 钉死校验



伪代码级规划：

端口选择从策略落到可实现规格。我按"约定 → die 级第一跳 → 分带预计算 → 组装 `port_for` → 多跳接力"五段写,全部伪代码级,坐标约定严格对齐现有代码\(router\_utils\.cpp:`x=pos%GRID_X, y=pos/GRID_X`;`EAST`=x\+,`NORTH`=y\+\)。

1. 约定与身份

    ```Plain Text
    // 局部核坐标(die 内),沿用现有约定
    lx(c) = c % GRID_X ;  ly(c) = c / GRID_X
    // die 级坐标
    dx(d) = d % DIE_X  ;  dy(d) = d / DIE_X
    // 全局地址两级
    des_die   = gid / GRID_SIZE
    des_local = gid % GRID_SIZE

    // 端口身份:朝外 tile + 边
    Port = { tile: int(局部核id), side: {N,S,E,W}, role, dir, bw, lat, buf }
    ports_dir[D] = { p ∈ PortTable | p.role==C2C && p.dir==D }   // D 方向的候选端口集
    ```

    一个规格级不变量先立死:**`port_for`**** 的 key 是 ****`(tile, D)`**** 而不是 ****`(source_core, D)`**。因为多跳接力时,中间 die 上做选择的"当前 tile"是入口端口 tile,不是原始源核——同一张表两处复用,这是整个方案能简洁接力的关键。

2. die 级第一跳方向:`dst_die → D`

    die 级维序路由,先 X 后 Y,是 GetNextHop 的 die 级镜像:

    ```Plain Text
    firstHopDieDir(my_die, dst_die):
        if my_die == dst_die: return CENTER          // 已在目的 die
        if dx(dst_die) != dx(my_die):                // 先对齐 die 的 X
            return (dx(dst_die) > dx(my_die)) ? EAST : WEST
        else:                                         // X 齐了再走 Y
            return (dy(dst_die) > dy(my_die)) ? NORTH : SOUTH
    ```

    维序保证 die 级路由图无环 → 从根上排除跨 die wormhole 死锁\(前面讲的三层防环里的一层\)。

3. 分带划分\(垂直轴均衡分带\)

    D 方向的候选端口只在**垂直轴**上分布:E/W → 沿 y;N/S → 沿 x。把该轴 `[0, N)` 均衡切成 k 段,余数按固定规则给低编号段 → 完全确定、可复现:

    ```Plain Text
    // 把 [0,N) 连续均分成 k 段,返回 coord→band 的映射
    bandOf(coord, N, k):
        base = N / k ;  rem = N % k
        // 前 rem 段各多 1;段 j 覆盖 [lo_j, hi_j)
        // 用累加边界定位 coord 落在哪段(确定的整数划分)
        edge = 0
        for j in 0..k-1:
            width = base + (j < rem ? 1 : 0)
            if coord < edge + width: return j
            edge += width
        return k-1

    // 把 D 方向端口按其垂直坐标排序,段 j ← 第 j 个端口(有序赋值)
    buildBandMap(D):
        P = sort(ports_dir[D], key = (D∈{E,W}) ? p.tile 的 ly : p.tile 的 lx)
        k = len(P)
        axisLen = (D∈{E,W}) ? GRID_Y : GRID_X
        return (P, k, axisLen)   // 段 j 的目标端口 = P[j]
    ```

    > `banded_nearest`\(推荐默认\)= 等宽段 \+ 有序赋值:负载均衡\(每端口 = 轴长/k\),且段 j 拿到的正是坐标序第 j 个端口 → 局部性也够。纯局部性变体 `nearest` 只需把 `bandOf` 换成 Voronoi\(相邻端口坐标中点切界\),但端口聚集时段不均。二者同一接口下切换。
    >
    >

4. 组装 `port_for[tile][D]`

    预计算,建仿真前跑一次:

    ```Plain Text
    precomputePortFor():
        for each D in {N,S,E,W} where ports_dir[D] nonempty:
            (P, k, axisLen) = buildBandMap(D)
            for tile in 0..GRID_SIZE-1:
                perp = (D∈{E,W}) ? ly(tile) : lx(tile)   // 取垂直轴坐标
                j = bandOf(perp, axisLen, k)
                port_for[tile][D] = P[j]
        // 校验:die-mesh 里实际存在的每个邻居方向都必须有端口
        for D in requiredDirs(DIE_X, DIE_Y):
            assert ports_dir[D] nonempty : "方向 D 无 C2C 端口,邻 die 不可达"
    ```

    `select_policy=tag_hash/hybrid/dynamic` 时,`port_for[tile][D]` 存的不是单个端口而是候选集 `P`,注入时再按 tag 散列/最少负载选;但**只在源 die 注入那一刻选一次并钉死**\(per\-flow pinning 铁律\)。

5. 出口 → 邻 die 落点\(点对点配对\)

    E 侧端口 tile `(GRID_X-1, y)` 物理连到东邻 die 的 W 侧 tile `(0, y)`——**垂直坐标守恒,边翻转**:

    ```Plain Text
    landing(port):                       // 出口端口 → (邻 die, 入口 tile)
        switch port.side:
          E: nd = die(dx+1, dy); it = tile(0,          ly(port.tile))
          W: nd = die(dx-1, dy); it = tile(GRID_X-1,   ly(port.tile))
          N: nd = die(dx, dy+1); it = tile(lx(port.tile), 0)
          S: nd = die(dx, dy-1); it = tile(lx(port.tile), GRID_Y-1)
        return (nd, it)
    ```

    因为所有包走同一出口端口 → 同一物理链路 → **同一入口 tile** 落到邻 die,这天然保证了下一跳选择的一致性\(见下\)。

6. 统一路由决策\(源 = 中间 = 目的,一套逻辑\)

    把整个跨 die 路由收敛成**每块 die 上的同一个函数**。包头带 `des_die + des_local`。router 在"当前 tile `cur`"处:

    ```Plain Text
    routeStep(cur, des_die, des_local):
        if des_die == my_die:
            // 已在目的 die:退化成现有片内 XY,直到 des_local
            return GetNextHop(des_local, cur)          // 复用现有实现,零改动
        D = firstHopDieDir(my_die, des_die)            // 该往哪个邻 die
        exit = port_for[cur][D]                         // 该 die 的出口端口(查同一张表)
        if cur == exit.tile:
            return EGRESS(exit)                         // 到端口 tile:出 C2C 链路
        else:
            return GetNextHop(exit.tile, cur)          // 片内 XY 先走到端口 tile
    ```

    > ⚠ 下面的「免费/自动成立」已被文末**第二轮修正 #4** 覆盖：`cur` 逐跳变、Router 看不到 WorkerCore 的 flow context，pinning **不自动成立**。须改为「每进 die 选一次出口、写进包头、离 die 前不重选」，flow key = `(source_global_id, tag, subflow)`，并修 `source_` 未初始化 bug。

    **多跳接力的答案就在这里,而且是"免费"的**:

    - **源 die**:`cur=源核` → 选 `port_for[源核][D0]` → 片内 XY 到该端口 → egress。

    - **中间 die**:包从 `landing()` 的固定入口 tile `it` 注入,router 对它跑**同一个 ****`routeStep`**:`des_die` 还没到 → 重算 `D1 = firstHopDieDir(this_die, des_die)`\(die 级 XY 会自然接力到下一跳方向\)→ 查 `port_for[it][D1]` → 走到该 die 的下一个出口端口。**中间 die 的端口选择 = 用"入口 tile"当 key 查同一张表**,不需要任何额外机制。

    - **目的 die**:`des_die==my_die` → 直接 `GetNextHop(des_local, cur)` 到目的核。

    per\-flow 一致性**自动成立**:整条路径是 `(源核, des_die)` 的确定函数——源端选端口\(静态确定\)→ 固定物理链路 → 固定入口 tile → 中间 die 又是静态确定 → ……每个 DATA 包走**逐字节相同**的多 die 路径,不会乱序。

    > 唯一例外:`select_policy=dynamic`。动态选择只允许发生在**源 die 注入那一刻**;中间 die 一律走静态 `banded_nearest`。否则中间 die 按运行时负载重选,会让同一条流的不同包在中间分叉 → 乱序。要么"只源端动态、中间静态",要么把整条路径的端口序列在流启动时一次算全、塞进包头当 source\-route 头携带。**默认取前者**\(简单且仍 per\-flow 确定\)。
    >
    >

7. 规格自洽性检查\(实现时的断言\)

    1. `firstHopDieDir` 对 die 级维序无环 → 多跳必然收敛,不会绕圈。

    2. `landing` 垂直坐标守恒 → 入口 tile 一定落在邻 die 的合法边缘 tile 上。

    3. `port_for` 对**每个** tile\(含所有可能的入口 tile\)都有定义 → 中间接力查表不会 miss。

    4. `precomputePortFor` 的方向校验兜住"邻居方向无端口"→ 编译期\(建模期\)暴露,不留运行时死锁。



---

跨die原语用表：

跨 die `Send` 触发时,路由变成"先查表定端口,再复用片内 XY":

```Plain Text
1. 目标 global_id → 目标 die → die 级 XY 算出第一跳方向 D (N/S/E/W)
2. exit_port = port_for[src_core][D]          // 选定本 die 的 C2C 出口端口
3. 片内:用现有 XY 路由把包送到 exit_port 的 tile(真实 NoC 跳+拥塞)
4. 到达该 tile → 从 side 侧 egress 进 C2C 链路(端口 bw/latency/buffer,带 output_lock 争用)
5. 邻 die 的对应入端口注入 → 重新进 GetNextHop:到目的 die 就转片内 XY 到目的核,否则回到步骤 2 继续下一跳 die
```

包头因此要带两级目的:`des_die` \+ `des_local`。router 判断 `des_die != my_die` → 路由到 `port_for[cur][D]`;`des_die == my_die` → 走原片内 XY。

---

接上现有代码：

- `GetNextHop` 加一个前置分支\(router\_utils\.cpp:34\):`des_die != my_die` 时返回"朝 exit\_port tile 的方向",否则原逻辑不动。

- 端口出链 = 泛化今天的 "WEST margin → HOST"\(router\.cpp:344\)。今天只有西边 margin 的出链接到 `host_buffer`;改成:任一边缘 tile 的朝外方向出链,按 `port_table` 接到 port unit——`role==host` → host\_buffer\(保留\)、`role==mem` → 内存控制器、`role==c2c` → D2D 链路。四种出口共用一套 buffer\+lock 代码。

- 单 die / 全 mem 配置下退化成今天的行为\(回归安全\)。





### 拥塞叠加

**cycle\-accurate情况下**

需要仿真出三种流量对于片上NoC的争用

**behavioral情况下**

把这三段折成一个 `wait(hops_src*hop_lat + bytes/d2d_bw + d2d_latency + hops_dst*hop_lat)`,无争用。（按发起跨die通信的核的真实坐标与端口之间的距离算hop数）



### 避免死锁

> ⚠ 本节「独立通道与 buffer（暂时不用）」的定位已被文末**第二轮修正 #7** 覆盖：组合网络无环**不能**由「片内无环 ⊕ die 级无环」推出，独立通道是**正确性要求（MVP 阻塞级）**，不是可选性能优化。**MVP 采用 store-and-forward**（整流入端口 buffer 后才释放片内锁），escape VC 作后续。SAF 的代价与影响见 #7。

- 维度序路由\+原语

    - 维序路由本身无环\(XY 先东西后南北\),片内已经靠它避免死锁,die 级也用同样维序 → die 级图也无环。

    - 原语层规避接`REQ→ACK→DATA` 握手的环形等待死锁：收端先 post recv 再回 ACK\(奇发偶收那套\),别用会形成对角环等待的置换模式



- 独立通道与buffer（暂时不用，后续再考虑添加）

    - 片内 \+ die 级混合时，入端口→片内→出端口这段如果和片内本地流共享 buffer，仍可能耦合出环。

    - 稳妥做法: D2D 流量走独立虚通道/独立 buffer\(port tile 的入/出缓冲与普通核 buffer 物理分开\),即"逃逸通道"思路。

    - 代价是每个 margin 核多一组 buffer，有面积开销

> 单 die 内这是纯协议级死锁\(跟 buffer 满不满无关\)。跨 die 后会叠加第二种死锁:
>
> - REQ/ACK/DATA 这些包现在要穿更长的路径\(源 mesh → 端口 → D2D 链路 → 目的 mesh\),沿途占用一串有限缓冲\(router 的 `MAX_BUFFER_PACKET_SIZE=3` \+ 端口 buffer\)。
>
> - 如果跨 die 流量图里有环,协议级 rendezvous 死锁和wormhole buffer 死锁会同时发生,而且后者就算你把原语改成奇发偶收也未必能躲——它是网络层的循环 buffer 依赖。
>
>



## 仿真器扩展\-D2D Link

> 两档行为建模
>
>

**0\. 一条跨 die 传输 = 一条串联的限速流水线**

```Plain Text
源核 ──①NoC信道──▶ 端口入buf ──②端口注入(位宽)──▶ [D2D link ③带宽+延迟] ──▶ 邻die端口入buf ──④NoC信道──▶ 目的核
       rate=noc_bw            rate=w_p(端口位宽)      rate=b_l, +L(延迟)         rate=noc_bw
```

**用户的三个限制项各自对应一段的速率**:

sustained 吞吐 = **各段速率的最小值**。这就是那个 min\(\) 的物理来源。两档的区别只在:**behavioral 显式算 min\(\),cycle\-accurate 让 min\(\) 从背压里自然涌现**。

---

**1\. 关键澄清:"端口数×端口位宽"要靠条带化才拿得到**

这一点必须先钉死,否则模型会撒谎。per\-flow pinning 铁律说**一条流只走一个端口** → 单条流的带宽**永远** ≤ 一个端口的 `w_p`,跟这个方向上有多少端口无关。

要吃到 `端口数 × 端口位宽` 的聚合带宽,一次逻辑传输必须**条带化成 ≥ k 条子流**\(每子流独立 tag,由前面的 `tag_hash`/striping 策略钉到不同端口\)。所以:

- **单流传输**:`BW = min(noc_信道, w_p, b_l)`\(port\_count 不参与\)。

- **条带化到 k 端口**:`BW = min(k×noc信道, k×w_p, k×b_l)`,即 `k=port_count` 时才是用户说的那个式子。

实现上:`Send` 原语加一个 `stripe` 字段\(或按 tensor 大小自动切\),把大 tensor 拆成 k 个 sub\-flow。**这是"端口数"这一维能生效的前提**,要在文档里写明——否则用户配了 8 个端口却发单流,只会得到 1 个端口的带宽,看着像 bug 其实是建模正确。

---

**2\. Behavioral 档:显式 roofline min\(\)**

> ⚠ 下面的 `BW_eff` 与 `L_route` 已被文末**第二轮修正 #8、#10** 覆盖：`k×noc` 高估条带化（stripe 共享源核单注入端口），应改 **min-cut**；`L_route` 含 `hops` 会与「包实际穿 Router」重复计算，公式**只补 `Σ link_latency`**，bulk service 单点记账。

一次\(可能条带化的\)跨 die 传输,`N_pkt` 个包、条带到 `k` 端口:

```Plain Text
BW_eff = min( k × noc_payload_per_cycle,     // NoC 供给(k 条不相交路径)
              k × w_p,                        // 端口注入总位宽
              k × b_l )                        // D2D link 总带宽
                                              // 单位:包/cycle

T_serial = ceil(N_pkt / k / BW_per_port) × CYCLE    // 传输(搬运)时间
         = ceil( N_pkt / BW_eff ) × CYCLE

L_route  = (hops_src + hops_dst) × on_die_hop_latency   // 两侧片内跳(已有常量)
         + L                                            // D2D 固定延迟(SerDes+线)

T_transfer = L_route + T_serial
```

落地方式:**复用现有 ****`wait(roofline_packets × CYCLE)`**** 通道**——把跨 die 传输的等效 `roofline_packets` 设成 `ceil(N_pkt / BW_eff × noc_payload_per_cycle)` 再加上 `L_route/CYCLE` 的头延迟。即"beha D2D = beha NoC 的 roofline,只是 BW 换成三段 min、并多一个 D2D 头延迟"。和 `SPEC_USE_BEHA_NOC` 同一条代码路径,零新机制。**对端口争用盲**\(和 beha NoC 一样\),这正是它作对照基线的价值。

---

**3\. Cycle\-accurate 档:端口单元 = 限速器 \+ 背压,min 自然涌现**

> ⚠ 下面 `accept up to w_p packets ... this cycle` 已被文末**第二轮修正 #9** 覆盖：单 `sc_bv<256>` 信号每周期最多 1 包，`w_p>1` 无法真并行。改用 **token-bucket credit**（每拍累加小数信用、够 1 才发），可精确表达 `b_l<1`、以时间平均近似 `w_p>1`。

**不算 min\(\),让它从结构里长出来**。给每个 C2C 端口建一个 SystemC 单元\(SC\_THREAD\),三段有限 buffer \+ 逐周期 drain:

```Plain Text
port_unit(每 cycle):
    // ② 注入段:NoC → 入buf,受 w_p 和 buf 容量限
    accept up to w_p packets from router output link this cycle
        if in_buf full → 背压回 router(router 的 output_lock 就地串行化 → 体现①的 NoC 限制)
    // ③ link 段:入buf → 在途队列,受 b_l 限
    move up to b_l packets from in_buf into link_fifo, stamp arrival = now + L
    // 到达:link_fifo 中 arrival ≤ now 的包 → 邻 die 入端口 buf
        if 邻die in_buf full → 背压(④ 目的 NoC 满 → 回压到 link → 回压到本端口 → 回压到源 NoC)
```

三段速率不同 → **最慢那段的 buffer 先满 → 背压逐级回传 → 整条流的稳态吞吐自动 = min\(段速率\)**。谁是瓶颈由配置决定,仿真器不需要知道,这就是 cycle\-accurate 的意义。而且:

- **端口争用**:多条子流抢同一端口 → 复用 router 的 `output_lock`\(router\.cpp:344\)在端口入口串行化。

- **NoC 限制①**:端口 in\_buf 满 → 背压使源 mesh 的包堵在路上 → 和本地流量抢链路 → 诱发的片上流量真实发生。

- **D2D link 限制③**:`b_l` 通常 ≪ 片上带宽\(片外串行\),多数情况下它就是那个 min,link\_fifo 常满、稳态吞吐 = `k×b_l`。

---

**4\. 配置参数\(挂到 die\_ports\.c2c 下\)**

```Plain Text
"c2c": {
  "port_width":   4,    // w_p: 片上→端口注入,包/cycle
  "link_bw":      2,    // b_l: 跨die串行带宽,包/cycle(通常最小 → 主瓶颈)
  "link_latency": 20,   // L: SerDes+线延迟,cycle
  "buffer_depth": 8     // 每段 buf 深度(决定背压松紧,类比 MAX_BUFFER_PACKET_SIZE=3)
}
```

`port_width`/`link_bw` 单位与 `noc_payload_per_cycle` 一致\(包/cycle\),三者可直接比大小,min\(\) 一眼可判瓶颈。

---

**5\. 像其他模块一样"评估出跨die传输时间"**

沿用项目现有的三条输出通道,不新造:

1. **总时间**:跨 die 传输的延迟自然计入 `sc_time_stamp()`,end\-core DONE 到齐时的 `[CATCH TEST] ... finished` 就是含 D2D 的总周期\(和 noc\_congestion 完全一样\)。

2. **逐传输 trace**:端口单元在流"进 link / 出 link / 背压停顿"时打 `events.json`\(Chrome\-tracing\),像现有 router flow 停顿一样可视化——直接能在 streaming\_trace\_viewer 里看到 D2D 段。

3. **带宽利用率日志**:端口单元按 `NETWORK` 日志风格打 `实际 drain / b_l` 利用率,佐证哪一段是瓶颈。

对照实验\(复刻 noc\_congestion 打法\):固定计算、只改 `link_bw` 或 `stripe`\(端口数\),beha 档三档几乎同周期\(盲\)、cycle 档随瓶颈变化 → 定量"跨 die 传输时间 = f\(NoC, 端口, D2D link\)"。

---

**6\. 和现有开关的接缝**

- 复用 `SPEC_USE_BEHA_NOC`:true → 走第 2 节 roofline min\(\);false → 走第 3 节端口限速器。**同一个开关同时管片内 NoC 和 D2D**,语义一致。

- 单 die / 无 c2c 端口配置下,端口单元不实例化 → 完全退化成今天\(回归安全\)。



---

# 落地文件清单

## 两个前置事实

1. **CMake 用 `GLOB_RECURSE ./llm/*.cpp`**（`CMakeLists.txt:37`）——新建的 `.cpp` 放在 `llm/` 下任意位置就自动纳入编译，只需重跑一次 cmake。可以放心开新模块目录。
2. **`Msg` 只有单级 `des_`**（`llm/include/common/msg.h:13`），包序列化成 `sc_bv<256>`（`llm/src/utils/msg_utils.cpp`）。两级目的 `des_die + des_local` 需要动这两处。

**建议新开一个 `die/` 模块**（对齐现有 `link/` 是 chip 级、`router/` 是片内级的分层），D2D 的端口 / 链路 / 选择都归到这里。以下按本文档章节逐块映射。

## A. 编址 + 全局状态（对应「编址」）

> **⚠ 前置任务 A0（排在所有其它步骤之前）：拓扑重构。** 见第二轮修正 #1：现网络是 **torus（环绕）**，边缘链路已被占用、不是自由边；且只支持方阵。必须先 (a) 支持矩形 X×Y（Y 轴 `%GRID_Y`、去 `GRID_Y=GRID_X`），(b) 改成**开边 mesh**（边缘朝外不 wrap），空出边缘出链才能接端口。此任务未完成前，E 节「边缘出链绑定」无从谈起。
>
> **⚠ 前置任务 A1：`GRID_SIZE` 四重超载拆分。** 编址地基，B/C/D/E/G 全依赖。拆分内容见「编址」章节 checklist + 「tag 语义」；采用**方案 A（扁平全局数组 `[TOTAL_CORES]`）**。验收：单 die（DIE_COUNT=1）逐位回归不变。

| 文件 | 存量/新建 | 实现什么 |
| --- | --- | --- |
| `llm/include/defs/spec.h` + `llm/src/defs/spec.cpp` | 存量，改造 | 引入 `CORES_PER_DIE / DIE_X / DIE_Y / DIE_COUNT / TOTAL_CORES / HOST_ENDPOINT_ID`；保留 `GRID_X/Y`；`GRID_SIZE` 逐步退役（过渡期可 `#define GRID_SIZE CORES_PER_DIE` 兜底，最终清掉）。默认单 die：`DIE_COUNT=1` |
| `llm/src/utils/config_utils.cpp` `ParseHardwareConfig` | 存量，追加 | 解析 `"die": {x,y}` → 设 `DIE_X/Y/DIE_COUNT`；`CORES_PER_DIE=GRID_X*GRID_Y`；`TOTAL_CORES=CORES_PER_DIE*DIE_COUNT`；`HOST_ENDPOINT_ID=TOTAL_CORES`；缺省 `DIE_COUNT=1` |
| 核级数组分配处（`monitor.cpp`、`router.cpp`、`system_utils.cpp`、cache 等 ~97 处 `[GRID_SIZE]`） | 存量，扫描替换 | 数组维度一律 `GRID_SIZE → TOTAL_CORES`（方案 A 扁平全局数组）；`des_/source_ == GRID_SIZE → HOST_ENDPOINT_ID`；`recv_tag < GRID_SIZE` **单独审**（tag/core-id 耦合，勿机械替换） |

`global_core_id ↔ (die_id, local_id)` 的编解码（`gid/CORES_PER_DIE`、`gid%CORES_PER_DIE`、`die_x=die%DIE_X`）做成 inline helper，放 `llm/include/utils/router_utils.h`（它已是路由地址工具的家）。

## B. 端口配置解析 + 两张表（对应「端口指派」「内部成表」「伪代码 1–4」）

**新建模块 `die/`，方案主体：**

| 文件 | 实现什么 |
| --- | --- |
| `llm/include/die/port.h`（新） | `enum PortRole{HOST,MEM,C2C}`、`struct Port{tile,side,role,dir,bw,latency,buf}`、`struct DieConfig`；全局 `port_table[]`、`port_for[tile][dir]` 声明 |
| `llm/src/die/port_config.cpp`（新） | 解析 `die_ports`：`edges` 整边默认 + `overrides` 逐端口覆盖 → 展开成 `port_table`；`{side,idx}`→线性 `port_id` 映射；方向可达性校验（`\|ports_D\|≥1` 否则报错）。对应「配置形式」「配置校验」 |
| `llm/src/die/port_select.cpp`（新） | `bandOf` / `buildBandMap` / `precomputePortFor` / `firstHopDieDir` / `landing`——伪代码 2/3/4/5 逐条落这里；`select_policy` 分派（banded_nearest/tag_hash/hybrid/dynamic）。**★第三轮修正 次-7**：同时预计算 `port_for_host[core]`（HOST 也是端口，非硬编西边）；解析期加 **endpoint 类型解码**（`des_` 在保留区 → core/host/mem）与 **HOST 可达性校验**（≥1 HOST 端口且每核可达，全被 C2C override 则报错） |

调用时机：`precomputePortFor()` 在建 SystemC 模块前调一次（`config_utils.cpp` 解析完 hardware config 之后，或主流程 `llm/unittest/npusim.cpp` 里）。

## C. 路由决策：`routeStep` / 两级 XY（对应「两级XY路由」「伪代码 6」）

| 文件 | 实现什么 |
| --- | --- |
| `llm/src/utils/router_utils.cpp` `GetNextHop` | 加前置分支：先取 `des_die`；`des_die==my_die` 走原逻辑不动；`des_die!=my_die` → `D=firstHopDieDir()` → `exit=port_for[cur][D]` → `cur==exit.tile ? EGRESS(exit) : GetNextHop(exit.tile,cur)`。即 `routeStep`，直接嵌进现有函数，单 die 时 `des_die` 恒等 my_die → 零行为变化 |
| `llm/include/utils/router_utils.h` | 声明新增的 `firstHopDieDir`、`RouteStep` 返回类型（Direction + 可能的 EGRESS 标记） |

`GetNextHop` 需要知道当前 router 属于哪个 die（`my_die`）。router 实例带 `rid`（局部核 id），`my_die` 应作为 router 成员传入 → 见 E。

## D. 包头两级目的（对应「跨die原语用表」结尾）

> **修正（原「加独立 `des_die_` 字段」不采用）**：包字段实测只剩 **17 位空闲**——256 位已用 239 位（`M_D_DES=16`、`M_D_SOURCE=16`，`macros.h:98-109`；`msg_utils.cpp:43` 的 `range(255,239)` 补零）。加独立字段会和「加宽 `des_`」抢这 17 位。**正确做法：把 `global_core_id` 直接编进现有 `des_`，用 `des_/CORES_PER_DIE` 拆出 (die,local)，不加新字段。**

| 文件 | 实现什么 |
| --- | --- |
| `llm/include/common/msg.h` `class Msg` | `des_`/`source_` 语义改为**全局 id**；`(die,local)` 由 helper 从 global_id 拆，不加 `des_die_` 字段 |
| `llm/include/macros/macros.h` `M_D_DES/M_D_SOURCE` | 规模 > 65535 endpoint 时需加宽（当前各 16 位，仅 17 位余量 → 两者各加到 24 位即 +8+8=16 刚好用满，再无空间）；记录最大规模上限 |
| `llm/src/utils/msg_utils.cpp` `SerializeMsg/DeserializeMsg` | 若加宽字段，同步改位段布局（否则无需改，`des_` 位宽不变、只是承载全局 id） |

## E. Router 硬件：egress 泛化 + 端口单元接线（对应「端口放置」「接上现有代码」「cycle-accurate 拥塞」）

| 文件 | 实现什么 |
| --- | --- |
| `llm/src/router/router.cpp`（≈ L344 出口仲裁、L46 margin 建 buffer） | 把「WEST margin→HOST」泛化成「任一边缘朝外出链→port unit」：边缘 tile 的每个朝外 `buffer_o[dir]` 按 `port_table` 接到 port unit；`role==HOST`→保留 host_buffer，`role==MEM`→内存控制器，`role==C2C`→D2D link。`output_lock` 逻辑原样复用（端口争用免费获得） |
| `llm/include/router/router.h` | RouterUnit 加 `int my_die;` 成员；边缘端口方向的 out buffer 绑定 port unit 指针 |

## F. D2D Link 单元：cycle-accurate 限速+背压（对应「D2D Link 第 0/3 节」）

**新建，归入 `die/`：**

| 文件 | 实现什么 |
| --- | --- |
| `llm/include/die/d2d_link.h`（新） | `PortUnit`（SC_MODULE）：`in_buf / link_fifo`、参数 `w_p,b_l,L,buf_depth`；`DieMesh` 顶层把各 die router 阵列 + 端口按 `landing()` 配对连线 |
| `llm/src/die/d2d_link.cpp`（新） | 第 3 节 `port_unit(每 cycle)` 三段 drain + 背压：① NoC→in_buf（credit 限速，见二轮 #9）② in_buf→link_fifo（credit，盖 arrival=now+L）③ 到达注入邻 die 入口 tile（满则逐级回压）。`role==MEM` 端口转内存事务（接 `mem_interface.cpp`）。**★第三轮修正 次-6**：全双工独立 tx/rx；**反向链路载跨 die ACK**（配二轮 #6 控制子通道）；**在途 FIFO ≥ BDP=`link_bw×2L`** 否则吞吐塌；本地限速 credit 与跨链路背压 credit(RTT=2L) 不可混 |
| （F 节配置）`die_ports.c2c` / phase-2 `Link{}` | **★第三轮修正 次-5**：MVP 每端口 = 一条独立 D2D link（`link_bw` per-port）；共享 bundle = phase-2 放进 `Link{tx_bw,rx_bw,latency,...}` |

## G. 跨 die 原语 + 条带化（对应「跨die传输原语」「条带化」）

| 文件 | 实现什么 |
| --- | --- |
| ~~`llm/include/defs/enums.h` `enum SEND_TYPE` 加 `SEND_DIE`~~ | **不做**（第二轮修正 #5）：跨 die 是寻址属性，`des_id` 改全局 endpoint，router 靠 `des_die!=my_die` 自动判 |
| `llm/include/prims/norm_prims.h` `Send_prim`/`Recv_prim` | `Send_prim` 加全局寻址（`des_id` 用 global_id）+ `int stripe`（条带子流数） |
| `llm/src/prims/norm_prims/send_prim.cpp` / `recv_prim.cpp` | 按 `stripe` 把 tensor 拆成 k 条子流，每条独立 tag，发出时把**全局** des 填好。这是「端口数×位宽」能生效的前提 |
| `llm/src/prims/norm_prims/recv_prim.cpp` + `llm/src/workercore/logic.cpp`（收端聚合） | **★第三轮修正 次-4（真功能）**：单 tag `Recv_prim` 收不了多 tag 子流且会报错（`logic.cpp:509/553`）。须加 **k 个并行 Recv + join** 或 **grouped-recv（tag 集合 + 全子流 end 完成）**；subflow 即二轮 #4 flow key 的 subflow 位。**与 stripe 同批实现，否则 striping 不可用** |
| `llm/src/monitor/config_helper_core.cpp:59`（tag 改写） | `recv_tag < GRID_SIZE` → `< TOTAL_CORES`；**跨 die tag 必须全局唯一**（用全局接收核 id 或 ≥TOTAL_CORES 的保留显式 tag），否则不同 die 同 local-id 流在 router per-tag `output_lock` 上错误合并（详见「编址」章节「tag 语义」）。`M_D_TAG_ID=16` 位，tag 同受 65535 上限 |

## H. Workercore：两档时间评估（对应「D2D Link 第 2 节 beha / 第 6 节开关」）

| 文件 | 实现什么 |
| --- | --- |
| `llm/src/workercore/logic.cpp`（beha roofline 在 L58-61 / L147） | `SPEC_USE_BEHA_NOC==true` 且流是跨 die 时，把等效 `roofline_packets` 换成 `ceil(N_pkt/BW_eff)` 并加 `L_route/CYCLE` 头延迟（第 2 节 min() 公式）；`false` 则包真实走 F 的 port unit。复用同一个 `SPEC_USE_BEHA_NOC` 开关同时管 NoC 和 D2D |

## I. 常量 + 宏（已预留）

| 文件 | 实现什么 |
| --- | --- |
| `llm/include/defs/const.h` / `const.cpp` | 复用已存在的 `on_die_hop_latency`、`average_die_to_edge_distance`（beha 的 `hops×hop_lat` 直接用它） |
| `llm/include/macros/macros.h` | `M_D_DATA=128`、`CYCLE=2` 复用；端口 buffer 深度类比 `MAX_BUFFER_PACKET_SIZE`（在 `spec.h`） |

## J. 测试 + 文档（对应「评估」）

| 文件 | 实现什么 |
| --- | --- |
| `llm/test/d2d_transfer/`（新，仿 `llm/test/noc_congestion/`） | hardware（含 `die`+`die_ports`）、sim（beha/cycle 两份）、workload（跨 die SEND_DIE）、`run_test_d2d.py`；对照变量 = `link_bw`/`stripe`/`select_policy` |

## 建议实现次序（依赖顺序）

```Plain Text
A0 拓扑重构(torus→开边mesh+矩形) → A1 GRID_SIZE 四重超载拆分 → B 端口表+选择(die/)
  → D 包头(全局id+修source_) → C 路由routeStep(每进die选一次出口,写包头)
  → E router egress泛化 → F D2D link单元(cycle,credit限速)+控制子通道 → G 原语(全局endpoint+条带,不加SEND_DIE)
  → 死锁:store-and-forward → H logic双档(min-cut,单点延迟记账) → J 测试
```

A0/A1 是纯结构/编址地基（含第二轮修正 #1、#4 的 `source_` bug、#5、#6、#7、#8、#9、#10 的落地）；前段（到 E）跑通「包能路由到端口并出去」；F 之后才有真实 D2D 时序（SAF 保正确性、credit 表达带宽）；H 决定两档；J 验证。每一步都保持**单 die / 无 die_ports 配置退化成今天**，逐步可测。

---

# 第二轮 Review 修正（权威，覆盖前文相冲突处）

> 以下 10 条基于对现有代码的逐点核实，**优先级高于前文任何相冲突的措辞**。前文若与此矛盾，以此为准（相关处已在原地打「⚠ 已被本节 #N 覆盖」标记）。

## #1 先做拓扑重构（torus→开边 mesh + 支持矩形）——新增最底层前置任务 A0

现网络是 **torus（环绕）不是 mesh**：`monitor.cpp:250` 把 `channel_i` 连到 `GetInputSource` 的取模结果，`router_utils.cpp:8` 四方向全 `%GRID_X` 环绕——**边缘链路已被占用，不是可拿来接端口的自由边**。且 Y 轴也用 `%GRID_X`+`GRID_Y=GRID_X`，**只支持方阵**。

**改**：在「编址」前置任务之前再加 **A0 拓扑重构**：
- `GetInputSource`/`GetNextHop`/monitor 互连支持矩形 X×Y（Y 轴改 `%GRID_Y`、去掉 `GRID_Y=GRID_X`）；
- **引入边界模式**：die 内网络从 torus 改成**开边 mesh**（边缘朝外方向不 wrap、不互连），空出的边缘出链才能接 port unit。E 节「在现有边缘输出上追加绑定」不成立——必须先解开边。
- 验收：单 die + 开边 mesh 下既有测试仍通过（注意：若既有测例依赖 torus 环绕语义，需甄别）。

## #2 N/S 坐标写反（文档 bug）

`router_utils.cpp:11` `NORTH: y=(y+1)` → **N=高 y**。正确编号：`S side=row 0`、`N side=row(Y-1)`、`W side=col 0`、`E side=col(X-1)`。`landing()` 代码本身是对的（北出→落邻 die 南边 y=0），只需修端口编号那段散文。

## #3 side 与 dir 独立 → landing() 按 side 选邻居会路由错

配置允许 `{side:N, dir:E}`，但 `landing()` 按 side 送北邻。**MVP：强制 `side==dir`**，config 解析时校验，违反即报错；landing() 按 side（==dir）自洽。**phase-2：换显式物理链路表**，可校验端点成对/带宽共享/全双工：
```
Link { local_port, remote_die, remote_port, tx_bw, rx_bw, latency }
```

## #4 「per-flow pinning 自动成立」不成立——推翻伪代码 6 的「免费」结论

`routeStep` 每跳 `port_for[cur][D]` 的 `cur` 逐跳变，静态 banded 在端口不规则时也可能中途换出口；且路由决策在 **Router** 内，WorkerCore 的 flow context 对 Router 不可见。另外 `msg.h:23` 的 DATA 构造函数**没初始化 `source_`**，`msg_utils.cpp:27` 却照样序列化它（现存 bug）。

**改**：
- 出口选择改为**每进入一个 die 时选一次**，把 `selected_exit_port`（或 `ingress_anchor`）**写进包头/路由状态**；die 内只朝这个钉死的 tile 路由，**离开该 die 前不得重选**；跨到下一 die 才重选。
- Flow key = **`(source_global_id, tag, subflow/epoch)`**，不能只有 tag。
- 前置修 bug：DATA 路径必须初始化 `source_`。
- 删除「多跳接力免费/自动成立」措辞。

## #5 SEND_DIE 不该进 SEND_TYPE

`enums.h:18` 的 `SEND_TYPE` 是**协议阶段**（REQ/DATA/DONE）；跨 die 是**寻址属性**。**改**：不加 `SEND_DIE` 枚举；REQ/ACK/DATA 阶段不变，`des_id` 改成**全局 endpoint**，router 靠 `des_die!=my_die` 自动判跨 die。（覆盖 G 节「enums.h 加 SEND_DIE」那一行。）

## #6 控制网络未纳入 D2D 模型——跨 die 握手闭不了环

`router.h:84`：REQUEST/ACK/DONE 走**完全独立的 `ctrl_channel/ctrl_buffer`** 网络；而 F 节 port_unit 只建模了 DATA。**改（F 节加子节）**：D2D link 镜像片内的控制/数据分离——端口加**独立控制子通道**（小 buffer + 独立背压），明确：控制包不占 DATA 带宽、控制优先于数据仲裁、远端控制 buffer 满时回压。

## #7 死锁论证不成立——独立通道是正确性，不是可选性能【采用方案：MVP=store-and-forward】

「片内 XY 无环 ⊕ die 级 XY 无环 ⇒ 组合无环」是**错的**：组合路径资源类别每进一个 die 重新开始，叠加 `router.cpp:359` output_lock 首包持到尾包，可成新 channel dependency cycle。奇发偶收只挡协议级 rendezvous 死锁，挡不了网络 buffer 死锁。故把「独立 VC」从「暂时不用/后续」**提升为 MVP 阻塞级正确性要求**。

**采用 MVP = store-and-forward（SAF）**，后续升级 escape VC。

**SAF 是什么**：死锁环的本质是 **flow 级 output_lock**——一条流从首包（`seq_id==1`）锁到尾包（`is_end`）解锁，把源 die 一串输出链路**从头持有到尾**；跨 die 时「源 die 锁持有」依赖「远端 die 锁获取」，互等成环。SAF 让**整条流先落进端口 buffer、释放源 die 的锁，再由端口独立向远端重新发起一条新锁定流**——用 buffer 把源 die 流与远端 die 流剪成两段独立电路，切断跨 die 的 hold-and-wait。

**SAF 对整体通信的影响（已知代价，需写入实现说明）**：
1. **缓冲需求（核心代价）**：要真正保证无死锁，端口 buffer 必须**能装下一整条 flow**；否则 buffer 满时源 die 包仍堵在网络、锁还握着 → 退回 hold-and-wait，只降概率不消除。仿真器里这只是个可增长的 `queue`（易实现），但建模上意味着**很大的 beachfront 缓冲**，realism 上是妥协。
2. **延迟变差（可预期的悲观）**：SAF 把流水打断成**串行三段**（灌满端口 → 过链路 → 灌进远端），而 wormhole 是流水重叠；**每多穿一个 die 边界多一次整流 store-drain 串行**，多跳跨 die 流延迟明显偏悲观。
3. **聚合吞吐基本不变**：稳态多流下只要 link_bw 是瓶颈（D2D 常态），链路一直忙，总吞吐与 wormhole 几乎一样；挨打的是单流延迟。
4. **与控制网络解耦**：SAF 只作用于 DATA 流；REQ/ACK 走独立控制通道（#6），握手正常闭环。
- 一句话：用「大边界 buffer + 偏悲观的单流延迟」换 MVP 阶段确定的正确性，总带宽不亏；escape VC 后可拿回「有界 buffer + 流水延迟」。

## #8 带宽公式高估条带化收益

`min(k×noc, k×port, k×link)` 假设 k 条独立注入 + 不相交路径；但同源 tensor 的所有 stripe **先共享源核那一个注入端口**，再可能共享 NoC 链路 → 拿不到 `k×noc`。**改（覆盖 D2D 第 2 节公式）**：behavioral 至少写成 **min-cut**：
```
BW_eff = min( source_injection_bw,          // 源核单注入端口
              destination_ejection_bw,      // 目的核单弹出端口
              shared_NoC_cut_bw,            // 共享 NoC 割集
              Σ selected_port_bw,           // 所选端口带宽之和
              Σ independent_D2D_link_bw )    // 独立 D2D 链路带宽之和
```
若 MVP 不求路径 min-cut，则**显式标注 `k×min` 是理想上界（upper bound），不是准确吞吐**。

## #9 CA 的 w_p>1 当前无法实现【采用方案：token-bucket credit】

`router.cpp:158`：每方向输出是单个 `sc_bv<256>` 信号，**每周期最多 1 包**；`noc_payload_per_cycle` 只在 `msg_utils.cpp:80` 压缩逻辑包数（roofline），**不让 CA Router 一周期发多包**。

**采用 credit（token-bucket）**。**是什么**：给端口一个信用计数器，每拍按配置带宽**累加信用（可小数）**，发一个包**扣 1 信用**，不足则等：
```
每 cycle:
    credit += link_bw                     // 例如 0.5 → 每 2 拍攒够 1 个信用
    credit = min(credit, bucket_depth)    // 桶深，限突发
    if credit >= 1 and in_buf 非空 and 远端可收:
        forward 1 packet;  credit -= 1
```
把「信号每拍只能载 1 包」与「链路想建模的带宽」**解耦**：`link_bw=0.5`→每 2 拍服务 1 包；`w_p=2`→信用攒快、在空闲拍突发追上（**时间平均**）。

**为什么选它**：D2D 建模的全部意义是「片外链路比 NoC 慢」即 **`b_l<1` 是主场景**，credit 正是「慢于每拍一包」的天然工具，改动仅一个计数器，复用现有单信号。**已知局限**：无法表达*真并行* `w_p>1`（单信号所限），只能做时间平均——真并行留给后续多 lane。（对比：多 lane 信号=改动大；一 flit 携多 packet=搅乱逐包路由/锁；限 ≤1/cycle=b_l 被迫=1 失去 D2D 作瓶颈的意义。）

## #10 behavioral 延迟漏算/重复算——定唯一记账点

`logic.cpp:58`、`logic.cpp:562`：behavioral **不绕过 Router**，聚合包仍逐跳穿 Router，收发端各自再 roofline wait。**把 `L_route`（含 `hops_src/hops_dst`）塞进 `roofline_packets` 会与「包实际穿 Router 的跳延迟」重复计算**。多跳还需每个中间 die 片内路径 + 每条 link latency + 是否流水重叠。

**改（覆盖 D2D 第 2 节 `L_route` 与落地方式）**：定**唯一延迟记账点**。因聚合包已实际穿越 router 网络拿到逐跳延迟，公式**不再加 `hops_src/hops_dst`**，只补 D2D 专属增量：`Σ 各条 link_latency + 端口序列化`；bulk service time 只由**发送端 / 网络包 / 接收端其一**负责。中间 die 片内跳同样由「包实际穿越」承担，公式只补 link latency。

---

# 第三轮 Review 修正（次级：澄清 / 护栏 / phase-2 / 两条真功能）

> 次级细化，同样优先于前文相冲突处。其中 **次-4（收端聚合）、次-6（反向/BDP）是真功能**，需在 G / F 节补规格；其余为澄清、护栏或 phase-2 标注。

## 次-1 banded 的「均衡」只在流量按行均匀时成立（澄清）

banded 保证的是**每端口分到的源核数相等（结构均衡）**，**不是流量均衡**——热行/热核会让对应端口变热。**精确承诺**：banded = "equal source-core count per port"。**流量倾斜时升级到 L1 `tag_hash` 或 L2 `dynamic`**（计划已有这两层）。不改算法，只把承诺讲准。

## 次-2 端口集中在少数坐标时「额外跳数 ≤ 半带宽」不成立（限定条件）

那个界只在**端口沿轴大致均布**时成立；端口聚集时某带的目标端口可能离该带很远，跳数可到全轴。**改**：把界标注为**有条件**；端口聚集时改用 `nearest`（Voronoi 中点切界）变体——它把跳数界到「到最近端口的距离」而非半带宽。

## 次-3 `k > axisLen` 时 bandOf 产生零宽 band（护栏，联动一轮 #3）

伪代码真 bug。但 **MVP `side==dir` 约束下 `ports_dir[D]` 全在同一条边 → k ≤ axisLen，零宽 band 不会发生**。**改**：(a) `bandOf`/`buildBandMap` 加 `assert k ≤ axisLen`；(b) 写明「按垂直轴分带」这个前提**依赖 side==dir**——phase-2 换 `Link{}` 表、允许 side≠dir 后，分带轴的定义需重想。

## 次-4 striping 多 tag，但 Recv_prim 按单 tag 检查（★真功能，联动二轮 #4）

**代码核实**：`Recv_prim` 只有一个 `tag_id`，完成条件 `end_cnt == recv_cnt && recv_cnt >= max_recv`（`logic.cpp:553`），收到不匹配 tag 会**直接报错**（`logic.cpp:509`）。→ 把传输条带成 k 个**不同 tag** 的子流，单个 Recv **收不了且报错**；striping 现只是发送端半拉子方案。

**改（G 节补收端聚合）**，二选一：
- **(a) k 个并行 `Recv_prim`**（每子流一个 tag）+ 一个 join 完成点；
- **(b) 新增 grouped-recv**：接受一个 **tag 集合/范围**，完成 = 所有 k 子流 end 包到齐。

每子流仍各自 REQ/ACK/DATA（ACK 按 REQUEST 源逐个回，只要每 tag 都 post 了 recv）。**subflow 索引 = 二轮 #4 flow key `(source_global_id, tag, subflow/epoch)` 的 subflow 位**。**没有收端聚合，striping 不能用**——从「次级」提一档，与 stripe 字段同批实现。

## 次-5 link_bw 是每端口独立还是多端口共享一条物理 link（定义）

**MVP：每个 C2C 端口 = 一条独立 D2D link**（`port_width`/`link_bw` 均 per-port，二轮 #8 的 `Σ independent_D2D_link_bw` 成立）。**共享 bundle**（多端口分一条物理链路预算，给该 Σ 封顶）= phase-2，放进 #3 的 `Link{}` 表。

## 次-6 缺半/全双工、反向带宽、在途 FIFO 容量、credit 往返（★真功能，联动二轮 #6/#9）

**反向带宽是刚需**：跨 die 的 **ACK 走反向链路**（二轮 #6 控制子通道），不建模反向，握手过不去。**改（F 节新增「link 物理参数」小节）**：
- **全双工（MVP）**：独立 tx/rx；`Link{}` 已有 `tx_bw/rx_bw`。半双工（共享+仲裁）= phase-2。
- **在途 FIFO 容量 ≥ 带宽时延积 `BDP = link_bw × 2L`**，否则稳态吞吐 < `link_bw`（链路填不满）——**必须写明的约束**。
- **辨清两种 credit**：二轮 #9 的 token-bucket 是**本地限速器，无往返**；跨链路**背压**往返 = `2L` → 背压若用 credit 回授，则 outstanding credit 池必须 ≥ BDP，否则吞吐塌。两者别混。

## 次-7 HOST/mem/C2C 同框架，缺 port_for_host / endpoint 类型 / 可达性校验（联动二轮 A1）

统一成端口 role 后三个缺口：
- **`port_for_host[core]`**：HOST 不再硬编西边 margin，而是一/多个端口 → 核需像 `port_for[core][dir]` 一样查「去 HOST 走哪个端口」；mem 同理（若移到边缘）。
- **endpoint 类型解码**：`des_` 落在保留区（A1 的 `HOST_ENDPOINT_ID` 及其上的 host/mem 段）时能判「core / host / mem」。
- **可达性校验**：断言**至少 1 个 HOST 端口存在且每核可达**（镜像 C2C 方向可达性检查）；若配置把所有 margin 端口 override 成 C2C → HOST 不可达 → 报错。
