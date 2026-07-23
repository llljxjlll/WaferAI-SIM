# D2D Link V5 开发记录

> 分支：`feat/d2d-v5`
>
> 基线：V4 main `9b3ae64`，冻结基线 `d2d-v4-baseline`
>
> 目标：在 V4 单端口模型之上完成多物理端口、1/2/4 路 striping、共享 link group、
> 条带化 whole-flow SAF、Behavioral 多端口 min-cut 与 active-flow dynamic，并保持 V0–V4
> 和 NoC 冻结契约精确通过。

## 1. 固定契约

- `multi_port=true` 必须显式选择 `nearest|banded_nearest|tag_hash|hybrid|dynamic`；缺省配置
  继续走 V1–V4 单端口行为。
- 一个逻辑 flow 由 `(source,tag)` 标识，subflow 只用于条带化区分；`output_lock` 仍按接收槽
  `tag`，不能错误迁移为 FlowKey。
- striping 配额使用 `q=F/k,r=F%k`，前 `r` 条子流取 `q+1`；V5 支持 `k=1/2/4`，
  `F<k` 在发送 REQUEST 前拒绝。
- 每个 subflow 在每个 die 内必须固定一个出口；dynamic 也只能按 flow pin，禁止逐包重选。
- 多条独立 link 可以并行；相同 link_group 的端口必须共享一个物理 DATA cut，CTRL 独立。
- bounded SAF 对所有 subflow 和所有 hop 完整预检后一次提交，失败不得留下部分 reservation。
- Behavioral 只计一次端到端 bulk service，多 hop 只增加固定 link latency；真实 min-cut 必须包含
  源/目的 NoC、端口、独立 link 和去重后的 link_group。

## 2. 增量顺序

```text
V5-a  配置、wire tagged-union、stripe 原语与选择接缝
V5-b  多端口生产接线和 per-subflow pin
V5-c  sender q/r striping、receiver grouped completion、完整性探针
V5-d  条带化原子 SAF 与共享 link-group 仲裁
V5-e  Behavioral 多端口 min-cut 与独立 Python oracle
V5-f  active-flow dynamic pin、释放、活性和确定性
V5-g  1/2/4/8 die 规模门、全版本聚合门、冻结
```

## 3. V5-a：配置与编码（完成）

- 端口增加 `link_group`，镜像端必须一致；`multi_port=true` 时允许每方向多个 C2C 端口。
- REQUEST/ACK/DATA 把 CONFIG 专用两位 tagged-union 为 `subflow=0..3`，不扩宽 256-bit wire；
  原语 spare bits 承载 stripe，旧全零编码仍表示 1。
- workload 增加 `cast.stripe` 与 `recv_stripe`，配置生成路径同步 REQUEST/ACK/DATA。
- 静态选择由 `(source,tag,subflow,seed)` 决定；固定 seed 可复现。

验证：纯函数从 300 增至 305；V4 聚合门与 NoC 冻结值不变。

## 4. V5-b/V5-c：生产 striping 闭环（完成）

- sender 为每条 subflow 独立发送 REQUEST、等待 ACK，再用固定 exit 发送独立 seq 空间 DATA。
- receiver 按 `(source,subflow)` 收尾；全部 sender×stripe 尾包完成后只产生一次逻辑完成。
- `[V5_SUBFLOW]` 按 `(link,source,tag,subflow)` 记录包数、顺序 hash、完整 payload checksum、
  连续 seq 和唯一尾包，避免合法交织被旧全局探针误判为乱序。
- `F=7,k=1/2/4` 配额为 `[7]`、`[4,3]`、`[2,2,2,1]`；k=4 精确使用四条 link，
  两次运行路径和统计一致，所有残留为 0。

自审修复：Behavioral 代表 DATA 必须同时 `seq=1` 和 `is_end=true`，才能成对获取/释放 Router
锁；修复后 V4 Behavioral 继续 13/13。

## 5. V5-d：条带化 SAF 与共享 group（完成）

- sender 在首个 REQUEST 前计算全部 subflow 配额和逐 hop 物理路径；先聚合物理 link 与共享
  group 需求，完整预检成功后统一提交。
- 释放前同时校验 link/group 对同一 FlowKey 的 reservation，再成对扣账，防止半释放。
- 相同有向 die pair/group 的 DATA 用确定性 round-robin 共享一个 packet/cycle cut；CTRL 不限速。
- `F=31,k=4`：独立 group 692 ns，共享 group 702 ns；四成员 group stall 为
  `19/9/17/16`，配额 `[8,8,8,7]` 全部排空。

自审修复：补 `ReservedFor` 预检；V5 改写的 SAF 错误文本后来恢复 V3 冻结契约，同时保留
V5 的首个 REQUEST 前 link/group 诊断。

## 6. V5-e：Behavioral 多端口 min-cut（完成）

- 每个 die-hop 聚合实际选中的端口速率和与独立 link 速率和；相同 link_group 只计一次，
  再与源/目的单 lane NoC cut=`1/1` 取最小，禁止 `k*single_lane` 虚假加速。
- 完整逻辑 flow 元数据只注册一次，由 subflow0 首链代表包消费；bulk service 只计一次，
  其余代表包只承担固定 latency。
- 独立 Python oracle 不读取 C++ 统计。`F=31,k=4,rate=1/4`：独立链
  `R=1,S=31`，共享 group `R=1/4,S=124`；C++ ledger 精确为
  `(1,31,31,21,52)` 与 `(1,31,124,21,145)`，完成 572/742 ns。

## 7. V5-f：active-flow dynamic（完成）

- 动态 key 为 `(local_die,dir,FlowKey)`；选择最少 active-flow 的物理端口，固定 hash/seed
  只用于同负载 tie-break。同一 REQUEST/DATA 重查只返回缓存，不增加负载。
- DATA 尾包与 ACK 真正从 C2C egress 发出后分别释放正向和反向 pin；缺失/重复释放、下溢、
  错 die/方向均硬失败。active pin 纳入 drain 与 watchdog。
- 多 hop 用 REQUEST 控制面在每个 die 首次到达时惰性建 pin，后续 DATA 查缓存；因此不是逐包
  动态重选，等价于 DATA 前建立整条逐 die 固定路径。

验证：

- 单流 `F=31,k=4` selection/release=`8/8`；
- 两并发流=`16/16`，每流四端口配额 `[8,8,8,7]`，最终 load 全 0；
- 3×1 两 hop 单流=`16/16`，8 条 forward subflow-hop link 完整，
  typed=`(8,8,8,8,62,62)`、repin=`78/78/0`；
- 连续两次路径、统计与 sim-time 一致，无 watchdog；纯函数 307/307，V5 runner 21/21。

自审补强：增加同 key 不重复计数、重复 release 不破坏状态，以及多 hop 每 die 独立 pin/release；
历史聚合门发现 V3 SAF 文案回归后恢复兼容错误契约，V3 回到 16/16。

## 8. V5-g：规模门与冻结（完成）

1/2/4/8 dies 真实 production smoke：

| dies | cores/router/worker | directed link units | wall time | peak RSS |
|---:|---:|---:|---:|---:|
| 1 | 16 | 0 | 0.138 s | 135256 KiB |
| 2 | 32 | 8 | 0.190 s | 247808 KiB |
| 4 | 64 | 24 | 0.402 s | 467328 KiB |
| 8 | 128 | 56 | 0.929 s | 906624 KiB |

多 die 均把 `F=7,k=4` 从 die0 发到最远 die，逐 subflow-hop 数据完整、dynamic
selection==release、Router/Link/SAF/group/pin 全排空。资源门使用 30 s、1.5 GiB 绝对上限和
宽松相对增长护栏；它是当前容器的 smoke regression，不是跨机器 benchmark。

统一入口：

```bash
python3 llm/test/run_v5_exit.py
```

冻结门：

| 门 | 结果 |
|---|---:|
| V0–V2 历史 D2D | 67/67；pure 307/307；Link 37/37 |
| V3 production bounded SAF | 16/16 |
| V4 independent oracle | 8/8 |
| V4 production/calibration | 13/13 |
| V5 multi-port/striping/dynamic | 23/23 |
| NoC frozen four scenarios | 14781/29109、14833/45441 |

聚合结果：`AGGREGATE EXIT=0`。冻结 tag：`d2d-v5-baseline`。

## 9. V5 冻结范围与边界

冻结保证：cycle/Behavioral 多端口、1/2/4 subflow、非整除配额、静态和 dynamic pin、独立
link 与共享 group、条带化原子 SAF、逐 die 多 hop、统一逻辑完成、完整性/活性/排空和
1/2/4/8 die smoke scale。

不承诺：8-way 以上 striping、任意 `rate>1` 的单 wire、多路径自适应重路由、跨 flow 的
Behavioral 拥塞近似，以及把当前容器的 RSS/wall 数字当作其他机器的性能指标。

## 10. 冻结后独立评审

评审结论：**V5 无阻塞问题，冻结范围完整可信，不需要修改生产代码、测试判据或冻结 tag。**

评审复核结果：

| 门 | 结果 |
|---|---:|
| Pure-function self-test | 307/307 |
| Standalone Link self-test | 37/37 |
| V5 runner | 23/23 |
| `run_v5_exit.py` | `AGGREGATE EXIT=0` |
| NoC frozen baseline | 14781/29109、14833/45441 |

评审特别确认：

- 多端口 Behavioral min-cut 从源 NoC `1/1` 起算；端口和独立 link 聚合均封顶为 `1/1`，
  相同 `link_group` 只计一次，不会产生 `k×single-lane` 虚假带宽。
- quotient/remainder striping、per-subflow pin、重组和 `F<stripe_count` 启动期拒绝符合契约。
- striped SAF 对全部 subflow 的全部 hop 做 link/group 原子 admission，失败发生在首个
  REQUEST/DATA 前，成功路径的两类 reservation 均成对释放。
- V0–V4 配置仍由显式 `multi_port` 门保护；cycle、bounded、Behavioral 历史冻结值未漂移。
- Behavioral runtime 与独立 Python oracle 精确一致；dynamic 在单跳、多流和多跳下均
  per-flow/per-die 固定、可复现并完全释放。
- 1/2/4/8 dies 的层级、link 数、完整性、排空和资源 smoke bound 均通过。

工作树中的 `notes/extensions/DTE/` 是独立 DTE 建模计划，未纳入
`d2d-v5-baseline`，也不应在 D2D 的冻结后维护提交中误收。

本节属于冻结后的文档回填；`d2d-v5-baseline` 继续固定指向原冻结提交，不移动 tag。
