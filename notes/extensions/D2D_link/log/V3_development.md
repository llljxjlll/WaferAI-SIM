# V3 开发总结（有限缓冲、背压、瓶颈与 whole-flow SAF）

> 记录 WaferAI-SIM D2D Link **V3** 的开发、验证、审查修复与冻结过程。规划见
> `../D2D_link_test.md`、`../D2D通信建模计划.md`，测试入口见 `llm/test/d2d_link/README.md`。
>
> **一句话**：V3 把 V1/V2 的功能性无限 Link 升级为生产可用的有限 SAF/inflight/RX/CTRL 流水线，
> 用整数 token bucket 建模端口与链路速率，用 DATA/CTRL 信用表达背压，并以全路径原子 whole-flow
> SAF reservation 防止有限缓冲下的跨 die hold-and-wait。

## 1. 目标、选择与边界

- 默认 `functional_v2` 必须保持 V1/V2 冻结行为；只有显式 `mode=bounded_saf` 才进入 V3。
- 网络安全机制固定为 **whole-flow store-and-forward**，不在同一版本摇摆到 escape VC。
- `saf_buffer_depth >= F` 是安全正确性条件；`link_inflight_depth >= BDP` 是传输窗口/利用率条件，
  二者不可替代。
- 单个 `sc_bv<256>` 每拍最多一包，因此 V3 只支持 `0<rate<=1`、每方向一个 C2C 端口；
  多 lane、多端口、striping 留给 V5，Behavioral 快速模型留给 V4。

## 2. 增量顺序

```text
V3-a  模式、速率、四类容量与安全策略配置契约
V3-b  standalone 有限 Link、token bucket、信用与 BDP 边界
V3-c  REQUEST flow_packets + WholeFlowSafAdmission 原子账本
V3-d  生产 SAF/inflight/RX/CTRL 流水线 + 全路径预留 + DATA/CTRL 信用
V3-e  瓶颈、争用、混合 NoC、背压与最小缓冲安全压力
Review 修复  控制 ready 越界、连续信用事件、DATA 生产信用、later-hop 回滚测试
Freeze  全版本聚合门 + 文档 + tag d2d-v3-baseline
```

## 3. 实现

### 3.1 配置与消息协议

- `functional_v2` / `bounded_saf` 字段严格分区；有限模式必须显式
  `safety=whole_flow_saf`。
- `port_rate/link_rate` 用整数有理数，拒绝 0、负数和 `>1`。
- `saf_buffer_depth`、`link_inflight_depth`、`rx_buffer_depth`、`ctrl_buffer_depth`
  四类容量分别必填。
- REQUEST 用 tagged union 复用 24-bit roofline 字段携带 `flow_packets`，消息仍为 256 bit；
  dataflow 归一化阶段将 REQ 与后续同 `(dest,tag)` 的 DATA 配对。

### 3.2 原子 whole-flow SAF

- REQUEST 注入前，沿 die-level XY 路径为每条**有向** link 原子预留 `F` 个 SAF 槽。
- 任一 later hop 容量不足时回滚所有 earlier-hop 预留，并在 REQUEST/DATA 注入前报错。
- 每个 Link 收到 REQUEST 后记录 F；DATA 按 `FlowKey(source,tag,subflow)` 收齐，核对尾包和包数，
  整条 flow 完成后才进入物理 Link。
- 每跳 SAF 排空该 flow 时独立释放该跳 reservation；成功结束必须
  `[SAF] reserved_packets=0`。

### 3.3 生产有限流水线与信用

```text
Router -> whole-flow SAF -> port limiter -> link limiter/inflight -> RX -> Router
```

- port/link 两个整数 token bucket 独立，稳态吞吐由最慢段自然决定。
- SAF、inflight、RX、CTRL 分别有限；RX 满会阻塞 mature inflight，继而阻塞 SAF drain。
- DATA credit 表示 SAF 空位：初始为 SAF depth，发送扣一，包从 SAF 进入 inflight 时归还。
- CTRL credit 表示控制 FIFO 空位：初始为 ctrl depth，交付远端后归还。
- 信用回程采用每次归还**翻转 event bit**，避免连续周期的布尔高电平被合并；Router 检查
  下溢/上溢，结束时要求 DATA/CTRL 信用均恢复容量。
- whole-flow SAF 在完整 flow 落地时已释放源 NoC 锁，故远端背压止于 SAF drain；新 flow 无容量时
  在源端 admission 拒绝，而不是让已整流 flow 继续持有源锁。

### 3.4 可观测性

- `[D2D_BOUND]`：每条有向 link 的 SAF/inflight/RX 峰值、满周期及 port/link/inflight/RX/downstream stall。
- `[NOC_ACT]`：逐 die 成功 mesh send/stall 与 D2D 源端 credit stall。
- `[FLOW_DONE]`：按 `(source,tag,dest)` 的 DATA tail 到达 cycle。
- `[SAF_ADMIT]`、`[SAF]`：admission 成功/拒绝与剩余 reservation。
- `[CREDIT]`：DATA/CTRL 信用是否恢复；`[DRAIN]`：Router/Link residual。

## 4. 定量验证

- **瓶颈**：128 包长流 goodput 为 1、0.501976、0.250493，对应理论
  `min(1,port,link)=1、1/2、1/4`，误差 <1%；非瓶颈变化不改吞吐。
- **争用**：共享 Link 两 flow 都完成且总包数守恒；独立 Link 并行；全双工相反方向容量独立。
- **源 die 混合流量**：shared/disjoint NoC stall `11/0`，shared D2D 更晚完成。
- **中间 die 混合流量**：shared/disjoint stall `7/0`，D2D tail cycle `397/380`。
- **生产背压**：中间 die 热点在 `rx=1,inflight=4` 下得到
  `inflight_full=85、rx_full=95、inflight_stall=60、rx_stall=63、downstream_stall=63`，
  连续两次一致并最终排空。
- **多跳/多方向**：3×1 两跳每个 forward SAF 均存满 32 包；2×2 双向对角覆盖 E/N/W/S；
  四流固定置换在 `SAF=F=4、inflight=BDP=4、rx=1、ctrl=1` 下使八条有向 Link 均承载
  4 DATA 并完成。
- **浅控制缓冲**：`ctrl_depth=1` 下 8 并发 flow 的 8 REQUEST、8 ACK、32 DATA 全部完成，
  信用平衡且无越界。

## 5. 代码审查发现与修复

1. **生产 CTRL 只用 registered ready 会越界**：`ctrl_depth=1`、8 flow 压力复现
   `bounded CTRL producer exceeded advertised capacity`。改为真实控制信用后通过。
2. **布尔 pulse 可能吞掉连续归还**：单周期拉高无法可靠表达连续事件。改为 toggle event bit，
   并增加 Router 容量上界与结束态信用平衡检查。
3. **生产 DATA 仍只靠 reservation+ready，声明弱于 V3 契约**：补齐 SAF 槽位 DATA credit，
   registered ready 只保留诊断镜像。
4. **原回滚测试没有真的部分成功**：旧用例让第一跳就失败，却声称测到 later-hop rollback。
   改为先占满第二跳，再申请两跳路径；验证第一跳临时预留被回滚。
5. **BDP 措辞过度声称生产逐包 credit**：生产 DATA 使用全路径 reservation + SAF credit；
   BDP 公式是 standalone 已验证、生产继续强制的保守 transport-window 下界，文档已收紧。

## 6. 冻结准入

统一命令：`python3 llm/test/run_v0_exit.py`。冻结点必须同时通过：

- 纯函数/路由自测 **284/284**；
- Link SystemC 自测 **32/32**；
- 历史 D2D runner **67/67**；
- V3 production runner **16/16**；
- NoC 四场景 **14781/29109、14833/45441** 精确不变；
- `git diff --check` 干净。

冻结分支：`feat/d2d-v3`；冻结 tag：`d2d-v3-baseline`。tag 指向包含实现、测试和本日志的
V3 freeze 提交；`main` 只通过 fast-forward 合并，不重写 V0/V1/V2 tag。

## 7. 明确留待后续

- V4：Behavioral D2D 与更一般的解析 oracle/校准。
- V5：多 lane、多端口聚合、striping 与 subflow。
- escape VC、chunked SAF、`rate>1`、非 dataflow 多 die、逐事务 trace。
- 当前同一 `(source,tag,subflow)` active flow 重入会明确拒绝；未引入独立 flow-instance 序号。
