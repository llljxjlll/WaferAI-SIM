# P5 开发记录：`DTE_SEND`/`DTE_RECV` 真实字节 P2P 数据通路

阶段：P5——`DTE_SEND`/`DTE_RECV` 真实字节 P2P 数据通路

阶段状态：完成；P5 实现、三轮评审修改、P5 专项退出条件、最新共享 binary 定向回归和 full CTest 均已闭环

记录日期：2026-08-12（UTC）

提交/变更集：工作区实现，尚未提交

## 计划确认与阶段边界

- 本阶段严格对照《编译产物指令集开发计划》11.1～11.4，把 `DTE_SEND`/`DTE_RECV` 从 loader/时序占位升级为真实 payload 的生产数据通路。
- public endpoint 只发布 P2P `UNICAST_TX/UNICAST_RX`；集合记录和 keyed internal child 由 P6 whole-artifact lowering 生成，但复用本阶段同一 payload、session、NoC/D2D 和 SRAM 写入路径，不建立集合专用假数据旁路。
- `DTE_ISSUE` 的 local direction 仍属于 P4；P5 不新增 remote `DTE_ISSUE` 外部表达。SRAM/HBM 作为 send source、SRAM 作为 receive destination，通过唯一的 endpoint Prim/runtime 表达。
- P5 不改变 NoC Router、D2D functional/bounded/behavioral backend 和 SRAM/HBM media 的计费责任。endpoint 负责 source read、分片/重组、协议完成与 destination write，不重复计 hop、link 或 memory media 时延。
- 目的端采用完整校验后一次提交：seq、tail、length、fsm、source、checksum 任一失败均不暴露部分 SRAM 写入，并走显式 abort/rollback 清理 session、buffer、token 和 transport tag。

## 按计划完成的开发

### 外部 record、lowering 与 strict Prim

- `DTE_SEND`/`DTE_RECV` codec 固定 mode、completion、fsm、token、length、peer、source/destination address、datatype 和 P2P/collective metadata；standalone P2P 要求 collective key 全零且 peer 为非 self，keyed record 不允许绕过 whole-artifact lowering。
- lowering 生成 strict `DTE_SEND_ENDPOINT`/`DTE_RECV_ENDPOINT` Prim，保留 SRAM region+offset、SRAM absolute、HBM absolute三种已发布地址形态；HBM source 禁止 region 伪装，receive destination 保持 SRAM 所有权。
- SYNC token 固定为零，ASYNC token 必须非零；`fsm_id`、length、UINT8 datatype、NONE reduce、peer 和地址形态均在进入 runtime 前严格验证。
- direct lowering、Prim wire、factory 和 legacy transport guard 均有正负例；非法 keyed record、非法 enum、reserved/padding、截断、额外段和地址二选一错误确定性拒绝。

### 真实 payload wire 与 source/destination memory

- TX 使用生产 `AccessUnit` 读取准确 `length` 字节：SRAM source 以 `Initiator::kDte/Command::kRead` 访问，HBM source 走绑定的 HBM runtime；不再用固定 16-byte 访问或只算延迟后丢弃内容。
- payload 使用明确的 endpoint discriminator、REQUEST declaration、每 fragment DATA、sequence、tail length、总字节数、CRC32C 和独立 ACK kind；业务 128-bit payload 与 V2b streaming timing sideband 不共用同一位域。
- endpoint DATA 每个 16-byte fragment 真实经过现有 Router/NoC/D2D。physical、streaming、bounded SAF 和 behavioral backend 使用相同业务字节；behavioral 也不折叠成常量 representative payload。
- RX 以 source/fsm/tag/length/seq/tail/checksum 完整重组，全部校验成功后再用 `Initiator::kNocRx/Command::kWrite` 写目的 SRAM；completion 只在最后一字节可见后发布。
- P5 memory probe 在仿真前用确定性非零 affine pattern 播种 SRAM/HBM source 与目的 sentinel，仿真后逐字节核对 payload 和前后 sentinel；probe apply 失败会回滚已写入的测试状态。

### session、fsm、token 与完成语义

- 每核 `P2pEndpointSessionRuntime` 以完整 32-bit `fsm_id`、lifetime-unique 16-bit transport tag、round 和方向标识 session；`fsm=1` 与 `fsm=0x10001` 不碰撞。
- transport tag 从 1 单调分配、取消后不复用、耗尽后明确失败而不 wrap，从而阻断 stale REQUEST/ACK 的 ABA；`RemainingTransportTags()` 使用无下溢的精确剩余量。
- REQUEST/CTS admission 在 DATA 前完成；容量不足时 REQUEST 等待且反向流仍可前进，不能用 early DATA 越过 admission。queued/active/terminal identical retransmission 为 exactly-once no-op，冲突 declaration 可恢复拒绝且不消费正确状态。
- TX local completion 与 remote completion ACK 是两道独立门：SYNC send 等本地发送完成，ASYNC WAIT 只消费本地 token，transport session/tag 保留到 completion ACK；ACK 先到或 local retire 先到均安全。
- RX post-before-send、send-before-post、commit-ready、SYNC retire、ASYNC WAIT、queued cancel、active cancel 拒绝、token reuse 与 late ACK compare-and-erase 均有状态机覆盖。
- 正常和失败路径的 session、async token、allocated tag、pending REQUEST、reassembly、commit-ready、reserved bytes、early DATA 与 timing sideband 全部纳入 residual/drain。

### NoC/D2D backend、统计与 trace

- endpoint REQUEST/ACK/DATA 复用现有 NoC 路由、跨 die exit/step、D2D link 和 bounded SAF credit；没有直接 peer SRAM copy 或 test-only payload side channel。
- bounded SAF 对 endpoint DATA 按 fragment 流送，避免把大于 SAF capacity 的完整 P2P payload 错当 legacy whole-flow reservation，同时保留有限 inflight/RX/control backpressure 与 drain。
- behavioral endpoint 使用有界 DATA/CTRL event table。ready/valid 都是 clocked `sc_signal`，runtime 显式保留 delayed grant，并为已发布但尚未到达的 grant 预留 slot，避免默认容量 3 在 4 KiB/32 KiB 流上多接收一个 fragment。
- source read、wire、destination write、admission、local completion、completion ACK、checksum、byte/flit count、stats 和每核/global drain 均有 production trace/oracle；runner 同时拒绝 `PROTO_WAIT` 和非零 residual。

## 三轮评审发现与修改闭环

### 第一轮：ABI、真实数据路径与生命周期

1. 审核 record→Prim→Worker→memory/Msg 的全链，确认不是 loader-only；补齐真实 `kDte` source read、endpoint DATA transport、完整重组后的 `kNocRx` write 和逐字节 memory probe。
2. 将 fsm 身份从可能发生 low16 碰撞的表达升级为 ACK payload 中完整 32-bit `fsm_id`，transport tag 独立且 lifetime-unique；补 `1/0x10001` 并发、out-of-order ACK 和 token ABA 测试。
3. 将 admission ACK 与 completion ACK 设为显式 kind，严格校验 endpoint marker、kind、source/destination/tag/fsm、reserved 和 wire round-trip，避免 legacy ACK/EVENT/普通流量被误识别。
4. 固化“完整校验后提交”：checksum、sequence、tail 或 declaration 失败不产生目的写，并原子回收所有 session/reassembly/token/tag 状态。

### 第二轮：事务恢复、队列冲突与错误 ACK

1. 评审发现相同 flow 的冲突 REQUEST 可能在 queued、active 或 terminal lifetime 上误消费正确条目；引入统一 ingress classification：identical duplicate 是 no-op，conflict 是可恢复拒绝，且 good REQUEST 随后仍可 admission、传输并 drain。
2. REQUEST queue、runtime admission 和 Worker ingress 按同一分类契约处理，容量/溢出检查发生在 commit 前；失败前后 residual、queue、reserved bytes、stats/timing/trace 保持原子。
3. malformed admission/completion ACK 的 fatal cleanup 先做 exact-fsm 匹配，再只允许 lifetime-unique flow fallback；kind/phase/flow 不匹配的 decoy session 不得被清理。
4. selftest 将 malformed ACK 先 `SerializeMsg/DeserializeMsg` 再注入，证明错误字段确实在 wire 后仍存在；同时覆盖错误 kind、错误 phase、wrong-fsm、wrong-tag、duplicate/stale ACK 和 decoy isolation。
5. 为 pending REQUEST capacity、early fragment/byte、seen request identity 和 transport tag 剩余量建立有界计算与 overflow guard；失败不遗留、计数不下溢。

### 第三轮：独立最终复审

- 第三轮只读复审重新检查 conflict REQUEST 的 queued/active/terminal recovery、事务/queue/runtime/timing/stats/trace 一致性、malformed ACK exact-fsm→lifetime-unique fallback、kind/phase/decoy 安全，以及 `RemainingTags/seen` 上界。
- 同时回看前两轮的 full-fsm identity、admission-before-DATA、local/remote completion 双门、token ABA、checksum commit 和 abort drain。结论无 BLOCKER/HIGH，允许解除 base/streaming/bounded/behavioral 四类 runtime CTest 门禁。
- 四类 runtime 全绿后，behavioral 4 KiB 回归暴露 event capacity 的 registered-ready 在途 grant 漏算；已按上述有界握手修复，并新增 4 KiB/256 fragment、32 KiB/2048 fragment、反复背压、逐 wire 有序无丢重、峰值/满周期和最终 drain 测试。
- 合并完整 D2D V1 selftest 时又发现 legacy BDP fixture 用裸整数冒充 256-bit Msg，新增严格 stats decode 后在 `@2ns` 报 unknown message type。修复只把 fixture 改为 canonical DATA/REQUEST `SerializeMsg`，输出再 decode `seq_id`；production 未放宽，独立完整入口 `41/41 PASS`。

## 最终回归证据

本节记录 2026-08-12 实际执行的阶段回归。开发中遵守各轮任务约束，没有在子任务内启动共享 build；最终由最新共享 binary 补跑 D2D V1 正式入口和 full CTest，关闭先前的 fixture/旧 oracle 两项待办。

### P5 专项与 memory/runtime

执行：

```text
/opt/cmake/bin/ctest --test-dir build --output-on-failure -R '^(p2p_payload_selftest|p2p_session_selftest|p5_memory_probe_selftest|program_p5_endpoint_runtime_(base|streaming|bounded|behavioral)|hbm_r[0-4]_selftest|sram_r[0-6]_selftest)$'
build/npusim --p2p-payload-selftest
build/npusim --p2p-session-selftest
build/npusim --p5-memory-probe-selftest
```

结果：

| 验证项 | 结果 | 证据 |
|---|---:|---|
| 最新共享 binary full CTest | 53/53 PASS | `ctest --test-dir build --output-on-failure`，全注册测试通过 |
| 定向 CTest 组 | 19/19 PASS | payload/session/probe、四 runtime、HBM R0～R4、SRAM R0～R6 |
| P2P payload | 201431 checks PASS | ABI、长度/pattern、非对齐、CRC、strict REQUEST/DATA、seq/tail、reassembler bound、timing sideband |
| P2P session | 190 checks PASS | admission、fsm/tag/token、sync/async、错误/abort、容量、ABA、drain |
| P5 memory probe | PASS | 仿真前 source/sentinel 播种与仿真后逐字节/checksum/sentinel 验证 |
| endpoint runtime base | PASS | production physical baseline |
| endpoint runtime streaming | PASS | timing sideband 与业务 payload 分离 |
| endpoint runtime bounded | PASS | finite SAF/credit/backpressure/drain |
| endpoint runtime behavioral | PASS | 真实 endpoint fragments、内容与完成语义 |

四 runtime 覆盖 same-die SRAM SYNC 129 B、cross-die SRAM ASYNC 4 KiB、HBM source SYNC 129 B，并逐项核对 source/wire/destination CRC、bytes/fragments、`kDte` read、`kNocRx` write、local/remote completion、stats 和 residual。

### DTE V0～V4 runners

执行 `python3 llm/test/dte/run_test_dte_v0.py`、`run_test_dte_v1.py`、`run_test_dte_v2.py`、`run_test_dte_v2b.py`、`run_test_dte_v3.py`、`run_test_dte_v3b.py`、`run_test_dte_v4.py`。

| runner | 结果 |
|---|---:|
| DTE V0 | config/oracle/SystemC 全 PASS；SystemC 66 checks |
| DTE V1 | 13 项全部 PASS，含 same/cross-die、stripe、multi-source、repeated lifecycle、heterogeneous config |
| DTE V2a | 15/15 PASS |
| DTE V2b | 18/18 PASS |
| DTE V3a | 14/14 PASS；SystemC selftest 33/33 checks |
| DTE V3b | 16/16 PASS |
| DTE V4 | 18/18 PASS |

DTE V3a runner 的精确输出 oracle 已由旧 `31/31` 更新为当前 `33/33`；runner 定向复跑 14/14 PASS，selftest 和全部 13 个 runtime/integration 场景同时通过。

### D2D V0～V5 与 NoC congestion

执行：

```text
python3 llm/test/run_v5_exit.py
build/npusim --d2d-link-selftest
/usr/bin/c++ ... /tmp/d2d_link_selftest_main.cpp ... -Werror -o /tmp/d2d_full_selftest
/tmp/d2d_full_selftest
```

| 验证项 | 结果 |
|---|---:|
| D2D V0 pure functions | 308/308 PASS |
| D2D V0～V2 historical runner | 原 66/67 的唯一 fixture 失败已关闭 |
| D2D V1 最新共享 binary 正式入口 | 41/41 PASS |
| D2D V3 bounded SAF | 16/16 PASS |
| D2D V4 independent oracle | 8/8 PASS |
| D2D V4 production/calibration | 13/13 PASS |
| D2D V5 multi-port/striping/dynamic | 23/23 PASS |
| NoC congestion frozen oracle | 4/4 PASS |

NoC 精确冻结值保持：no-congestion behavioral/cycle=`14781/29109 ns`，congestion behavioral/cycle=`14833/45441 ns`。原 D2D V1 失败根因是 legacy BDP fixture 用裸整数冒充 Msg；fixture 改为 canonical DATA/REQUEST wire 后，先由独立 Werror 完整入口验证，再由最新共享 binary 正式执行 `--d2d-link-selftest`，legacy BDP/SAF/link 与 P5 behavioral 4 KiB/32 KiB 全部 `41/41 PASS`。

### SRAM/HBM runners 与 selftests

- CTest：HBM R0～R4 5/5 PASS，SRAM R0～R6 7/7 PASS。
- `run_test_sram_pipeline.py`：checksum `130560`，LSU/DTE blocking/double timing、memory/compute overlap 与完整 staged lifecycle PASS。
- `run_test_sram_compat.py`：legacy NpuBase helper path PASS。
- `run_test_sram_legacy.py`：LSU/DTE payload checksum `130560` 与 legacy-private overlap PASS。
- `run_test_sram_numa.py`：core0 LSU/core16 DTE remote-home checksum `130560`，D2D in/out `672/672` PASS。
- `run_test_sram_dramsys.py`：LSU/DTE payload checksum `130560` 与 distributed DRAMSys overlap PASS。

## 对照开发计划 11.1～11.4

- 11.1：核间通信已从协议/时序记账升级为真实 payload 搬运；NoC/D2D/SRAM/HBM 继续拥有唯一资源和时序责任。
- 11.2：record/lowering、strict endpoint Prim、Worker dispatch、source read、fragment Msg、session/fsm/token、destination commit、completion、stats/trace/drain 均进入 production；physical/bounded/behavioral/streaming 没有 payload 旁路。
- 11.3.1：payload 自测覆盖 1、15、16、17、127、128、129、fragment boundary、4 KiB、32 KiB，多 pattern、非对齐 slice、checksum；program runtime 验证目的逐字节、source/wire/destination checksum 和 sentinel。
- 11.3.2：same-die/cross-die、SRAM/HBM source、physical/streaming/bounded/behavioral、stripe/D2D/NoC 回归已有证据；timing sideband 与业务 payload wire 分离。
- 11.3.3：SYNC/ASYNC、recv-first/send-first、多个 fsm、full-fsm collision、token reuse、backpressure、cancel、duplicate/unknown/stale completion 均有状态机与 runtime 覆盖。
- 11.3.4：零/超长 length、地址/来源/fsm/tag/seq/tail/checksum、冲突 REQUEST、malformed ACK、容量/overflow、部分提交和所有 residual 失败路径均有负例。
- 11.3.5：source read、endpoint service、NoC/D2D、destination write 与 completion trace 可观测；DTE/D2D/NoC frozen timing 与资源 oracle 无行为回退。
- 11.3.6：DTE V0～V4、D2D V0～V5、NoC congestion、SRAM/HBM runner/selftest 已按上表执行；D2D fixture 与 DTE V3 count oracle 均已修改、复跑并关闭，最新 full CTest 53/53 PASS。
- 11.4：三轮评审的真实通路、wire payload、计费唯一性、RX 可见性、错误原子性、跨 die 内容和有限缓冲 backpressure 均已闭环，无遗留 BLOCKER/HIGH。

## 阶段退出条件

- [x] same-die、cross-die、SYNC、ASYNC P2P 均通过逐字节/checksum/sentinel 校验。
- [x] SRAM/HBM 已发布 source 和 SRAM destination 均有 production `kDte` read、wire payload、`kNocRx` write 与 completion trace。
- [x] payload 与 streaming timestamp/behavioral timing sideband 分离；physical、bounded、behavioral 不使用常量 payload 或直接 SRAM copy。
- [x] fsm、token、transport tag、admission/local/remote completion 生命周期明确，full-fsm collision 与 ABA 已封闭。
- [x] seq/tail/checksum/REQUEST/ACK/capacity 错误原子失败；正常与异常路径 session/buffer/tag/credit/timing residual 可清零。
- [x] base、streaming、bounded、behavioral 四类 P5 runtime CTest PASS。
- [x] payload 201431 checks、session 190 checks、memory probe、SRAM/HBM 与 DTE/D2D/NoC P5相关行为回归通过。
- [x] P6 可直接复用 endpoint payload/session/NoC/D2D/SRAM 数据通路，不需要集合假数据路径。
- [x] 最新共享 binary 的 `--d2d-link-selftest` 41/41 PASS，原 66/67 aggregate 中唯一 fixture 失败已关闭。
- [x] 非 P5 DTE V3 runner 精确 oracle 已更新为 33/33，定向 runner 14/14 PASS。
- [x] 最新共享 binary full CTest 53/53 PASS。

是否允许继续后续集成：允许。P5 数据通路退出条件已满足，第三轮评审无 BLOCKER/HIGH，最新共享 binary 定向与 full CTest 门禁全部通过。

结论：P5 已建立唯一的真实字节 P2P production 通路，same/cross-die、SRAM/HBM、SYNC/ASYNC、physical/streaming/bounded/behavioral 的值、时序、信用、trace 和 drain 已由专项与回归证据闭环。最新共享 binary 的 D2D V1 41/41、DTE V3 runner 14/14 和 full CTest 53/53 均通过；P5 阶段正式完成，数据面可作为 P6 collective child 的复用基础。
