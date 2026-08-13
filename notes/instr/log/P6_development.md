# P6 开发记录：九种通信语义与对称分解

阶段：P6——九种通信语义（P2P 与八种集合通信）、core group 与对称分解

阶段状态：完成；实现、评审修改、真实字节矩阵与正式 CTest 门禁均已闭环

完成日期：2026-08-12（UTC）

提交/变更集：工作区实现，尚未提交

## 计划确认

- 本阶段严格对照《编译产物指令集开发计划》12.1～12.4，在 P5 唯一真实字节 endpoint 通路之上实现 TX×RX 3×3 通信矩阵。
- baseline 只使用 unicast endpoint 与 `COLLECTIVE_DATA_V1`；不编程 multicast tree、不进入 DCA。P7 只替换正交 backend，不改九种高层语义。
- 非 P2P 集合 group 必须同 die；standalone P2P 保留跨 die。普通 Reduce 固定单 root，ReduceScatter/AllReduce 使用 per-target 对称分解。
- internal phase barrier、public `GROUP_SYNC`、aggregate token、child fsm/token 使用独立命名空间；receive-post、send-issue、wait、phase barrier 的顺序在 immutable program image 中冻结。

## 实际完成事项

### whole-artifact graph、规划与 lowering

- whole-artifact graph 按 `(group_id, collective_id, epoch)` 聚合 record，验证成员、rank、root、mode、长度、dtype、reduce op、expected source、token/fsm 与地址布局。
- planner 为九种语义生成 canonical `(chunk, source, destination)` child flow、tight SRAM layout、有限容量 greedy wave 和双 phase；输入 record 顺序变化不改变产物。
- AllToAll、AllGather、ReduceScatter、AllReduce 均展开为 per-core action stream，不生成运行时 `COLLECTIVE_CALL`；ReduceScatter/AllReduce 无 root 中转。
- `SRAM_REGION` ABI 固定为 `value=physical byte base`、`size=bounded extent`、relocation addend 为 region-local offset。在丢失 symbol 边界前验证 endpoint `L`、scatter/gather/reduce `N*L`、result `L` 及绝对地址跨度。
- child、action、wave 与 derived-byte 限额在任何大规模 reserve/resize/循环前检查，覆盖 exactly-limit、limit+1、chunk/N² 放大与溢出。

### internal Prim、执行与生命周期

- strict-only `COLLECTIVE_DATA_V1` 的 LOCAL_COPY/REDUCE 使用真实 `AccessUnit` 一次读、一次写；UINT8/INT32/INT64 小端 SUM/MAX、负数 MAX 和 wrapping 规则有纯函数与 SystemC 数据测试。
- `COLLECTIVE_PHASE_BARRIER_V1` 用完整 key+phase 隔离 N=1/2/4、连续 epoch 和不同 group；barrier 有界、可 abort/reset，正常和异常路径均不遗留 waiter/state。
- 实现 `COLLECTIVE_LAUNCH_V1`、child endpoint/data/phase materializer、immutable image、per-core executor、wave admission、aggregate runtime 与 final-phase gate。
- aggregate public token 仅在所有 child 的 local+transport 双门及本地 work 完成后 READY；WAIT/FENCE/CANCEL、连续 epoch、普通 DTE token 与 internal child token 冲突均有事务测试。
- helper 在验证、relocation、graph/image 构建和 wire 预序列化全部成功后一次 commit；失败保持 helper、CoreConfig、group registry、label table 和旧 image 不变。

### 真实字节 runtime 与可观测性

- child 完整复用 P5 REQUEST/CTS/DATA/ACK、CRC32C、Router/NoC、SRAM `kDte` read 与 `kNocRx` write，无 rank 合成值或测试旁路。
- multi-region probe 在仿真前播种每核 input/staging/result，仿真后逐字节验证结果、checksum 和完整范围外 sentinel。
- fixture 覆盖九宫格 N=1/2/4、1/17/1KiB/8KiB/32KiB、UINT8/INT32/INT64 SUM/MAX；每个 active core 末尾带 FENCE，避免 passive participant 提前 DONE。
- runtime 输出 action/wave、aggregate/admission/barrier/endpoint/session 与每核/global drain；runner 精确核对软件 oracle 和零残留。

## 评审、修改、复测与再评审

1. 配置 ACK 的 flow id 可在上一轮 ACK 尚未收齐时前移。修复为旧集合未完成时保留已收 ACK，仅在旧集合完整后进入新 phase 时清零；跨 phase ACK 回归和 N=4 AllReduce 通过。
2. collective 地址转为 absolute 后会丢失 symbol extent，且 `SRAM_REGION value/addend` 存在重复加 base 的歧义。统一文档与 loader，加入 `offset+span<=size`、负 addend、base/offset/span overflow 和 relocation 后二次完整校验。
3. planner 原先先构造巨量派生状态、后由 image 限额拒绝。将限额与 checked 预估下沉到所有 materialization 之前；ASan/UBSan/leak 和 limit+1 用例通过。
4. 1 KiB AllGather 首次暴露 `expected=61 actual=62`：Router 连续包在同一 SystemC delta 写 `valid=false` 再写 `true`，下一跳看不到新的 posedge。`requires_pulse_gap` 让 bit255 endpoint wire 在每一跳获得显式低周期；原必现 1 KiB、32 KiB 和完整 37 场景复测通过。修复的是脉冲协议，没有扩大 buffer。
5. 最终复审确认 baseline trace 为 unicast/endpoint，tree/DCA 为零；非 P2P cross-die 在 loader 拒绝；完成后的 endpoint、aggregate、barrier、Router 与 probe residual 全为零。

## 自测与集成证据

执行：

```text
build/npusim --isa-v1-selftest
/opt/cmake/bin/ctest --test-dir build -L p6 --output-on-failure
python3 llm/test/program/run_p6_collective_program.py --runtime-all ...
```

- unified ISA 全绿：manifest/factory 975、record codec 917、Prim wire 585、Program Format 656、record lowering 299、program helper 182。
- P6/P7 纯测全绿：planner 135、data 30、graph 78、data lowering 22、child 31、aggregate 55、wave 25、final gate 23、executor 59、profile 341、profile image 46、topology 35、tree batch 37、phase 25、image 85。
- 正式 P6 CTest：runner、fixture/loader、runtime matrix `3/3 PASS`。
- fixture/loader：37 个正例、6 个 malformed artifact loader 拒绝、1 个 codec 原子拒绝通过。
- 真实 runtime `37/37 PASS`；每例检查真实字节、checksum、sentinel、action/wave trace 和 drain。包含 32 KiB INT64 MAX、8 KiB INT64 SUM、1 KiB INT32 SUM/MAX。
- 原必现 1 KiB AllGather 与 32 KiB Broadcast 定向复测通过；相关文件 `git diff --check` 无诊断。

## 与计划的差异和边界

- 实现采用独立 `coll_plan_v1`、whole-artifact graph 与 immutable image，保持 legacy `coll_plan.h`/codec 路径不变，降低兼容风险。
- P6 baseline 完整闭环；真实 multicast/DCA、tree program/erase 与四档 profile runtime 归 P7，P6 不以旧 metadata-only multicast 或 synthetic-rank DCA 冒充完成。
- JSON 不具备等价九宫格 whole-artifact 表达，因此 P6 由独立软件 byte oracle 验证；P3 已另行完成计算链 JSON/program 等价测试。

## 阶段退出条件

- [x] 计划确认、代码、自测、集成测试和评审修改完成。
- [x] 九种模式 lowering、真实值、exactly-once 和 drain 通过。
- [x] ReduceScatter/AllReduce 对称分解；普通 Reduce 单 root。
- [x] baseline 不触发 multicast/DCA，且无静默 fallback。
- [x] 非 P2P 跨 die 明确拒绝；P2P 跨 die 保留。
- [x] dtype、tail、1/8/32 KiB、N=1/2/4、有限容量和错误矩阵通过。
- [x] loader/helper/image 失败原子，派生状态在分配前有界。
- [x] endpoint、fsm/token、aggregate、barrier、Router、SRAM probe 全部 drain。
- [x] 契约变化与 Router pulse 修复均完成定向复测和完整矩阵再评审。

是否允许继续后续集成：允许。

依据：P6 正式 CTest 3/3、真实 runtime 37/37 和 unified ISA 全绿；计划 12.4 的退出条件全部有自动化证据，P7 可在相同语义 image 上替换 backend。
