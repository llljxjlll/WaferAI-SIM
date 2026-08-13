# P4 开发记录：访存、异步控制、SRAM 生命周期与同步

阶段：P4——访存、异步控制、SRAM 生命周期与同步

阶段状态：完成；访存、异步控制、SRAM 生命周期、同步以及 fresh Release/Debug 63/63 均已闭环

记录日期：2026-08-11（UTC）

提交/变更集：工作区实现，尚未提交

## 计划确认与阶段边界

- 本阶段严格对照《编译产物指令集开发计划》10.1～10.4，完成 v1 的阻塞 LSU、非阻塞本地 DTE、SRAM region 生命周期、静态 core group、public `GROUP_SYNC` 和 `EVENT_SET/WAIT`。
- P0 D-19 已冻结：public `DTE_ISSUE` 只发布 `SPM_TO_SPM`、`SPM_TO_DRAM`、`DRAM_TO_SPM` 三个本地 direction；`SPM_TO_REMOTE`、`REMOTE_TO_SPM`、`DRAM_TO_REMOTE` 不通过 `DTE_ISSUE` 编码，留给唯一的 `DTE_SEND/RECV` endpoint 路径。P4 没有为 remote direction 建立第二套外部表达。
- P0 D-22 已冻结：P4 `GROUP_SYNC` 只做 group barrier，不 program/erase multicast/reduce tree；按批 tree 生命周期仍属于 P7，不能把 P4 的同步实现记作树容量问题已经完成。
- P0 D-24 已冻结为 one-shot `SRAM_BIND`：中间 MEM/SYNC 不消费 binding，下一条 COMPUTE 消费；P4 只补 region rename/free/resize/targeted-clear 与 pending binding 的联合语义，legacy `Set_addr` 仍是 persistent。
- P0 D-27 已冻结：`GROUP_SYNC` 是 public group barrier，collective phase barrier 保持 internal，二者即使复用底层计数逻辑也使用互不碰撞的 key/epoch 和独立 trace 语义。
- P0 D-28 已冻结 `N=group_size`。P4 的静态 group registry 为后续 P6 提供稳定成员表和 group size；本阶段没有声称完成 collective 的本地 1 份、远端 `N-1` 份贡献 lowering 或计数 oracle。
- P0 D-25 的 `load_expert`、`switch_data`、`parse_input`、`parse_output` 继续保持 internal MEM；D-26 的 `Sram_pipeline_prim`、`Set_batch` 继续保持 internal。它们作为生产/回归资产使用，不占新的 public opcode。

## 按计划完成的开发

### LSU：只发布阻塞访问

- public `LSU_LOAD`/`LSU_STORE` 只 lowering 到既有 `LOAD_BLOCKING`/`STORE_BLOCKING`，沿用 `CoreLsuUnit` 的数据、地址、size 和时序路径。
- v1 artifact 不发布 LSU ISSUE/WAIT/POLL/FENCE/CANCEL；known-but-unsupported/非法组合由 codec、manifest 或 lowering 确定性拒绝，不能借内部 Prim 绕过。
- LSU 完成后才放行后续 compute/store；它与 DTE 的异步 token 状态、trace 和计费完全分离。

### SRAM lifecycle：RegionTable 薄封装与 targeted clear

- public `SRAM_ALLOC/FREE/RESIZE/RENAME/CLEAR` lowering 到单一 internal `Sram_lifecycle`，`PrimId=53 (SRAM_LIFECYCLE)`；内部 op variant 为 `ALLOC/FREE/RESIZE/RENAME/CLEAR_TARGETED`。
- strict Prim wire 固定四段且每段携带 ID 53，校验 segment 数、逐段 ID、op/lifetime、label ID、inactive fields、reserved/padding、截断、额外段和重复 decode；legacy Prim transport 明确拒绝。
- `ALLOC` 直接复用 `RegionTable` 的 allocator，支持 allocator-chosen address、请求 alignment、lifetime 和 spillable；以绝对地址验证 alignment，并处理 align-up 溢出、零大小、容量不足、重名和非 2 的幂 alignment。
- `RESIZE` 支持 grow/shrink，保留有效前缀数据；busy 或 grow capacity/bad-alloc 失败时 span、size、locator 和 allocation metadata 原子不变。缩小后旧尾部访问按新边界拒绝。
- `RENAME` 同步更新 locator 与 `RegionTable`，重名冲突原子拒绝，并同步改写尚未消费的 P3 one-shot input/output label。
- `FREE` 只释放 metadata，不伪造 SRAM 数据面周期，也不清底层字节；double free、missing、busy、lifetime 不匹配以及仍被 pending one-shot binding 引用均拒绝。
- `CLEAR_TARGETED` 先经 `AccessUnit` 清零 allocation 的逻辑字节，再释放 label/allocation；只接受 D-16 允许的 task-lifetime、spillable、非 busy region。相邻 sentinel 不被清除，和纯 `FREE` 的“字节保留”语义、时序与 trace 分开。
- legacy `Clear_sram` 的 clear-all 行为保持不变；P4 targeted clear 没有把旧 clear-all 隐式改成按 label 清除。
- 对 pending one-shot binding，`FREE/RESIZE/CLEAR_TARGETED` 均在改变状态前拒绝；`RENAME` 原子更新 pending label，因此中间 MEM 操作仍不消费 binding，而下一条 COMPUTE 能解析新名字。

### DTE：双 region lowering 与每核本地 token 状态

- `DTE_ISSUE` 继续使用每核本地 `DteAsyncTracker`，`ISSUE` 返回/占用 token，`WAIT(token)` 只等待指定 token，无参 `FENCE` drain 全部 token，`CANCEL` 按 queued/active/completed/unknown 状态给出确定结果。
- `DTE_POLL` 未形成可被后续 artifact 消费的正确性结果，因此没有发布；`FENCE_ALL` 没有作为第二个 public alias 重复占 opcode、trace 或计费。
- 本地 tracker 覆盖 token duplicate/unknown/early reuse、max outstanding、队列 backpressure、refill、RAW/WAR/WAW hazard、完成回收和最终 residual；错误路径不会留下半注册 token 或半提交 hazard。
- `SPM_TO_SPM` 增加独立 source/destination named-region 编码：source 使用 `sram_region+sram_offset`，destination 使用 `destination_sram_region+destination_sram_offset`；两侧分别经 `RegionTable` 解析，避免把一个 label 同时当源和目的。
- source/destination region wire 分别使用独立 marker 和 64-byte 名字段，保留绝对地址兼容路径；严格检查 direction、地址形态互斥、offset、padding/reserved、截断、重复 decode 和旧 JSON 缺省行为。
- `SPM_TO_DRAM`、`DRAM_TO_SPM` 仍走唯一 local DTE 数据路径；三个 remote direction 在 P4 loader/codec 边界拒绝，后续只由 `DTE_SEND/RECV` 实现，符合 D-19。
- ISSUE 按实际访存分类为 MEM，WAIT/FENCE/CANCEL 为 SYNC；正常尾部 tracker 清零，故意漏 WAIT/FENCE 的程序由 drain gate 确定性报 residual。

### 静态 core group、GROUP_SYNC 与 EVENT

- 新增不可变 `CoreGroupRegistry`：加载期校验 group ID、成员唯一性、成员存在性和同 die 约束；跨 die group、unknown group、非成员执行在进入正常屏障路径前拒绝。
- `GROUP_SYNC{group_id,sync_seq}` 通过独立 runtime 维护成员到达、per-member next sequence 和完成状态；支持 N=1/2/4、非连续成员和 1000 次连续同步。
- 最快成员等待最慢成员，最后一个成员到达后统一放行；duplicate arrival、sequence jump/rollback、非成员和 unknown group 确定性失败且不推进序号。
- public group barrier 使用保留的 group-sync namespace；internal collective barrier 使用另一 namespace，同 group 并发不碰撞、不重复计数。高频 `GROUP_SYNC` 不 program/erase collective tree，符合 D-22/D-27。
- `EVENT_SET/WAIT` 走专用 EVENT control message/runtime，不构造违反 DATA 契约的零长度包；验证 source/destination/tag/count、同 die endpoint、重复/未知完成、提前 SET、先 WAIT、多 credit 原子消费和有界队列。
- token、barrier、event 和 region 均纳入 trace/drain；定向 runtime 测试确认正常路径 residual 为零，异常路径状态可回收。

### Program fixture 与 artifact E2E

- `--p4-lifecycle` 生成单核 `SRAM_ALLOC -> SRAM_RESIZE -> SRAM_RENAME -> SRAM_CLEAR`，region-name/`SRAM_LABEL` symbol、envelope、START/terminal/ACK/DONE 完整且编码确定。
- `--p4-sync-event` 生成同 die 两核静态 group：core0 执行 `EVENT_SET -> GROUP_SYNC`，core1 执行 `EVENT_WAIT(count=1) -> GROUP_SYNC`，每核 start、terminal、ACK/DONE 闭合。
- `--p4-lifecycle-dangling` 以 dangling lifecycle symbol 构造加载期负例；`--p4-sync-bad-endpoint` 以 event executing endpoint mismatch 构造负例。unknown group 另由 codec/helper/runtime 定向负例覆盖。
- fixture 只使用 public record/program encoder，默认 P3 fixture 行为未改变；正向用例走真实 artifact decode/lowering/runtime，负向用例在预期阶段 fail-fast。

## 评审发现与修改闭环

### lifecycle 原子性和 P3 one-shot 交互

1. 初版按计划复用 `RegionTable`，随后评审补齐 requested alignment 的绝对地址断言、align-up 溢出、busy resize 原子性、grow capacity/bad-alloc 原子性、rename 冲突原子性和 FREE 字节 sentinel。
2. 评审发现对象 `prim_context` 与 runtime 传入 `context` 可能分裂，lifecycle 执行统一使用实际传入 context，避免 locator/pending binding 属于不同 core context。
3. P3 one-shot 联合评审后，rename 改为同步更新 pending labels；free/resize/targeted clear 在任何 metadata/字节变化前拒绝被 pending binding 引用的 region。
4. manager/locator/RegionTable 路径补了 allocation 后 find/read/free 一致性和 alloc-id 生命周期检查，避免 metadata 能找到但 manager 释放错误对象。

### legacy duplicate-label playground 回归

- 回归暴露的根因是：legacy `parse_input` 会把新一轮输入重命名为当前 `INPUT_LABEL`，历史行为允许覆盖旧目标；P4 为 public `SRAM_RENAME` 新增的严格 duplicate-label 拒绝被公共 `changePairName` 默认路径继承后，迭代 playground 在第二轮 rename 处失败。
- 修复将 `SramPosLocator::changePairName` 拆成两种明确语义：默认 `replace_existing=false`，供 public lifecycle 使用并保持 duplicate rename 原子拒绝；只有 legacy `parse_input` 显式传 `replace_existing=true`，保留旧 overwrite 行为。
- replacement 路径协调 locator 与 `RegionTable`：先把旧目标 allocation 临时隔离，再 rename source、释放被替换 allocation；异常时恢复原 label/metadata，避免 allocation 泄漏、双 label 或 locator/RegionTable 分裂。
- `deletePair` 同步释放对应 `RegionTable` allocation；SRAM R6 增加 strict conflict 原子性、legacy replacement 和无 allocation leak 检查。
- 该根因完成代码修复与定向评审后，latest-source 历史构建已获得 full CTest 63/63；后续工作树仍有独立源码变化，因此本文只把 63/63 记为历史门，不虚报为最新工作树的最终 clean Release/Debug 证据。

## 测试设计与已发生的定向证据

下表只记录已经实际发生的阶段性证据。它们证明对应能力在当时构建点通过；duplicate-label 修复后的 latest-source 历史 full CTest 为 63/63，但不等价于后续源码变化后的最终 clean Release/Debug 已完成。

| 验证项 | 已发生结果 | 覆盖内容 |
|---|---:|---|
| P3/P4 program artifact 定向集 | 22/22 PASS | P4 lifecycle、sync/event 正向链，dangling symbol/bad endpoint 等 WILL_FAIL，兼顾 P3 fixture 不回退 |
| SRAM R0～R6 | 全部 PASS，`failures=0` | allocator/manager、地址与字节、pipeline、lifetime、busy/capacity/alignment、targeted clear、one-shot 交互和 duplicate-label 修复定向项 |
| DTE V3 selftest | 31/31 PASS | async ISSUE/WAIT/FENCE/CANCEL、overlap、token、hazard、队列/refill |
| DTE V3b selftest | 21/21 PASS | arbitration/aggregation/backpressure、trace 与完成回收 |
| DTE V4 selftest | 23/23 PASS | 资源/公式、local direction、双 named region、wire/JSON 兼容和 residual |
| 官方 DTE runner | PASS | `run_test_dte_v3.py`、`run_test_dte_v3b.py`、`run_test_dte_v4.py` 的 WorkerCore/trace/负例路径 |
| 官方 SRAM runner | PASS | `run_test_sram_pipeline.py` 的 LSU blocking、DTE/compute overlap、数据 checksum、stage trace 与 region lifecycle |

定向正负例还覆盖：

- LSU 的 load-before-compute、compute-before-store、HBM/SRAM 非零字节和 async LSU 加载期拒绝；
- DTE 指定 token wait、无参 fence、cancel 状态矩阵、double/unknown/early reuse、队列满、RAW/WAR/WAW、漏 drain 和三个 remote `DTE_ISSUE` direction 拒绝；
- SRAM allocate/read/write/grow/shrink/rename/free/clear，零大小、重名、missing、double free、busy、capacity、alignment、越界、缩小后旧尾部、nonspillable/lifetime 和数据 sentinel；
- group 的 N=1/2/4、非连续成员、1000 次 sync、fast/slow member、sequence jump/rollback、duplicate/nonmember/unknown/cross-die，以及 tree program/erase 计数不变；
- event 的 set-before-wait、wait-before-set、一对一、多 credit、重复 tag、count mismatch、endpoint mismatch、队列容量和 EVENT/DATA 路径隔离；
- 正常程序 token/region/barrier/event residual 为零，故意漏 wait/fence 或成员不到达由 drain/watchdog 确定性捕获。
- duplicate-label 修复后的 latest-source 历史 unified 分项包含 program helper 183、published NPU ops 41、NPU cost model 22，full CTest 为 63/63 PASS；最新工作树 clean Release/Debug 仍待复核。

## 对照开发计划 10.1～10.4 的完成度

- 10.1 的能力边界已落地：artifact 只发布 blocking LSU；local DTE 使用每核 async token；region、group 和 event 为后续阶段提供可复用的稳定基础。
- 10.2 的 P4-owned 实现已完成：lifecycle、D-19 local DTE 双 region、WAIT/FENCE/CANCEL、静态 group、public GROUP_SYNC、EVENT 和 residual 均已接入；POLL、LSU async、remote DTE_ISSUE、数据准备 helper 和 SRAM pipeline 按冻结结论保持不发布/internal。
- 10.3 的阶段内 unit、wire、runtime、program fixture、SRAM R0～R6、DTE V3/V3b/V4 与官方 runner 已有上述定向证据；duplicate-label 修复后的 latest-source 历史 full CTest 63/63 已记录，最终 clean 门仍以最新工作树为准。
- 10.4 的阻塞边界、token 生命周期、group namespace、region metadata/数据面分离、错误原子性和 drain 已完成代码评审与修改；最终退出仍受下一次统一重建和三项回归约束。
- D-22 的 batch tree program/erase、D-28 的 collective N 份贡献以及 remote P2P 真实字节通路分别属于 P7、P6、P5，本文明确不将其计入 P4 完成度。

## 阶段退出条件

- [x] v1 artifact 不能发起 LSU async；`LSU_LOAD/STORE` 只执行 blocking 路径。
- [x] public `DTE_ISSUE` 只包含 D-19 三个 local direction；remote direction 没有第二套编码。
- [x] DTE WAIT、无参 FENCE、CANCEL、容量/backpressure/hazard/trace/drain 具有定向正负例；POLL 和重复 fence alias 未发布。
- [x] SRAM lifecycle 复用 `RegionTable`，metadata 与数据面计费分离，FREE/targeted clear 字节语义明确且失败原子。
- [x] P3 one-shot binding 与 rename/free/resize/clear 的联合语义已实现并有定向覆盖。
- [x] 静态 group、GROUP_SYNC 和 EVENT 的成员、序号、控制消息、namespace、tree 隔离和 residual 已有定向覆盖。
- [x] P4 program fixture 正负链已纳入 22/22 阶段性定向 program 证据。
- [x] SRAM R0～R6、DTE V3/V3b/V4 selftest 和相应官方 runner 已在阶段开发中通过。
- [x] legacy duplicate-label 根因已定位，strict public rename 与 legacy replacement 语义已分离并完成代码修复。
- [x] duplicate-label 修复后的 latest-source 历史构建与 full CTest 63/63 已通过。
- [x] fresh Release/Debug unified ISA 与 full CTest 均为 63/63，确认后续源码变化无回退。
- [x] 同一 fresh Release/Debug 候选中的 `playground_legacy_baseline` 均通过，START_DATA 4/4、数据、trace 和 residual 闭合。

是否允许继续后续集成：允许；P4 最终验收已关闭。

结论：P4 完成；LSU/DTE、region、sync/event 与 P0 契约一致，fresh Release/Debug full CTest 63/63 及 playground 均通过。
