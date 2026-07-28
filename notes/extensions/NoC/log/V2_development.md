# NoC 集合通信 V2 开发日志

日期：2026-07-26

## 交付

V2 在 V1 Tier0 数据搬运上加入有限 Gather RX reorder 模型：

- `GatherReorderBuffer` 按 `PacketKey` 匹配预期 slot，支持跨源/跨 chunk 的乱序驻留；
- `gather_reorder_depth` 为物理容量，`0` 保持 V1 ideal/unbounded 路径；
- 单 commit 端口每周期最多按目标顺序提交一个 slot；
- 满容量返回显式 backpressure，统计 stall cycle；
- 为当前队头预留一个准入槽，避免后续乱序包占满容量后阻塞唯一可推进项；
- 重复 resident/committed packet、未知 PacketKey、非法 offset 均明确拒绝；
- trace 输出 accept/stall/commit/drain、occupancy、peak、stall、received/committed/missing bitmap；
- Gather、AllGather、AllToAll 的每个远端 flow 完成后进入 reorder marker，最终状态 drain 并擦除；
- 非 Gather RX 操作配置 `gather_reorder_depth` 会在启动期失败。

## 关键文件

- `llm/include/dte/coll_reorder.h`
- `llm/src/dte/coll_reorder.cpp`
- `llm/include/dte/coll_runtime.h`
- `llm/src/dte/coll_runtime.cpp`
- `llm/src/dte/coll_v2_selftest.cpp`
- `llm/src/prims/norm_prims/collective_prim.cpp`
- `llm/src/monitor/config_helper_core.cpp`
- `llm/test/noc_collective/run_test_coll_v2.py`

## 验收

- build：PASS。
- V2 finite reorder self-test：14/14。
- V2 runner：8/8。
- Gather/AllGather/AllToAll × cycle/behavioral：全部通过。
- 每个场景核对 flow、reorder accept、最终 drain、router residual 和 data/control credit。
- 当前端到端矩阵观测到所有 flow 均为 accept 后立即 commit，最终 `stalls=0`；这符合 V1 source-rank phase 串行契约，不把零 stall 误写成端到端 backpressure 覆盖。
- 非 Gather RX 配置有限 depth 的负例：PASS。
- V1 runner：24/24，冻结周期值不变。
- V0：32/32；DTE V0：64/64；DTE V3b：21/21；DTE V4：19/19。
- NoC congestion：4/4，冻结值 14781/29109、14833/45441。
- D2D V0 runner：67/67 test groups（含 pure self-test 308/308）。

## 覆盖边界

- V1 的确定性 source phase 每次只允许一个源搬运，并在 phase barrier 后才进入下一源，因此端到端 Gather flow 按序、逐个到达；运行路径中的 `FULL`/stall 分支在当前 V1 planner 下不可达。
- 真实乱序到达、buffer 满、stall cycle、缺失队头后的无死锁推进、depth=1、重复包、未知 PacketKey 和非法 offset，均由 C++ self-test 主动注入覆盖；runner 只覆盖 marker wire 往返及真实运行路径的 Accept→Commit→Drain 常见分支。
- 因此 V2 完成的是“有限 reorder 模型及其有序运行路径接线”，不宣称 backpressure 已影响现有端到端时序；需要后续并行/多 chunk source 展开才能在真实数据路径触发该分支。
- 当前每条 V1 远端 flow 对应一个逻辑 chunk；状态机契约支持 `chunk_id`，多 chunk 网络展开留给后续算法流水化版本。
- Reduce RX 不在 V2 范围，归约族在 V3 实现。
