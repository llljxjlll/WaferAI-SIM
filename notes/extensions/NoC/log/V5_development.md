# NoC 集合通信 V5 开发日志

日期：2026-07-26
状态：已完成（协议、Router DCA datapath、Tier2 展开与端到端验收完成）

## 本阶段交付

- 新增独立两段 256-bit reduce operand wire，冻结 tree ID、CollectiveKey、phase/chunk、child、dtype、reduce op、有效元素数和 128-bit payload；
- 序列化拒绝 tree ID 0、NONE op、FP32 和 payload 容量溢出；反序列化校验 magic/version、段数、枚举范围和所有保留位；
- `CollReduceMatchKey` 固定为 `(CollectiveKey, phase_id, chunk_id)`；
- 新增有限 `CollOperandMatchBuffer`，分别限制 header 和 operand 容量，按 expected-child bitmap 检测到齐、重复、非预期 child、元数据不一致及满载背压；
- `Open` 在分配状态前拒绝 FP32、NONE、零元素和超过 128-bit 的 operand，避免产生永远不可消费的表项；
- 实现 UINT8/INT32/INT64 的 bit-accurate SUM/MAX：SUM 按 dtype 位宽模溢出，INT32/INT64 MAX 使用有符号比较；
- DCA 在所有 child 到齐后才能消费，服务时间为 `max(count × (child_count - 1), ceil(payload_bits / 128)) + 54`；消费返回聚合 payload，释放 operand/header 状态，表达结果重注入的契约边界。

## 验收

- build：PASS。
- `--coll-v5-selftest`：20/20。
- 覆盖 wire 往返、FP/保留位/容量非法输入、UINT8 模溢出、signed INT32 MAX、partial/ready、重复/缺失/非预期/mismatch、提前消费、有限背压、三 child 计算量、结果 payload/周期及最终 drain。
- 相邻回归：V4 contract 13/13，V3 self-test 14/14。

## 生产接线完成（2026-07-26）

- 配置阶段从 multicast tree 生成反向 reduce node：每个 Router 得到 expected ingress bitmap 和 parent output；group core（包括 root）均包含 CENTER operand。
- WorkerCore 按 128-bit payload 分 chunk 注入两段 operand wire；严格 framing 同时检查 magic/version/tree/segment-count/reserved，避免普通 `Msg` 偶然碰撞。
- Router 组装 header/payload 后进入有限 Match Buffer（16 header、64 operand），重复、非预期 child、元数据不一致和容量满均不会破坏状态。
- header 与 operand 容量满均使用可重试 backpressure：header 满时保留 per-input pending header 和 input payload，不抛异常、不修改 match 状态；已有 match 消费释放容量后自动重试。
- 每 Router 使用一个公平、串行占用的 DCA 服务端；服务时间为 `max(comp,p/128)+54`，已匹配结果队列上限 16，满时向真实 input buffer 传播 backpressure。
- 聚合结果按反向树逐级重注入；root RX 对确定性整数 operand 做最终 bit-accurate SUM/MAX 验证。
- Tier2 planner 对 Reduce/ReduceScatter/AllReduce 不再生成 Tier0 root compute：Reduce 直接交付 root；ReduceScatter 聚合后差异化单播 scatter；AllReduce 聚合后复用 Tier1 单份 broadcast。
- collective primitive 禁止被通用 `prim_refill` 隐式重排；连续 epoch 只能由顶层显式声明，避免非 root 在 terminal root 完成后开启幽灵第二轮。
- `noc.collective.tier=2` 已放开；跨 die 和 FP32 仍启动期明确拒绝。

## 最终验收

- V5 contract：20/20（新增 header-full 无状态改变、stall、已有 match drain 和重试成功覆盖）。
- 三核 Tier2 Reduce/ReduceScatter/AllReduce：每项 3 个 TX operand、1 个 root verified result、最终 `router_residual=0`、credit balanced。
- 数值模式额外通过 INT32/MAX、INT64/SUM；UINT8/SUM 在三类归约中覆盖。
- V4/V5 production runner：9/9；V0–V3 全绿。
- NoC frozen：14781/29109、14833/45441；D2D shared Router self-test：308/308。

## 保留边界

- FP32 的舍入、NaN、溢出语义未冻结，继续拒绝。
- hierarchical cross-die collective 尚未设计，Tier1/Tier2 group 必须位于同一 die。
- `COLL_DATA` 是 timing/traffic header wire，不搬运普通 SRAM tensor；bit-accurate value 验证位于 V5 operand/DCA 路径。

## 关键文件

- `llm/include/dte/coll_innetwork_reduce.h`
- `llm/src/dte/coll_innetwork_reduce.cpp`
- `llm/src/dte/coll_v5_selftest.cpp`
- `llm/src/router/router.cpp`
- `llm/src/workercore/workercore.cpp`
- `llm/src/monitor/config_helper_core.cpp`
- `llm/test/noc_collective/run_test_coll_v4_v5.py`
