# P0 开发记录：冻结 v1 契约与回归基线

阶段：P0——冻结 v1 契约与回归基线
阶段状态：已完成
完成日期：2026-08-11（UTC）
提交/变更集：工作区实现，尚未提交

## 计划确认

- 阶段范围按开发计划第 6 节执行：关闭 D-01～D-32，冻结外部 Opcode、内部 PrimId、Program Format、lowering、控制 envelope、兼容边界并建立修改前基线。
- P0 只冻结 v1 合同与可行性结论；production codec、loader 和 runtime 压力测试分别由 P1、P2 及后续阶段完成。
- legacy JSON 必须保持可用；`--program` 使用独立的外部 record 层，禁止把现有 128-bit Prim wire 或进程内标签 ID 当作编译产物 ABI。

## 完成交付

- `notes/instr/编译产物指令集P0契约.md`：冻结 D-01～D-32，包含位宽、单位、错误语义、同步/完成协议和能力 gate。
- `notes/instr/isa_v1_manifest.md`：冻结 51 个内部 PrimId 与 44 个外部 Opcode 的 visibility、lifecycle、support、category、lowering 状态。
- 关键冻结结论：
  - 外部 ISA record 与内部 Prim wire 分层；artifact 使用 `NPUPRG1\0` 小端 section 容器、字符串/符号/语义重定位、core group、每核索引和控制 envelope。
  - `DTE_SEND/RECV` 使用新薄原语；legacy Send/Recv 字段不足，不能假定只改 loader。
  - `EVENT_SET/WAIT` 使用专用控制语义，禁止零长度 DATA 伪装。
  - public `SRAM_CLEAR` 为 targeted；legacy 无参 clear-all 保持 internal。
  - public `SRAM_BIND` 为 one-shot；legacy `Set_addr` 保持 persistent。
  - v1 不发布 POLL、GLOBAL、GPU、实验 stub；只发布无参 DTE_FENCE；MLA/PD 默认 capability gate 关闭。
  - local `DTE_ISSUE` 只支持三个本地方向；remote 统一由 DTE_SEND/RECV 表达。
  - 普通 Reduce 为单 root；ReduceScatter/AllReduce 才使用对称分解；整数 SUM 按模截断，MAX 按有符号小端解释。
  - 同 die路由冻结为 X-first XY；树分批必须具备真实 program/erase 生命周期或静态拒绝。
  - program 明确携带 active/source/terminal/expected ACK/expected DONE/empty-core 策略，不能从尾指令猜测。
- 仓库审计覆盖 51 个 `REGISTER_PRIM`、PrimFactory、NPU/GPU/LSU/DTE、SetAddr/Send/Recv、Monitor/MemInterface、路由树、collective 和 PD 路径。

## 基线测试

- 隔离工作目录串行执行既有 28 个 CLI selftest 入口，共 883 项检查，全部通过：

  | 测试组 | 入口 | 检查 | 结果 |
  |---|---:|---:|---|
  | D2D V0/link | 2 | 345 | 通过 |
  | DTE V0/V3/V3b/V4 | 4 | 135 | 通过 |
  | workload rendezvous | 1 | 6 | 通过 |
  | Collective V0～V6 | 7 | 116 | 通过 |
  | Collective R0～R8 | 9 | 178 | 通过 |
  | HBM R0～R4 | 5 | 103 | 通过 |
  | 合计 | 28 | 883 | 通过 |

- SRAM R0～R6 七个入口全部通过；这些 runner 只报告 failures=0，不输出内部检查总数。
- 全新 Release configure/build 成功，CTest 17/17 通过。
- 全新 Debug configure/build 成功，16/17 CTest 通过；唯一失败 `hbm_r3_selftest` 稳定复现为 DRAMSys Debug checker 的既有 HBM2 参数断言 `ranksPerChannel==2 -> burstLength==4`，与 ISA/Prim 变更隔离。ISA 及其余 16 项均通过。
- 隔离门命令全部在 `/tmp` 快照运行，其 source 根无 `events.json`/`Cchip_1.vcd`，Release/Debug 输出仅写各自隔离 build。并行期间工作区 `build/events.json`、`build/Cchip_1.vcd` 被根 agent 的工作区 CTest 更新；路径和哈希证明不是隔离门串写，故不把工作区 mtime 不变作为本轮证据。
- HBM selftest 的 `../DRAMSys/configs` 工作目录依赖已记录，隔离 runner 使用与 CTest 相同的目录布局后 5/5 通过。

## 评审、修改与复测

- 设计评审关闭了以下阻断：裸 128-bit artifact 与 loader lowering 的层次冲突；标签 ID 重定位；GlobalMemInterface 注入 assert；EVENT 零长度不可行；POLL 无可消费结果；FENCE 重复语义；PD 缺 program context；remote DTE 字段不足；collective Reduce/发布顺序/数据归约缺口。
- 评审后把计划依赖修正为 P8 同时依赖 P3 与 P7，并明确 P3/P4 的 SRAM_BIND/region ownership。
- P0 spike 只作为可行性证据，不再循环依赖 P2/P4 production 实现。
- 契约和 manifest 完成后重新审阅全部编号、区段、能力状态与分类，没有遗留需要用户选择的冲突。

## 阶段退出条件

- [x] 计划确认完成
- [x] D-01～D-32 全部关闭
- [x] 外部/内部格式与 manifest 冻结
- [x] 回归基线通过并隔离输出
- [x] Release/Debug clean build 证据完成
- [x] 设计评审问题关闭
- [x] 必须修改项完成复核
- [x] 后续阶段依赖和 ownership 明确

是否允许进入下一阶段：是。
结论：P0 已完成，P1 及以后必须以冻结契约和 manifest 为唯一基线；若改变已冻结外部 ABI，必须升级 format/ISA version 并重新评审。
