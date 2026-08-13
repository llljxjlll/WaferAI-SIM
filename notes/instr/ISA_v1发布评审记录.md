# NPU ISA v1 发布评审记录

> 评审结论：**GO**
>
> 本记录是自动化协作评审签署。候选为基线 commit 加当前 P0～P8 工作树；在提交/打 tag 时应把工作树内容固定为新的不可变 commit，并保留本记录中的日志 SHA。

## 1. 候选与材料

| 项 | 值 / 结论 |
|---|---|
| 基线 commit | `139ef73a803dc3637b28000bbc3a34016b2d5e45` |
| 候选时间 | `2026-08-12T13:21:02Z` |
| 工作树 | P0～P8 实现与文档尚未提交；implementation manifest `6a209f12c1c8d44b00a3ff8ef73aba2b6062f45e937a5b7451145120f4cdd49b`，status `db4f66fee4ba5dd3bcedd68b93b21b4e1cb2796b82f8a8772c674770676eef7c` |
| manifest SHA | `56a885b64af136c061a16156d773b70f0e36e9b68bb1ca357a8397107d9b37f8` |
| Program Format SHA | `db631bed4b223e8a6b581ed5bde4e74b80d8aceada641aaf6fc0732a37548749` |
| P0 contract SHA | `8d558f19c60475a1c35aaa40bdab615f5273acb38da123579728d047ac992956` |
| ABI/Golden policy SHA | `9084fb161583bc168f11278337f2c305a590045964aed65ed7afa6c7a2299a99` |
| lowering guide SHA | `a70ca7ee0a019ebbd46e72b706ce9f51546d0980f9b44a1c9bb4d89f98518b67` |
| migration guide SHA | `6c26041843fa394bb727bde6692079a0cc16db52b94c0bb212b4d6df74b15b29` |
| known limits SHA | `dd9da0530acdb07be97b7513d505dfc92160988dc3b6f649d04c543b5861c22a` |

材料审阅范围：manifest、Program Format、P0 contract、ABI/golden policy、compiler lowering、JSON→Program migration、Known Limits、P0～P8 logs、[最终测试报告](ISA_v1最终测试报告.md)、[性能报告](ISA_v1性能报告.md)。

## 2. 角色与签署

| 角色 | 自动化评审者 | 评审范围 | 结论 |
|---|---|---|---|
| 主线/ISA 接口 | Codex `/root` | P0～P8 集成、ABI、fresh gates、报告 | GO |
| NoC collective | Bacon `/root/p7_production_integration` | profile/topology/tree/DCA/multicast/Router/Worker | GO |
| 测试与兼容 | Parfit `/root/p7_32k_runtime_gate_impl` | 32 KiB、40+5 compatibility、oracle 审计 | GO |
| 独立发布审计 | Avicenna `/root/final_gate_matrix` | Router/HBM/CI/sanitizer/文档门 | GO |
| P5/P6 交叉审查 | 多阶段独立 agent 记录 | P2P、collective image/runtime、三轮缺陷复审 | GO |

这里的“签署”表示自动化工程评审结论，不冒充外部人工审批或组织授权。

## 3. ABI 与资源决议

- Opcode/Prim ID append-only；manifest/factory/wire/record codec unified selftest 全绿。
- Program Format 对 magic/version/endian/section/CRC/relocation、数量、长度和地址均有界；mutation 58/58。
- external REDUCE_COMPUTE 只由 whole-artifact lowering 到真实 data prim；legacy timing-only prim 不可替代。
- P2P payload 上限 1,048,560 bytes，transport tag 单调不复用；seen tombstone 有 topology×tag-space 硬上界。
- collective planner 在 reserve/resize 前检查 children/actions/waves/derived bytes；tree table 每 Router 64 entries，K 范围 1～64。
- SRAM region span、nonzero symbol/addend、HBM physical burst、byte-enable、logical stats 与异常原子性均闭合。
- 所有资源耗尽或 capability 不匹配均 fail-fast，不 silent fallback。

## 4. Capability 决议

| profile | 状态 | backend | 证据 |
|---|---|---|---|
| baseline | available | unicast + endpoint | P6 37/37、P8-B |
| broadcast_only | available | multicast + endpoint | P7 32 KiB、P8-B repeat3/20 |
| reduce_only | available | unicast + DCA | P7 32 KiB、P8-B repeat3/20 |
| reduce_broadcast | available | multicast + DCA | P7 32 KiB、P8-B repeat3/20 |

ReduceScatter+DCA 保持 **unsupported**，加载期精确拒绝；跨 die collective 也按当前已知限制拒绝。available 是正确性与生命周期承诺，不是相对 baseline 的性能 SLA。

## 5. 最终门禁

| Gate | 实际结果 | 证据 | 结论 |
|---|---|---|---|
| fresh Release | build 0；CTest 63/63 | log SHA `6ebc5968…374fd` | PASS |
| fresh Debug | build 0；CTest 63/63 | log SHA `a0603568…dc3a` | PASS |
| unified ISA | 全 suite PASS | log SHA `669b6c97…cc15` | PASS |
| Program mutation | 58/58 | Release CTest | PASS |
| P8-A | runner/runtime + sanitizer | Release/Debug/ASan logs | PASS |
| P6 | runtime 37/37 + sanitizer 37/37 | `0cd4b58e…00cd4` | PASS |
| P7 32 KiB | 8/8 + sanitizer 8/8 | `8baa688d…4556c` | PASS |
| P8-B repeat3 | 12/12 + sanitizer 12/12 | `e11c6411…e9df` | PASS |
| P8-B repeat20 | 80/80 | `beadafe4…f96c` | PASS |
| compatibility | main 40/40 + extra 5/5 | signed RESULTS/SHA lists | PASS |
| ASan/UBSan/LSan | 0/0/0，leak 0 bytes | fresh pthread-SystemC build | PASS |
| whitespace/links/YAML | diff-check、local links、PyYAML/Bash static checks | final audit | PASS |

CI workflow 已加入 clean Release/Debug、repeat20、完整 R/V、pthread sanitizer P6/P7/P8 和 always-upload evidence；首次远端 CI 仍应观察 hosted runner 总时长，但这不改变本地候选正确性结论。

## 6. 缺陷关闭

| ID | 严重度 | 关闭内容 | 复验 |
|---|---|---|---|
| B-01 | BLOCKER | collective span/ABI 与 derived-state 预分配界限 | unified + P6 + mutation |
| B-02 | BLOCKER | Router CTRL pulse、P2P full-flow lock、serialized value FIFO | P5/D2D/R/V/full CTest |
| B-03 | BLOCKER | P7 tree observer、DCA sources-ready、真实 SRAM byte path | P7 8/8 + P8-B |
| B-04 | BLOCKER | HBM 32B physical burst、byte-enable、Submit 原子性 | HBM R0～R4 + sanitizer |
| B-05 | HIGH | SystemC wire UB 与 KVTable/MemEndpoint leak | pthread ASan/UBSan/LSan |
| B-06 | HIGH | N=3 AllGather reserved-tag refill residual | V1 15 checks + R6/R7 |
| B-07 | HIGH | stale timing/behavioral-DCA compatibility oracle | compatibility 40/40 |

所有缺陷均有源码修复与定向/full 复验；没有接受开放的 P0/P1。

## 7. 发布检查单

- [x] B-01～B-07 全部关闭。
- [x] 最终测试报告每个 required gate 均 PASS，无 NOT RUN/0 tests。
- [x] 性能报告包含原始值、派生值、解释和 SLA 边界。
- [x] P0～P8 日志、规范、lowering、迁移和已知限制一致。
- [x] 四档 capability 与 production runtime 一致，无 fallback。
- [x] fresh Release/Debug、compatibility、repeat20、sanitizer 全绿。
- [x] `git diff --check`、文档链接、CI YAML/Bash 静态审计通过。
- [x] 没有开放 P0/P1 缺陷。

## 8. 最终决议

**GO**。允许将当前工作树固定为 ISA v1 release candidate commit。提交前不得丢弃现有修改；提交后应把新 commit 和 source manifest SHA 补入发布制品元数据。后续扩大 ReduceScatter+DCA、跨 die collective 或更广 performance golden 时，必须走新的 ABI/capability 评审，不能回写本次 GO 的支持边界。
