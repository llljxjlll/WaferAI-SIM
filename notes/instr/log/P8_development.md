# P8 开发记录

> 阶段状态：**完成**
>
> 范围：P8 端到端集成、双缓冲代表程序、混合代表程序、最终回归、sanitizer、性能与发布材料。

## 1. 交付内容

### P8-A：双缓冲程序

单一 Program Format artifact 覆盖：

`LSU_LOAD(blocking) → DTE 81 → BIND/MATMUL A → WAIT 81 → DTE 82 → BIND/MATMUL B → WAIT 82 → LSU_STORE → FENCE`。

硬件布局避开 MATMUL 内部静态 LSU block；probe 在仿真前事务初始化多 HBM/SRAM range，在仿真后逐字节验证两批覆盖、HBM output 和所有范围外 sentinel。runner 从 trace 证明 public LSU blocking 以及两次 DTE/compute overlap，并核对 token/stats/drain。

### P8-B：复合程序

单 die 四个非连续 core 的一个 artifact 同时包含：

- compute 与真实 P2P；
- AllGather 与 INT32 SUM AllReduce；
- 每核 ALLOC/BIND、前 compute、三次阶段 GROUP_SYNC；
- RENAME 8→9、按新 label rebind、后 compute、FREE 9；
- 末尾第四次 GROUP_SYNC 与 final FENCE。

四档 profile 使用同一 artifact：baseline、broadcast_only、reduce_only、reduce_broadcast。runner 用 profile-aware sidecar 检查 P2P/collective source、destination、mixed DCA staging、checksum、sentinel、backend marker、trace、lifecycle 与全部 drain；`--require-accelerated` 禁止静默回退。

### 集成与门禁

- CMake 注册 P7 32 KiB、P8-A、P8-B、mutation 与 sanitizer。
- CI 使用 clean Release/Debug、完整 compatibility、repeat20 与 pthread-SystemC runtime sanitizer，并始终上传 evidence。
- 发布文档已完成：lowering、迁移、Known Limits、最终测试、性能和发布评审。

## 2. 集成期间关闭的问题

1. Program Format 对 DTE_ISSUE inactive address 误校验：按 direction 只验证 active SRAM 端和 relocation。
2. P8-A MATMUL 静态 LSU 污染 probe region：调整 SRAM layout 与 shape，保持 byte/sentinel oracle。
3. P8-B P2P seq3 丢失：定位到 Worker 单槽 event 竞态；全部 normal/DCA/multicast wire 迁入有界 value FIFO 和 ticket completion。
4. Router normal endpoint DATA tag-only lock：改为完整 `(source,destination,tag,subflow)` 锁直到 tail。
5. Router CTRL 跨 hop 电平重复/丢包：所有方向增加显式 low-cycle pulse gate并计 residual。
6. P7 plan 过早 retire 与 late observer：complete-but-queryable + 每 Worker exactly-once observer latch。
7. DCA root 提前 RX_WAIT：增加全 source ready 门，禁止清 residual 旁路。
8. DCA staging source 的 expected-after：schema 显式提供变换后 bytes/checksum，仍严格验证 mixed 镜像。
9. distributed HBM 路径构造 legacy KV table 的单位/下溢问题：分布式路径不创建 legacy table，误用 fail-fast。
10. HBM2 小请求产生非法 BL：logical byte request 拆为合法 32B physical burst，write 使用 byte-enable，read 聚合回拷。
11. HBM Submit 部分提交：局部完整构造后 splice commit，trace/active/stats 有 rollback。
12. sanitizer 发现 SystemC subref UB、KVTable 泄漏与 MemEndpoint callback 环：逐 bit codec、显式释放和断环。
13. N=3 AllGather endpoint residual：reserved collective tag 禁止普通 prim refill，ACK 传播原 tag。
14. legacy playground worklist 断边与旧 timing goldens：修配置/实现后，只在结构性谓词仍严格成立时更新 exact ns。

## 3. 最终候选证据

| Gate | 实际结果 | 状态 |
|---|---|---|
| fresh Release configure/build | exit 0 | PASS |
| fresh Release full CTest | 63/63，39.34 s | PASS |
| fresh Debug configure/build | exit 0 | PASS |
| fresh Debug full CTest | 63/63，98.55 s | PASS |
| unified ISA | manifest 975、codec 917、format 656、helper 183、profile 341、FIFO 280 等全部 PASS | PASS |
| P8-A | Release/Debug runtime + sanitizer | PASS |
| P8-B repeat3 | 四档 12 isolated processes | PASS |
| P8-B repeat20 | 四档 80 isolated processes | PASS |
| P7 32 KiB | 8/8；sanitizer 8/8 | PASS |
| P6 | 37/37；sanitizer 37/37 | PASS |
| mutation | 58/58 | PASS |
| compatibility | 主矩阵 40/40 + 补充 5/5 | PASS |
| sanitizer | format/P8-A/P6/P7/P8-B；ASan=0、UBSan=0、LSan=0 | PASS |
| whitespace/docs/CI | diff-check、links、YAML/Bash static audit | PASS |

Release CTest 日志 SHA `6ebc5968273c5f4ac82c001d6135abb7c68f3047dd6a97d2c3047ccc446374fd`；Debug 为 `a0603568538ca83acb93d258e2d272d832081dab99dbb9d634c20767ed85dc3a`。

P8-B 稳定 digest：

- baseline `c20a0d033dcf037c51e1f117f8ad2314d993fff7fe820c35d5ecd34803661843`
- broadcast_only `e04b620a13a6c69ed33ab585842cc86e447677f604d141679dd038ca13f4c39f`
- reduce_only `fdadebe1de80785066a8a04e0287ee6ea98dd6ad3451196916c6151b465eb295`
- reduce_broadcast `f067f45becf35be858ded5fc1e919fd87c55f6200b0fe6a4703e71cd1bbf64e3`

compatibility `RESULTS.tsv` SHA `1ff8dc2c108fca29e93fdb6801f168fc1fdd2dc2f9d1d361fc9840e9e4d81f95`，40 个日志的 `sha256sum -c` 全部 OK。

## 4. 交叉评审

- ISA/主线：Codex `/root`，GO。
- P7/NoC：Bacon `/root/p7_production_integration`，GO。
- runtime/compatibility：Parfit `/root/p7_32k_runtime_gate_impl`，40/40，GO。
- final safety/CI：Avicenna `/root/final_gate_matrix`，GO。
- P5/P6：阶段开发和多轮独立只读复审均无剩余 BLOCKER/HIGH。

以上为自动化协作评审，不冒充外部人工签名。

## 5. 退出条件

- [x] P0～P8 均有实现、测试和评审记录。
- [x] manifest、artifact、golden、lowering、迁移、Known Limits 与示例齐全。
- [x] 已发布 Opcode 可执行；未发布能力 fail-fast。
- [x] P8-A/P8-B byte、sentinel、trace、stats、lifecycle 和 drain 全部通过。
- [x] 四档 production backend 正交生效，无 silent fallback。
- [x] 32 KiB、多 tree、批级 program/erase、DCA/multicast 有 production 证据。
- [x] fresh Release/Debug full CTest 与全部 compatibility runner 全绿。
- [x] ASan/UBSan/LSan representative runtime 与四档 repeat20 全绿。
- [x] 五个技术面完成交叉审查。
- [x] [最终测试报告](../ISA_v1最终测试报告.md)、[性能报告](../ISA_v1性能报告.md)、[发布评审](../ISA_v1发布评审记录.md) 已写入真实证据。

## 6. 阶段结论

P8 完成，发布工程判定 **GO**。ReduceScatter+DCA 和跨 die collective 仍保持明确 unsupported/fail-fast；更广性能曲线属于后续扩展，不回退本阶段正确性结论。
