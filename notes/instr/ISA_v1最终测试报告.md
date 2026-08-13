# NPU ISA v1 最终测试报告

> 文档状态：**FINAL / GO**
>
> 候选是 `HEAD` 加当前 P0～P8 工作树。工作树尚未提交，因此 commit 只标识基线；发布身份还必须结合本报告的源码状态和证据 SHA。未删除或覆盖用户原有未跟踪文件。

## 1. 候选身份

| 字段 | 值 |
|---|---|
| 基线 commit | `139ef73a803dc3637b28000bbc3a34016b2d5e45` |
| 候选时间 | `2026-08-12T13:21:02Z` |
| 主机 | Linux `6.6.87.2-microsoft-standard-WSL2`, x86_64 |
| C++ | `/usr/bin/g++`, GCC 11.4.0 |
| CMake | `/opt/cmake/bin/cmake`, 3.31.3 |
| Release | `build-release-candidate`，fresh configure/build |
| Debug | `build-debug-candidate`，fresh configure/build |
| sanitizer | `build-asan-candidate`，RelWithDebInfo，ASan+UBSan+LSan，pthread SystemC 2.3.3 |
| implementation-files manifest | `6a209f12c1c8d44b00a3ff8ef73aba2b6062f45e937a5b7451145120f4cdd49b` |
| candidate status | `db4f66fee4ba5dd3bcedd68b93b21b4e1cb2796b82f8a8772c674770676eef7c` |

配置 SHA-256：

- P6 hardware `577905d23b1abd84c9c68859f636a0a27ac45668d14d61035c4dd17b5c02d03f`
- simulation `32bce414f159454eff62612d46ffb2d9b85d75fad8e31973af94329bb7b5ce3c`
- mapping `99a357b646bc6d0d81ac188c8bfffcbf6ab8f8f72a5d262fe81624f6f9a9a66c`
- P8-B hardware `9993c0e5adc66740c97d498be6f5a71243e0e922720e87fcc1d828ada429ab3e`
- P6/P7/P8-B oracle：`ee0177c9570f9b59e45ce615a1bf95b68656d58c038c19bc7eab7a344ec9fe06` / `087e092ca0a6d9827f05c27a52b48c42ee008da8ba7f6db7565421429760dd43` / `92e2bcdf4a129dc8c3c1183d28c73f6133c0054360307d34db3ed1d56e4f21b4`

## 2. Clean build 与全量 CTest

| Gate | 结果 | 日志 SHA-256 | 状态 |
|---|---|---|---|
| Release configure | exit 0 | `5ff3126931235589d52d74b98428c0efc14ceda85826edf05bd41af6248604b5` | PASS |
| Release build | exit 0 | `eae5a13851db4537014ec8ad8422cc49e4caf273f9992ae1e3836c221ddf9ee2` | PASS |
| Release CTest | **63/63**，39.34 s | `6ebc5968273c5f4ac82c001d6135abb7c68f3047dd6a97d2c3047ccc446374fd` | PASS |
| Debug configure | exit 0 | `3e9d40f8372f55d561987dd9ac8101dce439bf2b20f89fed2ec0a67461f85b79` | PASS |
| Debug build | exit 0 | `065479b40fdaa3865576d2bc912343acd27072f31a35387b6c8415ac95a00a2e` | PASS |
| Debug CTest | **63/63**，98.55 s | `a0603568538ca83acb93d258e2d272d832081dab99dbb9d634c20767ed85dc3a` | PASS |
| Release registration | 63 tests；含 P7 32 KiB、P8-A、P8-B、mutation、playground | `12c3d5154b43a018019f7dfcc9015220e1ccc6e010852287a428ae7f581c87bc` | PASS |
| whitespace | `git diff --check` 无输出 | final audit | PASS |

`playground_legacy_baseline` 在两套 full CTest 内均通过；旧 worklist 断边和发送单槽问题已分别关闭，没有通过放宽 watchdog 掩盖。

## 3. ISA、格式、ABI 与负例

Release unified ISA selftest 全绿，日志 SHA-256 `669b6c9774cf4c9233b897a42fff0cb4005f50379a8151cf65e0304e6865cc15`。主要计数：

- manifest/factory 975；external record codec 917；Prim wire 585；Program Format 656；record lowering 299；helper 183；planner 135。
- P7 profile 341、acceleration runtime 40、topology 35、tree batch 38、serialized FIFO 280。
- Program image 85、executor 59、aggregate 55、phase 25、child endpoint 31。
- published ops 41；cost-specific 22；combined cost suite 63。

固定 seed mutation 58/58 在加载成功标记前拒绝；坏 magic/version/endian/section/CRC/relocation、截断、reserved、未知 Opcode/Prim、非法枚举和越界均 fail-fast。ABI 文档 SHA：manifest `56a885b64af136c061a16156d773b70f0e36e9b68bb1ca357a8397107d9b37f8`，Program Format `db631bed4b223e8a6b581ed5bde4e74b80d8aceada641aaf6fc0732a37548749`。

## 4. 真实数据与运行时门

### P8-A：LSU blocking + DTE double buffer

Release 与 Debug CTest 均通过。HBM→SRAM、两批覆盖、SRAM→HBM、全范围 sentinel、两个 DTE/compute overlap、blocking LSU 顺序、token issue/wait/fence exactly once 和所有 residual=0 均由 runner 检查。sanitizer 复跑同样通过。

### P5/P6

- P5 same/cross-die、SYNC/ASYNC、SRAM/HBM source、tail 与大 payload 均通过；payload/session focused 历史门为 201431/190 checks。
- P6 runtime **37/37**，覆盖 TX×RX 九宫格、N=1/2/4、1/17/1024/8192/32768 bytes、UINT8/INT32/INT64、SUM/MAX、rank-major Gather、single-root Reduce 与对称 collective。
- source read、wire、`kNocRx` write、checksum、sentinel、child/FSM/token/barrier/session/router drain 均严格核对。

### P7 四档与 32 KiB

固定 CTest `program_p7_accelerated_32k_runtime_matrix` 通过 **8/8**：AllGather/AllReduce × baseline/broadcast_only/reduce_only/reduce_broadcast，K=1，8 个隔离进程。每例验证真实 SRAM 字节、顺序或 INT32 SUM、tree begin/end、program=erase、occupancy_after=0、backend marker、无 fallback 与全 drain。

P8-B 四档 repeat3 为 12/12；repeat20 为 **80/80**：

| profile | backend | repeat3 | repeat20 | digest |
|---|---|---:|---:|---|
| baseline | unicast + endpoint | PASS | PASS | `c20a0d033dcf037c51e1f117f8ad2314d993fff7fe820c35d5ecd34803661843` |
| broadcast_only | multicast + endpoint | PASS | PASS | `e04b620a13a6c69ed33ab585842cc86e447677f604d141679dd038ca13f4c39f` |
| reduce_only | unicast + DCA | PASS | PASS | `fdadebe1de80785066a8a04e0287ee6ea98dd6ad3451196916c6151b465eb295` |
| reduce_broadcast | multicast + DCA | PASS | PASS | `f067f45becf35be858ded5fc1e919fd87c55f6200b0fe6a4703e71cd1bbf64e3` |

repeat20 日志 SHA-256：`beadafe416e299ca53dbf4af5e6f72295bf7ddfd0c27517bd08590949d1ef96c`。四档均检查真实 P2P、AllGather、AllReduce、四次 GROUP_SYNC、ALLOC/RENAME/rebind/FREE、byte/sentinel、trace 与 residual；`--require-accelerated` 保证无静默回退。

## 5. 兼容矩阵

主矩阵严格串行执行 40 个入口，**40/40 exit 0**，累计 72.463 s。包含 NoC collective R0～R8/V0～V6、DTE V0～V4（含 V2b/V3b）、D2D V0/V3/V4/V5、SRAM pipeline 与 R0～R6、HBM R0～R4、NoC congestion。

- 证据目录：`/tmp/isa-v1-compat-green`
- `RESULTS.tsv` SHA：`1ff8dc2c108fca29e93fdb6801f168fc1fdd2dc2f9d1d361fc9840e9e4d81f95`
- `SHA256SUMS` SHA：`cb1fa696561e6a0b081779bee6dcab755b2684e1af7475e60a6281634a09b05a`
- `sha256sum -c SHA256SUMS`：全部 OK。

补充 SRAM compat/legacy/NUMA/DRAMSys 和 legacy exit 共 **5/5**；RESULTS SHA `a494a2445dad30acec758cf1b5cdf7b239b668ce6c0ef384f36b8e285f9f9164`。

## 6. Sanitizer

使用 pthread coroutine SystemC，避免 quickthreads 自定义栈造成 ASan 假阳性；ASan strict/leak、LSan exitcode、UBSan halt 均开启。

| Gate | 结果 | ASan | UBSan | LSan | 日志 SHA-256 |
|---|---|---:|---:|---:|---|
| build | 三目标完成 | 0 | 0 | 0 | `a011acba595ba7d9a3e964c14bfce6869bdfb1c8717261ff5242452790a678be` |
| format + P8-A | 2/2 | 0 | 0 | 0 bytes | `f8945db35d6932ae7f632002b26f2ddc99ac6c66f8d5064db02054e75c93c415` |
| P6 | 37/37 | 0 | 0 | 0 bytes | `0cd4b58e56ec96193a41dcea022f0c78eb0a7d7d1d761d0fc0b3f8667a000cd4` |
| P7 32 KiB | 8/8 | 0 | 0 | 0 bytes | `8baa688d9758888f6a3184fdba66e6e8f316cbf7a105c49b19a7ebe87654556c` |
| P8-B | 12/12 | 0 | 0 | 0 bytes | `e11c6411227e088a56766b63b3a0bd18def3e08392ac5570129b2de6f3f5e9df` |

## 7. 已关闭缺陷

发布审计发现并关闭：P2P tag/ACK/REQUEST 生命周期、Router CTRL pulse、endpoint full-flow lock、serialized-wire ownership、collective endpoint refill、P7 observer/DCA readiness、HBM 32-byte physical burst 与 Submit 异常原子性、SystemC/HBM wire UB、KVTable/MemEndpoint 泄漏、collective address span/derived-state bounds、旧 timing oracle 与 behavioral-SRAM DCA 合同。每项均有 focused 回归，并在 fresh full/compat/sanitizer 门中复验。

当前无开放 P0/P1。ReduceScatter+DCA 仍是明确 unsupported capability，加载期拒绝，属于已发布限制而非缺陷。

## 8. 最终结论

**GO**：P0～P8 的实现、负例、真实字节、性能语义、兼容性、fresh Release/Debug、repeat20 和 sanitizer 门均闭合。功能可用性不构成相对 baseline 的性能 SLA；支持边界以 [ISA v1 已知限制](ISA_v1已知限制.md) 为准。
