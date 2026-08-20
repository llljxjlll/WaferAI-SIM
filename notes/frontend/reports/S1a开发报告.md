# S1a 开发报告：最小持久状态与数据搬运底座

## 1. 阶段状态

阶段 1a 已完成，并已冻结 `stage1a-persistent-state-v1` 可重建证据。parameter、cross-action KV
和 synthetic PD 的状态搬运均已从前端语义层执行到 C++ simulator，且通过 byte-exact
seed/probe、流量、依赖和 fail-closed 门禁。阶段 1a 不改变 S1 的研究覆盖分数：当前仍为
3.5/6、顶层验收仍为 1/3。完整 decode 与真实 PD/KV handoff 分别属于阶段 3、4。

## 2. 需求理解与范围

阶段 1a 的目标是为小型 decode、PD 和后续训练建立可执行的数据底座，而不是提前实现完整
负载。必须提供：

- parameter、逐层 KV 和未来 optimizer state 的稳定逻辑身份；
- 与逻辑身份分离的 HBM 物理 backing；
- HBM 与 SRAM 之间显式、可计时、可统计字节的 DMA；
- STEP/PERSISTENT 两种最小 HBM state scope；
- ProgramIo 对 behavioral HBM 的 seed/probe/rollback；
- state、backing、DMA、manifest、runtime 的逐层 exact closure。

本阶段明确不做：地址扩宽、SRAM lifetime reuse、spill、alias、异步预取、双缓冲、在线请求
调度、动态 RegionCall、真实多 instance PD placement、KV reshard、optimizer 执行。optimizer
只允许声明保留身份，不允许生成 action 或流量。

## 3. 模型形式化

### 3.1 状态与 backing 分离

逻辑 state ID 只由 kind、instance、静态 request slot、layer、logical name、shard/generation、
shape、dtype、layout、scope、lifecycle 和 mutability 决定。物理 HBM 地址不得进入逻辑 state
ID；placement 或地址变化只能改变 backing ID 及其下游 artifact digest。

HBM backing 不复用 SRAM `BufferBinding`。后者继续严格表示 core-local、named-region、带 bank
集合和 core-order lifetime 的 SRAM root。HBM backing 不得携带 core、bank、SRAM storage、alias
或 core-order lifetime 字段。

### 3.2 最小状态机

```text
DECLARED -> BOUND -> SEEDED -> HBM_VALID
HBM_VALID -> DMA_IN_FLIGHT -> SRAM_VALID
SRAM_VALID -> SRAM_DIRTY
SRAM_DIRTY -> DMA_OUT_FLIGHT -> HBM_VALID
HBM_VALID -> PROBED
PROBED -> RELEASED             # 仅 STEP 在 artifact 结束后
```

READ_ONLY parameter 禁止进入 DIRTY/DMA_OUT。PERSISTENT 只保证一次 ProgramArtifact 内使用同一
backing；跨 simulator invocation 必须显式 checkpoint/reload，不能依赖进程内残留状态。

### 3.3 DMA 语义

第一版固定使用后端已有的阻塞 LSU 通路：

```text
DMA_IN  -> LSU_LOAD   # HBM -> SRAM
DMA_OUT -> LSU_STORE  # SRAM -> HBM
```

`DMA_IN` 只有一个本地 SRAM WRITE 端点，`DMA_OUT` 只有一个本地 SRAM READ 端点。HBM 端由
独立 state backing 引用，不创建伪造的 SRAM `TaskBufferUse`。异步 `DTE_ISSUE+WAIT` 保留为
阶段 1b/优化策略接口。

### 3.4 数值公式

```text
B_tensor    = product(shape) * dtype_bytes
B_HBM_read  = sum(DMA_IN.bytes)
B_HBM_write = sum(DMA_OUT.bytes)
B_HBM_total = B_HBM_read + B_HBM_write
T_HBM_floor = max_die ceil(B_HBM_die / C_HBM_die)
```

Host ProgramIo seed/probe 不计入 DMA traffic。只有实际 LSU record 执行产生 HBM/SRAM bytes 和
timeline。地址必须满足无 uint64 overflow、完整落在 home range、满足 backing alignment；当前
checked-in hardware 的 HBM alignment 固定为 64 B。

## 4. 软件架构

```text
IR0 state declaration/access
  -> IR1 HBM region/backing placement
  -> IR2 StateIoOrigin + DmaContract
  -> IntraDieSchedule SRAM use + HBM state use
  -> GlobalAction exact quotient
  -> CommandFragment StateABI + LSU_LOAD/STORE
  -> LinkedProgramManifest exact state/address closure
  -> ProgramArtifact blocking LSU
  -> ProgramIo HBM seed/probe + runtime byte oracle
```

Physical fabric 必须规范化 hardware JSON 的 HBM home range、capacity、alignment、home die、
backend 与 bytes/cycle。Stage 1a 只支持 behavioral HBM 的 functional seed/probe；DRAMSys
缺少相同 debug backing 时明确 fail closed。

## 5. 已完成的独立实现切片

1. State identity/lifecycle/HBM backing 独立 schema：strict serde、稳定 ID、组合/地址/重叠
   负测；不改现有编译链。
2. Hardware loader 规范化 HBM address space；2x1/2x2 home range 数值门禁。
3. IR0/IR1 state 与 backing：parameter、逐层 KV、optimizer-reserved 身份和 exact placement。
4. IR2/GlobalAction DMA contract：正确的单 SRAM 端点、依赖和状态机门禁。
5. Python lowering/manifest：`LSU_LOAD/STORE`、StateABI、state operand closure。
6. C++ finalizer 与 ProgramIo：strict parse、behavioral HBM seed/probe/rollback、LSU stats。
7. parameter、cross-action KV、synthetic PD 三个 case 的数值与 runtime 集成。
8. capability/baseline 收口：新建 `stage1a-persistent-state-v1`，不覆盖 Stage 0。

每片依次执行最小实现、focused unit、数值 oracle、相邻集成、全量回归、重构和必要的性能
检查；任何片未绿时不进入后续片。

## 6. 冻结的最小 case

- Parameter DMA+compute：FP16 `[8,8]` weight 为 128 B，HBM read=128 B，GEMM FLOPs=128，
  D2D=0，16 B/cycle 时 HBM capacity floor=8 cycles；compute 只声明 timing。
- Cross-action KV：L=2、T=2、KVH=2、DH=4，四个 K/V state 各 32 B；HBM write/read
  各128 B，四个目的 payload 逐字节匹配。
- Synthetic PD state：2 die、K/V 各32 B；D2D=64 B，P 侧 HBM read=64 B，D 侧 HBM
  write=64 B，completion 必须支配 synthetic decode-start。
- Fail-closed：unknown/conflicting identity、wrong-home、misalignment、home-range crossing、
  uint64 overflow、READ_ONLY writeback、未完成 DMA 的消费、SRAM 越界。

这些 case 新增为零计分 `s1.foundation.*` 能力。它们证明底座，不得升级
`s1.kv_handoff`、`s1.pd`、F-D1 或 F-PDS 的能力状态。

## 7. Digest 与证据政策

Stage 1a 未修改 `stage0-policy-provenance-v1`，并已一次性生成
`stage1a-persistent-state-v1`，保存三个正向 case、负例 evidence、CaseMatrix、CapabilityManifest
以及 E1/E2 的 old/new digest 和结构差异。冻结器拒绝覆盖已有目录，发布前再次验证 Stage 0
tree digest，并以临时目录原子 rename；禁止因测试变红直接更新 golden。

## 8. 端到端实现结果

三个最小 case 均通过同一条 production 链路：IR1/IR2 → schedule → GlobalAction → lowering →
linker → C++ finalizer → ProgramIo resolver → `npusim`。两次 runtime 的 artifact、marker、内存统计、
probe 和 makespan 均稳定，LSU/DTE residual 均为 0。

| Case | Artifact（bytes/records/relocations） | HBM read/write | D2D | Makespan | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| P1 | 1,694 / 9 / 16 | 128 B / 0 B | 0 B | 163 cycles | parameter shard LOAD 与 SRAM probe 通过 |
| K1 | 16,266 / 122 / 223 | 128 B / 128 B | 0 B | 1,291 cycles | 4 个 32 B K/V payload 的 HBM 与 SRAM probe 全部 byte-exact |
| PD1 | 4,272 / 30 / 44 | 64 B / 64 B | 64 B | 466 cycles | K/V source LOAD→SEND/RECV→destination STORE 与 completion 支配通过 |

冻结的 runtime report 分别为：

- P1：`stage1a_runtime_report_84e2d9df77b0cedd`；
- K1：`stage1a_runtime_report_e6d91348f83de3da`；
- PD1：`stage1a_runtime_report_31b3a31c90104a7a`。

## 9. E1/E2 reviewed regression

E1/E2 均保持 success→success。analytic FLOPs、op counts、D2D、flow、ACK/DONE、credit、drain 和
validation mode 被列为强不变量；只允许 persistent state、显式 DMA/LSU、SRAM lifecycle、版本和
makespan 等已审查变化。完整逐字段差异保存在 `e1_e2_reviewed_diff.json`，ID 为
`reviewed_baseline_diff_61ca064b714cda17`。

| 指标 | E1：Stage 0 → Stage 1a | E2：Stage 0 → Stage 1a |
| --- | ---: | ---: |
| artifact bytes | 25,276 → 27,676 | 155,552 → 156,256 |
| records | 204 → 220 | 1,304 → 1,304 |
| relocations | 322 → 354 | 1,948 → 1,980 |
| ProgramIo initializations / probes | 58 / 2 → 70 / 6 | 356 / 4 → 372 / 20 |
| max SRAM end / core | 22,528 → 13,440 B | 62,784 → 16,960 B |
| observed HBM read / write | 0 / 0 → 18,688 / 256 B | 0 / 0 → 41,984 / 1,024 B |
| observed D2D | 1,024 B → 1,024 B | 8,192 B → 8,192 B |
| makespan | 1,728 → 4,292 cycles | 5,333 → 8,273 cycles |

artifact SHA-256 分别为：E1
`f4607bcf478a51854adaa7182507094a4b2b456bcc10d9c197ea811007f796bf` →
`fdd1f19839881c285e6f5a7a4deb4db8f4be663be92c39ef32a175658be7a7be`；E2
`ec7295762a8349ac80e71b013598c59d30e69d5c3eccdb7c32035bbeed2bbe77` →
`1caba2e72b7e3888d54e5f61097f54098434c8e81c42e846aff33f678ba815ed`。

## 10. 冻结证据与 fail-closed

证据目录为 `notes/frontend/baselines/stage1a-persistent-state-v1`，关键稳定 ID 为：

- CaseMatrix：`case_matrix_c7622e16e257a475`；
- CapabilityManifest：`capability_manifest_5ddb7d5f6d78f90a`；
- baseline review：`baseline_review_fee7f3e6b8842317`；
- negative evidence：`stage1a_negative_evidence_779343de02591d5c`。

negative runner 实际执行并锁定 9 个失败见证：重复 state identity、HBM misalignment、错误
home/range、uint64 overflow、READ_ONLY writeback、空 DMA access refs、缺 DMA completion
dependency、SRAM 越界，以及 DRAMSys debug-peek pre-simulation fail-closed。最后一项绑定
finalizer/resolver/npusim/hardware/simulation/mapping digest，要求精确错误
`HBM backend does not support debug peeking`，且失败日志中不得出现 `SIM_RESULT` 或
`PROGRAM_MEMORY`。

冻结目录不保存 `.npup`；artifact 由 SHA-256、size、record/relocation count、manifest closure 和
finalizer 双跑闭合。发布后的第二次 freezer 调用已验证会拒绝覆盖；严格 serde 后从冻结的
oracle/runtime/negative evidence 重建 CaseMatrix、CapabilityManifest、E1/E2 diff 和 review，结果
逐字匹配 checked evidence，Stage 0 tree digest 也保持不变。

## 11. 验证门禁与能力边界

- Stage1a P1/K1/PD1 production runtime CTest：3/3 通过；
- capability/evidence focused：19/19 通过；
- baseline freezer 与 checked-evidence rebuild focused：5/5 通过；
- E1 CTest 与 E2 direct wrapper：各 1/1 通过；
- 阶段主线 Python 回归：548/548 通过；
- full freeze、overwrite rejection、zero-`.npup`、strict rebuild：全部通过。

这些结果证明的是 timing execution、byte-exact state transport、显式 HBM/LSU 流量和控制闭合，
不是模型 functional output。PD1 仍是单 instance、TP2、同 group、rank0→rank1 的 synthetic case；
没有实现真实多 instance PD placement、KV reshard、完整 decode 或 optimizer execution。因此
`s1.kv_handoff` 与 `s1.pd` 仍为 unsupported，S1 覆盖仍为 3.5/6、顶层验收仍为 1/3。
