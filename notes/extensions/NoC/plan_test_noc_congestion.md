# Plan: `test_noc_congestion` — 评估 WaferAI-SIM 的 NoC 拥塞建模

## Context (为什么做这件事)

用户想在 test 文件夹中新增一个 `test_noc_congestion` 测例，用来评估仿真器
（WaferAI-SIM）对片上网络（NoC）拥塞的建模能力。需求三点：(1) 4×4 核阵列上跑一个
简单的分布式 GEMM；(2) 设计两种片上通信情况——一种无拥塞、一种严重拥塞；(3) 分别
运行仿真器得到输出并对比。

关键调研结论（决定了本方案的形态）：

- **仿真器是配置驱动的**：一个"测例" = 4 份配置文件（hardware / simulation /
  workload / mapping）喂给已编译好的 `npusim`（`build/npusim` 已存在）。没有 C++
  单测框架，不需要新增 `.cpp`。入口 `llm/unittest/npusim.cpp`。
- **4×4 阵列**：hardware config 里 `"x": 4` → `GRID_X=GRID_Y=4`，16 核方形 mesh
  （`config_utils.cpp:50-139`）。workload 不写 `mode` 字段时默认 `dataflow` 模式
  （`config_helper_core`），即纯核阵列分布式 GEMM，用 `Matmul_f` 原语 + `cast`/`recv`
  核间消息。
- **拥塞只在周期精确 NoC 下才出现**：默认的行为级 NoC（`use_beha_noc: true`）是纯
  roofline 模型，`wait(roofline_packets * CYCLE)`，**既不看跳数也不看跨流争用**
  （`workercore/logic.cpp:58-61,147,563-564`）。真正的拥塞来自周期精确 router
  （`use_beha_noc: false`）：XY 维序路由（`router_utils.cpp:34-62`）、有限缓冲
  `MAX_BUFFER_PACKET_SIZE=3`（`defs/spec.h:112`，背压）、以及**每条输出链路按 flow 的
  tag 上锁**——tag 不同的流会被 `continue` 阻塞（`router.cpp:344-347`），这就是拥塞/
  串行化机制。
- **输出指标**：仿真结束（所有 end-core 的 DONE 到齐 → `sc_stop()`，
  `config_helper_core.cpp:543-551`）时的 `sc_time_stamp()` = 总周期数（每周期 `CYCLE=2` ns），
  会打成 `[CATCH TEST] ... finished` 日志并追加进 `simulation_result_df_pd.txt`；另有
  `events.json`（Chrome-tracing 时间线，含 router flow 停顿）和 `NETWORK` 类日志
  （router 上锁/解锁）可作拥塞佐证。

已与用户确认的三个设计选择：
1. **对照方式 = 通信模式**：GEMM 计算量两场景完全一致，只改 `cast` 通信图（不动放置、不动带宽）。
2. **额外做双 NoC 对比**：每个场景各跑 `use_beha_noc=true` 与 `false`，显式展示行为级
   NoC 对拥塞"盲区"、周期精确 NoC 能捕获拥塞。
3. **手写精简合成 GEMM**：手写 16 核 workload，每核一个 `Matmul_f` 分片 + 受控 `cast`。

---

## 目录与产物

新建自包含目录 **`test/noc_congestion/`**（用户预先创建的顶层 `test/` 目录，当前为空）。
所有路径以从 `build/` 运行为准（沿用 `renew_tests.py` 的惯例）。

```
test/noc_congestion/
├── hardware/core_4x4.json          # x=4；noc_payload_per_cycle 固定（两场景相同）
├── sim/sim_cycle.json              # use_beha_noc=false  （周期精确，能建模拥塞）
├── sim/sim_beha.json               # use_beha_noc=true   （行为级 roofline，对照）
├── mapping/identity.spec           # 恒等映射（空文件即恒等，逻辑核=物理核）
├── workload/gemm_no_congestion.json    # 场景A：近邻/树状规约，链路不相交
├── workload/gemm_congestion.json       # 场景B：all-to-one 集中规约，链路争用
├── run_test_noc_congestion.py      # 运行器：4 次运行(2 场景 × 2 NoC 模型) + 汇总对比
└── README.md                       # 说明与结果解读
```

---

## 设计细节

### 1. 硬件 `hardware/core_4x4.json`
拷贝 `llm/test/hardware_config/default/8x8.json` 的结构，改 `"x": 4`，保留单个 `cores[0]`
模板（会被克隆到 16 核）。`noc.noc_payload_per_cycle` 取一个**两场景共用的中等值**
（如 4；不作为对照变量，故两场景相同）。

### 2. 仿真配置 `sim/sim_cycle.json` 与 `sim/sim_beha.json`
以 `llm/test/simulation_config/default_spec.json` 为基底，**只翻转** `noc.use_beha_noc`
（cycle=false / beha=true）。其余保持：`router_pipe=false`、`send_recv_parallel=false`、
`fast_warmup=true`、内存维持 `use_beha_sram/dram=true`、`use_dramsys=true`（DRAMSys 配置
路径 `../DRAMSys/configs/hbm3-example.json` 与 `ttf_file ../font/...` 相对 `build/` 有效）。
workload 里所有 `*_data` DRAM 尺寸设 0 → 无 DRAM 流量，把延迟差异**隔离到 NoC**。

### 3. 映射 `mapping/identity.spec`
恒等映射（空 `.spec` 即恒等，见 `default_mapping.spec`）。两场景都用它，保证逻辑 `cast`
目标 = 物理位置，通信模式直接决定物理链路占用。

### 4. 两个 workload（核心）——同算力、异通信
两份 workload 的 `vars` 与每核的 `Matmul_f` 分片**逐字节相同**（如 `B,T,C,OC` 一致），
仅 `cast`/`recv` 通信图不同。参照最小可用 dataflow 样例
`llm/test/workload_config/paper/other/mla_example.json` 的结构：
`source[]` 注入输入 → 每核 `worklist` 项 = `recv_cnt`/`recv_tag` + `prims`(含 `Matmul_f`) +
`cast:[{dest,tag}]`；发给 host 用 `dest:-1, loopout:"true"`（该核成为 end-core，发 `SEND_DONE`
触发结束，`config_helper_core.cpp:328-368`）。

- **场景B `gemm_congestion.json`（严重拥塞，集中式 all-to-one）**：
  `source` 给 16 核各注入一片输入；核 0..14 各做一次 `Matmul_f` 后 `cast` 到同一个**角落
  hub 核**（如物理核 15）；hub 核 `recv_cnt=15`（+ 自身），做一次 `Matmul_f`/`Merge_matmul`
  后 `cast dest:-1 loopout:true`。XY 路由下 15 条流全部汇聚到通往核 15 的同一批链路 →
  被 `output_lock` 串行化 → 严重拥塞。

- **场景A `gemm_no_congestion.json`（无拥塞，近邻/树状规约）**：
  每核**同样**做一次 `Matmul_f`，但把规约拆成**近邻树**（16→8→4→2→1，共 4 级；每级只在
  物理相邻核之间成对通信）。每一步的流都是短跳、空间上互不相交的链路 → 争用最小。根核
  完成后 `cast dest:-1 loopout:true` 结束。
  - 若树状规约在手写时 recv/tag 配对过于繁琐，回退到**成对近邻 + 线性收尾**：8 对相邻核
    (0↔1,2↔3,…) 并发 1 跳交换（完全不相交），再由一条轻量链把结果汇到 end-core。
  - 认定标准：两场景**总的 `Matmul_f` 次数与尺寸一致**，只有链路占用/争用不同。

> tag 约定：每条 flow 用唯一 `tag`（接收方 `recv_tag` 与发送方 `cast.tag` 对应，
> `recv_cnt` = 发送方数量），避免死锁与错配。实现时逐核核对。

### 5. 运行器 `run_test_noc_congestion.py`
仿 `llm/test/tool_script/renew_tests.py`：对 `{congestion, no_congestion} × {cycle, beha}`
共 4 组，`subprocess` 调 `./npusim --workload-config … --hardware-config … --simulation-config …
--mapping-config …`，流式读 stdout、抓取 `[CATCH TEST]` 的结束周期数，汇总成一张表写入
`noc_congestion_summary.txt`。

预期结果（即"评估结论"）：

| 场景 \ NoC 模型 | 行为级 beha (roofline) | 周期精确 cycle |
| --- | --- | --- |
| 无拥塞 | 基线周期 | ≈ 基线（略高，有跳数/争用但很小）|
| 严重拥塞 | **≈ 与无拥塞相同**（拥塞盲区）| **明显 > 无拥塞**（捕获拥塞）|

即：行为级 NoC 对两种通信模式给出几乎相同的周期（说明它不建模链路争用）；周期精确
NoC 下"严重拥塞"场景周期显著增大（说明它建模了拥塞）。这正面回答了"评估仿真器对 NoC
拥塞的建模情况"。

---

## 关键复用点（不要重复造轮子）
- 结构模板：`llm/test/workload_config/paper/other/mla_example.json`（最小 dataflow 样例，
  含 `source`/`recv_cnt`/`cast dest:-1 loopout` 用法）。
- 硬件模板：`llm/test/hardware_config/default/8x8.json`。
- 仿真模板：`llm/test/simulation_config/default_spec.json`。
- 运行器模板：`llm/test/tool_script/renew_tests.py`（globbing + subprocess + 抓末行）。
- 拥塞机制源码（写 README 解读时引用）：`llm/src/router/router.cpp:344-416`（tag 上锁）、
  `llm/src/utils/router_utils.cpp:34-62`（XY 路由）、`llm/src/workercore/logic.cpp:58-61,147`
  （beha roofline）、`llm/src/monitor/config_helper_core.cpp:300-368,543-551`（发/收/结束）。

---

## 验证（Verification）
从 `build/` 目录执行（`npusim` 已存在，无需重编）：

1. **先验证周期精确 NoC 能跑通**（第一大风险——所有随包配置都用 beha，cycle 路径较少被使用）：
   先用**无拥塞 + cycle** 单独跑一次，确认能正常 `[CATCH TEST] ... finished` 而不死锁/崩溃。
   ```bash
   cd build && ./npusim \
     --workload-config ../test/noc_congestion/workload/gemm_no_congestion.json \
     --hardware-config  ../test/noc_congestion/hardware/core_4x4.json \
     --simulation-config ../test/noc_congestion/sim/sim_cycle.json \
     --mapping-config    ../test/noc_congestion/mapping/identity.spec
   ```
   - 若 cycle 路径不稳定/死锁：先修 workload 的 tag/recv_cnt 配对；仍不行则在 README 记录
     限制，并至少用 beha 模型跑通两场景（此时用 `noc_payload_per_cycle` 高/低作为可见的
     拥塞代理指标，作为降级方案）。
2. **跑全套并对比**：
   ```bash
   cd build && python3 ../test/noc_congestion/run_test_noc_congestion.py
   ```
   检查 `noc_congestion_summary.txt`：确认 (a) 两场景在 cycle 精确下周期数明显不同
   （拥塞 > 无拥塞），(b) 两场景在 beha 下周期数几乎相同。两条都成立即达成目标。
3. **拥塞佐证（可选）**：在拥塞+cycle 那次运行中，`grep NETWORK` 核日志看 router 上锁/停顿，
   或在 trace viewer 里看 `events.json` 的 flow 停顿，写入 README。

## 交付时对用户说明
- 明确指出"拥塞只有在 `use_beha_noc=false`（周期精确 router）下才被建模"，这是本测例的核心发现之一。
- 给出汇总表与一句话结论；附上如何自行改 `noc_payload_per_cycle` / 通信模式做进一步实验。

---

## 实现结果（落地记录，2026-07-12）

**最终落地目录**：`llm/test/noc_congestion/`（**不是**顶层 `test/`——顶层 `/test/` 被
`.gitignore` 第 14 行忽略、且 `*.txt` 全局忽略，属临时/不可追踪目录；`llm/test/` 才是与
既有全部配置一致、被 git 追踪的规范测试目录）。

**设计相对原计划的两处修正（实测驱动）：**
1. 原计划的"all-to-one 集中规约"会引入**接收端串行化**（`recv_cnt=15`，一个核顺序收 15 条），
   这一开销**行为级和周期精确 NoC 都会体现**，无法干净隔离"链路拥塞"。改为
   **二分匹配（8 奇核发、8 偶核收，每收核负载=1）**，两场景收发消息数完全对称，差异只剩
   路由是否共享中心链路 → 干净隔离链路争用。
2. 原设想的"任意置换/反射 `perm(i)=15-i`"因核间 `REQ→ACK→DATA` 握手（发送阻塞等 ACK）
   形成**环形等待死锁**（已实测超时）。故采用**奇发偶收的二分结构**（偶核先 post 接收即可回
   ACK）规避死锁。无拥塞=奇`i`→偶`i-1`（近邻不相交）；严重拥塞=奇`i`→偶`15-i`（对角穿中心）。

**实测结果**（`noc_payload_per_cycle=4`，`B=1,T=256,C=64,OC=1024`；总仿真时间 ns）：

| 场景 | 行为级 beha | 周期精确 cycle | cycle−beha |
| --- | --- | --- | --- |
| no_congestion | 14781 | 29109 | 14328 |
| congestion | 14833 | 45441 | 30608 |

- 行为级 NoC：拥塞/无拥塞 = **1.00×**（对拥塞盲）。
- 周期精确 NoC：拥塞/无拥塞 = **1.56×**（捕获链路争用）。
- 隔离出的纯链路拥塞开销 ≈ **16280 ns**（仅周期精确可见）。

**运行**：`python3 llm/test/noc_congestion/run_test_noc_congestion.py`（自动切到 `build/`，
跑 4 组并写 `noc_congestion_summary.txt`）。单次最慢运行 ~0.4s。README 在
`llm/test/noc_congestion/README.md`。
