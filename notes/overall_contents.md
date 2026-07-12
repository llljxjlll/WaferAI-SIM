# WaferAI-SIM / NPU-SIM 项目总览

> 整理日期：2026-07-12
> 论文：[From Principles to Practice: A Systematic Study of LLM Serving on Multi-core NPUs](https://arxiv.org/abs/2510.05632)
> 文档：https://npu-sim.readthedocs.io/zh-cn/latest/

---

## 1. 项目定位

**WaferAI-SIM**（可执行文件名 `npusim`）是一个面向 **多核 NPU（Neural Processing Unit）** 的
轻量级、大规模、多层级仿真框架，主要用于对 **LLM 推理服务（serving）** 做系统级性能分析。

核心特性：

- **🔬 多层级仿真**：既支持 transaction-level（事务级，精确）仿真，也支持 performance-model（性能模型，快速）仿真。
- **🏗️ Wafer-Scale 建模**：支持基于 hybrid bonding、分布式内存的下一代硬件建模。
- **🧩 灵活的并行策略**：可探索多种 Tensor Parallelism (TP) 策略；同时支持 PP/DP。
- **📍 可定制核放置（core placement）**：用户自定义核到物理网格的映射策略。
- **💾 多样的内存管理**：可仿真不同的内存管理方式（SRAM/DRAM/Cache）。
- **🔄 可配置数据流**：可在 **PD-disaggregation（分离）** 与 **PD-fusion（融合）** 之间切换（PD = Prefill/Decode）。
- **📊 实时可视化**：基于 SFML/Cairo 的交互式 GUI，可监控仿真过程。

底层基于 **SystemC**（离散事件仿真内核）+ **DRAMSys**（DRAM 时序模型）构建，C++17。

---

## 2. 顶层目录结构

| 路径 | 说明 |
| --- | --- |
| `llm/` | **核心仿真器源码**（include + src + test + unittest + config_generation） |
| `DRAMSys/` | 集成的第三方 DRAM 时序仿真库（作为 CMake 子项目） |
| `doc/` | Sphinx/ReadTheDocs 文档（`.rst`，中文为主，含英文 locale）与配图 |
| `plot/` | 论文实验的绘图脚本（matplotlib，`*.py` + 输出 pdf/png） |
| `scripts/` | 批量跑实验的 shell 脚本（`run_test*.sh`） |
| `streaming_trace_viewer/` | 基于 trace.json 的流式 trace 可视化前端（Python） |
| `font/` | GUI 渲染用字体 |
| `build/` | CMake 构建产物 |
| `CMakeLists.txt` | 顶层构建脚本 |
| `Dockerfile` | 一致化的构建/运行环境（推荐用法，约 3 分钟构建） |
| `notes/` | （本文件所在）整理的项目笔记 |

### 2.1 `llm/` 内部结构（include 与 src 一一对应）

| 模块 | 职责 |
| --- | --- |
| `defs/` | 全局定义：`const`（常量）、`enums`（枚举）、`global`（全局变量/初始化）、`spec`（硬件规格）、`types` |
| `common/` | `config`（配置解析）、`memory`、`msg`（消息）、`pd`、`system`、`display` |
| `workercore/` | **计算核模型**：`WorkerCore`（sc_module）+ `WorkerCoreExecutor`（原语执行引擎）；`logic.cpp` 执行逻辑 |
| `prims/` | **原语（primitive）库**——仿真的最小任务单元，见第 5 节 |
| `memory/` | 内存层次模型：`sram/`、`dram/`（含 DCache）、`gpu/`（L1/L2 Cache）、`MemoryManager` |
| `link/` | 芯片/节点级互联与全局内存：`chip_*`、`config_*`、`global_mem_interface`、`nvlink_packet`、`instr/` |
| `router/` | NoC 路由器模型（`RouterMonitor`） |
| `monitor/` | **顶层协调者 `Monitor`** + 各种 `config_helper_*`（core/gpu/pd/pds 场景装配）+ `mem_interface` |
| `trace/` | 事件引擎 `Event_engine` + `Trace_event`（生成 VCD/trace 供可视化） |
| `utils/` | 各类工具：config/display/memory/msg/print/router/system utils + `simple_flags`（命令行解析） |
| `macros/` | 宏定义 |
| `unit_module/` | 可独立编译的子模块：`sram_manager`、`dram_kvtable`（各带自己的 main/CMake，用于单元验证） |
| `config_generation/` | 配置生成器（`config_gen.cpp`，独立 main） |
| `test/` | **各类配置文件** + 工具脚本，见第 6 节 |
| `unittest/` | 仿真主入口 `npusim.cpp`（`sc_main`） |

---

## 3. 整体架构与仿真流程

系统是一个 **2D 网格（mesh）的多核 NPU**：每个 `WorkerCore` 通过 `Router` 组成 NoC，
核之间靠消息（`Msg`）通信；核外有全局内存（`ChipGlobalMemory` / `GlobalMemInterface`）。
`Monitor` 作为顶层 sc_module 把所有核、路由、内存接口装配并连线起来。

### 主入口 `llm/unittest/npusim.cpp`（`sc_main`）执行流程：

```
1. simple_flags::parse_args        // 解析 4 个 --*-config 命令行参数
2. 清理上次运行的日志文件
3. InitGrid(workload, hardware, simulation, mapping)  // 解析 4 类配置 → 建立网格
4. InitGlobalMembers()             // 初始化全局成员
5. InitializeMemorySpec()          // 初始化内存规格
6. new Event_engine(...)           // 创建事件/trace 引擎
7. Monitor monitor(...)            // 装配：WorkerCore[]、RouterMonitor、MemInterface、GlobalMemInterface
8. sc_clock clk("clk", CYCLE, SC_NS)
9. sc_start()                      // 启动 SystemC 离散事件仿真
10. 收集结果 → simulation_result_df_pd.txt，输出总耗时
```

### 数据流抽象（workload 层面）

一个 workload 被描述为若干 `chips → cores → worklist → prims` 的层级结构（见 `playground.json`）：

- `vars`：所有形状/尺寸变量（B、T、DH、NH、HS、L、IS、以及 TP 切分后派生的量如 `BTP/4`）。
- `source`：初始数据注入（dest 核 + size）。
- 每个 core 有一个 `worklist`，每个 work 项包含：`recv_cnt`（需接收几个包）、`cast`（发往哪些 dest + tag）、
  `prims`（该步要顺序执行的原语列表，每个原语带 `sram_address` / `dram_address` 及形状参数）。
- `loop` 字段控制该核 worklist 的循环执行（对应解码逐 token 循环）。

---

## 4. 核心运行模型：WorkerCore + Executor + 原语

- **`WorkerCore`**（`workercore.h`）：一个物理计算核的 sc_module，持有
  `DCache`、`DynamicBandwidthRamRow`（SRAM 阵列）、`sram_manager`、GPU L1/L2 cache 等硬件子模型。
- **`WorkerCoreExecutor`**：核内的**原语执行引擎**，持有 `PrimCoreContext`（元数据）。
  用 SystemC 的 `sc_event` 做细粒度事件驱动：
  - `ev_block`/`prim_block`：指示/切换当前原语是否在执行（阻塞执行流）。
  - `ev_send` / `ev_recv` / `ev_comp` / `ev_systolic` 等：各类阶段事件。
  - 按 `MSG_TYPE` 分类的接收事件数组 `ev_recv_msg_type_[]` 与 `msg_buffer_[]` 队列。
- 一个核的执行 = 三类核心动作（`CORE_PRIM`）：`PRIM_RECV → PRIM_COMP → PRIM_SEND`，
  由 worklist 驱动、逐条执行原语。

---

## 5. 原语（Primitives）系统

原语是仿真的**最小任务/算子单元**，位于 `llm/{include,src}/prims/`。

### 类型（`enums.h::PRIM_TYPE`，可位或组合）

`NORM_PRIM=0`, `COMP_PRIM=1`, `GPU_PRIM=1<<1`, `NPU_PRIM=1<<2`, `PD_PRIM=1<<3`, `MOE_PRIM=1<<4`

### 基类（`prims/base.h`）

- `PrimBase`：所有原语基类，虚接口 `taskCoreDefault(context)` / `serialize` / `deserialize` / `printSelf`。
- `CompBase : PrimBase`：计算类原语基类，增加 `initialize()` / `initializeDefault()` / `parseJson()`，
  管理输入输出尺寸、参数（`param_name`/`param_value`）、数据类型等。

### 按目录分类的原语

| 目录 | 原语 |
| --- | --- |
| `base/` | `npu_base` / `gpu_base`（NPU/GPU 后端基类） |
| `comp_prims/` | 算子：`attention_forward`、`matmul_forward`(+ `_mla`)、`conv_forward`、`layernorm`/`rmsnorm`/`batchnorm`、`gelu`/`relu`/`silu`/`swiglu`、`rope_forward`、`residual`、`gate_forward`、`max_pool`；<br>切分/合并：`split_matmul`/`merge_matmul`、`split_conv`/`merge_conv`；<br>数据搬运：`parse_input`/`parse_output`、`send/recv_global_memory`、`switch_data`、`dummy_prim` |
| `gpu_prims/` | GPU 版本算子：`attention_forward_gpu`(+`_pd`)、`matmul_forward_gpu`(+`_pd`)、`gelu`/`layernorm`/`residual`_gpu |
| `pd_prims/` | PD（Prefill/Decode）专用：`attention_forward_pd`、`matmul_forward_pd`、`rope_forward_pd` |
| `moe_prims/` | MoE：`load_expert`、`matmul_forward_moe` |
| `norm_prims/` | 底层控制原语：`load`/`store`/`send`/`recv`、`clear_sram`、`set_addr`/`set_batch` |

> 新增原语的方法见文档 `doc/source/getting_started/primitive_detail.rst` 与 `advanced_primitive_detail.rst`。

---

## 6. 配置系统（四类 config 文件）

运行时通过 4 个命令行参数传入，互相解耦：

| 参数 | 示例路径 | 作用 |
| --- | --- | --- |
| `--workload-config` | `llm/test/workload_config/gpu/pd_serving.json` | **负载**：模型形状、TP/PP/DP、数据流图（chips/cores/worklist/prims） |
| `--hardware-config` | `llm/test/hardware_config/core_4x4.json` | **硬件**：网格尺寸 `x`、NoC 带宽、每核算力（exu/sfu/vec/sa_cnt）、SRAM/DRAM 尺寸与带宽 |
| `--simulation-config` | `llm/test/simulation_config/default_spec.json` | **仿真开关**：行为级 vs 精确级（`use_beha_sram/dram/noc`、`use_dramsys`）、GEMM 性能模型、日志级别 |
| `--mapping-config` | `llm/test/mapping_config/default_mapping.spec` | **映射**：逻辑核 → 物理网格位置（linear / ring / mesh / interleave 等策略） |

### 关键配置字段速查

- **workload.vars**：`B`(batch) `T`(seqlen) `DH`(head dim) `NH`(head num) `KVH`(kv head) `HS`(hidden) `L`(layers) `IS`(intermediate) `pp`/`dp` + TP 派生量。
- **hardware.cores[]**：`exu_x`/`sfu_x`/`vec_x`（各单元宽度）、`sa_cnt`（systolic array 数）、`vec_cnt`、`sram_bitwidth`、`dram_bw`。
- **hardware.operand**：`comp_util`（算力利用率）、`core_credit`、`pd_ratio`。
- **simulation.memory**：`use_beha_sram/dram`（行为模型开关）、`use_dramsys`（是否用 DRAMSys 精确时序）。

### 工作负载自动生成

`llm/test/tool_script/workload_autogen.py`：参数化快速生成 workload JSON。
关键参数：`B/T/DH/NH/KVH/HS/L/IS`、`pp/dp/tp`（tp 格式 `mn_dim_k_dim`，如 `1_1`）、`avg_output`、`model`(`gpt`/`qwen`)。

支持的模型：**LLAMA 系列**（Llama-2/3, 7B~70B）、**GPT 系列**、**Qwen 系列**、**MoE**。

---

## 7. 仿真模式（`enums.h::SIM_MODE`）

| 模式 | 含义 |
| --- | --- |
| `SIM_DATAFLOW` | 纯数据流仿真（NPU 多核数据流） |
| `SIM_GPU` | GPU 模式 |
| `SIM_PD` | PD 分离 / 相关调度 |
| `SIM_PDS` | PD serving |
| `SIM_GPU_PD` | GPU + PD |

对应有不同的 `config_helper_*`（base/core/gpu/gpu_pd/pd/pds）负责各模式下的核装配。

**PD 相关**（Prefill/Decode 分离服务）：

- `PD_PHASE`：`PREFILL` / `DECODE` / `UNTOUCHED` / `PD_DONE`
- `PD_JOB`（每核任务）：`JOB_PREFILL` / `JOB_DECODE` / `JOB_BOTH` / `JOB_NONE`
- 数据流可选 **PD-fusion**（同核先 prefill 再 decode）或 **PD-disaggregation**（不同核分别负责）。

---

## 8. 内存层次

- **SRAM**（核内）：`DynamicBandwidthRamRow`（多 bank、动态带宽）、`multiport_ram_array`、`arbiter_ram_bank`、`Mem_access_unit`；由 `sram_manager` 管理。
- **DRAM**（核外/HBM）：`DCache` / `DCacheCore` / `DummyDCache`；可接 **DRAMSys** 做精确时序（`hbm3-example.json`），或用行为模型（`use_beha_dram`）。
- **GPU Cache**：`GPU_L1L2_Cache`（L1/L2）、`gpu_cache_system`。
- **全局内存**：`ChipGlobalMemory` + `GlobalMemInterface`（跨核共享，经 NoC/NVLink 访问）。
- **MemoryManager**（`common`/`memory`）：地址与 tag 管理范式，详见 `doc/.../memory_detail.rst`。

---

## 9. 构建与运行

### Docker（推荐）

```bash
docker build -t waferai-sim:latest .        # 约 3 分钟
docker run -it waferai-sim:latest           # 进入后当前目录即有可执行文件 npusim
```

### 直接构建（需 `SYSTEMC_HOME` 环境变量）

- CMake ≥ 3.10，C++17；链接 `systemc cairo sfml-* m`。
- 主目标：`add_test_executable(npusim "./llm/unittest/npusim.cpp")`（见 `CMakeLists.txt:131`）。
- 可选参数：`L1CACHESIZE` / `L2CACHESIZE`（默认 4MB / ~14.4MB）、`BUILD_DEBUG_TARGETS`。
- 构建时排除 `unittest/`（除 npusim）、`test/`、`config_generation/`、`build/` 下的源文件。

### 运行示例（Quick Start）

```bash
./npusim \
    --workload-config   ../llm/test/workload_config/gpu/pd_serving.json \
    --simulation-config ../llm/test/simulation_config/default_spec.json \
    --hardware-config   ../llm/test/hardware_config/core_4x4.json \
    --mapping-config    ../llm/test/mapping_config/default_mapping.txt
```

批量实验：`scripts/run_test*.sh`；论文各图对应配置在 `llm/test/*/paper/figXX/` 与 `llm/test/chores/*.json`。

---

## 10. 可视化与文档

- **GUI**：仿真过程实时可视化（SFML + Cairo 渲染，配图见 `doc/images/gui_demo.gif`）。Demo：PD-fusion, TP=4, PP=7。
- **Trace 引擎**：`Event_engine` 生成 VCD（`Cchip_1`）/ trace，供 `streaming_trace_viewer/`（`main.py` + `static/`）流式查看。
- **文档**（`doc/source/**/*.rst`，Sphinx/ReadTheDocs）：
  - `getting_started/`：安装、quickstart、run、各 config 详解、TP mapping、原语/内存/进阶实现细节、实验分析、仿真器验证。
  - `architecture_trends/`：架构趋势综述（convergence / product_landscape / significance）。
- **绘图**：`plot/*.py` 复现论文图（TP 维度、mapping、cache 命中率、SRAM 利用率、PD 核分配等）。

---

## 11. 关键源码索引（快速定位）

| 想看什么 | 去哪 |
| --- | --- |
| 仿真主入口 | `llm/unittest/npusim.cpp` |
| 顶层装配 / 连线 | `llm/include/monitor/monitor.h` + `llm/src/monitor/*` |
| 计算核模型 & 执行引擎 | `llm/include/workercore/workercore.h`, `llm/src/workercore/{workercore,logic}.cpp` |
| 原语基类 | `llm/include/prims/base.h` |
| 具体算子实现 | `llm/src/prims/comp_prims/*.cpp`（如 `attention_forward.cpp`） |
| 全局枚举 / 常量 | `llm/include/defs/{enums,const,global,spec}.h` |
| 配置解析 | `llm/src/common/config.cpp`, `llm/src/utils/config_utils.cpp` |
| 内存管理 | `llm/src/memory/*`, `llm/include/memory/**` |
| NoC 路由 | `llm/include/router/router.h`, `llm/src/router/router.cpp` |
| 事件 / trace | `llm/src/trace/{Event_engine,Trace_event}.cpp` |

---

## 12. 术语速查

- **NPU**：Neural Processing Unit，多核 AI 加速器。
- **NoC**：Network-on-Chip，片上网络（2D mesh），核间通过 Router 通信。
- **TP / PP / DP**：Tensor / Pipeline / Data Parallelism。
- **PD**：Prefill / Decode，LLM 推理两阶段；可 fusion（融合于同核）或 disaggregation（分离到不同核）。
- **MoE**：Mixture of Experts，含专家加载策略（random/hot/best）。
- **原语（Primitive）**：仿真最小任务单元（算子 / 数据搬运 / 控制）。
- **WorkerCore**：单个物理计算核的 SystemC 模型。
- **Wafer-Scale**：晶圆级、hybrid bonding、分布式内存的下一代硬件形态。
- **SystemC**：C++ 离散事件仿真库，本项目的仿真内核。
- **DRAMSys**：集成的 DRAM 精确时序仿真库。
