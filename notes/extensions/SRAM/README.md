# SRAM 扩展使用说明

## 启用条件

真实数据路径同时接入 `distributed_hbm` 与 `legacy_private`。`distributed_hbm` 可选择 behavioral 或 DRAMSys backend，并保留远端 NUMA 的 NoC/C2C 路由；`legacy_private` 通过 DCache/DRAMSys 收费，并用 transport 内的 byte backing 补足默认 `NoStorage` 配置的功能数据。默认配置关闭，因此旧 workload 的时序不变。

```json
{
  "memory": {
    "sram_size": 33554432,
    "sram": {
      "capacity_bytes": 33554432,
      "real_data_path": true,
      "manual_regions": true,
      "manual_memory_schedule": true,
      "allocation_alignment_bytes": 1024,
      "bank_count": 16,
      "bank_interleave_bytes": 256,
      "read_base_latency_cycles": 1,
      "write_base_latency_cycles": 1,
      "queue_depth": 16,
      "lsu": {
        "queue_depth": 8,
        "max_outstanding": 2,
        "issue_latency_ns": 2
      },
      "dte_memory": {
        "queue_depth": 16,
        "workers": 2
      },
      "ports": {
        "compute": {
          "read": {"count": 1, "width_bits": 2048},
          "write": {"count": 1, "width_bits": 2048}
        },
        "dte": {
          "read": {"count": 1, "width_bits": 2048},
          "write": {"count": 1, "width_bits": 2048}
        },
        "lsu": {
          "read": {"count": 1, "width_bits": 2048},
          "write": {"count": 1, "width_bits": 2048}
        }
      },
      "regions": [
        {"name": "input", "base_bytes": 0, "size_bytes": 8388608,
         "allocator": "block", "spillable": true,
         "access": ["compute", "dte", "lsu"]},
        {"name": "intermediate", "base_bytes": 8388608,
         "size_bytes": 8388608, "allocator": "block",
         "spillable": false, "access": ["compute", "dte", "lsu"]},
        {"name": "comm", "base_bytes": 16777216, "size_bytes": 4194304,
         "allocator": "block", "spillable": false,
         "access": ["compute", "dte", "noc_rx"]},
        {"name": "double_a", "base_bytes": 20971520,
         "size_bytes": 4194304, "allocator": "fixed",
         "spillable": false, "access": ["compute", "dte", "lsu"]},
        {"name": "double_b", "base_bytes": 25165824,
         "size_bytes": 4194304, "allocator": "fixed",
         "spillable": false, "access": ["compute", "dte", "lsu"]},
        {"name": "temp", "base_bytes": 29360128, "size_bytes": 4194304,
         "allocator": "block", "spillable": false,
         "access": ["compute", "dte", "lsu"]}
      ]
    }
  }
}
```

未提供 `regions` 时，`sram_size` 自动映射为覆盖整个容量的 `legacy` block region。统一 SRAM API、region、LSU/DTE primitive 的地址与大小均为 byte，区间为半开区间；仅历史 `SramPosLocator::AddrPosKey.pos` 和 `context.sram_addr` 仍是 SRAM word index，并在进入统一数据面时按 `sram_bitwidth / 8` 显式换算。

## Workload 原语

LSU 阻塞 load：

```json
{
  "type": "Lsu_mem",
  "op": "load_blocking",
  "hbm_addr": 1048576,
  "sram_region": "double_a",
  "sram_offset": 0,
  "size_bytes": 4096
}
```

`queue_depth` 限制可保留的 descriptor/token 总数，`max_outstanding` 只限制并行执行的 worker 数；例如 8/2 配置可以接收 8 笔请求、最多同时运行 2 笔。

异步 LSU load 使用 `op: issue`、`direction: HBM_TO_SRAM` 和非零 `token`；随后用同 token 的 `wait`、`poll` 或 `cancel`，也可用 `fence` 排空全部请求。store 方向为 `SRAM_TO_HBM`。

DTE 的 `DRAM_TO_SPM`、`SPM_TO_DRAM` 和本地 `SPM_TO_SPM` 同样支持 `hbm_addr`、`sram_region`、`sram_offset`、`spm_size`；real-memory 模式要求 payload byte 数与 `spm_size` 完全相等。`SPM_TO_SPM` 使用 `sram_region+sram_offset`（或解析后的 `spm_addr`）作为源、`remote_addr` 作为目标 SRAM 绝对地址。DTE token 只有在 endpoint、HBM 和 SRAM commit 都完成后才完成。

旧 `Load_prim`/`Store_prim` 现在是阻塞 LSU 兼容别名；其地址字段按 byte 解释。历史空描述符（`size=0`）仍为 no-op。

## 计算原语内手动调度

计算原语可通过 `TaskCoreContext` 使用：

- `lsu_memory`：`IssueLoad/IssueStore/Wait/Poll/Fence/Cancel`；
- `dte_memory`：现有 DTE token API；
- `sram_regions`：按 region 名解析地址或进行子分配；
- `sram_access`：计算核直接读写 SRAM；
- `compute_timeline`：`RunTile/RunCycles`。

`manual_memory_schedule=true` 时，`NpuBase` 不再自动装载输入、删除输入标签或执行粗粒度输出写回；计算 primitive 必须显式完成所需搬运。

双缓冲基本顺序是：预取 tile 0；等待当前 slot；向另一 slot 发起下一 tile；读取当前 slot 并计算；最后排空 store。描述符在 issue 时即声明地址范围租约，所以提前读取或覆盖同一 slot 会按 RAW/WAR/WAW 依赖阻塞，不会读取半提交数据。

## 时序、统计与错误

DTE、LSU、compute 和兼容 helper 共用每核 `SramAccessUnit`，共享 bank、端口、队列和 hazard 时间线。可读取：

- `SramAccessUnit::stats()`：按 initiator 的 requests/bytes、queue wait、service 和 stall；
- `CoreLsuUnit::stats()`：HBM/SRAM 读写字节与 token 状态；
- `DteMemoryBridge::stats()`：DTE HBM/SRAM 字节；
- `ComputeTimeline::stats()`：tile、compute cycle 和 SRAM read byte。

生产 trace sink 输出 `SRAM_queue`、`SRAM_read`、`SRAM_write`、`SRAM_bank_wait`；`LSU_issue`、`LSU_hbm`、`LSU_sram`、`LSU_wait`；`DTE_mem_hbm`、`DTE_mem_axi`、`DTE_mem_spm`、`DTE_mem_commit`；`SRAM_region_alloc`、`SRAM_region_free`、`SRAM_region_spill`、`SRAM_region_reload` 和 `Compute_tile` B/E 事件；异常路径也闭合已经开始的阶段。四个数据面模块同时提供结构化 `trace()` 记录。逐 bank/port 的 beats、bytes、service 和 stall 可由 `SramAccessUnit::stats()` 读取。

越界、region 权限错误、未初始化读取、重复 token 和未消费 token 默认报错。real-data 模式关闭时，旧 SRAM/DTE endpoint-only 路径保持原行为。

## 生产验收

仓库提供以下可直接运行的 WorkerCore workload：

```bash
python3 llm/test/sram/run_test_sram_pipeline.py
python3 llm/test/sram/run_test_sram_dramsys.py
python3 llm/test/sram/run_test_sram_legacy.py
python3 llm/test/sram/run_test_sram_numa.py
python3 llm/test/sram/run_test_sram_compat.py
```

第一项同时验证 LSU/DTE 的 blocking 与 double-buffer 调度、非零 payload 校验和、`events.json` 中的计算/搬运重叠、完整分阶段 trace，以及 `input`、`intermediate`、`comm` 标签的分配、搬移与释放。其余用例分别冻结 distributed DRAMSys、legacy_private、跨 die 远端 NUMA 和扩展关闭的旧 NpuBase/helper 路径。

默认启动和完整 legacy 基线从 `build/` 执行：

```bash
./npusim --trace-window 1000000
./npusim --trace-window 1000000 \
  --workload-config ../llm/test/workload_config/playground.json \
  --hardware-config ../llm/test/hardware_config/default/8x8.json \
  --simulation-config ../llm/test/simulation_config/default_spec.json \
  --mapping-config ../llm/test/mapping_config/default_mapping.spec
```

两者都要求无 `PROTO_WAIT`；最终输出的 `START_DATA` 六阶段计数必须相等。P0/P1 详细验收见 `log/P0_P1_default_startup_closure_2026-08-06.md`。
