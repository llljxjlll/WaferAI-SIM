# DTE 专用控制核测试与验证记录

> 日期：2026-08-25  
> 性质：开发前只读审计与最小验证方案  
> 关联方案：`notes/extensions/DTE_control_core/DTE专用控制核开发方案.md`

## 1. 当前测试入口审计

根目录 `CMakeLists.txt` 使用 `file(GLOB_RECURSE ... CONFIGURE_DEPENDS ./llm/*.cpp)` 收集 `npusim` 源文件。因此新增 `llm/src/dte/*_selftest.cpp` 会自动编入 `npusim`，但仍需在 `llm/unittest/npusim.cpp` 增加 CLI flag 和函数分发，CTest 也需要显式 `add_test()` 才能发现它。

当前 DTE V0-V4 并不是每版都有独立 C++ CLI：

| 能力 | 直接入口 | WorkerCore/trace 集成入口 | 当前冻结重点 |
|---|---|---|---|
| V0 | `./build/npusim --dte-v0-selftest` | `python3 llm/test/dte/run_test_dte_v0.py` | 配置、payload、channel/shared bus、oracle |
| V1 | 无独立 V1 CLI | `python3 llm/test/dte/run_test_dte_v1.py` | DTE on/off、每核并行、跨 die、精确完成时间 |
| V2a | 复用 V0 自测 | `python3 llm/test/dte/run_test_dte_v2.py` | parallel SEND、channel 上限、共享总线 |
| V2b | 复用 V0 自测 | `python3 llm/test/dte/run_test_dte_v2b.py` | streaming 三段流水与配置负例 |
| V3a | `./build/npusim --dte-v3-selftest` | `python3 llm/test/dte/run_test_dte_v3.py` | async token、WAIT/FENCE/CANCEL、345 ns overlap |
| V3b | `./build/npusim --dte-v3b-selftest` | `python3 llm/test/dte/run_test_dte_v3b.py` | aggregation、fan-out、V3a 345 ns 冻结 |
| V4 | `./build/npusim --dte-v4-selftest` | `python3 llm/test/dte/run_test_dte_v4.py` | command slots、端口背压、V3a 345 ns 冻结 |

现有 CTest 没有注册上述 DTE V0/V3/V3b/V4 CLI，也没有注册 DTE Python runner。`BUILD_TESTING=ON` 的当前 `build` 共注册 92 项；其中默认 smoke 是 `default_workload_smoke`，命令为：

```bash
/opt/cmake/bin/ctest --test-dir build \
  -R '^default_workload_smoke$' --output-on-failure
```

它目前只检查 START_DATA 计数和不存在 `[PROTO_WAIT]`，没有冻结最终 sim-time。开发前现有 `build/npusim` 的默认 workload 基线为：

```text
All requests finished: 112 ns
[SIM_RESULT] makespan_cycles=56
[START_DATA] enqueued=1 injected=1 accepted=1 delivered=1 consumed=1 completed=1
```

## 2. 建议新增的最小测试结构

为缩短开发闭环，建议只新增一个 C++ selftest 和一个 Python 端到端 runner：

```text
llm/include/dte/dte_control_core_selftest.h
llm/src/dte/dte_control_core_selftest.cpp
llm/test/dte/run_test_dte_control_core.py
llm/test/dte/hardware/control_core_legacy.json
llm/test/dte/hardware/control_core_dedicated.json
llm/test/dte/hardware/control_core_heterogeneous.json
```

CLI 建议统一为：

```text
--dte-control-core-selftest
```

并在 `BUILD_TESTING` 段注册：

```cmake
add_test(NAME dte_control_core_selftest
         COMMAND npusim --dte-control-core-selftest)
set_tests_properties(dte_control_core_selftest PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    TIMEOUT 60
    LABELS "dte;control-core;config")
```

### 2.1 C++ selftest 的最小覆盖

配置契约测试不需要启动完整 workload，直接构造 JSON 并调用解析边界：

1. 缺失 `control_cores` 得到 `legacy_shared` 和 `16/1/0/0` 默认值。
2. 顶层 `dual_dte_dedicated` 对全部 core 生效。
3. `cores[].control_cores` merge-patch 覆盖单个字段；显式 core 未写该字段时回到顶层默认。
4. core ID 间隙和尾部自动补齐时完整复制有效控制核配置。
5. 单 core 覆盖回 `legacy_shared`，其余 core 仍为 dedicated。
6. 未知 mode、depth=0、width=0、width>depth、负延迟和 legacy 下非默认专用字段均拒绝，并检查错误包含字段路径与 core ID。
7. `TOTAL_CORES`、core ID 和 NoC endpoint 数不因 dedicated 模式改变。

控制核 SystemC fixture 使用固定服务时间的假 backend，避免把 DTE engine 时间混入 controller 断言：

1. depth=1 连续提交两条命令，第二条产生 queue-full stall，不丢命令。
2. depth=2、width=1/2 时，同一控制周期最大派发数分别为 1/2。
3. `dispatch_latency_ns=1/2/3` 在当前 `CYCLE=2 ns` 下分别按明确规则得到 2/2/4 ns；测试必须冻结向上取整规则。
4. backend 完成后仅在 `completion_notify_latency_ns` 到期时对提交者可见。
5. notify 先于 waiter 注册时，完成状态仍可被消费，不依赖一次性 event 唤醒。
6. 两个 core 的 FIFO、command ID、token、统计互相隔离。
7. legacy backend 不创建 FIFO、不等待 delta cycle，controller trace/统计均为空或 0。

### 2.2 Python 端到端 runner 的最小覆盖

建议复用 `v3_overlap.json`、`v3_blocking.json` 和 `v3_four.json`，不要再造复杂 workload：

1. 缺省配置与显式 `legacy_shared` 各运行一次，比较退出码、最终时间、DTE/async 关键 trace 序列；二者均须保持 V3a overlap `345 ns`，且不得出现 `DTE_CTRL_*` trace。
2. dedicated + 0 latency 运行 overlap，断言 compute 与 DTE transmit 真实重叠，所有 controller residual 为 0，并出现 queue/dispatch/notify 证据。
3. dedicated 分别增加 1 个 controller cycle 的 dispatch/notify 延迟。用简单 ISSUE→WAIT 关键路径断言 DTE pending 或 WAIT 可见时刻精确后移 2 ns；不要用包含大量 slack 的 workload推断总时间。
4. dedicated depth=1 与较大 depth 运行 `v3_four`，断言 depth=1 有 queue stall/high-watermark=1，所有命令最终仍各完成一次。
5. width=1/2 运行同一 burst，断言同周期最大 dispatch 数分别为 1/2；只比较 controller 段，不把 DTE channel/shared-bus 的瓶颈误算成控制核效果。
6. overlap 版仍满足 `transmit_begin < compute_end` 且比 blocking 版更快；同时校验 transfer bits、compute span 和 token drain 一致，防止只优化时间却丢工作。

## 3. legacy 时序冻结门禁

最小门禁应同时覆盖“不含 DTE 的默认 workload”和“包含 async DTE 的冻结 workload”：

```bash
/opt/cmake/bin/ctest --test-dir build \
  -R '^(default_workload_smoke|dte_control_core_selftest)$' \
  --output-on-failure -j1

python3 llm/test/dte/run_test_dte_v3.py
```

其中 `run_test_dte_v3.py` 已冻结 blocking `421 ns`、overlap `345 ns`、两者差值 `76 ns`，并检查相同 transfer bits 和相同 compute 工作量。建议新增的 control-core runner 再冻结：

- 缺省 hardware 与显式 legacy 的关键 DTE trace 完全相同；
- 默认 smoke 仍为 `112 ns / 56 cycles`；
- legacy 模式没有 `DTE_CTRL_queue_wait/dispatch/barrier/notify`；
- 程序结束 controller/DTE/token residual 全部为 0。

不能只以进程返回 0 作为 legacy 兼容结论。

## 4. 可直接执行的构建与测试命令

仓库要求先设置 `SYSTEMC_HOME`。首次或 CMake 注册变化后：

```bash
export SYSTEMC_HOME=/opt/systemc-2.3.3
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target npusim --parallel 2
```

快速开发门（串行，避免共享 trace 冲突）：

```bash
./build/npusim --dte-control-core-selftest
/opt/cmake/bin/ctest --test-dir build \
  -R '^(default_workload_smoke|dte_control_core_selftest)$' \
  --output-on-failure -j1
python3 llm/test/dte/run_test_dte_control_core.py
python3 llm/test/dte/run_test_dte_v3.py
```

合入前 DTE 回归：

```bash
./build/npusim --dte-v0-selftest
./build/npusim --dte-v3-selftest
./build/npusim --dte-v3b-selftest
./build/npusim --dte-v4-selftest

python3 llm/test/dte/run_test_dte_v0.py
python3 llm/test/dte/run_test_dte_v1.py
python3 llm/test/dte/run_test_dte_v2.py
python3 llm/test/dte/run_test_dte_v2b.py
python3 llm/test/dte/run_test_dte_v3.py
python3 llm/test/dte/run_test_dte_v3b.py
python3 llm/test/dte/run_test_dte_v4.py
```

相关生产路径的最小补充门：

```bash
/opt/cmake/bin/ctest --test-dir build \
  -R '^(p2p_payload_selftest|p2p_session_selftest|sram_r4_selftest|default_workload_smoke)$' \
  --output-on-failure -j1
```

上述 Python runner 全部硬编码使用 `build/npusim`，并共享 `build/events.json`；不能并发执行，也不能在不修改 runner 的情况下验证 `build-debug-*` 等其他构建目录。

## 5. 本次审计实测基线

使用当前已有 `/workspace/build/npusim` 实测：

| 命令 | 结果 |
|---|---|
| `--dte-v0-selftest` | PASS，66 checks |
| `--dte-v3-selftest` | PASS，33/33 |
| `--dte-v3b-selftest` | PASS，21/21 |
| `--dte-v4-selftest` | PASS，23/23 |
| CTest `default_workload_smoke` | PASS |
| 默认 workload 直接运行 | 112 ns，56 cycles |

这组结果仅作为开发前参考：现有 `build/npusim` 时间戳早于部分 dirty source，不能替代改动后的重新构建与门禁。

## 6. dirty tree 与并行开发风险

审计时工作树已有大量未提交修改和未跟踪构建产物；其中直接与本扩展重叠的文件至少包括：

- `llm/include/workercore/workercore.h`；
- `llm/src/workercore/workercore.cpp`；
- `llm/include/dte/dte_async_types.h`；
- 多个 P2P/sync/collective runtime 与 selftest；
- `llm/unittest/npusim.cpp`。

风险控制建议：

1. 不用 `git checkout/reset` 清理共享工作树，不覆盖未知改动。
2. 新增 CLI 前先重新查看 `npusim.cpp` diff，避免覆盖并行开发的 flag 和分发逻辑。
3. Python runner 串行运行；每次失败保留 stdout，并确认 `events.json` 来自本次命令。
4. 新增 `.cpp` 后重新运行 CMake configure；虽然 `CONFIGURE_DEPENDS` 可触发重新扫描，但不能依赖旧二进制证明新源已编入。
5. 验收记录同时写明源码 commit/diff、binary 时间戳和构建目录，避免把旧 binary 的 PASS 误当成新实现结果。
