# NPU ISA v1 ABI 与 Golden 变更策略

> 状态：发布门禁
> 适用范围：Program Format v1、外部 Opcode/record、semantic relocation、内部 PrimId/wire、capability、编译器后端与 `--program` loader

## 1. 兼容性边界

ISA v1 同时维护两层版本，评审时不得混为一层：

- 外部 ABI 是编译器写入 artifact 的 Program Format、Opcode、record operand、symbol/relocation 与 control envelope；
- 内部 ABI 是 loader lowering 后交给 Worker 的 PrimId 和 128-bit segment wire，不供外部编译器直接生成。

外部 ABI 的稳定依据是 `编译产物指令集P0契约.md`、`isa_v1_manifest.md` 和 `program_format_v1.md`。内部 wire 的稳定依据是显式 PrimId manifest、factory 类型检查和 prim-wire golden。进程内标签 ID、C++ 对象布局、JSON 字段顺序、静态注册顺序均不是 ABI。

## 2. 版本与编号规则

以下变化必须提升 Program Format major 或 ISA major，旧 v1 decoder 必须拒绝新 major，不得猜测兼容：

- 改变 header、section descriptor、required section 或 record header 的既有布局、字节序、单位或校验范围；
- 改变已发布 Opcode 数值、operand 位宽/含义、默认值、合法域或完成语义；
- 让相同 artifact 在相同 capability 下选择不同的正确性相关 lowering；
- 改变 symbol kind、semantic operand ID、relocation 加数或 SRAM region `value/size` 语义；
- 将原先会拒绝的保留字段解释为有副作用的字段。

只有旧 decoder 能按既有规则安全跳过时，才可增加 optional section。新增 public Opcode 只能使用未分配区，不得占用 reserved/tombstone；删除已发布 Opcode 后必须保留 tombstone。新 PrimId 只能从当前最大值之后追加，禁止填洞或依赖静态初始化顺序。

patch/minor 级修复只允许收紧明显非法输入、改善诊断、修复未定义行为或增加不影响既有合法编码的 capability。若合法 golden bytes、数据副作用、完成点、计费责任或 trace oracle 改变，仍按 ABI/性能语义变更评审，不得以“内部重构”规避。

## 3. Golden 集合

每次发布候选至少冻结并自动比较以下 golden：

1. 外部 Opcode manifest：数值、名称、类别、visibility/lifecycle/support、capability；
2. 内部 Prim manifest：PrimId、factory name、类别、状态与 creator 类型；
3. 每个外部 record schema 的 minimal、typical、max 编码和 max+1 拒绝；
4. 每个多段 Prim 的 segment 数、每段 PrimId/ordinal、reserved/padding 和 serialize→deserialize→serialize；
5. Program Format 完整 artifact：header、7 个 required section、section/whole CRC32C、符号和重定位；
6. lowering golden：external record 序列到内部 Prim 类型、字段、段数与确定性顺序；
7. 运行 golden：逐字节 source/destination、范围外 sentinel、token/fsm/tree/barrier 生命周期、ops/cycle 和最终 residual；
8. legacy golden：JSON 路径的 wire、关键完成时刻和冻结 runner 结果。

golden 不只比较成功结果。unknown、reserved、known-unsupported、capability-disabled、版本不匹配、越界和 CRC 错误必须各有稳定错误类别；mutation 测试必须在任何 CONFIG/全局提交前失败。

## 4. 变更审批清单

提交涉及 ISA/Program Format 时，评审说明必须逐项回答：

- 变更属于外部 ABI、内部 wire、capability、性能语义还是纯实现；
- 哪些编号、bit range、单位、合法域、reserved 位、完成点和错误类别变化；
- 是否需要新 major/minor；若不升级，为什么旧合法 artifact 逐字节和语义仍兼容；
- 哪些 manifest、codec、wire、artifact、lowering、runtime、legacy golden 被更新；
- 编译器后端、loader、Worker 和旧 JSON 各自需要什么迁移；
- mutation、边界、ASan/UBSan、Release/Debug、重复运行和 full regression 证据；
- 是否新增 experimental/unsupported 能力，其 capability 默认值和无 silent fallback 证据；
- 回滚方法是否只回滚本变更，且不会复用已经发布的编号。

涉及正确性、编号或格式的修改至少需要 ISA/编译器接口与 simulator/runtime 两个视角交叉评审；涉及 DTE/SRAM/NoC 的修改还需要对应 owner 审查数据落点、计费责任和 drain。测试作者不得只验证自己复制的公式或路由算法，oracle 应复用生产纯函数或由独立软件模型计算。

## 5. 自动门禁

发布候选必须通过：

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j8
ctest --test-dir build -L isa --output-on-failure -j1
python3 llm/test/dte/run_test_dte_v4.py
python3 llm/test/sram/run_test_sram_pipeline.py
python3 llm/test/noc_collective/run_test_coll_r6_r7.py
python3 llm/test/run_v5_exit.py
git diff --check
```

CI 另以 `BUILD_SANITIZER_TESTS=ON` 构建 focused Program Format parser，使用 ASan、UBSan 和 leak detection 运行格式/mutation 负例。共享 `events.json`、VCD 或仿真全局状态的 runner 必须串行或使用独立目录。

门禁失败时不得更新 golden 以“接受新结果”。必须先证明新结果符合已批准的规格变更；否则修复实现并复跑。ABI 变更的 golden 更新必须与实现、规范、迁移说明在同一评审中出现。

## 6. 发布后处理

- 已发布编号永久保留；移除项转 tombstone，decoder 返回稳定的 retired/unsupported 诊断；
- experimental 能力默认关闭，只有数据、时序、压力和 drain 专项通过后才能在新 capability 中开放；
- 发现错误解码、错误数据、死锁或无界资源时立即阻塞发布，不允许仅记录为性能已知限制；
- 纯性能偏差只有在正确性不受影响、唯一计费责任明确、trace/oracle 更新并完成评审后，才可作为已知限制随版本发布；
- JSON 与 `--program` 至少并存一个发布周期，废弃必须先有使用统计、迁移工具/文档和完整 legacy 回归替代方案。
