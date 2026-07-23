# D2D Link V4 开发记录

> 分支：`feat/d2d-v4`
>
> 基线：`e1a8c02`（V3 冻结 `ba9b7f8` 之后纳入 mixed-NoC 与 production-SAF probe）
>
> 目标：在不改变 V3 cycle 默认路径的前提下，增加无跨-flow争用的快速 Behavioral
> D2D、独立解析 oracle，以及 Behavioral/cycle 的定量校准。

## 1. 固定边界

- `backend=cycle` 是默认值，继续承载 `functional_v2` 与 `bounded_saf`。
- `backend=behavioral` 不建模有限 FIFO、credit、背压、SAF reservation 或网络死锁。
- Behavioral 首版仍是每方向单端口、`0<rate<=1`；多 lane、多端口和 striping 属 V5。
- Router hop 若由代表包实际穿越 Router 取得，解析式不得重复加入 hop latency。
- Behavioral 只在一个责任点计入 bulk service，并另加每条 D2D link 的固定 latency。
- 无争用时用 cycle 校准；shared/disjoint 有争用差异是模型定义，不要求 Behavioral 伪装成 cycle。

## 2. 增量顺序

```text
V4-a  backend/字段分区/默认兼容契约
V4-b  独立 oracle.py + C++ 纯函数 estimate
V4-c  Behavioral REQUEST/ACK/DATA 单流运行时闭环
V4-d  多跳、消息大小、latency/rate 扫描
V4-e  Behavioral/cycle 校准 + shared/disjoint 语义对照
V4-f  聚合门、文档、冻结
```

## 3. V4-a：配置接缝（完成）

- `D2DLinkConfig` 新增 `backend=cycle|behavioral`，缺省为 `cycle`。
- Behavioral 必须显式给出 `port_rate`、`link_rate`、`link_latency`。
- Behavioral 禁止 cycle-only `mode`、SAF/depth 与 legacy latency/bw/buffer 字段，防止
  “看似 bounded、实际忽略”的静默降级。
- `rate>1` 继续明确拒绝，留待 V5 多 lane。
- 本增量尚未接生产运行时：合法 Behavioral 配置在 topology gate 明确报错，而不是退回 cycle。

验证：

- `npusim --d2d-v0-selftest`：**293/293**；
- Link SystemC：**37/37**；历史 D2D：**67/67**；V3 production：**16/16**；
- NoC 冻结值：**14781/29109、14833/45441**；
- `git diff --check`：通过。

自审修正：backend 与 cycle mode 明确正交；字段按 backend/mode 分区；无 runtime 时 default-closed。

## 4. V4-b：独立 oracle 与纯函数 estimate（完成）

- 新增 header-only C++ `behavioral.h`：构造真实有向 link 路径、统计代表包片内 Router hops、
  使用整数有理数 min-cut 计算 fixed/service/first/last/transaction 分解。
- 新增独立 `behavioral/oracle.py`：直接解析 hardware JSON，不复用 C++ 结果；支持 CLI JSON
  输出与 8 项自测。
- forwarding contract 固定为 **end-to-end pipelined min-cut**：
  `R=min(1,port,link)`、`S(F)=ceil(F/R)`、`T_D2D=3*H*L+S(F)`。
- Router hop 只报告不重复计时；bulk service 只记一次；不维护跨-flow争用状态。

验证：

- C++ pure-function 总门：**300/300**；Python oracle：**8/8**；
- 2×1 单跳、3×1 两跳、2×2 X-first、missing peer、同 die 与非法 packet 边界均覆盖；
- 例：`F=7,R=1/4,L=7`，单跳事务 D2D-only `49 cycle`，两跳 `70 cycle`；多跳
  只增加 `3*L` fixed，不重复 28-cycle bulk service。

自审修正：乘法在发生前做 `LLONG_MAX/den` 检查；ceil 用商余数避免末端加法溢出；
Python/C++ 保持独立实现，防止测试只是复述生产函数。
