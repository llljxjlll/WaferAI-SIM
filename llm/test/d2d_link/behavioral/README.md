# V4 Behavioral D2D oracle

`oracle.py` 是独立于 C++ 仿真器实现的解析参考。它直接读取 hardware JSON 的 die mesh、
单方向 C2C port、rate 和 latency，按 X-first die routing 构造路径。

## 公式契约

V4 MVP 每方向只有一个端口、无 striping，因此单 flow min-cut 为：

```text
R = min(1 packet/cycle source/destination NoC cut,
        port_rate,
        link_rate)
S(F) = ceil(F / R)
Lphase = H * link_latency
```

对于 REQUEST → ACK → DATA 事务：

```text
DATA first = Lphase + S(1)
DATA last  = Lphase + S(F)
D2D-only transaction = 3 * Lphase + S(F)
```

三倍 fixed latency 来自 REQUEST、ACK、DATA 三个因果串联阶段各跨 `H` 条 link；bulk service
只由 DATA 计一次。多跳采用 end-to-end pipelined min-cut，不按 hop 重复 `S(F)`。

代表包仍实际穿过仿真器 Router，因此 oracle 输出 `intra_die_hops` 用于解释路径，但不再次
加入时间，避免 Router hop latency 重复记账。

Behavioral 不维护跨 flow 的 port/link busy state，不建模有限 FIFO、credit、backpressure、
SAF reservation 或 deadlock；shared/disjoint 的争用差异由 cycle 模型负责。

## 使用

```bash
python3 llm/test/d2d_link/behavioral/oracle.py --selftest

python3 llm/test/d2d_link/behavioral/oracle.py \
  --hardware path/to/hardware.json --source 0 --dest 32 --packets 128
```

输出为机器可读 JSON，包含：

- die/direction path；
- D2D 与片内 Router hop 数；
- effective rate；
- fixed latency、首包/尾包与完整事务的 D2D-only cycle 分解。

## V4-b 门

- Python oracle self-test：**8/8**；
- C++ route/estimate 与配置纯函数总门：**300/300**；
- 两者对固定的 2×1、3×1、2×2 解析样例给出相同路径和 cycle 值。
