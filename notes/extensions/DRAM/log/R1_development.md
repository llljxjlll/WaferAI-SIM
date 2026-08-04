# 分布式 HBM R1 开发记录

日期：2026-08-04
状态：MEM wire、CoreMemAdapter 和合成 endpoint 路径已完成

## 1. 阶段范围

R1 冻结功能性内存 transaction 协议，并在合成 SystemC testbench 中跑通共享 backing
store。请求强制经过真实 256-bit codec，但尚未进入生产 Router link/credit。

## 2. MEM wire

新增独立的 `MEM_REQ/WDATA/RDATA/RESP` tagged wire：

- 每个物理 flit 固定 256 bit，与现 Router 数据宽度一致；
- 请求携带 64-bit local byte address、command、length、source/home/stack/channel 和
  32-bit txid；
- R1 初版数据 flit 携带 16-byte payload、sequence、tail/end 和逐 byte enable；R4 为给
  每个独立 flit 增加 source/home/stack/channel 路由元数据，将 payload 调整为 14 byte；
- 支持最长 65535-byte transaction，解码严格检查 magic/version/kind/txid/sequence/
  tail/长度，不能借用只有 8-bit offset 的通用 `Msg`。

## 3. Adapter 与 endpoint

- `CoreMemAdapter` 使用 live txid 集合避免回绕碰撞，异常路径也会回收；
- logical burst 按 home/range/stack/channel/local-address 连续性自动拆包并按原顺序重组；
- 即使合成传输使用方法调用，请求和响应也先 serialize 再 deserialize，避免 codec 与
  功能路径分叉；
- `MemEndpointUnit` 使用有界 FIFO、dispatch/completion 进程、max outstanding 和响应
  event，提供 queue backpressure 与 transaction 匹配。

## 4. 测试结果

- `npusim --hbm-r1-selftest`：18/18；
- 覆盖 64-bit 最大地址 codec、40-byte 多 flit+byte-enable、单核 read-after-write、
  跨核同址共享、双 channel 隔离、跨 64-byte stripe burst、并发请求和 txid 回绕；
- 已注册为 CTest `hbm_r1_selftest`。

## 5. 已知边界

`CoreMemAdapter → MemEndpointUnit` 仍是进程内交付，不消耗真实 NoC/C2C 带宽，也没有
request/response 虚通道或 Router credit。生产 EP_MEM 路由与特殊模式迁移属于 R4；
因此 R1 不能描述为“真实 NoC 已跑通”。
