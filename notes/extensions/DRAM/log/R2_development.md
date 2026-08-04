# 分布式 HBM R2 开发记录

日期：2026-08-04
状态：共享队列与 behavioral HBM 核心已完成

## 1. 阶段范围

R2 在统一 `HBMBackend` 入口后实现快速行为模型，目标是正确表达同一 channel 的共享
竞争、带宽上限、固定延迟和读写切换开销。真实 NoC 瓶颈不在合成链路中伪造。

## 2. Behavioral backend

- byte-address backing store 支持重叠和非对齐访问；写入遵守逐 byte enable；
- 同一 backend 通过 `next_available` 串行化服务，时间为
  `base_latency + bytes/(bandwidth_GBps×efficiency)`；
- read→write、write→read turnaround 独立配置；
- 导出 requests、reads、writes、bytes、completed、failed 和 service time。

## 3. Endpoint 队列与统计

`MemEndpointUnit` 将队列等待与 backend service 分开，支持有限 queue depth 和有限
outstanding，导出 requests/bytes/completed、queue stalls、queue wait、response time。
多个 core 绑定同一 endpoint/backend 时共享同一 backing store 和同一 channel 带宽。

## 4. 测试结果

- `npusim --hbm-r2-selftest`：18/18；
- 单 channel 饱和吞吐等于配置有效带宽；四核共享不会得到四倍带宽且无永久饥饿；
- 两个独立 channel 获得约 2 倍扩展；queue depth 不改变 backend 本身的 service time；
- 覆盖长短 burst、读写混合、部分/重叠写、两个方向 turnaround 及统计口径；
- 已注册为 CTest `hbm_r2_selftest`。

## 5. 已知边界

尚未接生产 Router，因而没有 on-die NoC useful-bandwidth、link stalls 或真实 NoC 瓶颈
数据；这些指标只能在 R4 接入后测量。
