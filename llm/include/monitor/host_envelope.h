#pragma once
// V0b-2B0：HOST 消息信封 + legacy backend。
// config_helper 只决定「发给哪个全局核 + 什么 Msg」，不再决定物理 HOST lane / write_buffer row。
// 物理落队交给 backend：2B0 用 LegacyHostEnqueue（die0 西边 row，逐位不变）；
// 2B1 换成 per-die HostAttachment 路由。
#include "common/msg.h"
#include <queue>
#include <vector>

struct HostEnvelope {
    int dest_global_id; // 目的全局核 id（config_helper 决定）
    Msg msg;
};

// 单消息入口：经 HOST 挂载表解析 lane（HostLaneOfCore）并校验范围后直接入队。
// 供大批量下发（如权重）**逐条生成、逐条入队**，避免先把整批信封缓存进 vector 再复制到
// write_buffer 造成的双份峰值内存（大 workload 下权重包数量可观）。
void HostEnqueue(const HostEnvelope &env, std::queue<Msg> *q);

// 批量版本：等价于对每个信封依次调用 HostEnqueue（lane 解析与校验完全一致）。
void LegacyHostEnqueue(const std::vector<HostEnvelope> &envs, std::queue<Msg> *q);
