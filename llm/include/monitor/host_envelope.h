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

// Legacy backend：把信封按 die0 西边 row（dest/GRID_X）落入 write_buffer，保持原行为。
// 仅对 die0（dest < CORES_PER_DIE）有效；多 die 路由属 2B1。
void LegacyHostEnqueue(const std::vector<HostEnvelope> &envs, std::queue<Msg> *q);
