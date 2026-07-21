#include "monitor/host_envelope.h"
#include "defs/spec.h"
#include "die/port.h"
#include <stdexcept>
#include <string>

void LegacyHostEnqueue(const std::vector<HostEnvelope> &envs,
                       std::queue<Msg> *q) {
    // 经 HOST 挂载表映射到 lane（legacy: dest/GRID_X = 全局行，时序等价）。
    // 入队前显式校验 lane 合法，非法 dest 抛异常而非静默写 q[-1]。
    for (const auto &env : envs) {
        int lane = HostLaneOfCore(env.dest_global_id);
        if (lane < 0 || lane >= HOST_LANES)
            throw std::runtime_error(
                "HostEnqueue: dest_global_id " +
                std::to_string(env.dest_global_id) +
                " has no valid HOST lane");
        q[lane].push(env.msg);
    }
}
