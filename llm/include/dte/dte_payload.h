#pragma once

#include "common/msg.h"
#include "macros/macros.h"
#include "prims/norm_prims.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

inline uint64_t CeilDivU64(uint64_t numerator, uint64_t denominator) {
    if (denominator == 0)
        throw std::invalid_argument("DTE division denominator must be > 0");
    return numerator / denominator + (numerator % denominator != 0);
}

inline uint64_t NanosecondsToDteCycles(uint64_t nanoseconds) {
    return CeilDivU64(nanoseconds, uint64_t(CYCLE));
}

// physical NoC 中一条 Msg 代表自身 length_；behavioral NoC 中一条代表包
// 代表 roofline_packets_ 个逻辑包，其中只有最后一个可能是短尾包。
inline uint64_t ComputeMsgLogicalPayloadBits(const Msg &msg,
                                             bool behavioral_noc) {
    if (msg.length_ <= 0 || msg.length_ > M_D_DATA)
        throw std::invalid_argument(
            "DTE Msg.length_ must be in [1, M_D_DATA] bits");
    if (!behavioral_noc)
        return uint64_t(msg.length_);
    if (msg.roofline_packets_ <= 0)
        throw std::invalid_argument(
            "DTE behavioral representative requires roofline_packets_ > 0");
    return uint64_t(msg.roofline_packets_ - 1) * uint64_t(M_D_DATA) +
           uint64_t(msg.length_);
}

// SEND_DATA 的逻辑 payload，单位 bit。不能从 behavioral NoC 的代表包
// length_ 反推，因为一个代表包可能对应多个真实网络包。
inline uint64_t ComputeSendPayloadBits(const Send_prim &prim) {
    if (prim.max_packet < 0)
        throw std::invalid_argument("DTE max_packet must be >= 0");
    if (prim.max_packet == 0) {
        if (prim.end_length != 0)
            throw std::invalid_argument(
                "DTE empty transfer requires end_length == 0");
        return 0;
    }
    if (prim.end_length <= 0 || prim.end_length > M_D_DATA)
        throw std::invalid_argument(
            "DTE end_length must be in [1, M_D_DATA] bits");

    if (prim.packet_scale <= 0 || prim.packet_scale > 255)
        throw std::invalid_argument("DTE packet_scale must be in [1,255]");
    if (prim.packets_in_last_group <= 0 ||
        prim.packets_in_last_group > prim.packet_scale)
        throw std::invalid_argument(
            "DTE packets_in_last_group must be in [1, packet_scale]");

    const uint64_t grouped_prefix = uint64_t(prim.max_packet - 1);
    if (grouped_prefix >
        (std::numeric_limits<uint64_t>::max() -
         uint64_t(prim.packets_in_last_group)) /
            uint64_t(prim.packet_scale))
        throw std::overflow_error("DTE raw packet count overflows");
    const uint64_t raw_packets =
        grouped_prefix * uint64_t(prim.packet_scale) +
        uint64_t(prim.packets_in_last_group);
    const uint64_t full_packets = raw_packets - 1;
    if (full_packets >
        (std::numeric_limits<uint64_t>::max() -
         uint64_t(prim.end_length)) /
            uint64_t(M_D_DATA))
        throw std::overflow_error("DTE SEND_DATA payload bit count overflows");

    return full_packets * uint64_t(M_D_DATA) +
           uint64_t(prim.end_length);
}

// REQUEST 只在 DTE 开启时携带严格校验后的逻辑 payload。关闭路径必须是 no-op，
// 不能让 DTE 的额外合法性约束改变 legacy workload 的功能或异常行为。
inline void AttachRequestDtePayload(Msg &request, const Send_prim &prim,
                                    bool dte_enabled) {
    if (dte_enabled)
        request.dte_payload_bits_ = ComputeSendPayloadBits(prim);
}
