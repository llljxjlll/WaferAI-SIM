#pragma once

#include "dte/coll_types.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

enum class CollActionKind : uint8_t {
    SEND = 0, RECV = 1, BARRIER = 2, REDUCE_COMPUTE = 3
};
struct CollAction {
    CollActionKind kind = CollActionKind::BARRIER;
    uint16_t phase_id = 0;
    uint16_t peer_rank = 0;
    uint64_t payload_bits = 0;
    uint64_t offset_bits = 0;
};

inline std::vector<CollAction> PlanTier0Collective(const CollDescriptor &d,
                                                   uint16_t self_rank) {
    ValidateCollDescriptor(d);
    if (self_rank >= d.group.size()) throw std::invalid_argument("planner self rank outside group");
    const uint16_t n = static_cast<uint16_t>(d.group.size());
    std::vector<CollAction> out;
    auto barrier = [&](uint16_t phase) {
        out.push_back({CollActionKind::BARRIER, phase, 0, 0, 0});
    };
    auto pair = [&](uint16_t phase, uint16_t src, uint16_t dst,
                    uint64_t bits, uint64_t offset) {
        if (self_rank == src) out.push_back({CollActionKind::SEND, phase, dst, bits, offset});
        if (self_rank == dst) out.push_back({CollActionKind::RECV, phase, src, bits, offset});
    };

    if (CollIsReduction(d.op)) {
        ValidateTier0ReductionDescriptor(d);
        for (uint16_t src = 0; src < n; ++src) {
            if (src != d.root_rank)
                pair(src, src, d.root_rank, d.chunk_bits, 0);
            barrier(src);
        }
        if (self_rank == d.root_rank)
            out.push_back({CollActionKind::REDUCE_COMPUTE, n, 0, 0, 0});
        barrier(n);

        if (d.op == CollOp::REDUCESCATTER) {
            for (uint16_t dst = 0; dst < n; ++dst) if (dst != d.root_rank) {
                const auto part = CollRankCountOffset(d.count, n, dst);
                pair(n + 1, d.root_rank, dst,
                     part.first * CollDTypeBits(d.dtype),
                     part.second * CollDTypeBits(d.dtype));
            }
            barrier(n + 1);
        } else if (d.op == CollOp::ALLREDUCE) {
            for (uint16_t dst = 0; dst < n; ++dst) if (dst != d.root_rank)
                pair(n + 1, d.root_rank, dst, d.chunk_bits, 0);
            barrier(n + 1);
        }
        return out;
    }

    if (d.op == CollOp::P2P) {
        if (n != 2) throw std::invalid_argument("P2P requires exactly two ranks");
        pair(0, 0, 1, d.chunk_bits, 0);
        barrier(0);
        return out;
    }
    if (d.op == CollOp::BROADCAST || d.op == CollOp::SCATTER) {
        for (uint16_t dst = 0; dst < n; ++dst) if (dst != d.root_rank) {
            uint64_t bits = d.chunk_bits, offset = 0;
            if (d.op == CollOp::SCATTER) {
                bits = d.chunk_bits;
                offset = uint64_t(dst) * d.stride_bits;
            }
            pair(0, d.root_rank, dst, bits, offset);
        }
        barrier(0);
        return out;
    }
    if (d.op == CollOp::GATHER) {
        for (uint16_t src = 0; src < n; ++src) {
            if (src != d.root_rank) pair(src, src, d.root_rank, d.chunk_bits,
                                         uint64_t(src) * d.stride_bits);
            barrier(src);
        }
        return out;
    }
    if (d.op == CollOp::ALLGATHER || d.op == CollOp::ALLTOALL) {
        for (uint16_t src = 0; src < n; ++src) {
            for (uint16_t dst = 0; dst < n; ++dst) if (dst != src) {
                uint64_t bits = d.chunk_bits;
                uint64_t offset = uint64_t(src) * d.stride_bits;
                if (d.op == CollOp::ALLTOALL) {
                    bits = d.chunk_bits;
                    offset = uint64_t(dst) * d.stride_bits;
                }
                pair(src, src, dst, bits, offset);
            }
            barrier(src);
        }
        return out;
    }
    throw std::invalid_argument("V1 planner does not implement reduction collectives");
}
