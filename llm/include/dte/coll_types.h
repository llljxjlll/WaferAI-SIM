#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

enum class CollTxKind : uint8_t { UNICAST = 0, SCATTER = 1, BROADCAST = 2 };
enum class CollRxKind : uint8_t { UNICAST = 0, GATHER = 1, REDUCE = 2 };
enum class CollOp : uint8_t {
    P2P = 0, SCATTER = 1, GATHER = 2, BROADCAST = 3,
    ALLTOALL = 4, ALLGATHER = 5, REDUCE = 6,
    REDUCESCATTER = 7, ALLREDUCE = 8
};
enum class CollAlgorithm : uint8_t { DIRECT = 0, REDUCE_ROOT_SCATTER = 1, REDUCE_ROOT_BROADCAST = 2 };
enum class CollReduceOp : uint8_t { NONE = 0, SUM = 1, MAX = 2 };
enum class CollDType : uint8_t {
    UINT8 = 0, INT32 = 1, INT64 = 2, FP32 = 3, FP16 = 4, FP8 = 5
};

constexpr uint16_t COLL_TAG_BASE = 0x8000u;
constexpr uint16_t COLL_TAG_MAX = 0xfffeu;

struct CollectiveKey {
    uint32_t group_id = 0;
    uint32_t collective_id = 0;
    uint32_t epoch = 0;
    bool operator==(const CollectiveKey &o) const {
        return std::tie(group_id, collective_id, epoch) ==
               std::tie(o.group_id, o.collective_id, o.epoch);
    }
    bool operator<(const CollectiveKey &o) const {
        return std::tie(group_id, collective_id, epoch) <
               std::tie(o.group_id, o.collective_id, o.epoch);
    }
};

struct PacketKey {
    CollectiveKey collective;
    uint16_t phase_id = 0;
    uint32_t chunk_id = 0;
    uint16_t src_rank = 0;
    uint16_t dst_rank = 0;
    bool operator==(const PacketKey &o) const {
        return collective == o.collective && phase_id == o.phase_id &&
               chunk_id == o.chunk_id && src_rank == o.src_rank &&
               dst_rank == o.dst_rank;
    }
};

struct CollDescriptor {
    CollOp op = CollOp::P2P;
    CollAlgorithm algorithm = CollAlgorithm::DIRECT;
    CollDType dtype = CollDType::UINT8;
    CollReduceOp reduce_op = CollReduceOp::NONE;
    CollectiveKey key;
    std::vector<uint16_t> group;
    uint16_t root_rank = 0;
    uint16_t self_rank = 0;
    uint64_t count = 0;
    uint64_t src_addr = 0;
    uint64_t dst_addr = 0;
    uint64_t chunk_bits = 0;
    uint64_t stride_bits = 0;
    uint32_t gather_reorder_depth = 0; // 0 means ideal/unbounded.
    // Simulation-only endpoint vector load used to exercise the R5 shared
    // tile arbiter. It is carried by Collective_data_prim, not the frozen
    // base descriptor wire.
    uint32_t core_contention_beats = 0;
};

inline bool CollIsReduction(CollOp op) {
    return op == CollOp::REDUCE || op == CollOp::REDUCESCATTER ||
           op == CollOp::ALLREDUCE;
}

inline void ValidateCollDescriptor(const CollDescriptor &d);

inline uint64_t CollDTypeBits(CollDType dtype) {
    switch (dtype) {
    case CollDType::UINT8: return 8;
    case CollDType::INT32: return 32;
    case CollDType::INT64: return 64;
    case CollDType::FP32: return 32;
    case CollDType::FP16: return 16;
    case CollDType::FP8: return 8;
    }
    throw std::invalid_argument("unknown collective dtype");
}

inline void ValidateTier0ReductionDescriptor(const CollDescriptor &d) {
    ValidateCollDescriptor(d);
    if (!CollIsReduction(d.op))
        throw std::invalid_argument("Tier0 reduction validator requires reduction op");
    if (d.dtype == CollDType::FP32 || d.dtype == CollDType::FP16 ||
        d.dtype == CollDType::FP8)
        throw std::invalid_argument(
            "V3 Tier0 reduction does not support floating-point semantics");
    const uint64_t bits = CollDTypeBits(d.dtype);
    if (d.count > UINT64_MAX / bits || d.chunk_bits != d.count * bits)
        throw std::invalid_argument(
            "V3 reduction chunk_bits must equal count*dtype_bits");
    const uint64_t align_bytes = bits / 8;
    if (d.src_addr % align_bytes != 0 || d.dst_addr % align_bytes != 0)
        throw std::invalid_argument("V3 reduction address is not dtype aligned");
}

inline uint64_t CollTier0ReduceComputeCycles(const CollDescriptor &d,
                                             uint64_t lanes = 128) {
    ValidateTier0ReductionDescriptor(d);
    if (lanes == 0)
        throw std::invalid_argument("Tier0 reduction lanes must be positive");
    const uint64_t peers = d.group.size() - 1;
    if (peers != 0 && d.count > UINT64_MAX / peers)
        throw std::overflow_error("Tier0 reduction operation count overflows");
    const uint64_t operations = d.count * peers;
    return operations / lanes + (operations % lanes != 0);
}

inline void ValidateCollDescriptor(const CollDescriptor &d) {
    if (d.group.empty()) throw std::invalid_argument("collective group must not be empty");
    if (!std::is_sorted(d.group.begin(), d.group.end()))
        throw std::invalid_argument("collective group must be sorted");
    if (std::adjacent_find(d.group.begin(), d.group.end()) != d.group.end())
        throw std::invalid_argument("collective group contains duplicate core IDs");
    if (d.root_rank >= d.group.size() || d.self_rank >= d.group.size())
        throw std::invalid_argument("collective rank is outside group");
    if (d.count == 0) throw std::invalid_argument("collective count must be positive");
    if (d.chunk_bits == 0) throw std::invalid_argument("collective chunk_bits must be positive");
    if (CollIsReduction(d.op) && d.reduce_op == CollReduceOp::NONE)
        throw std::invalid_argument("reduction collective requires reduce_op");
    if (!CollIsReduction(d.op) && d.reduce_op != CollReduceOp::NONE)
        throw std::invalid_argument("non-reduction collective forbids reduce_op");
    if (d.op == CollOp::REDUCESCATTER && d.algorithm != CollAlgorithm::REDUCE_ROOT_SCATTER)
        throw std::invalid_argument("V0 ReduceScatter algorithm is fixed");
    if (d.op == CollOp::ALLREDUCE && d.algorithm != CollAlgorithm::REDUCE_ROOT_BROADCAST)
        throw std::invalid_argument("V0 AllReduce algorithm is fixed");
    if (d.op != CollOp::REDUCESCATTER && d.op != CollOp::ALLREDUCE &&
        d.algorithm != CollAlgorithm::DIRECT)
        throw std::invalid_argument("V0 operation requires direct algorithm");
}

inline std::pair<uint64_t, uint64_t> CollRankCountOffset(uint64_t count,
                                                         size_t ranks,
                                                         size_t rank) {
    if (ranks == 0 || rank >= ranks) throw std::invalid_argument("invalid collective rank partition");
    const uint64_t base = count / ranks;
    const uint64_t extra = count % ranks;
    const uint64_t local = base + (rank < extra ? 1 : 0);
    const uint64_t offset = rank * base + std::min<uint64_t>(rank, extra);
    return {local, offset};
}
