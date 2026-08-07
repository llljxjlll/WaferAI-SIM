#pragma once

#include "dte/coll_latency.h"
#include "dte/coll_types.h"
#include "defs/enums.h"
#include "systemc.h"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

constexpr uint16_t COLL_REDUCE_MAGIC = 0xc5d2;
constexpr uint8_t COLL_REDUCE_VERSION = 0;
constexpr uint8_t COLL_REDUCE_PAYLOAD_SEGMENT = 1;

inline bool IsCollReduceHeaderWire(const sc_bv<256> &w) {
    return w.range(15, 0).to_uint() == COLL_REDUCE_MAGIC &&
           w.range(23, 16).to_uint() == COLL_REDUCE_VERSION &&
           w.range(39, 24).to_uint() != 0 &&
           w.range(239, 232).to_uint() == 2 &&
           !w.range(255, 240).or_reduce();
}
inline bool IsCollReducePayloadWire(const sc_bv<256> &w) {
    return w.range(143, 128).to_uint() == COLL_REDUCE_MAGIC &&
           w.range(151, 144).to_uint() == COLL_REDUCE_VERSION &&
           w.range(159, 152).to_uint() == COLL_REDUCE_PAYLOAD_SEGMENT &&
           !w.range(255, 160).or_reduce();
}
constexpr size_t COLL_REDUCE_NODES_PER_ROUTER = 64;

struct CollReduceOperand {
    uint16_t tree_id = 0;
    CollectiveKey collective;
    uint16_t phase_id = 0;
    uint32_t chunk_id = 0;
    uint16_t child_id = 0;
    CollDType dtype = CollDType::UINT8;
    CollReduceOp op = CollReduceOp::NONE;
    uint16_t valid_elements = 0;
    sc_bv<128> payload = 0;
};

inline std::vector<sc_bv<256>> SerializeCollReduceOperand(const CollReduceOperand &o) {
    if (o.tree_id == 0 || o.op == CollReduceOp::NONE ||
        o.dtype == CollDType::FP32 || o.dtype == CollDType::FP16 ||
        o.dtype == CollDType::FP8)
        throw std::invalid_argument("invalid in-network reduce operand mode");
    const uint64_t width = CollDTypeBits(o.dtype);
    if (o.valid_elements == 0 || uint64_t(o.valid_elements) * width > 128)
        throw std::invalid_argument("reduce operand valid elements exceed payload");
    std::vector<sc_bv<256>> w(2);
    w[0].range(15, 0) = COLL_REDUCE_MAGIC; w[0].range(23, 16) = COLL_REDUCE_VERSION;
    w[0].range(39, 24) = o.tree_id;
    w[0].range(71, 40) = o.collective.group_id;
    w[0].range(103, 72) = o.collective.collective_id;
    w[0].range(135, 104) = o.collective.epoch;
    w[0].range(151, 136) = o.phase_id; w[0].range(183, 152) = o.chunk_id;
    w[0].range(199, 184) = o.child_id;
    w[0].range(207, 200) = static_cast<uint8_t>(o.dtype);
    w[0].range(215, 208) = static_cast<uint8_t>(o.op);
    w[0].range(231, 216) = o.valid_elements; w[0].range(239, 232) = 2;
    w[1].range(127, 0) = o.payload;
    w[1].range(143, 128) = COLL_REDUCE_MAGIC;
    w[1].range(151, 144) = COLL_REDUCE_VERSION;
    w[1].range(159, 152) = COLL_REDUCE_PAYLOAD_SEGMENT;
    return w;
}

inline CollReduceOperand DeserializeCollReduceOperand(const std::vector<sc_bv<256>> &w) {
    if (w.size() != 2 || w[0].range(15, 0).to_uint() != COLL_REDUCE_MAGIC ||
        w[0].range(23, 16).to_uint() != COLL_REDUCE_VERSION ||
        w[0].range(239, 232).to_uint() != 2 || w[0].range(255, 240).or_reduce() ||
        !IsCollReducePayloadWire(w[1]) || w[1].range(255, 160).or_reduce())
        throw std::invalid_argument("invalid in-network reduce wire framing");
    CollReduceOperand o;
    o.tree_id = w[0].range(39, 24).to_uint();
    o.collective.group_id = w[0].range(71, 40).to_uint64();
    o.collective.collective_id = w[0].range(103, 72).to_uint64();
    o.collective.epoch = w[0].range(135, 104).to_uint64();
    o.phase_id = w[0].range(151, 136).to_uint(); o.chunk_id = w[0].range(183, 152).to_uint64();
    o.child_id = w[0].range(199, 184).to_uint();
    const unsigned dtype = w[0].range(207, 200).to_uint();
    const unsigned op = w[0].range(215, 208).to_uint();
    if (dtype > unsigned(CollDType::FP8) || op > unsigned(CollReduceOp::MAX))
        throw std::invalid_argument("in-network reduce enum out of range");
    o.dtype = static_cast<CollDType>(dtype); o.op = static_cast<CollReduceOp>(op);
    o.valid_elements = w[0].range(231, 216).to_uint(); o.payload = w[1].range(127, 0);
    (void)SerializeCollReduceOperand(o);
    return o;
}

struct CollReduceMatchKey {
    CollectiveKey collective; uint16_t phase_id = 0; uint32_t chunk_id = 0;
    bool operator<(const CollReduceMatchKey &o) const {
        return std::tie(collective, phase_id, chunk_id) < std::tie(o.collective, o.phase_id, o.chunk_id);
    }
};

inline sc_bv<128> ReduceIntegerOperands(const std::vector<sc_bv<128>> &values,
                                        CollDType dtype, CollReduceOp op,
                                        uint16_t count) {
    if (values.empty() || dtype == CollDType::FP32 ||
        dtype == CollDType::FP16 || dtype == CollDType::FP8 ||
        op == CollReduceOp::NONE)
        throw std::invalid_argument("unsupported integer reduction");
    const unsigned width = CollDTypeBits(dtype);
    if (count == 0 || uint64_t(count) * width > 128) throw std::invalid_argument("invalid reduction count");
    sc_bv<128> out = 0;
    for (unsigned e = 0; e < count; ++e) {
        uint64_t acc = values[0].range((e + 1) * width - 1, e * width).to_uint64();
        for (size_t i = 1; i < values.size(); ++i) {
            uint64_t v = values[i].range((e + 1) * width - 1, e * width).to_uint64();
            if (op == CollReduceOp::SUM) acc += v;
            else if (dtype == CollDType::UINT8) acc = std::max(acc, v);
            else {
                const int64_t sa = width == 32 ? int32_t(acc) : int64_t(acc);
                const int64_t sv = width == 32 ? int32_t(v) : int64_t(v);
                if (sv > sa) acc = v;
            }
        }
        if (width < 64) acc &= (uint64_t(1) << width) - 1;
        out.range((e + 1) * width - 1, e * width) = acc;
    }
    return out;
}

enum class CollOperandStatus { ACCEPTED, READY, BACKPRESSURE, DUPLICATE, UNEXPECTED, MISMATCH };
struct CollDcaResult { sc_bv<128> payload = 0; uint64_t service_cycles = 0; };

class CollOperandMatchBuffer {
public:
    CollOperandMatchBuffer(size_t header_capacity, size_t operand_capacity)
        : hcap_(header_capacity), ocap_(operand_capacity) {
        if (!hcap_ || !ocap_) throw std::invalid_argument("operand/header capacity must be positive");
    }
    bool Open(const CollReduceMatchKey &key, uint64_t expected_children,
              CollDType dtype, CollReduceOp op, uint16_t count) {
        if (!expected_children || states_.count(key)) throw std::invalid_argument("invalid/duplicate match open");
        if (dtype == CollDType::FP32 || dtype == CollDType::FP16 ||
            dtype == CollDType::FP8 || op == CollReduceOp::NONE || count == 0 ||
            uint64_t(count) * CollDTypeBits(dtype) > 128)
            throw std::invalid_argument("unsupported match reduction mode");
        if (states_.size() == hcap_) return false;
        states_[key] = {expected_children, 0, dtype, op, count, {}};
        return true;
    }
    CollOperandStatus Accept(const CollReduceMatchKey &key, uint16_t child,
                             CollDType dtype, CollReduceOp op, uint16_t count,
                             const sc_bv<128> &payload) {
        auto it = states_.find(key); if (it == states_.end()) return CollOperandStatus::UNEXPECTED;
        State &s = it->second; if (child >= 64 || !(s.expected & (uint64_t(1) << child))) return CollOperandStatus::UNEXPECTED;
        if (s.received & (uint64_t(1) << child)) return CollOperandStatus::DUPLICATE;
        if (s.dtype != dtype || s.op != op || s.count != count) return CollOperandStatus::MISMATCH;
        if (operands_ == ocap_) return CollOperandStatus::BACKPRESSURE;
        s.received |= uint64_t(1) << child; s.values.push_back(payload); ++operands_;
        return s.received == s.expected ? CollOperandStatus::READY : CollOperandStatus::ACCEPTED;
    }
    CollDcaResult Consume(const CollReduceMatchKey &key) {
        auto it = states_.find(key); if (it == states_.end() || it->second.received != it->second.expected)
            throw std::runtime_error("consume before operand match completion");
        State &s = it->second;
        const uint64_t reductions = static_cast<uint64_t>(__builtin_popcountll(s.expected) - 1);
        CollDcaResult r{ReduceIntegerOperands(s.values, s.dtype, s.op, s.count),
            CollDcaServiceCycles(uint64_t(s.count) * CollDTypeBits(s.dtype),
                                 uint64_t(s.count) * reductions)};
        operands_ -= s.values.size(); states_.erase(it); return r;
    }
    size_t Residual() const { return states_.size() + operands_; }
private:
    struct State { uint64_t expected, received; CollDType dtype; CollReduceOp op; uint16_t count; std::vector<sc_bv<128>> values; };
    size_t hcap_, ocap_, operands_ = 0; std::map<CollReduceMatchKey, State> states_;
};

struct CollReduceTreeNode {
    uint8_t expected_inputs = 0;
    Directions parent_output = CENTER;
};
void ResetCollectiveReduceFabric();
void ProgramCollectiveReduceNode(uint16_t tree_id, uint16_t router_id,
                                 const CollReduceTreeNode &node);
size_t EraseCollectiveReduceTree(uint16_t tree_id);
size_t CollectiveReduceNodeCount();
CollReduceTreeNode LookupCollectiveReduceNode(uint16_t tree_id,
                                              uint16_t router_id);
