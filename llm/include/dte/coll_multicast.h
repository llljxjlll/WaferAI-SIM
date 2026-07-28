#pragma once

#include "dte/coll_types.h"
#include "defs/enums.h"
#include "systemc.h"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

constexpr uint16_t COLL_DATA_MAGIC = 0xc4d1;
constexpr uint8_t COLL_DATA_VERSION = 0;
constexpr uint8_t COLL_OUTPUT_MASK = (1u << DIRECTIONS) - 1u;
constexpr size_t COLL_TREE_ENTRIES_PER_ROUTER = 64;

inline bool IsCollDataWire(const sc_bv<256> &w) {
    return w.range(15, 0).to_uint() == COLL_DATA_MAGIC &&
           w.range(23, 16).to_uint() == COLL_DATA_VERSION &&
           w.range(39, 24).to_uint() != 0 &&
           w.range(247, 240).to_uint() != 0 &&
           !w.range(255, 249).or_reduce();
}

struct CollDataHeader {
    uint16_t tree_id = 0;
    PacketKey packet;
    uint32_t seq_id = 0;
    uint8_t length_bits = 0;
    bool is_end = false;
};

inline sc_bv<256> SerializeCollData(const CollDataHeader &h) {
    if (h.tree_id == 0) throw std::invalid_argument("COLL_DATA tree_id 0 is reserved");
    if (h.length_bits == 0 || h.length_bits > 128)
        throw std::invalid_argument("COLL_DATA length_bits must be in [1,128]");
    if (h.seq_id > 0xffffffu)
        throw std::overflow_error("COLL_DATA seq_id exceeds 24-bit wire");
    sc_bv<256> w = 0;
    w.range(15, 0) = COLL_DATA_MAGIC;
    w.range(23, 16) = COLL_DATA_VERSION;
    w.range(39, 24) = h.tree_id;
    w.range(71, 40) = h.packet.collective.group_id;
    w.range(103, 72) = h.packet.collective.collective_id;
    w.range(135, 104) = h.packet.collective.epoch;
    w.range(151, 136) = h.packet.phase_id;
    w.range(183, 152) = h.packet.chunk_id;
    w.range(199, 184) = h.packet.src_rank;
    w.range(215, 200) = h.packet.dst_rank;
    w.range(239, 216) = h.seq_id;
    w.range(247, 240) = h.length_bits;
    w[248] = h.is_end;
    return w;
}

inline CollDataHeader DeserializeCollData(const sc_bv<256> &w) {
    if (w.range(15, 0).to_uint() != COLL_DATA_MAGIC)
        throw std::invalid_argument("invalid COLL_DATA magic");
    if (w.range(23, 16).to_uint() != COLL_DATA_VERSION)
        throw std::invalid_argument("unsupported COLL_DATA version");
    if (w.range(255, 249).or_reduce())
        throw std::invalid_argument("COLL_DATA reserved bits are non-zero");
    CollDataHeader h;
    h.tree_id = w.range(39, 24).to_uint();
    h.packet.collective.group_id = w.range(71, 40).to_uint64();
    h.packet.collective.collective_id = w.range(103, 72).to_uint64();
    h.packet.collective.epoch = w.range(135, 104).to_uint64();
    h.packet.phase_id = w.range(151, 136).to_uint();
    h.packet.chunk_id = w.range(183, 152).to_uint64();
    h.packet.src_rank = w.range(199, 184).to_uint();
    h.packet.dst_rank = w.range(215, 200).to_uint();
    h.seq_id = w.range(239, 216).to_uint64();
    h.length_bits = w.range(247, 240).to_uint();
    h.is_end = w[248].to_bool();
    if (h.tree_id == 0 || h.length_bits == 0 || h.length_bits > 128)
        throw std::invalid_argument("invalid COLL_DATA field range");
    return h;
}

struct CollectiveTreeKey {
    uint16_t tree_id = 0;
    uint16_t router_id = 0;
    uint8_t ingress = 0;
    bool operator<(const CollectiveTreeKey &o) const {
        return std::tie(tree_id, router_id, ingress) <
               std::tie(o.tree_id, o.router_id, o.ingress);
    }
};

class CollectiveTreeTable {
public:
    explicit CollectiveTreeTable(size_t capacity) : capacity_(capacity) {
        if (capacity == 0) throw std::invalid_argument("tree table capacity must be positive");
    }
    void Program(const CollectiveTreeKey &key, uint8_t outputs) {
        if (key.tree_id == 0 || key.ingress >= DIRECTIONS)
            throw std::invalid_argument("invalid collective tree key");
        if (outputs == 0 || (outputs & ~COLL_OUTPUT_MASK) != 0 ||
            (outputs & (1u << key.ingress)) != 0)
            throw std::invalid_argument("invalid collective tree output bitmap");
        auto it = entries_.find(key);
        if (it != entries_.end()) {
            if (it->second != outputs)
                throw std::runtime_error("conflicting collective tree programming");
            return;
        }
        if (entries_.size() == capacity_)
            throw std::runtime_error("collective tree table capacity exhausted");
        entries_.emplace(key, outputs);
    }
    uint8_t Lookup(const CollectiveTreeKey &key) const {
        auto it = entries_.find(key);
        if (it == entries_.end()) throw std::runtime_error("collective tree entry missing");
        return it->second;
    }
    size_t EraseTree(uint16_t tree_id) {
        size_t erased = 0;
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->first.tree_id == tree_id) { it = entries_.erase(it); ++erased; }
            else ++it;
        }
        return erased;
    }
    size_t Size() const { return entries_.size(); }
private:
    size_t capacity_;
    std::map<CollectiveTreeKey, uint8_t> entries_;
};

struct CollBranchLockKey {
    uint16_t tree_id = 0;
    CollectiveKey collective;
    uint16_t phase_id = 0;
    uint32_t chunk_id = 0;
    bool operator==(const CollBranchLockKey &o) const {
        return tree_id == o.tree_id && collective == o.collective &&
               phase_id == o.phase_id && chunk_id == o.chunk_id;
    }
};

class AtomicMulticastFork {
public:
    bool CanCommit(uint8_t outputs, const bool available[DIRECTIONS],
                   const CollBranchLockKey &key) const {
        for (int d = 0; d < DIRECTIONS; ++d) if (outputs & (1u << d))
            if (!available[d] || (locked_[d] && !(keys_[d] == key))) return false;
        return outputs != 0;
    }
    void Commit(uint8_t outputs, bool first, bool tail,
                const CollBranchLockKey &key) {
        for (int d = 0; d < DIRECTIONS; ++d) if (outputs & (1u << d)) {
            if (first) { locked_[d] = true; keys_[d] = key; ++refs_[d]; }
            if (tail) {
                if (!locked_[d] || refs_[d] == 0 || !(keys_[d] == key))
                    throw std::runtime_error("multicast branch tail without matching lock");
                if (--refs_[d] == 0) locked_[d] = false;
            }
        }
    }
    size_t Residual() const {
        size_t r = 0; for (int d = 0; d < DIRECTIONS; ++d) r += refs_[d]; return r;
    }
private:
    bool locked_[DIRECTIONS] = {};
    unsigned refs_[DIRECTIONS] = {};
    CollBranchLockKey keys_[DIRECTIONS] = {};
};

// Production collective fabric registry. It is programmed once while the
// workload is expanded, then read by RouterUnit instances during simulation.
void ResetCollectiveFabric();
void ProgramCollectiveTreeEntry(const CollectiveTreeKey &key, uint8_t outputs);
size_t EraseCollectiveTree(uint16_t tree_id);
uint8_t LookupCollectiveTreeEntry(const CollectiveTreeKey &key);
size_t CollectiveTreeEntryCount();
void ValidateCollectiveTree(uint16_t tree_id, uint16_t root,
                            const std::vector<uint16_t> &targets);

struct CollFabricLinkStat {
    uint16_t tree_id = 0;
    uint16_t router_id = 0;
    uint8_t output = 0;
    uint64_t committed_flits = 0;
    uint64_t stalled_attempts = 0;
};
void RecordCollectiveForkAttempt(uint16_t tree_id, uint16_t router_id,
                                 uint8_t outputs, bool committed);
std::vector<CollFabricLinkStat> CollectiveFabricLinkStats();

struct CollSharedLinkStat {
    uint16_t router_id = 0;
    uint8_t output = 0;
    uint64_t normal_flits = 0;
    uint64_t collective_flits = 0;
};
void RecordCollectiveSharedOutput(uint16_t router_id, uint8_t output,
                                  bool collective);
std::vector<CollSharedLinkStat> CollectiveSharedLinkStats();
