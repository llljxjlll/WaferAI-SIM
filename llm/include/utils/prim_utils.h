#pragma once
#include "common/system.h"
#include "isa/prim_id.h"
#include "isa/prim_manifest.h"
#include "systemc.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "prims/base.h"
#include "prims/comp_prims.h"
#include "prims/gpu_prims.h"
#include "prims/norm_prims.h"
#include "utils/print_utils.h"

class PrimFactory {
public:
    using CreatorFunc = std::function<PrimBase *()>;

    static PrimFactory &getInstance() {
        static PrimFactory instance;
        return instance;
    }

    void registerPrim(const std::string &type, PrimId id,
                      CreatorFunc creator) {
        if (type.empty())
            throw std::invalid_argument("primitive name must not be empty");
        if (id == PrimId::INVALID)
            throw std::invalid_argument("PrimId::INVALID cannot be registered");
        if (!creator)
            throw std::invalid_argument("primitive creator must not be empty");
        if (creators_.count(type) != 0 || type_to_id_.count(type) != 0)
            throw std::logic_error("duplicate primitive name: " + type);

        const int raw_id = static_cast<int>(PrimIdValue(id));
        const PrimManifestEntry *manifest_entry = LookupPrim(id);
        if (manifest_entry == nullptr)
            throw std::invalid_argument(
                "unassigned PrimId cannot be registered: " +
                std::to_string(raw_id));
        if (manifest_entry->factory_name != type)
            throw std::invalid_argument(
                "primitive name/ID pair disagrees with frozen manifest: " +
                type + "/" + std::to_string(raw_id));
        if (id_to_type_.count(raw_id) != 0)
            throw std::logic_error("duplicate primitive id: " +
                                   std::to_string(raw_id));

        creators_.emplace(type, std::move(creator));
        type_to_id_.emplace(type, raw_id);
        id_to_type_.emplace(raw_id, type);
    }

    PrimBase *createPrim(const std::string &type, bool need_init = true,
                         bool track = true) {
        auto it = creators_.find(type);
        if (it != creators_.end()) {
            PrimBase *prim = it->second();
            if (need_init)
                prim->prim_context = make_shared<PrimCoreContext>();

            if (track)
                g_prim_stash.push_back(prim);
            return prim;
        }

        LOG_ERROR(prim_utils.h) << "Unregistered primitive type " << type;
        throw std::invalid_argument("unregistered primitive type " + type);
    }

    PrimBase *createPrim(int id, bool need_init = true, bool track = true) {
        if (id <= 0 || id > UINT8_MAX)
            throw std::out_of_range("primitive ID is outside the 8-bit wire");
        auto it = id_to_type_.find(id);
        if (it != id_to_type_.end()) {
            return createPrim(it->second, need_init, track);
        }

        LOG_ERROR(prim_utils.h) << "Unregistered primitive ID " << id;
        throw std::invalid_argument("unregistered primitive ID " +
                                    std::to_string(id));
    }

    int getPrimId(const std::string &type) const {
        auto it = type_to_id_.find(type);
        if (it != type_to_id_.end()) {
            return it->second;
        }
        return -1;
    }

    const std::string &getPrimType(int id) const {
        static const std::string empty_string = "";
        auto it = id_to_type_.find(id);
        if (it != id_to_type_.end()) {
            return it->second;
        }
        return empty_string;
    }

    size_t registeredCount() const { return type_to_id_.size(); }

    std::vector<int> registeredIds() const {
        std::vector<int> ids;
        ids.reserve(id_to_type_.size());
        for (const auto &entry : id_to_type_)
            ids.push_back(entry.first);
        std::sort(ids.begin(), ids.end());
        return ids;
    }

private:
    std::unordered_map<std::string, CreatorFunc> creators_;
    std::unordered_map<std::string, int> type_to_id_;
    std::unordered_map<int, std::string> id_to_type_;

    PrimFactory() = default;
};

namespace prim_wire {
inline constexpr uint32_t kTrailerMagic = 0x31575250U; // "PRW1"
inline constexpr uint8_t kTrailerVersion = 1;
inline constexpr size_t kCarryBytesPerSegment = 15;

// Legacy JSON workloads predate the strict all-segment PrimId framing. The
// compatibility switch is process-wide because a simulation selects exactly
// one loader path. Program Format keeps it disabled; InitGrid enables it.
inline bool &LegacyCompatibilityEnabled() {
    static bool enabled = false;
    return enabled;
}

inline void SetLegacyCompatibility(bool enabled) {
    LegacyCompatibilityEnabled() = enabled;
}

inline bool HasStrictTrailer(const std::vector<sc_bv<128>> &wire) {
    return wire.size() >= 3 &&
           wire.back().range(39, 8).to_uint64() == kTrailerMagic &&
           wire.back().range(47, 40).to_uint64() == kTrailerVersion;
}

inline bool IsLegacyWrappedPrimId(uint8_t id) {
    switch (static_cast<PrimId>(id)) {
    case PrimId::COLLECTIVE_DATA:
    case PrimId::COLLECTIVE:
    case PrimId::DTE_ASYNC:
    case PrimId::LSU_MEM:
    case PrimId::REDUCE_COMPUTE:
    case PrimId::SRAM_PIPELINE:
        return true;
    default:
        return false;
    }
}

inline bool IsLegacyNpuPrimId(uint8_t id) {
    return (id >= PrimIdValue(PrimId::ATTENTION_F) &&
            id <= PrimIdValue(PrimId::SWITCH_DATA)) ||
           (id >= PrimIdValue(PrimId::LOAD_EXPERT) &&
            id <= PrimIdValue(PrimId::MATMUL_FORWARD_MOE)) ||
           (id >= PrimIdValue(PrimId::ATTENTION_FORWARD_PD) &&
            id <= PrimIdValue(PrimId::ROPE_FORWARD_PD));
}

inline bool IsLegacyGpuPrimId(uint8_t id) {
    return id >= PrimIdValue(PrimId::ATTENTION_F_GPU) &&
           id <= PrimIdValue(PrimId::RESIDUAL_F_GPU);
}

inline uint8_t RequireRegisteredId(const std::string &name) {
    const int raw = PrimFactory::getInstance().getPrimId(name);
    if (raw <= 0 || raw > std::numeric_limits<uint8_t>::max())
        throw std::logic_error("primitive has no valid 8-bit PrimId: " + name);
    return static_cast<uint8_t>(raw);
}

inline std::vector<sc_bv<128>>
WrapSegments(std::vector<sc_bv<128>> legacy, const std::string &name) {
    if (legacy.size() < 2)
        throw std::invalid_argument(
            name + " multi-segment Prim wire requires at least two segments");
    if (legacy.size() > std::numeric_limits<uint16_t>::max())
        throw std::overflow_error(name + " Prim wire has too many segments");
    const uint8_t id = RequireRegisteredId(name);
    if (legacy.front().range(7, 0).to_uint64() != id)
        throw std::logic_error(name + " Prim wire header has the wrong ID");

    std::vector<uint8_t> displaced;
    displaced.reserve(legacy.size() - 1);
    for (size_t i = 1; i < legacy.size(); ++i) {
        displaced.push_back(
            static_cast<uint8_t>(legacy[i].range(7, 0).to_uint64()));
        legacy[i].range(7, 0) = sc_bv<8>(id);
    }
    const size_t carry_count =
        (displaced.size() + kCarryBytesPerSegment - 1) /
        kCarryBytesPerSegment;
    if (carry_count > std::numeric_limits<uint16_t>::max())
        throw std::overflow_error(name + " Prim wire trailer is too large");
    for (size_t base = 0; base < displaced.size();
         base += kCarryBytesPerSegment) {
        sc_bv<128> carry = 0;
        carry.range(7, 0) = sc_bv<8>(id);
        const size_t count = std::min(
            kCarryBytesPerSegment, displaced.size() - base);
        for (size_t slot = 0; slot < count; ++slot) {
            const int low = static_cast<int>(8 + slot * 8);
            carry.range(low + 7, low) = sc_bv<8>(displaced[base + slot]);
        }
        legacy.push_back(carry);
    }

    sc_bv<128> trailer = 0;
    trailer.range(7, 0) = sc_bv<8>(id);
    trailer.range(39, 8) = sc_bv<32>(kTrailerMagic);
    trailer.range(47, 40) = sc_bv<8>(kTrailerVersion);
    trailer.range(63, 48) = sc_bv<16>(legacy.size() - carry_count);
    trailer.range(79, 64) = sc_bv<16>(carry_count);
    legacy.push_back(trailer);
    return legacy;
}

inline std::vector<sc_bv<128>>
UnwrapStrictSegments(const std::vector<sc_bv<128>> &wire,
                     const std::string &name) {
    if (!HasStrictTrailer(wire))
        throw std::invalid_argument(name + " Prim wire trailer is missing");
    const uint8_t id = RequireRegisteredId(name);
    for (const auto &segment : wire)
        if (segment.range(7, 0).to_uint64() != id)
            throw std::invalid_argument(
                name + " Prim wire contains inconsistent segment IDs");

    const auto &trailer = wire.back();
    if (trailer.range(127, 80).or_reduce())
        throw std::invalid_argument(
            name + " Prim wire trailer reserved bits are set");
    const size_t original_count = trailer.range(63, 48).to_uint64();
    const size_t carry_count = trailer.range(79, 64).to_uint64();
    const size_t expected_carry = original_count < 2
        ? 0
        : (original_count - 1 + kCarryBytesPerSegment - 1) /
              kCarryBytesPerSegment;
    if (original_count < 2 || carry_count != expected_carry ||
        wire.size() != original_count + carry_count + 1)
        throw std::invalid_argument(
            name + " Prim wire trailer segment count is inconsistent");

    std::vector<sc_bv<128>> legacy(wire.begin(),
                                   wire.begin() + original_count);
    for (size_t index = 0; index < original_count - 1; ++index) {
        const auto &carry = wire[original_count +
                                 index / kCarryBytesPerSegment];
        const int low = static_cast<int>(
            8 + (index % kCarryBytesPerSegment) * 8);
        legacy[index + 1].range(7, 0) = carry.range(low + 7, low);
    }
    const size_t used_last = (original_count - 1) % kCarryBytesPerSegment;
    if (used_last != 0) {
        const auto &last_carry = wire[original_count + carry_count - 1];
        const int first_reserved = static_cast<int>(8 + used_last * 8);
        if (last_carry.range(127, first_reserved).or_reduce())
            throw std::invalid_argument(
                name + " Prim wire carry padding is non-zero");
    }
    return legacy;
}

inline std::vector<sc_bv<128>>
UnwrapSegments(const std::vector<sc_bv<128>> &wire,
               const std::string &name) {
    // The selected loader fixes one transport before SystemC starts. Legacy
    // transport is explicitly unframed, so never guess a trailer from data.
    if (LegacyCompatibilityEnabled())
        return wire;
    return UnwrapStrictSegments(wire, name);
}

inline void RequireStrictSegmentIds(const std::vector<sc_bv<128>> &wire,
                                    uint8_t id,
                                    const std::string &name) {
    if (wire.empty())
        throw std::invalid_argument(name + " Prim wire has no segments");
    for (const auto &segment : wire)
        if (segment.range(7, 0).to_uint64() != id)
            throw std::invalid_argument(
                name + " strict Prim wire contains inconsistent segment IDs");
}

inline std::vector<sc_bv<128>>
LegacySetAddrSegments(const std::vector<sc_bv<128>> &wire, uint8_t id,
                      const std::string &name) {
    constexpr size_t kStrictLabelsPerSegment = 3;
    constexpr size_t kLegacyLabelsPerSegment = 4;
    const size_t strict_inputs =
        (MAX_SPLIT_NUM + kStrictLabelsPerSegment - 1) /
        kStrictLabelsPerSegment;
    const size_t legacy_inputs =
        (MAX_SPLIT_NUM + kLegacyLabelsPerSegment - 1) /
        kLegacyLabelsPerSegment;
    if (wire.size() != 1 + strict_inputs + 1)
        throw std::invalid_argument(name + " strict segment count is invalid");
    RequireStrictSegmentIds(wire, id, name);

    std::vector<uint32_t> labels;
    labels.reserve(MAX_SPLIT_NUM);
    for (size_t index = 0; index < MAX_SPLIT_NUM; ++index) {
        const size_t segment = 1 + index / kStrictLabelsPerSegment;
        const int low = static_cast<int>(
            8 + (index % kStrictLabelsPerSegment) * 32);
        labels.push_back(static_cast<uint32_t>(
            wire[segment].range(low + 31, low).to_uint64()));
    }

    std::vector<sc_bv<128>> legacy(1 + legacy_inputs + 1);
    for (auto &segment : legacy)
        segment = 0;
    legacy[0] = wire[0];
    for (size_t index = 0; index < labels.size(); ++index) {
        const size_t segment = 1 + index / kLegacyLabelsPerSegment;
        const int low = static_cast<int>(
            (index % kLegacyLabelsPerSegment) * 32);
        legacy[segment].range(low + 31, low) = sc_bv<32>(labels[index]);
    }
    legacy.back().range(31, 0) = wire.back().range(39, 8);
    return legacy;
}

inline std::vector<sc_bv<128>>
LegacySetBatchSegments(const std::vector<sc_bv<128>> &wire, uint8_t id,
                       const std::string &name) {
    constexpr size_t kStagesPerSegment = 5;
    if (wire.empty())
        throw std::invalid_argument(name + " Prim wire has no segments");
    const size_t batch_size = wire[0].range(23, 8).to_uint64();
    const size_t expected =
        1 + (batch_size + kStagesPerSegment - 1) / kStagesPerSegment;
    if (wire.size() != expected)
        throw std::invalid_argument(name + " strict segment count is invalid");
    RequireStrictSegmentIds(wire, id, name);

    std::vector<sc_bv<128>> legacy(wire.size());
    for (auto &segment : legacy)
        segment = 0;
    legacy[0] = wire[0];
    for (size_t index = 0; index < batch_size; ++index) {
        const size_t segment = 1 + index / kStagesPerSegment;
        const int strict_low = static_cast<int>(
            8 + (index % kStagesPerSegment) * 22);
        const int legacy_low = static_cast<int>(
            (index % kStagesPerSegment) * 22);
        legacy[segment].range(legacy_low + 21, legacy_low) =
            wire[segment].range(strict_low + 21, strict_low);
    }
    return legacy;
}

inline std::vector<sc_bv<128>>
LegacyNpuSegments(const std::vector<sc_bv<128>> &wire, uint8_t id,
                  const std::string &name) {
    RequireStrictSegmentIds(wire, id, name);
    std::vector<sc_bv<128>> legacy = wire;
    sc_bv<128> metadata = 0;
    metadata.range(7, 0) = sc_bv<8>(id);
    metadata.range(8, 8) = wire[0].range(8, 8);
    metadata.range(24, 9) = wire[0].range(25, 10);
    metadata.range(40, 25) = wire[0].range(57, 42);
    metadata.range(56, 41) = wire[0].range(89, 74);
    legacy[0] = metadata;
    return legacy;
}

inline std::vector<sc_bv<128>>
LegacyGpuSegments(const std::vector<sc_bv<128>> &wire, uint8_t id,
                  const std::string &name) {
    RequireStrictSegmentIds(wire, id, name);
    std::vector<sc_bv<128>> legacy = wire;
    sc_bv<128> metadata = 0;
    metadata.range(7, 0) = sc_bv<8>(id);
    metadata.range(8, 8) = wire[0].range(8, 8);
    metadata.range(24, 9) = wire[0].range(25, 10);
    metadata.range(56, 41) = wire[0].range(57, 42);
    legacy[0] = metadata;
    return legacy;
}

inline std::vector<sc_bv<128>>
LegacyTransportSegments(const std::vector<sc_bv<128>> &wire,
                        const std::string &name) {
    const uint8_t id = RequireRegisteredId(name);
    if (id == PrimIdValue(PrimId::SRAM_BIND_ONESHOT) ||
        id == PrimIdValue(PrimId::SRAM_LIFECYCLE) ||
        id == PrimIdValue(PrimId::GROUP_SYNC) ||
        id == PrimIdValue(PrimId::EVENT_CONTROL) ||
        id == PrimIdValue(PrimId::DTE_SEND_ENDPOINT) ||
        id == PrimIdValue(PrimId::DTE_RECV_ENDPOINT) ||
        id == PrimIdValue(PrimId::COLLECTIVE_DATA_V1) ||
        id == PrimIdValue(PrimId::COLLECTIVE_PHASE_BARRIER_V1) ||
        id == PrimIdValue(PrimId::COLLECTIVE_LAUNCH_V1))
        throw std::invalid_argument(
            name + " is strict-only and cannot use legacy transport");
    if (IsLegacyWrappedPrimId(id))
        return UnwrapStrictSegments(wire, name);
    if (id == PrimIdValue(PrimId::SET_ADDR))
        return LegacySetAddrSegments(wire, id, name);
    if (id == PrimIdValue(PrimId::SET_BATCH))
        return LegacySetBatchSegments(wire, id, name);
    if (IsLegacyNpuPrimId(id))
        return LegacyNpuSegments(wire, id, name);
    if (IsLegacyGpuPrimId(id))
        return LegacyGpuSegments(wire, id, name);
    return wire;
}

inline std::vector<sc_bv<128>> LegacyTransportSegments(PrimBase *prim) {
    if (prim == nullptr)
        throw std::invalid_argument(
            "cannot serialize a null Prim for legacy transport");
    return LegacyTransportSegments(prim->serialize(), prim->name);
}
} // namespace prim_wire

// 所有原语的注册函数
#define REGISTER_PRIM(prim_type, prim_id)                                      \
    static bool registered_##prim_type = []() {                                \
        PrimFactory::getInstance().registerPrim(                               \
            prim_type().name, prim_id, []() { return new prim_type(); });       \
        return true;                                                           \
    }();
