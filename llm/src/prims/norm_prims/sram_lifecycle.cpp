#include "prims/sram_lifecycle_prim.h"

#include "common/memory.h"
#include "memory/sram/sram_access_unit.h"
#include "memory/sram/sram_region.h"
#include "utils/memory_utils.h"
#include "utils/prim_utils.h"

#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

REGISTER_PRIM(Sram_lifecycle, PrimId::SRAM_LIFECYCLE);

namespace {
constexpr size_t kLegacyWireSegments = 4;
constexpr size_t kAllocAtWireSegments = 5;
constexpr size_t kRegionNameMaxBytes = 64;
constexpr size_t kLabelMaxBytes = 255;

uint8_t ExpectedId() {
    const int registered =
        PrimFactory::getInstance().getPrimId("Sram_lifecycle");
    if (registered != static_cast<int>(
                          PrimIdValue(PrimId::SRAM_LIFECYCLE)))
        throw std::logic_error(
            "Sram_lifecycle factory ID does not match PrimId 53");
    return static_cast<uint8_t>(registered);
}

void RequireStrictTransport() {
    if (prim_wire::LegacyCompatibilityEnabled())
        throw std::invalid_argument(
            "Sram_lifecycle is strict-only and rejects legacy transport");
}

bool IsPowerOfTwo(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

bool IsUnset(const std::string &value) {
    return value.empty() || value == UNSET_LABEL;
}

void ValidateText(const std::string &value, size_t max_bytes,
                  const char *what) {
    if (value.size() > max_bytes)
        throw std::invalid_argument(
            std::string("Sram_lifecycle ") + what + " is too long");
}

void RequireMetadataOnly(const Sram_lifecycle &prim,
                         bool allow_new_label = false) {
    if (!prim.region_name.empty() ||
        (!allow_new_label && !prim.new_label.empty()) ||
        prim.region_offset_bytes != 0 ||
        prim.alignment_bytes != 0 ||
        prim.lifetime != sram::AllocationLifetime::kTask || prim.spillable)
        throw std::invalid_argument(
            "Sram_lifecycle inactive fields must be canonical zero");
}

void ValidatePrim(const Sram_lifecycle &prim) {
    const auto raw_op = static_cast<uint8_t>(prim.op);
    if (raw_op > static_cast<uint8_t>(
                     SramLifecycleOp::ALLOC_AT))
        throw std::invalid_argument("Sram_lifecycle operation is invalid");
    if (static_cast<uint8_t>(prim.lifetime) >
        static_cast<uint8_t>(sram::AllocationLifetime::kPersistent))
        throw std::invalid_argument(
            "Sram_lifecycle allocation lifetime is invalid");
    ValidateText(prim.region_name, kRegionNameMaxBytes, "region name");
    ValidateText(prim.label, kLabelMaxBytes, "label");
    ValidateText(prim.new_label, kLabelMaxBytes, "new label");

    switch (prim.op) {
    case SramLifecycleOp::ALLOC:
        if (IsUnset(prim.region_name) || IsUnset(prim.label) ||
            !prim.new_label.empty() || prim.region_offset_bytes != 0 ||
            prim.size_bytes == 0 ||
            !IsPowerOfTwo(prim.alignment_bytes))
            throw std::invalid_argument(
                "Sram_lifecycle ALLOC fields are invalid");
        return;
    case SramLifecycleOp::ALLOC_AT:
        if (IsUnset(prim.region_name) || IsUnset(prim.label) ||
            !prim.new_label.empty() || prim.size_bytes == 0 ||
            !IsPowerOfTwo(prim.alignment_bytes))
            throw std::invalid_argument(
                "Sram_lifecycle ALLOC_AT fields are invalid");
        return;
    case SramLifecycleOp::FREE:
        RequireMetadataOnly(prim);
        if (IsUnset(prim.label) || prim.size_bytes != 0)
            throw std::invalid_argument(
                "Sram_lifecycle FREE fields are invalid");
        return;
    case SramLifecycleOp::RESIZE:
        RequireMetadataOnly(prim);
        if (IsUnset(prim.label) || prim.size_bytes == 0)
            throw std::invalid_argument(
                "Sram_lifecycle RESIZE fields are invalid");
        return;
    case SramLifecycleOp::RENAME:
        RequireMetadataOnly(prim, true);
        if (IsUnset(prim.label) || IsUnset(prim.new_label) ||
            prim.label == prim.new_label || prim.size_bytes != 0)
            throw std::invalid_argument(
                "Sram_lifecycle RENAME fields are invalid");
        return;
    case SramLifecycleOp::CLEAR_TARGETED:
        RequireMetadataOnly(prim);
        if (IsUnset(prim.label) || prim.size_bytes != 0)
            throw std::invalid_argument(
                "Sram_lifecycle CLEAR_TARGETED fields are invalid");
        return;
    }
    throw std::invalid_argument("Sram_lifecycle operation is invalid");
}

uint32_t AddLabel(const std::string &label) {
    if (label.empty()) return 0;
    const int raw = g_addr_label_table.addRecord(label);
    if (raw <= 0)
        throw std::overflow_error(
            "Sram_lifecycle label table ID is invalid");
    return static_cast<uint32_t>(raw);
}

std::string ReadOptionalLabel(uint64_t raw) {
    if (raw == 0) return {};
    if (raw > g_addr_label_table.table.size())
        throw std::invalid_argument(
            "Sram_lifecycle Prim wire label ID is unknown");
    return g_addr_label_table.findRecord(static_cast<int>(raw));
}

void ValidateWireIdentity(const vector<sc_bv<128>> &segments) {
    if (segments.size() != kLegacyWireSegments &&
        segments.size() != kAllocAtWireSegments)
        throw std::invalid_argument(
            "Sram_lifecycle Prim wire segment count mismatch");
    const uint8_t id = ExpectedId();
    for (const auto &segment : segments) {
        if (segment.range(7, 0).to_uint64() != id)
            throw std::invalid_argument(
                "Sram_lifecycle Prim wire has inconsistent segment IDs");
    }
}

SramPosLocator &RequireLocator(Sram_lifecycle &prim) {
    if (prim.prim_context == nullptr)
        throw std::logic_error(
            "Sram_lifecycle requires a PrimCoreContext");
    if (prim.prim_context->sram_pos_locator_ == nullptr)
        throw std::runtime_error(
            "Sram_lifecycle requires an SRAM label table");
    return *prim.prim_context->sram_pos_locator_;
}

AddrPosKey &FindManagedKey(SramPosLocator &locator,
                           const std::string &label) {
    const auto found = locator.data_map.find(label);
    if (found == locator.data_map.end())
        throw std::out_of_range("unknown SRAM label: " + label);
    if (found->second.region_allocation_id == 0)
        throw std::invalid_argument(
            "SRAM lifecycle requires a region-managed label");
    return found->second;
}

bool PendingReferences(const PrimCoreContext &context,
                       const std::string &label) {
    if (!context.sram_bind_pending_) return false;
    for (int index = 0; index < MAX_SPLIT_NUM; ++index) {
        if (context.sram_bind_pending_labels_.indata[index] == label)
            return true;
    }
    return context.sram_bind_pending_labels_.outdata == label;
}

void RejectPendingReference(const PrimCoreContext &context,
                            const std::string &label) {
    if (PendingReferences(context, label))
        throw std::logic_error(
            "SRAM lifecycle cannot mutate a pending SRAM_BIND label");
}

AddrDatapassLabel RenamedPendingLabels(const PrimCoreContext &context,
                                       const std::string &old_label,
                                       const std::string &new_label) {
    AddrDatapassLabel result = context.sram_bind_pending_labels_;
    if (!context.sram_bind_pending_) return result;
    for (int index = 0; index < MAX_SPLIT_NUM; ++index) {
        if (result.indata[index] == old_label)
            result.indata[index] = new_label;
    }
    if (result.outdata == old_label) result.outdata = new_label;
    return result;
}

uint64_t RuntimeAlignment(uint64_t requested, uint64_t word_bytes) {
    const uint64_t divisor = std::gcd(requested, word_bytes);
    const uint64_t multiplier = word_bytes / divisor;
    if (requested > std::numeric_limits<uint64_t>::max() / multiplier)
        throw std::overflow_error(
            "Sram_lifecycle runtime alignment overflows uint64_t");
    return requested * multiplier;
}

int CheckedLogicalSize(uint64_t size_bytes) {
    if (size_bytes > static_cast<uint64_t>(
                         std::numeric_limits<int>::max()))
        throw std::out_of_range(
            "Sram_lifecycle size exceeds locator integer range");
    return static_cast<int>(size_bytes);
}

void RollBackAllocation(sram::RegionTable &regions,
                        const sram::Allocation &allocation) {
    regions.Free(allocation.id, allocation.lifetime);
}
} // namespace

void Sram_lifecycle::printSelf() {}

vector<sc_bv<128>> Sram_lifecycle::serialize() {
    RequireStrictTransport();
    ValidatePrim(*this);

    const size_t segment_count = op == SramLifecycleOp::ALLOC_AT
                                     ? kAllocAtWireSegments
                                     : kLegacyWireSegments;
    vector<sc_bv<128>> segments(segment_count);
    const uint8_t id = ExpectedId();
    for (auto &segment : segments) {
        segment = 0;
        segment.range(7, 0) = sc_bv<8>(id);
    }
    segments[0].range(10, 8) =
        sc_bv<3>(static_cast<uint8_t>(op));
    segments[0].range(12, 11) =
        sc_bv<2>(static_cast<uint8_t>(lifetime));
    segments[0].range(13, 13) = sc_bv<1>(spillable);

    segments[1].range(39, 8) = sc_bv<32>(AddLabel(region_name));
    segments[1].range(71, 40) = sc_bv<32>(AddLabel(label));
    segments[1].range(103, 72) = sc_bv<32>(AddLabel(new_label));
    segments[2].range(71, 8) = sc_bv<64>(size_bytes);
    segments[3].range(71, 8) = sc_bv<64>(alignment_bytes);
    if (op == SramLifecycleOp::ALLOC_AT)
        segments[4].range(71, 8) = sc_bv<64>(region_offset_bytes);
    return segments;
}

void Sram_lifecycle::deserialize(vector<sc_bv<128>> segments) {
    RequireStrictTransport();
    ValidateWireIdentity(segments);
    const auto decoded_op = static_cast<SramLifecycleOp>(
        segments[0].range(10, 8).to_uint64());
    const size_t expected_segments =
        decoded_op == SramLifecycleOp::ALLOC_AT
            ? kAllocAtWireSegments
            : kLegacyWireSegments;
    if (segments.size() != expected_segments)
        throw std::invalid_argument(
            "Sram_lifecycle Prim wire segment count mismatches operation");
    if (segments[0].range(127, 14).or_reduce())
        throw std::invalid_argument(
            "Sram_lifecycle Prim wire metadata padding is non-zero");
    if (segments[1].range(127, 104).or_reduce())
        throw std::invalid_argument(
            "Sram_lifecycle Prim wire label padding is non-zero");
    if (segments[2].range(127, 72).or_reduce() ||
        segments[3].range(127, 72).or_reduce() ||
        (segments.size() == kAllocAtWireSegments &&
         segments[4].range(127, 72).or_reduce()))
        throw std::invalid_argument(
            "Sram_lifecycle Prim wire numeric padding is non-zero");

    Sram_lifecycle decoded;
    decoded.op = decoded_op;
    decoded.lifetime = static_cast<sram::AllocationLifetime>(
        segments[0].range(12, 11).to_uint64());
    decoded.spillable = segments[0].range(13, 13).to_uint64() != 0;
    decoded.region_name = ReadOptionalLabel(
        segments[1].range(39, 8).to_uint64());
    decoded.label = ReadOptionalLabel(
        segments[1].range(71, 40).to_uint64());
    decoded.new_label = ReadOptionalLabel(
        segments[1].range(103, 72).to_uint64());
    decoded.size_bytes = segments[2].range(71, 8).to_uint64();
    decoded.alignment_bytes = segments[3].range(71, 8).to_uint64();
    if (decoded.op == SramLifecycleOp::ALLOC_AT)
        decoded.region_offset_bytes =
            segments[4].range(71, 8).to_uint64();
    ValidatePrim(decoded);

    op = decoded.op;
    region_name = std::move(decoded.region_name);
    label = std::move(decoded.label);
    new_label = std::move(decoded.new_label);
    region_offset_bytes = decoded.region_offset_bytes;
    size_bytes = decoded.size_bytes;
    alignment_bytes = decoded.alignment_bytes;
    lifetime = decoded.lifetime;
    spillable = decoded.spillable;
}

int Sram_lifecycle::taskCoreDefault(TaskCoreContext &context) {
#if USE_SRAM_MANAGER
    (void)context;
    throw std::runtime_error(
        "Sram_lifecycle requires the unified RegionTable allocator and "
        "does not support USE_SRAM_MANAGER");
#endif
    ValidatePrim(*this);
    SramPosLocator &locator = RequireLocator(*this);
    if (context.sram_regions == nullptr)
        throw std::runtime_error(
            "Sram_lifecycle requires an SRAM region table");
    auto &regions = *context.sram_regions;
    locator.BindRegionTable(&regions);

    switch (op) {
    case SramLifecycleOp::ALLOC:
    case SramLifecycleOp::ALLOC_AT: {
        if (locator.data_map.find(label) != locator.data_map.end())
            throw std::invalid_argument("duplicate SRAM label: " + label);
        const auto &region = regions.Region(region_name);
        if (region.spillable != spillable)
            throw std::invalid_argument(
                "Sram_lifecycle spillable flag disagrees with region");
        const int logical_size = CheckedLogicalSize(size_bytes);
        const uint64_t word_bytes = LegacySramWordBytes(context);
        const uint64_t runtime_alignment =
            RuntimeAlignment(alignment_bytes, word_bytes);
        AddrPosKey key;
        key.size = logical_size;
        key.spillable = spillable;
        key.allocation_lifetime = lifetime;
        key.preferred_region = region_name;
        const sram::Allocation allocation =
            op == SramLifecycleOp::ALLOC_AT
                ? regions.AllocateAt(region_name, region_offset_bytes,
                                     size_bytes, label, lifetime,
                                     runtime_alignment)
                : regions.Allocate(region_name, size_bytes, label, lifetime,
                                   runtime_alignment);
        try {
            if (allocation.range.address % word_bytes != 0 ||
                allocation.range.address / word_bytes >
                    static_cast<uint64_t>(std::numeric_limits<int>::max()))
                throw std::out_of_range(
                    "allocated SRAM address exceeds locator word range");
            key.pos = static_cast<int>(allocation.range.address / word_bytes);
            key.region_id = allocation.range.region_id;
            key.region_allocation_id = allocation.id;
            const auto inserted = locator.data_map.emplace(label, key);
            if (!inserted.second)
                throw std::invalid_argument(
                    "duplicate SRAM label during allocation");
        } catch (...) {
            RollBackAllocation(regions, allocation);
            throw;
        }
        return 0;
    }
    case SramLifecycleOp::FREE: {
        RejectPendingReference(*prim_context, label);
        (void)FindManagedKey(locator, label);
        locator.deletePair(label);
        return 0;
    }
    case SramLifecycleOp::RESIZE: {
        RejectPendingReference(*prim_context, label);
        AddrPosKey &key = FindManagedKey(locator, label);
        if (!key.valid || key.spill_size != 0)
            throw std::runtime_error(
                "cannot resize a non-resident SRAM label");
        const int logical_size = CheckedLogicalSize(size_bytes);
        (void)regions.ResizeAllocation(
            key.region_allocation_id, size_bytes);
        key.size = logical_size;
        return 0;
    }
    case SramLifecycleOp::RENAME: {
        (void)FindManagedKey(locator, label);
        if (locator.data_map.find(new_label) != locator.data_map.end())
            throw std::invalid_argument(
                "duplicate SRAM label: " + new_label);
        AddrDatapassLabel pending =
            RenamedPendingLabels(*prim_context, label, new_label);
        locator.changePairName(label, new_label);
        if (prim_context->sram_bind_pending_)
            std::swap(prim_context->sram_bind_pending_labels_, pending);
        return 0;
    }
    case SramLifecycleOp::CLEAR_TARGETED: {
        RejectPendingReference(*prim_context, label);
        if (context.sram_access == nullptr || context.sram_storage == nullptr)
            throw std::runtime_error(
                "targeted SRAM clear requires the unified data path");
        AddrPosKey &key = FindManagedKey(locator, label);
        if (!key.valid || key.spill_size != 0 || key.pos < 0 || key.size <= 0)
            throw std::runtime_error(
                "targeted SRAM clear requires a resident label");
        const auto &allocation =
            regions.FindAllocation(key.region_allocation_id);
        const auto &region = regions.Region(allocation.range.region_id);
        if (!region.spillable || !key.spillable)
            throw std::invalid_argument(
                "targeted SRAM clear rejects non-spillable regions");
        if (allocation.lifetime != sram::AllocationLifetime::kTask ||
            key.allocation_lifetime != sram::AllocationLifetime::kTask)
            throw std::invalid_argument(
                "targeted SRAM clear only accepts task-lifetime labels");
        const uint64_t address = LegacySramByteAddress(
            context, static_cast<uint64_t>(key.pos));
        const uint64_t logical_size = static_cast<uint64_t>(key.size);
        if (allocation.label != label || allocation.range.address != address ||
            allocation.range.size_bytes < logical_size ||
            allocation.range.region_id != key.region_id)
            throw std::logic_error(
                "SRAM locator and region allocation metadata disagree");
        const auto resolved = regions.ResolveAbsolute(
            address, logical_size, sram::Initiator::kCompute,
            sram::Command::kClear);
        if (resolved.region_id != allocation.range.region_id)
            throw std::logic_error(
                "targeted SRAM clear crosses its allocation region");
        if (context.sram_access->IsRangeBusy(
                {allocation.range.address, allocation.range.size_bytes}))
            throw std::runtime_error(
                "cannot clear SRAM allocation with outstanding accesses");
        sram::Request clear;
        clear.initiator = sram::Initiator::kCompute;
        clear.command = sram::Command::kClear;
        clear.address = address;
        clear.size_bytes = logical_size;
        context.sram_access->Access(clear);
        locator.deletePair(label);
        return 0;
    }
    }
    throw std::invalid_argument("Sram_lifecycle operation is invalid");
}
