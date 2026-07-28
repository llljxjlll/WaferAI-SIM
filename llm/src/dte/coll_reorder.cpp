#include "dte/coll_reorder.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {
constexpr size_t NPOS = std::numeric_limits<size_t>::max();
}

GatherReorderBuffer::GatherReorderBuffer(
    std::vector<GatherExpectedSlot> expected_slots, size_t depth,
    size_t commit_ports)
    : depth_(depth), commit_ports_(commit_ports) {
    if (expected_slots.empty())
        throw std::invalid_argument("gather reorder requires expected slots");
    if (depth == 0)
        throw std::invalid_argument("gather reorder depth must be positive");
    if (commit_ports == 0)
        throw std::invalid_argument("gather reorder commit ports must be positive");
    slots_.reserve(expected_slots.size());
    for (const auto &expected : expected_slots) {
        for (const auto &slot : slots_)
            if (slot.expected.key == expected.key)
                throw std::invalid_argument("gather reorder expected packet is duplicated");
        slots_.push_back({expected, false, false});
    }
}

size_t GatherReorderBuffer::Find(const PacketKey &key) const {
    for (size_t i = 0; i < slots_.size(); ++i)
        if (slots_[i].expected.key == key) return i;
    return NPOS;
}

GatherAcceptResult GatherReorderBuffer::Accept(const PacketKey &key,
                                                uint64_t offset_bits) {
    const size_t index = Find(key);
    if (index == NPOS)
        return {GatherAcceptStatus::UNKNOWN_PACKET, occupancy_};
    Slot &slot = slots_[index];
    if (slot.received || slot.committed)
        return {GatherAcceptStatus::DUPLICATE, occupancy_};
    if (slot.expected.offset_bits != offset_bits)
        return {GatherAcceptStatus::INVALID_OFFSET, occupancy_};
    // Keep one physical entry available for the head-of-line packet. Without
    // this reservation, later packets can fill the buffer and backpressure the
    // only packet that can make the ordered commit port progress.
    if (occupancy_ == depth_ ||
        (index != next_commit_ && occupancy_ + 1 == depth_ &&
         !slots_[next_commit_].received))
        return {GatherAcceptStatus::FULL, occupancy_};
    slot.received = true;
    ++occupancy_;
    peak_occupancy_ = std::max(peak_occupancy_, occupancy_);
    return {GatherAcceptStatus::ACCEPTED, occupancy_};
}

size_t GatherReorderBuffer::CommitCycle() {
    size_t committed = 0;
    while (committed < commit_ports_ && next_commit_ < slots_.size()) {
        Slot &slot = slots_[next_commit_];
        if (!slot.received) break;
        slot.received = false;
        slot.committed = true;
        --occupancy_;
        ++next_commit_;
        ++committed;
    }
    return committed;
}

void GatherReorderBuffer::RecordBackpressureStall() { ++stall_cycles_; }

std::string GatherReorderBuffer::ReceivedBitmap() const {
    std::string bitmap;
    bitmap.reserve(slots_.size());
    for (const auto &slot : slots_) bitmap.push_back(slot.received ? '1' : '0');
    return bitmap;
}

std::string GatherReorderBuffer::CommittedBitmap() const {
    std::string bitmap;
    bitmap.reserve(slots_.size());
    for (const auto &slot : slots_) bitmap.push_back(slot.committed ? '1' : '0');
    return bitmap;
}

std::string GatherReorderBuffer::MissingBitmap() const {
    std::string bitmap;
    bitmap.reserve(slots_.size());
    for (const auto &slot : slots_)
        bitmap.push_back(!slot.received && !slot.committed ? '1' : '0');
    return bitmap;
}
