#pragma once

#include "dte/coll_types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class GatherAcceptStatus : uint8_t {
    ACCEPTED = 0,
    FULL = 1,
    DUPLICATE = 2,
    UNKNOWN_PACKET = 3,
    INVALID_OFFSET = 4,
};

struct GatherExpectedSlot {
    PacketKey key;
    uint64_t offset_bits = 0;
};

struct GatherAcceptResult {
    GatherAcceptStatus status = GatherAcceptStatus::UNKNOWN_PACKET;
    size_t occupancy = 0;
};

// Finite Gather RX reorder model. Packets may arrive in any order, but the
// single ordered destination stream commits according to expected_slots.
class GatherReorderBuffer {
public:
    GatherReorderBuffer(std::vector<GatherExpectedSlot> expected_slots,
                        size_t depth, size_t commit_ports = 1);

    GatherAcceptResult Accept(const PacketKey &key, uint64_t offset_bits);
    size_t CommitCycle();
    void RecordBackpressureStall();

    size_t Depth() const { return depth_; }
    size_t Occupancy() const { return occupancy_; }
    size_t PeakOccupancy() const { return peak_occupancy_; }
    size_t StallCycles() const { return stall_cycles_; }
    size_t CommittedCount() const { return next_commit_; }
    bool Complete() const { return next_commit_ == slots_.size(); }
    std::string ReceivedBitmap() const;
    std::string CommittedBitmap() const;
    std::string MissingBitmap() const;

private:
    struct Slot {
        GatherExpectedSlot expected;
        bool received = false;
        bool committed = false;
    };

    size_t Find(const PacketKey &key) const;

    std::vector<Slot> slots_;
    size_t depth_ = 0;
    size_t commit_ports_ = 1;
    size_t occupancy_ = 0;
    size_t peak_occupancy_ = 0;
    size_t stall_cycles_ = 0;
    size_t next_commit_ = 0;
};
