#pragma once

#include "dte/dte_async_types.h"
#include "dte/dte_coalescing.h"
#include "dte/dte_unit.h"
#include "dte/dte_memory_bridge.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

class Event_engine;

struct DteAsyncRecord {
    uint32_t token = 0;
    uint64_t xfer_id = DTE_ASYNC_INVALID_XFER_ID;
    DteTransferContext *context = nullptr;
    uint64_t spm_addr = 0;
    uint64_t spm_size = 0; // byte, half-open [addr, addr + size)
    DteAsyncAccess access = DteAsyncAccess::NONE;
    uint64_t spm_read_addr = 0;
    uint64_t spm_read_size = 0;
    uint64_t spm_write_addr = 0;
    uint64_t spm_write_size = 0;
    uint64_t issue_sequence = 0;
    uint64_t payload_bits = 0;
    DteDir direction = DteDir::SPM_TO_REMOTE;
    uint32_t remote_peer = DTE_ASYNC_INVALID_REMOTE_PEER;
    uint64_t remote_addr = 0;
    uint32_t address_block = 0;
    uint64_t group_id = DTE_ASYNC_INVALID_XFER_ID;
    bool staged = false;
};

// Per-core logical-token layer above DTEUnit. V3b makes it a SystemC module so
// a partially filled aggregation group can flush at its exact timeout even
// while the WorkerCore dispatcher is executing an unrelated compute primitive.
class DteAsyncTracker : public sc_module {
public:
    SC_HAS_PROCESS(DteAsyncTracker);

    DteAsyncTracker(const sc_module_name &name, DTEUnit &unit,
                    const DteAggregationConfig &aggregation = {},
                    int core_id = -1, Event_engine *event_engine = nullptr);

    uint64_t IssueToken(
        uint32_t token, uint64_t payload_bits, DteDir dir,
        uint64_t spm_addr, uint64_t spm_size,
        uint32_t remote_peer = DTE_ASYNC_INVALID_REMOTE_PEER,
        uint64_t remote_addr = 0, uint32_t address_block = 0);
    void WaitToken(uint32_t token);
    bool PollToken(uint32_t token);
    void Fence();
    void CancelToken(uint32_t token);
    void BindMemoryBridge(DteMemoryBridge *bridge);

    size_t OutstandingCount() const { return records_.size(); }
    size_t OpenGroupCount() const { return open_groups_.size(); }
    bool HasToken(uint32_t token) const { return records_.count(token) != 0; }
    const DteAsyncRecord &Record(uint32_t token) const;
    std::vector<uint32_t> OutstandingTokens() const;
    const DteAggregationMetrics &AggregationMetrics() const {
        return aggregation_metrics_;
    }

private:
    using RecordMap = std::map<uint32_t, DteAsyncRecord>;
    using GroupKey = std::tuple<uint8_t, uint32_t, uint32_t>;

    struct OpenGroup {
        uint64_t group_id = 0;
        GroupKey key{};
        DteDir direction = DteDir::SPM_TO_REMOTE;
        std::vector<uint32_t> tokens;
        uint64_t total_payload_bits = 0;
        uint64_t total_payload_bytes = 0;
        uint64_t next_spm_addr = 0;
        uint64_t next_remote_addr = 0;
        uint64_t first_issue_sequence = 0;
        sc_time deadline = SC_ZERO_TIME;
    };

    struct PhysicalBatch {
        uint64_t xfer_id = 0;
        DteTransferContext *context = nullptr;
        size_t remaining_tokens = 0;
        size_t member_count = 0;
        uint64_t payload_bits = 0;
    };

    static DteAsyncAccess AccessForDirection(DteDir dir);
    static bool RangeOverlaps(uint64_t lhs_addr, uint64_t lhs_size,
                              uint64_t rhs_addr, uint64_t rhs_size);
    static bool HasHazard(const DteAsyncRecord &existing,
                          const DteAsyncRecord &incoming);
    static void PopulateSpmRanges(DteAsyncRecord &record);
    void ValidateIssue(uint32_t token, uint64_t payload_bits, DteDir dir,
                       uint64_t spm_addr, uint64_t spm_size,
                       uint32_t remote_peer, uint64_t remote_addr,
                       uint32_t address_block) const;
    bool CanAppend(const OpenGroup &group, uint64_t payload_bits,
                   uint64_t spm_addr, uint64_t spm_size,
                   uint64_t remote_addr) const;
    uint64_t IssuePhysical(const std::vector<uint32_t> &tokens,
                           uint64_t payload_bits, DteDir dir);
    uint64_t StartAggregationGroup(uint32_t token);
    uint64_t AppendToAggregationGroup(uint64_t group_id, uint32_t token);
    uint64_t FlushGroup(uint64_t group_id, const char *reason);
    void FlushAllOpenGroups(const char *reason);
    void CancelStagedToken(uint32_t token);
    void WaitForCompletion(uint32_t token, bool trace_wait);
    void WaitAndRelease(uint32_t token, bool trace_wait);
    void timeoutWorker();
    void Trace(const char *stage, const char *phase, uint32_t token,
               uint64_t xfer_id, const std::string &extra = "") const;

    DTEUnit &unit_;
    DteAggregationConfig aggregation_;
    int core_id_;
    Event_engine *event_engine_;
    DteMemoryBridge *memory_bridge_ = nullptr;
    RecordMap records_;
    std::map<uint64_t, OpenGroup> open_groups_;
    std::map<GroupKey, uint64_t> group_by_key_;
    std::map<uint64_t, PhysicalBatch> physical_batches_;
    DteAggregationMetrics aggregation_metrics_;
    uint64_t next_issue_sequence_ = 0;
    uint64_t next_group_id_ = 0;
    sc_event aggregation_changed_;
};

int RunDTEV3SelfTest();
int RunDTEV3bSelfTest();
