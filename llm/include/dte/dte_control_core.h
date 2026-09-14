#pragma once

#include "dte/dte_types.h"
#include "dte/dte_async_types.h"
#include "systemc.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <deque>
#include <exception>
#include <map>
#include <memory>

class DTEUnit;
class Event_engine;
class DteAsyncTracker;
class DteMemoryBridge;

enum class DteControlOpcode : uint8_t {
    ISSUE_TRANSFER = 0,
    WAIT_TRANSMIT_START,
    WAIT_TRANSFER,
    RELEASE_TRANSFER,
    ISSUE_TOKEN,
    WAIT_TOKEN,
    POLL_TOKEN,
    FENCE,
    CANCEL_TOKEN,
    COUNT,
};

constexpr size_t DteControlOpcodeCount =
    static_cast<size_t>(DteControlOpcode::COUNT);
const char *DteControlOpcodeName(DteControlOpcode opcode);

// Runtime form of control_cores.dte.  The hardware JSON parser deliberately
// converts into this type so this module does not depend on common/config.h.
struct DteControlCoreConfig {
    uint32_t command_queue_depth = 16;
    uint32_t dispatch_width = 1;
    sc_time dispatch_latency = SC_ZERO_TIME;
    sc_time completion_notify_latency = SC_ZERO_TIME;
};

struct DteTransferHandle {
    uint64_t value = 0;

    friend bool operator==(DteTransferHandle lhs, DteTransferHandle rhs) {
        return lhs.value == rhs.value;
    }
    friend bool operator!=(DteTransferHandle lhs, DteTransferHandle rhs) {
        return !(lhs == rhs);
    }
};

enum class DteControlTransferState : uint8_t {
    QUEUED = 0,
    DISPATCHING,
    BACKEND_ACTIVE,
    BACKEND_COMPLETED,
    COMPLETED,
    FAILED,
};

// Value-only view of a transfer.  In particular, this never exposes the
// lifetime of DteTransferContext owned by DTEUnit.
struct DteTransferSnapshot {
    DteTransferHandle handle{};
    DteControlTransferState control_state =
        DteControlTransferState::QUEUED;
    bool backend_valid = false;
    uint64_t xfer_id = 0;
    uint64_t payload_bits = 0;
    DteDir direction = DteDir::SPM_TO_REMOTE;
    DteTransferState backend_state = DteTransferState::PENDING;
    int channel_id = -1;
    sc_time enqueue_time = SC_ZERO_TIME;
    sc_time backend_issue_time = SC_ZERO_TIME;
    sc_time transmit_start_time = SC_ZERO_TIME;
    sc_time scheduled_completion_time = SC_ZERO_TIME;
    sc_time backend_completion_time = SC_ZERO_TIME;
    sc_time notification_time = SC_ZERO_TIME;
};

struct DteControlCoreStatistics {
    uint64_t enqueued = 0;
    uint64_t dispatched = 0;
    uint64_t completed = 0;
    uint64_t failed = 0;
    uint64_t queue_stalls = 0;
    size_t max_queue_occupancy = 0;
    uint64_t queue_stall_cycles = 0;
    uint64_t barrier_stall_cycles = 0;
    uint64_t dispatch_busy_cycles = 0;
    uint64_t completion_notify_cycles = 0;
    std::array<uint64_t, DteControlOpcodeCount> opcode_count{};
    std::array<sc_time, DteControlOpcodeCount> opcode_total_latency{};
};

struct DteControlResidual {
    size_t queued_commands = 0;
    size_t inflight_commands = 0;
    size_t active_transfers = 0;
    size_t logical_tokens = 0;
    size_t pending_notifications = 0;

    bool Drained() const {
        return queued_commands == 0 && inflight_commands == 0 &&
               active_transfers == 0 && logical_tokens == 0 &&
               pending_notifications == 0;
    }
};

// Dedicated DTE controller.  Public methods are called from SystemC processes
// and may wait as documented.  Issue only waits for FIFO space; DTE descriptor
// backpressure is consumed by the independent dispatch worker.
class DteControlCore : public sc_module {
public:
    SC_HAS_PROCESS(DteControlCore);

    DteControlCore(const sc_module_name &name, DTEUnit &unit,
                   const DteControlCoreConfig &config,
                   Event_engine *event_engine = nullptr);
    ~DteControlCore() override;

    DteTransferHandle Issue(uint64_t payload_bits, DteDir direction);
    void WaitTransmitStart(DteTransferHandle handle);
    void Wait(DteTransferHandle handle);
    bool Release(DteTransferHandle handle);
    DteTransferSnapshot Snapshot(DteTransferHandle handle) const;

    void BindAsyncTracker(DteAsyncTracker *tracker);
    void BindMemoryBridge(DteMemoryBridge *bridge);
    uint64_t IssueToken(
        uint32_t token, uint64_t payload_bits, DteDir direction,
        uint64_t spm_addr, uint64_t spm_size,
        uint32_t remote_peer = DTE_ASYNC_INVALID_REMOTE_PEER,
        uint64_t remote_addr = 0, uint32_t address_block = 0);
    void WaitToken(uint32_t token);
    bool PollToken(uint32_t token);
    void Fence();
    void CancelToken(uint32_t token);

    uint32_t BitWidth() const;
    size_t QueueOccupancy() const { return pending_.size(); }
    size_t OutstandingCount() const { return records_.size(); }
    const DteControlCoreConfig &config() const { return config_; }
    const DteControlCoreStatistics &statistics() const { return statistics_; }
    DteControlResidual Residual() const;

    static void ValidateConfig(const DteControlCoreConfig &config);
    static sc_time NormalizeLatency(const sc_time &latency);

private:
    struct Record;

    Record &Find(DteTransferHandle handle);
    const Record &Find(DteTransferHandle handle) const;
    void DispatchWorker();
    void PhysicalCommandWorker(uint64_t handle_value);
    void PhysicalControlCommandWorker(uint64_t handle_value);
    void LogicalCommandWorker(uint64_t handle_value);
    void CompleteRecord(Record &record);
    void FailRecord(Record &record, std::exception_ptr failure);
    void AdvanceLogicalOrder(Record &record);
    Record &Enqueue(std::unique_ptr<Record> record);
    void WaitLogicalAndConsume(DteTransferHandle handle);
    void Trace(const Record &record, const char *stage,
               const char *phase) const;
    static void ValidateRequest(uint64_t payload_bits, DteDir direction,
                                const DTEUnit &unit);

    DTEUnit &unit_;
    DteAsyncTracker *async_tracker_ = nullptr;
    Event_engine *event_engine_ = nullptr;
    DteControlCoreConfig config_;
    std::map<uint64_t, std::unique_ptr<Record>> records_;
    std::deque<Record *> pending_;
    uint64_t next_handle_ = 0;
    uint64_t next_logical_sequence_ = 0;
    uint64_t next_logical_to_start_ = 0;
    DteControlCoreStatistics statistics_;
    sc_event pending_changed_;
    sc_event queue_space_available_;
    sc_event logical_start_changed_;
};

int RunDteControlCoreSelfTest();
