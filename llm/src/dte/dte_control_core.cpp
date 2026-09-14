#include "dte/dte_control_core.h"

#include "dte/dte_async.h"
#include "dte/dte_memory_bridge.h"
#include "dte/dte_unit.h"
#include "macros/macros.h"
#include "trace/Event_engine.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
sc_time ControlCycle() { return sc_time(CYCLE, SC_NS); }

sc_time RoundUpToControlCycle(const sc_time &time) {
    if (time == SC_ZERO_TIME)
        return SC_ZERO_TIME;
    const auto cycle = ControlCycle().value();
    const auto ticks = time.value();
    return sc_time::from_value(((ticks + cycle - 1) / cycle) * cycle);
}

uint64_t TimeCyclesCeil(const sc_time &time) {
    if (time == SC_ZERO_TIME)
        return 0;
    const auto cycle = ControlCycle().value();
    return (time.value() + cycle - 1) / cycle;
}

bool IsTerminal(DteControlTransferState state) {
    return state == DteControlTransferState::COMPLETED ||
           state == DteControlTransferState::FAILED;
}

DteControlCoreConfig NormalizeConfig(DteControlCoreConfig config) {
    DteControlCore::ValidateConfig(config);
    config.dispatch_latency =
        DteControlCore::NormalizeLatency(config.dispatch_latency);
    config.completion_notify_latency =
        DteControlCore::NormalizeLatency(config.completion_notify_latency);
    return config;
}
} // namespace

const char *DteControlOpcodeName(DteControlOpcode opcode) {
    switch (opcode) {
    case DteControlOpcode::ISSUE_TRANSFER: return "issue_transfer";
    case DteControlOpcode::WAIT_TRANSMIT_START: return "wait_transmit_start";
    case DteControlOpcode::WAIT_TRANSFER: return "wait_transfer";
    case DteControlOpcode::RELEASE_TRANSFER: return "release_transfer";
    case DteControlOpcode::ISSUE_TOKEN: return "issue_token";
    case DteControlOpcode::WAIT_TOKEN: return "wait_token";
    case DteControlOpcode::POLL_TOKEN: return "poll_token";
    case DteControlOpcode::FENCE: return "fence";
    case DteControlOpcode::CANCEL_TOKEN: return "cancel_token";
    case DteControlOpcode::COUNT: break;
    }
    return "unknown";
}

struct DteControlCore::Record {
    DteTransferHandle handle{};
    uint64_t command_id = 0;
    DteControlOpcode opcode = DteControlOpcode::ISSUE_TRANSFER;
    uint64_t logical_sequence = 0;
    bool logical_order_advanced = false;
    DteTransferHandle target_handle{};

    uint64_t payload_bits = 0;
    DteDir direction = DteDir::SPM_TO_REMOTE;
    uint32_t token = 0;
    uint64_t spm_addr = 0;
    uint64_t spm_size = 0;
    uint32_t remote_peer = DTE_ASYNC_INVALID_REMOTE_PEER;
    uint64_t remote_addr = 0;
    uint32_t address_block = 0;
    uint64_t fence_watermark = 0;
    uint64_t uint_result = DTE_ASYNC_INVALID_XFER_ID;
    bool bool_result = false;

    DteControlTransferState state = DteControlTransferState::QUEUED;
    DteTransferContext *backend = nullptr;
    sc_time enqueue_time = SC_ZERO_TIME;
    sc_time backend_issue_time = SC_ZERO_TIME;
    sc_time notification_time = SC_ZERO_TIME;
    std::exception_ptr failure;
    sc_event dispatched;
    sc_event backend_bound;
    sc_event notified;
};

DteControlCore::DteControlCore(const sc_module_name &name, DTEUnit &unit,
                               const DteControlCoreConfig &config,
                               Event_engine *event_engine)
    : sc_module(name), unit_(unit), event_engine_(event_engine),
      config_(NormalizeConfig(config)) {
    SC_THREAD(DispatchWorker);
}

DteControlCore::~DteControlCore() = default;

void DteControlCore::ValidateConfig(const DteControlCoreConfig &config) {
    if (config.command_queue_depth == 0)
        throw std::invalid_argument(
            "DTE control command_queue_depth must be > 0");
    if (config.dispatch_width == 0)
        throw std::invalid_argument("DTE control dispatch_width must be > 0");
    if (config.dispatch_width > config.command_queue_depth)
        throw std::invalid_argument(
            "DTE control dispatch_width must not exceed command_queue_depth");
    if (config.dispatch_latency < SC_ZERO_TIME ||
        config.completion_notify_latency < SC_ZERO_TIME)
        throw std::invalid_argument("DTE control latencies must be >= 0");
}

sc_time DteControlCore::NormalizeLatency(const sc_time &latency) {
    if (latency < SC_ZERO_TIME)
        throw std::invalid_argument("DTE control latency must be >= 0");
    return RoundUpToControlCycle(latency);
}

void DteControlCore::ValidateRequest(uint64_t payload_bits, DteDir direction,
                                     const DTEUnit &unit) {
    if (payload_bits == 0)
        throw std::invalid_argument("DTE payload_bits must be > 0");
    if (direction < DteDir::SPM_TO_REMOTE ||
        direction > DteDir::DRAM_TO_REMOTE)
        throw std::invalid_argument("DTE transfer direction is invalid");
    if (!unit.config().fine_grained_resources &&
        direction != DteDir::SPM_TO_REMOTE &&
        direction != DteDir::REMOTE_TO_SPM)
        throw std::invalid_argument(
            "DTE V4 direction requires fine_grained_resources=true");
}

void DteControlCore::BindAsyncTracker(DteAsyncTracker *tracker) {
    if (tracker == nullptr)
        throw std::invalid_argument("cannot bind null async tracker");
    if (async_tracker_ != nullptr && async_tracker_ != tracker)
        throw std::logic_error("DTE control core async tracker already bound");
    async_tracker_ = tracker;
}

void DteControlCore::BindMemoryBridge(DteMemoryBridge *bridge) {
    if (async_tracker_ == nullptr)
        throw std::logic_error(
            "bind async tracker before DTE control memory bridge");
    async_tracker_->BindMemoryBridge(bridge);
}

DteControlCore::Record &
DteControlCore::Enqueue(std::unique_ptr<Record> record) {
    bool stalled = false;
    const sc_time stall_begin = sc_time_stamp();
    while (pending_.size() >= config_.command_queue_depth) {
        if (!stalled) {
            ++statistics_.queue_stalls;
            record->command_id = next_handle_;
            if (event_engine_ != nullptr)
                Trace(*record, "DTE_CTRL_queue_wait", "B");
            stalled = true;
        }
        wait(queue_space_available_);
    }
    if (stalled) {
        statistics_.queue_stall_cycles +=
            TimeCyclesCeil(sc_time_stamp() - stall_begin);
        if (event_engine_ != nullptr)
            Trace(*record, "DTE_CTRL_queue_wait", "E");
    }
    if (next_handle_ == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("DTE control command id space exhausted");

    record->handle.value = next_handle_;
    record->command_id = next_handle_++;
    record->enqueue_time = sc_time_stamp();
    Record *raw = record.get();
    records_.emplace(record->handle.value, std::move(record));
    pending_.push_back(raw);
    ++statistics_.enqueued;
    statistics_.max_queue_occupancy =
        std::max(statistics_.max_queue_occupancy, pending_.size());
    pending_changed_.notify(SC_ZERO_TIME);
    return *raw;
}

DteTransferHandle DteControlCore::Issue(uint64_t payload_bits,
                                        DteDir direction) {
    ValidateRequest(payload_bits, direction, unit_);
    auto record = std::make_unique<Record>();
    record->opcode = DteControlOpcode::ISSUE_TRANSFER;
    record->payload_bits = payload_bits;
    record->direction = direction;
    return Enqueue(std::move(record)).handle;
}

DteControlCore::Record &DteControlCore::Find(DteTransferHandle handle) {
    auto it = records_.find(handle.value);
    if (it == records_.end())
        throw std::out_of_range("unknown or released DTE control handle " +
                                std::to_string(handle.value));
    return *it->second;
}

const DteControlCore::Record &
DteControlCore::Find(DteTransferHandle handle) const {
    auto it = records_.find(handle.value);
    if (it == records_.end())
        throw std::out_of_range("unknown or released DTE control handle " +
                                std::to_string(handle.value));
    return *it->second;
}

void DteControlCore::DispatchWorker() {
    sc_time next_dispatch_time = SC_ZERO_TIME;
    while (true) {
        while (pending_.empty())
            wait(pending_changed_);
        if (sc_time_stamp() < next_dispatch_time)
            wait(next_dispatch_time - sc_time_stamp());

        const size_t count =
            std::min<size_t>(config_.dispatch_width, pending_.size());
        std::vector<Record *> batch;
        batch.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            Record *record = pending_.front();
            pending_.pop_front();
            record->state = DteControlTransferState::DISPATCHING;
            Trace(*record, "DTE_CTRL_dispatch", "B");
            batch.push_back(record);
        }
        queue_space_available_.notify(SC_ZERO_TIME);

        if (config_.dispatch_latency != SC_ZERO_TIME)
            wait(config_.dispatch_latency);
        statistics_.dispatch_busy_cycles +=
            TimeCyclesCeil(config_.dispatch_latency);
        for (Record *record : batch) {
            record->state = DteControlTransferState::BACKEND_ACTIVE;
            record->backend_issue_time = sc_time_stamp();
            ++statistics_.dispatched;
            const size_t opcode = static_cast<size_t>(record->opcode);
            ++statistics_.opcode_count.at(opcode);
            Trace(*record, "DTE_CTRL_dispatch", "E");
            const uint64_t handle_value = record->handle.value;
            if (record->opcode == DteControlOpcode::ISSUE_TRANSFER) {
                sc_spawn([this, handle_value] {
                    PhysicalCommandWorker(handle_value);
                }, sc_gen_unique_name("dte-control-physical"));
            } else if (record->opcode ==
                           DteControlOpcode::WAIT_TRANSMIT_START ||
                       record->opcode == DteControlOpcode::WAIT_TRANSFER ||
                       record->opcode ==
                           DteControlOpcode::RELEASE_TRANSFER) {
                record->dispatched.notify(SC_ZERO_TIME);
                sc_spawn([this, handle_value] {
                    PhysicalControlCommandWorker(handle_value);
                }, sc_gen_unique_name("dte-control-physical-command"));
            } else {
                record->dispatched.notify(SC_ZERO_TIME);
                sc_spawn([this, handle_value] {
                    LogicalCommandWorker(handle_value);
                }, sc_gen_unique_name("dte-control-logical"));
            }
        }
        next_dispatch_time = sc_time_stamp() + ControlCycle();
    }
}

void DteControlCore::PhysicalCommandWorker(uint64_t handle_value) {
    Record &record = Find(DteTransferHandle{handle_value});
    try {
        unit_.WaitForCredit();
        record.backend = &unit_.Issue(record.payload_bits, record.direction);
        record.backend_issue_time = record.backend->issue_time;
        record.backend_bound.notify(SC_ZERO_TIME);
        record.dispatched.notify(SC_ZERO_TIME);
        DteTransferContext &context = *record.backend;
        if (context.state != DteTransferState::COMPLETED &&
            context.state != DteTransferState::CANCELLED)
            wait(context.done);
        CompleteRecord(record);
    } catch (...) {
        FailRecord(record, std::current_exception());
    }
}

void DteControlCore::PhysicalControlCommandWorker(uint64_t handle_value) {
    Record &command = Find(DteTransferHandle{handle_value});
    try {
        Record &target = Find(command.target_handle);
        switch (command.opcode) {
        case DteControlOpcode::WAIT_TRANSMIT_START:
            while (target.backend == nullptr && !target.failure)
                wait(target.backend_bound | target.notified);
            if (target.failure)
                std::rethrow_exception(target.failure);
            if (target.backend->state != DteTransferState::TRANSMITTING &&
                target.backend->state != DteTransferState::COMPLETED &&
                target.backend->state != DteTransferState::CANCELLED)
                wait(target.backend->transmit_started | target.backend->done);
            if (target.backend->state == DteTransferState::CANCELLED)
                throw std::runtime_error(
                    "DTE transfer was cancelled before transmit start");
            break;
        case DteControlOpcode::WAIT_TRANSFER:
            while (!IsTerminal(target.state))
                wait(target.notified);
            if (target.failure)
                std::rethrow_exception(target.failure);
            break;
        case DteControlOpcode::RELEASE_TRANSFER:
            if (IsTerminal(target.state) && target.backend == nullptr) {
                command.bool_result = true;
                records_.erase(command.target_handle.value);
            } else if (IsTerminal(target.state) &&
                       target.backend != nullptr) {
                command.bool_result = unit_.Release(target.backend->xfer_id);
                if (command.bool_result)
                    records_.erase(command.target_handle.value);
            }
            break;
        default:
            throw std::logic_error(
                "unsupported physical DTE control opcode");
        }
        CompleteRecord(command);
    } catch (...) {
        FailRecord(command, std::current_exception());
    }
}

void DteControlCore::AdvanceLogicalOrder(Record &record) {
    if (record.logical_order_advanced)
        return;
    if (record.logical_sequence != next_logical_to_start_)
        throw std::logic_error("DTE logical command order is corrupt");
    record.logical_order_advanced = true;
    ++next_logical_to_start_;
    logical_start_changed_.notify(SC_ZERO_TIME);
}

void DteControlCore::LogicalCommandWorker(uint64_t handle_value) {
    Record &record = Find(DteTransferHandle{handle_value});
    while (record.logical_sequence != next_logical_to_start_)
        wait(logical_start_changed_);
    if (async_tracker_ == nullptr) {
        AdvanceLogicalOrder(record);
        FailRecord(record, std::make_exception_ptr(std::logic_error(
            "DTE async tracker is not bound to dedicated controller")));
        return;
    }

    try {
        switch (record.opcode) {
        case DteControlOpcode::ISSUE_TOKEN:
            record.uint_result = async_tracker_->IssueToken(
                record.token, record.payload_bits, record.direction,
                record.spm_addr, record.spm_size, record.remote_peer,
                record.remote_addr, record.address_block);
            AdvanceLogicalOrder(record);
            break;
        case DteControlOpcode::WAIT_TOKEN: {
            const sc_time barrier_begin = sc_time_stamp();
            bool complete = async_tracker_->TryRetireToken(record.token);
            const bool barrier_traced = !complete;
            if (barrier_traced)
                Trace(record, "DTE_CTRL_barrier", "B");
            while (!complete) {
                wait(async_tracker_->StateChangedEvent());
                complete = async_tracker_->TryRetireToken(record.token);
            }
            if (barrier_traced)
                Trace(record, "DTE_CTRL_barrier", "E");
            statistics_.barrier_stall_cycles +=
                TimeCyclesCeil(sc_time_stamp() - barrier_begin);
            AdvanceLogicalOrder(record);
            break;
        }
        case DteControlOpcode::POLL_TOKEN:
            record.bool_result = async_tracker_->PollToken(record.token);
            AdvanceLogicalOrder(record);
            break;
        case DteControlOpcode::FENCE: {
            const sc_time barrier_begin = sc_time_stamp();
            record.fence_watermark =
                async_tracker_->CaptureFenceWatermark();
            bool complete =
                async_tracker_->TryFenceThrough(record.fence_watermark);
            const bool barrier_traced = !complete;
            if (barrier_traced)
                Trace(record, "DTE_CTRL_barrier", "B");
            AdvanceLogicalOrder(record);
            while (!complete) {
                wait(async_tracker_->StateChangedEvent());
                complete =
                    async_tracker_->TryFenceThrough(record.fence_watermark);
            }
            if (barrier_traced)
                Trace(record, "DTE_CTRL_barrier", "E");
            statistics_.barrier_stall_cycles +=
                TimeCyclesCeil(sc_time_stamp() - barrier_begin);
            break;
        }
        case DteControlOpcode::CANCEL_TOKEN:
            async_tracker_->CancelToken(record.token);
            AdvanceLogicalOrder(record);
            break;
        default:
            throw std::logic_error(
                "unsupported opcode in DTE logical command worker");
        }
        CompleteRecord(record);
    } catch (...) {
        try {
            AdvanceLogicalOrder(record);
        } catch (...) {
        }
        FailRecord(record, std::current_exception());
    }
}

void DteControlCore::CompleteRecord(Record &record) {
    record.state = DteControlTransferState::BACKEND_COMPLETED;
    Trace(record, "DTE_CTRL_notify", "B");
    if (config_.completion_notify_latency != SC_ZERO_TIME)
        wait(config_.completion_notify_latency);
    statistics_.completion_notify_cycles +=
        TimeCyclesCeil(config_.completion_notify_latency);
    record.notification_time = sc_time_stamp();
    record.state = record.failure ? DteControlTransferState::FAILED
                                  : DteControlTransferState::COMPLETED;
    ++statistics_.completed;
    if (record.failure)
        ++statistics_.failed;
    statistics_.opcode_total_latency.at(
        static_cast<size_t>(record.opcode)) +=
        record.notification_time - record.enqueue_time;
    Trace(record, "DTE_CTRL_notify", "E");
    record.notified.notify(SC_ZERO_TIME);
}

void DteControlCore::FailRecord(Record &record,
                                std::exception_ptr failure) {
    record.failure = std::move(failure);
    CompleteRecord(record);
}

void DteControlCore::Trace(const Record &record, const char *stage,
                           const char *phase) const {
    if (event_engine_ == nullptr)
        return;
    std::ostringstream detail;
    detail << stage
           << " handle=" << record.command_id
           << " core=" << name()
           << " command_id=" << record.command_id
           << " opcode=" << DteControlOpcodeName(record.opcode)
           << " token=" << record.token
           << " width=" << config_.dispatch_width
           << " dispatch_latency=" << config_.dispatch_latency
           << " notify_latency=" << config_.completion_notify_latency
           << " status=" << static_cast<unsigned>(record.state)
           << " occupancy=" << pending_.size()
           << " depth=" << config_.command_queue_depth;
    event_engine_->add_event(
        name(), stage, phase, Trace_event_util(detail.str()), SC_ZERO_TIME,
        static_cast<unsigned>(record.command_id));
}

void DteControlCore::WaitTransmitStart(DteTransferHandle handle) {
    (void)Find(handle);
    auto command = std::make_unique<Record>();
    command->opcode = DteControlOpcode::WAIT_TRANSMIT_START;
    command->target_handle = handle;
    WaitLogicalAndConsume(Enqueue(std::move(command)).handle);
}

void DteControlCore::Wait(DteTransferHandle handle) {
    (void)Find(handle);
    auto command = std::make_unique<Record>();
    command->opcode = DteControlOpcode::WAIT_TRANSFER;
    command->target_handle = handle;
    WaitLogicalAndConsume(Enqueue(std::move(command)).handle);
}

bool DteControlCore::Release(DteTransferHandle handle) {
    (void)Find(handle);
    auto command = std::make_unique<Record>();
    command->opcode = DteControlOpcode::RELEASE_TRANSFER;
    command->target_handle = handle;
    const auto command_handle = Enqueue(std::move(command)).handle;
    Record &submitted = Find(command_handle);
    while (!IsTerminal(submitted.state))
        wait(submitted.notified);
    const bool result = submitted.bool_result;
    std::exception_ptr failure = submitted.failure;
    records_.erase(command_handle.value);
    if (failure)
        std::rethrow_exception(failure);
    return result;
}

void DteControlCore::WaitLogicalAndConsume(DteTransferHandle handle) {
    Record &record = Find(handle);
    while (!IsTerminal(record.state))
        wait(record.notified);
    std::exception_ptr failure = record.failure;
    records_.erase(handle.value);
    if (failure)
        std::rethrow_exception(failure);
}

uint64_t DteControlCore::IssueToken(
    uint32_t token, uint64_t payload_bits, DteDir direction,
    uint64_t spm_addr, uint64_t spm_size, uint32_t remote_peer,
    uint64_t remote_addr, uint32_t address_block) {
    auto record = std::make_unique<Record>();
    record->opcode = DteControlOpcode::ISSUE_TOKEN;
    record->logical_sequence = next_logical_sequence_++;
    record->token = token;
    record->payload_bits = payload_bits;
    record->direction = direction;
    record->spm_addr = spm_addr;
    record->spm_size = spm_size;
    record->remote_peer = remote_peer;
    record->remote_addr = remote_addr;
    record->address_block = address_block;
    const DteTransferHandle handle = Enqueue(std::move(record)).handle;
    Record &submitted = Find(handle);
    while (!IsTerminal(submitted.state))
        wait(submitted.notified);
    const uint64_t result = submitted.uint_result;
    std::exception_ptr failure = submitted.failure;
    records_.erase(handle.value);
    if (failure)
        std::rethrow_exception(failure);
    return result;
}

void DteControlCore::WaitToken(uint32_t token) {
    auto record = std::make_unique<Record>();
    record->opcode = DteControlOpcode::WAIT_TOKEN;
    record->logical_sequence = next_logical_sequence_++;
    record->token = token;
    const auto handle = Enqueue(std::move(record)).handle;
    WaitLogicalAndConsume(handle);
}

bool DteControlCore::PollToken(uint32_t token) {
    auto record = std::make_unique<Record>();
    record->opcode = DteControlOpcode::POLL_TOKEN;
    record->logical_sequence = next_logical_sequence_++;
    record->token = token;
    const auto handle = Enqueue(std::move(record)).handle;
    Record &submitted = Find(handle);
    while (!IsTerminal(submitted.state))
        wait(submitted.notified);
    const bool result = submitted.bool_result;
    std::exception_ptr failure = submitted.failure;
    records_.erase(handle.value);
    if (failure)
        std::rethrow_exception(failure);
    return result;
}

void DteControlCore::Fence() {
    auto record = std::make_unique<Record>();
    record->opcode = DteControlOpcode::FENCE;
    record->logical_sequence = next_logical_sequence_++;
    const auto handle = Enqueue(std::move(record)).handle;
    WaitLogicalAndConsume(handle);
}

void DteControlCore::CancelToken(uint32_t token) {
    auto record = std::make_unique<Record>();
    record->opcode = DteControlOpcode::CANCEL_TOKEN;
    record->logical_sequence = next_logical_sequence_++;
    record->token = token;
    const auto handle = Enqueue(std::move(record)).handle;
    WaitLogicalAndConsume(handle);
}

DteTransferSnapshot
DteControlCore::Snapshot(DteTransferHandle handle) const {
    const Record &record = Find(handle);
    DteTransferSnapshot result;
    result.handle = handle;
    result.control_state = record.state;
    result.payload_bits = record.payload_bits;
    result.direction = record.direction;
    result.enqueue_time = record.enqueue_time;
    result.backend_issue_time = record.backend_issue_time;
    result.notification_time = record.notification_time;
    if (record.backend != nullptr) {
        const DteTransferContext &context = *record.backend;
        result.backend_valid = true;
        result.xfer_id = context.xfer_id;
        result.backend_state = context.state;
        result.channel_id = context.channel_id;
        result.backend_issue_time = context.issue_time;
        result.transmit_start_time = context.transmit_start_time;
        result.scheduled_completion_time = context.scheduled_completion_time;
        result.backend_completion_time = context.completion_time;
    }
    return result;
}

DteControlResidual DteControlCore::Residual() const {
    DteControlResidual residual;
    residual.queued_commands = pending_.size();
    residual.active_transfers = unit_.InflightCount();
    residual.logical_tokens =
        async_tracker_ == nullptr ? 0 : async_tracker_->OutstandingCount();
    for (const auto &[id, owned] : records_) {
        (void)id;
        const Record &record = *owned;
        if (record.state != DteControlTransferState::QUEUED)
            ++residual.inflight_commands;
        if (record.state == DteControlTransferState::BACKEND_COMPLETED)
            ++residual.pending_notifications;
    }
    return residual;
}

uint32_t DteControlCore::BitWidth() const {
    return unit_.config().bit_width_bits;
}
