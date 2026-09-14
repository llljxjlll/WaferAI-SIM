#include "dte/dte_control_frontend.h"

#include "dte/dte_async.h"
#include "dte/dte_memory_bridge.h"
#include "dte/dte_unit.h"

#include <stdexcept>
#include <string>

struct DteControlFrontend::LegacyRecord {
    DteTransferHandle handle{};
    DteTransferContext *backend = nullptr;
};

DteControlFrontend::DteControlFrontend(DTEUnit &unit,
                                       DteAsyncTracker *async_tracker)
    : unit_(unit), async_tracker_(async_tracker) {}

DteControlFrontend::DteControlFrontend(DTEUnit &unit,
                                       DteControlCore &dedicated_core,
                                       DteAsyncTracker *async_tracker)
    : mode_(DteControlMode::DUAL_DTE_DEDICATED), unit_(unit),
      dedicated_core_(&dedicated_core), async_tracker_(async_tracker) {
    if (async_tracker_ != nullptr)
        dedicated_core_->BindAsyncTracker(async_tracker_);
}

DteControlFrontend::~DteControlFrontend() = default;

DteTransferHandle DteControlFrontend::Issue(uint64_t payload_bits,
                                            DteDir direction) {
    if (dedicated_core_ != nullptr)
        return dedicated_core_->Issue(payload_bits, direction);

    // This is deliberately identical to the old call sequence.  If credit is
    // already available (the common case), no delta cycle or timed wait occurs.
    unit_.WaitForCredit();
    DteTransferContext &context = unit_.Issue(payload_bits, direction);
    DteTransferHandle handle{next_legacy_handle_++};
    auto record = std::make_unique<LegacyRecord>();
    record->handle = handle;
    record->backend = &context;
    legacy_records_.emplace(handle.value, std::move(record));
    return handle;
}

DteControlFrontend::LegacyRecord &
DteControlFrontend::FindLegacy(DteTransferHandle handle) {
    auto it = legacy_records_.find(handle.value);
    if (it == legacy_records_.end())
        throw std::out_of_range("unknown or released legacy DTE handle " +
                                std::to_string(handle.value));
    return *it->second;
}

const DteControlFrontend::LegacyRecord &
DteControlFrontend::FindLegacy(DteTransferHandle handle) const {
    auto it = legacy_records_.find(handle.value);
    if (it == legacy_records_.end())
        throw std::out_of_range("unknown or released legacy DTE handle " +
                                std::to_string(handle.value));
    return *it->second;
}

void DteControlFrontend::WaitTransmitStart(DteTransferHandle handle) {
    if (dedicated_core_ != nullptr) {
        dedicated_core_->WaitTransmitStart(handle);
        return;
    }
    DteTransferContext &context = *FindLegacy(handle).backend;
    if (context.state != DteTransferState::TRANSMITTING &&
        context.state != DteTransferState::COMPLETED)
        wait(context.transmit_started);
}

void DteControlFrontend::Wait(DteTransferHandle handle) {
    if (dedicated_core_ != nullptr) {
        dedicated_core_->Wait(handle);
        return;
    }
    DteTransferContext &context = *FindLegacy(handle).backend;
    if (context.state != DteTransferState::COMPLETED &&
        context.state != DteTransferState::CANCELLED)
        wait(context.done);
}

bool DteControlFrontend::Release(DteTransferHandle handle) {
    if (dedicated_core_ != nullptr)
        return dedicated_core_->Release(handle);
    LegacyRecord &record = FindLegacy(handle);
    if (!unit_.Release(record.backend->xfer_id))
        return false;
    legacy_records_.erase(handle.value);
    return true;
}

DteTransferSnapshot
DteControlFrontend::Snapshot(DteTransferHandle handle) const {
    if (dedicated_core_ != nullptr)
        return dedicated_core_->Snapshot(handle);
    const DteTransferContext &context = *FindLegacy(handle).backend;
    DteTransferSnapshot result;
    result.handle = handle;
    result.control_state =
        (context.state == DteTransferState::COMPLETED ||
         context.state == DteTransferState::CANCELLED)
            ? DteControlTransferState::COMPLETED
            : DteControlTransferState::BACKEND_ACTIVE;
    result.backend_valid = true;
    result.xfer_id = context.xfer_id;
    result.payload_bits = context.payload_bits;
    result.direction = context.dir;
    result.backend_state = context.state;
    result.channel_id = context.channel_id;
    result.enqueue_time = context.issue_time;
    result.backend_issue_time = context.issue_time;
    result.transmit_start_time = context.transmit_start_time;
    result.scheduled_completion_time = context.scheduled_completion_time;
    result.backend_completion_time = context.completion_time;
    result.notification_time = context.completion_time;
    return result;
}

uint32_t DteControlFrontend::BitWidth() const {
    return dedicated_core_ != nullptr ? dedicated_core_->BitWidth()
                                      : unit_.config().bit_width_bits;
}

size_t DteControlFrontend::OutstandingTransferCount() const {
    return dedicated_core_ != nullptr ? dedicated_core_->OutstandingCount()
                                      : legacy_records_.size();
}

DteControlResidual DteControlFrontend::Residual() const {
    if (dedicated_core_ != nullptr)
        return dedicated_core_->Residual();
    DteControlResidual residual;
    residual.inflight_commands = legacy_records_.size();
    residual.active_transfers = unit_.InflightCount();
    residual.logical_tokens =
        async_tracker_ == nullptr ? 0 : async_tracker_->OutstandingCount();
    return residual;
}

void DteControlFrontend::BindMemoryBridge(DteMemoryBridge *bridge) {
    if (dedicated_core_ != nullptr) {
        dedicated_core_->BindMemoryBridge(bridge);
        return;
    }
    Async().BindMemoryBridge(bridge);
}

DteAsyncTracker &DteControlFrontend::Async() {
    if (async_tracker_ == nullptr)
        throw std::logic_error("DTE async tracker is not bound to frontend");
    return *async_tracker_;
}

const DteAsyncTracker &DteControlFrontend::Async() const {
    if (async_tracker_ == nullptr)
        throw std::logic_error("DTE async tracker is not bound to frontend");
    return *async_tracker_;
}

bool DteControlFrontend::HasToken(uint32_t token) const {
    return Async().HasToken(token);
}

uint64_t DteControlFrontend::IssueToken(
    uint32_t token, uint64_t payload_bits, DteDir direction,
    uint64_t spm_addr, uint64_t spm_size, uint32_t remote_peer,
    uint64_t remote_addr, uint32_t address_block) {
    if (dedicated_core_ != nullptr)
        return dedicated_core_->IssueToken(
            token, payload_bits, direction, spm_addr, spm_size, remote_peer,
            remote_addr, address_block);
    return Async().IssueToken(token, payload_bits, direction, spm_addr,
                              spm_size, remote_peer, remote_addr,
                              address_block);
}

void DteControlFrontend::WaitToken(uint32_t token) {
    if (dedicated_core_ != nullptr) {
        dedicated_core_->WaitToken(token);
        return;
    }
    Async().WaitToken(token);
}

bool DteControlFrontend::PollToken(uint32_t token) {
    if (dedicated_core_ != nullptr)
        return dedicated_core_->PollToken(token);
    return Async().PollToken(token);
}

void DteControlFrontend::Fence() {
    if (dedicated_core_ != nullptr) {
        dedicated_core_->Fence();
        return;
    }
    Async().Fence();
}

void DteControlFrontend::CancelToken(uint32_t token) {
    if (dedicated_core_ != nullptr) {
        dedicated_core_->CancelToken(token);
        return;
    }
    Async().CancelToken(token);
}

size_t DteControlFrontend::OutstandingTokenCount() const {
    return Async().OutstandingCount();
}
