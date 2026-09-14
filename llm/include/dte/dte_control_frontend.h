#pragma once

#include "dte/dte_control_core.h"
#include "dte/dte_async_types.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>

class DTEUnit;
class DteAsyncTracker;
class DteMemoryBridge;

enum class DteControlMode : uint8_t {
    LEGACY_SHARED = 0,
    DUAL_DTE_DEDICATED,
};

// One call surface for both hardware modes.  The legacy constructor performs
// exactly the old WaitForCredit()+Issue sequence and adds no SystemC wait.
class DteControlFrontend {
public:
    explicit DteControlFrontend(DTEUnit &unit,
                                DteAsyncTracker *async_tracker = nullptr);
    DteControlFrontend(DTEUnit &unit, DteControlCore &dedicated_core,
                       DteAsyncTracker *async_tracker = nullptr);
    ~DteControlFrontend();

    DteControlMode mode() const { return mode_; }
    DteTransferHandle Issue(uint64_t payload_bits, DteDir direction);
    void WaitTransmitStart(DteTransferHandle handle);
    void Wait(DteTransferHandle handle);
    bool Release(DteTransferHandle handle);
    DteTransferSnapshot Snapshot(DteTransferHandle handle) const;
    uint32_t BitWidth() const;
    size_t OutstandingTransferCount() const;
    DteControlResidual Residual() const;
    void BindMemoryBridge(DteMemoryBridge *bridge);

    // Logical-token facade. Dedicated mode submits these operations through
    // the same bounded controller FIFO; legacy mode remains a direct path.
    bool HasToken(uint32_t token) const;
    uint64_t IssueToken(
        uint32_t token, uint64_t payload_bits, DteDir direction,
        uint64_t spm_addr, uint64_t spm_size,
        uint32_t remote_peer = DTE_ASYNC_INVALID_REMOTE_PEER,
        uint64_t remote_addr = 0, uint32_t address_block = 0);
    void WaitToken(uint32_t token);
    bool PollToken(uint32_t token);
    void Fence();
    void CancelToken(uint32_t token);
    size_t OutstandingTokenCount() const;

private:
    struct LegacyRecord;

    LegacyRecord &FindLegacy(DteTransferHandle handle);
    const LegacyRecord &FindLegacy(DteTransferHandle handle) const;
    DteAsyncTracker &Async();
    const DteAsyncTracker &Async() const;

    DteControlMode mode_ = DteControlMode::LEGACY_SHARED;
    DTEUnit &unit_;
    DteControlCore *dedicated_core_ = nullptr;
    DteAsyncTracker *async_tracker_ = nullptr;
    uint64_t next_legacy_handle_ = 0;
    std::map<uint64_t, std::unique_ptr<LegacyRecord>> legacy_records_;
};
