#pragma once

#include "dte/dte_payload.h"
#include "dte/dte_types.h"
#include "systemc.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <vector>

class Event_engine;

DTEConfig MakeDTEConfig(uint32_t channel_count, uint32_t bit_width_bits,
                        int64_t gamma_ns, int64_t tau_launch_ns);

class DTEUnit : public sc_module {
public:
    SC_HAS_PROCESS(DTEUnit);

    DTEUnit(const sc_module_name &name, const DTEConfig &config,
            int core_id = -1, Event_engine *event_engine = nullptr);

    DteTransferContext &Issue(uint64_t payload_bits, DteDir dir);
    DteTransferContext *TryIssue(uint64_t payload_bits, DteDir dir);
    void WaitForCredit();
    bool CanAccept() const;
    bool Cancel(uint64_t xfer_id);
    bool Release(uint64_t xfer_id);

    const DTEConfig &config() const { return config_; }
    const DteStatistics &statistics() const { return statistics_; }
    size_t PendingCount() const { return pending_.size(); }
    size_t ActiveCount() const { return active_count_; }
    size_t MaxActiveCount() const { return max_active_count_; }
    size_t InflightCount() const { return inflight_count_; }
    uint64_t CompletedCount() const { return completed_count_; }
    bool BusBusy() const;
    bool PortBusy(DtePort port) const;
    double AveragePowerMw() const;

    static void ValidateConfig(const DTEConfig &config);
    static uint32_t RequiredPorts(DteDir dir, bool fine_grained);

private:
    void scheduler();
    void admitPending();
    bool finishLaunches();
    bool startReadyTransfers();
    bool finishPortServices();
    int findFreeSlot() const;
    int chooseReadySlot() const;
    bool resourcesAvailable(uint32_t mask) const;
    sc_time launchTime() const;
    sc_time serviceTime(uint64_t payload_bits, DtePort port) const;
    uint32_t portWidth(DtePort port) const;
    double portEnergyPerBit(DtePort port) const;
    double computeAreaUm2() const;
    size_t descriptorCapacity() const;
    void traceStage(const DteTransferContext &ctx, const char *stage,
                    const char *phase);
    void tracePort(const DteTransferContext &ctx, DtePort port,
                   const char *phase);
    void traceStatistics(const DteTransferContext &ctx);

    DTEConfig config_;
    int core_id_;
    Event_engine *event_engine_;

    std::list<std::unique_ptr<DteTransferContext>> contexts_;
    std::deque<DteTransferContext *> pending_;
    std::vector<DteTransferContext *> active_;
    std::array<DteTransferContext *,
               static_cast<std::size_t>(DtePort::COUNT)> resource_owners_{};
    size_t active_count_ = 0;
    size_t max_active_count_ = 0;
    size_t inflight_count_ = 0;
    uint64_t completed_count_ = 0;
    uint64_t next_xfer_id_ = 0;
    int last_served_slot_ = -1;
    DteStatistics statistics_;
    sc_time first_issue_time_ = SC_ZERO_TIME;
    sc_time last_completion_time_ = SC_ZERO_TIME;
    bool have_issue_time_ = false;

    sc_event state_changed_;
    sc_event credit_available_;
};

int RunDTEV0SelfTest();
int RunDTEV4SelfTest();
