#pragma once

#include "memory/dramsys_wrapper.h"
#include "memory/hbm_backend.h"

#include <deque>
#include <memory>
#include <list>
#include <string>
#include <string_view>
#include <tlm>
#include <tlm_utils/peq_with_cb_and_phase.h>
#include <tlm_utils/simple_initiator_socket.h>
#include <unordered_map>
#include <vector>

struct DRAMSysHBMAccessRecord {
    MemCommand command = MemCommand::kRead;
    uint64_t address = 0;
    unsigned channel = 0;
    unsigned rank = 0;
    unsigned bankgroup = 0;
    unsigned bank = 0;
    unsigned row = 0;
    unsigned column = 0;
    sc_core::sc_time submitted = sc_core::SC_ZERO_TIME;
    sc_core::sc_time issued = sc_core::SC_ZERO_TIME;
    sc_core::sc_time completed = sc_core::SC_ZERO_TIME;
    sc_core::sc_time service = sc_core::SC_ZERO_TIME;
    int status = -1;
};

class DRAMSysHBMBackend : public HBMBackend, public sc_core::sc_module,
                          public tlm::tlm_mm_interface {
public:
    SC_HAS_PROCESS(DRAMSysHBMBackend);
    DRAMSysHBMBackend(const sc_core::sc_module_name &n,
                      const std::string &dram_config_path,
                      std::string_view resource_directory =
                          "../DRAMSys/configs");
    ~DRAMSysHBMBackend() override;

    void Submit(const std::shared_ptr<HBMBackendTransaction> &tx) override;
    const HBMBackendStats &Stats() const override { return stats_; }
    uint64_t CapacityBytes() const;
    size_t InFlightCount() const { return inflight_.size(); }
    size_t PendingCount() const { return pending_.size(); }
    size_t PeakInFlightCount() const { return peak_inflight_; }
    const std::vector<DRAMSysHBMAccessRecord> &AccessTrace() const {
        return access_trace_;
    }
    DRAMSys::DecodedAddress DecodeBackendAddress(uint64_t address) const;
    uint64_t EncodeBackendAddress(DRAMSys::DecodedAddress address) const;

private:
    struct LogicalRequest {
        explicit LogicalRequest(std::shared_ptr<HBMBackendTransaction> t)
            : tx(std::move(t)) {}
        std::shared_ptr<HBMBackendTransaction> tx;
        size_t remaining = 0;
        size_t trace_index = 0;
        bool issued = false;
        sc_core::sc_time issued_at = sc_core::SC_ZERO_TIME;
        int status = 0;
        std::string error;
    };

    struct PhysicalRequest {
        std::shared_ptr<LogicalRequest> logical;
        uint64_t address = 0;
        std::vector<uint8_t> payload;
        std::vector<uint8_t> byte_enable;
        size_t logical_offset = 0;
        size_t burst_offset = 0;
        size_t copy_length = 0;
    };

    struct Inflight {
        Inflight(std::shared_ptr<PhysicalRequest> r,
                 tlm::tlm_mm_interface *mm)
            : request(std::move(r)), payload(mm) {}
        std::shared_ptr<PhysicalRequest> request;
        tlm::tlm_generic_payload payload;
        sc_core::sc_time issued = sc_core::SC_ZERO_TIME;
    };

    tlm::tlm_sync_enum nb_transport_bw(tlm::tlm_generic_payload &payload,
                                       tlm::tlm_phase &phase,
                                       sc_core::sc_time &delay);
    void peqCallback(tlm::tlm_generic_payload &payload,
                     const tlm::tlm_phase &phase);
    void TryIssue();
    void free(tlm::tlm_generic_payload *payload) override;
    void CleanupLoop();

    ::DRAMSys::Config::Configuration config_;
    gem5::memory::DRAMSysWrapper *dram_sys_wrapper_ = nullptr;
    tlm_utils::simple_initiator_socket<DRAMSysHBMBackend> initiator_socket_;
    tlm_utils::peq_with_cb_and_phase<DRAMSysHBMBackend> peq_;
    std::list<std::shared_ptr<PhysicalRequest>> pending_;
    std::unordered_map<tlm::tlm_generic_payload *, std::unique_ptr<Inflight>>
        inflight_;
    bool begin_req_in_progress_ = false;
    tlm::tlm_generic_payload *begin_req_payload_ = nullptr;
    std::deque<tlm::tlm_generic_payload *> retired_;
    sc_core::sc_event cleanup_event_;
    HBMBackendStats stats_;
    size_t peak_inflight_ = 0;
    std::vector<DRAMSysHBMAccessRecord> access_trace_;
    std::unordered_map<const HBMBackendTransaction *,
                       std::shared_ptr<LogicalRequest>>
        active_logical_;
};
