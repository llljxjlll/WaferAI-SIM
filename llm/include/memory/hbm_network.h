#pragma once

#include "defs/enums.h"
#include "memory/core_mem_adapter.h"
#include "memory/hbm_runtime.h"

#include <deque>
#include <map>
#include <memory>
#include <systemc>
#include <utility>
#include <vector>

class RouterMonitor;
class RouterUnit;
class HBMEndpointAgent;

struct HBMNetworkStats {
    uint64_t logical_requests = 0;
    uint64_t reads = 0;
    uint64_t writes = 0;
    uint64_t request_bytes = 0;
    uint64_t response_bytes = 0;
    uint64_t request_flits = 0;
    uint64_t response_flits = 0;
    uint64_t noc_hops = 0;
    uint64_t c2c_request_hops = 0;
    uint64_t c2c_response_hops = 0;
    uint64_t injection_stalls = 0;
    uint64_t shared_noc_contention = 0;
    uint64_t mem_mem_noc_contention = 0;
};

// R4 shared transport. Core adapters inject request flits into their local
// router; MEM attachment agents consume them at the home die and reinject the
// response on the independent control VC.
class HBMNetwork : public sc_core::sc_module, public MemTransport {
public:
    SC_HAS_PROCESS(HBMNetwork);
    HBMNetwork(const sc_core::sc_module_name &name, RouterMonitor &routers,
               HBMRuntime &runtime, unsigned workers_per_endpoint = 4,
               unsigned completed_queue_depth = 64);
    ~HBMNetwork() override;

    MemMsg Exchange(const MemMsg &request) override;
    bool TryAcceptRequestFlit(int router_id, const MemWireFlit &wire);
    bool TryAcceptResponseFlit(int router_id, const MemWireFlit &wire);
    void InjectResponse(int attachment_router,
                        const std::vector<MemWireFlit> &wire);
    void RecordHop(int router_id, int output_dir, const MemWireFlit &wire);
    void RecordSharedNocContention() { ++stats_.shared_noc_contention; }
    void RecordMemNocContention() { ++stats_.mem_mem_noc_contention; }

    const HBMNetworkStats &Stats() const { return stats_; }
    size_t Residual() const;

private:
    struct ResponseWaiter {
        sc_core::sc_event done;
        MemMsg response;
        bool complete = false;
    };
    using TxKey = std::pair<int, int>; // source_core, txid

    RouterMonitor &routers_;
    HBMRuntime &runtime_;
    std::map<TxKey, std::unique_ptr<ResponseWaiter>> waiters_;
    std::map<TxKey, std::vector<MemWireFlit>> response_assemblies_;
    std::map<std::pair<int, int>, HBMEndpointAgent *> endpoint_index_;
    std::vector<std::unique_ptr<HBMEndpointAgent>> agents_;
    HBMNetworkStats stats_;
};

HBMNetwork *ActiveHBMNetwork();

// Returns the output direction for one independently routable MEM flit.
Directions MemFlitNextHop(const MemWireFlit &wire, int router_id);
