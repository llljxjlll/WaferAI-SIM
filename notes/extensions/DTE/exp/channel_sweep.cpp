// Small, self-contained experiment: how does DTEUnit::channel_count affect
// the completion time of a fixed communication workload?
//
// Workload: a single core issues N back-to-back SPM_TO_REMOTE transfers
// (all at simulation time 0, mirroring how send_para_logic() in
// llm/src/workercore/logic.cpp fires a burst of SEND_DATA transfers without
// waiting between them). We drive the real DTEUnit scheduler (no
// reimplementation) with dte_channel_count swept over a range of values and
// record each transfer's completion cycle.
//
// This links directly against the project's own llm/src/dte/dte_unit.cpp
// object file, so it exercises the exact same admission/arbitration code
// path as the simulator (see build_and_run.sh for the compile command).

#include "dte/dte_unit.h"
#include "macros/macros.h"
#include "trace/Event_engine.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace {

// Fixed workload shared by every channel_count in the sweep.
constexpr int kNumTransfers = 12;
constexpr uint32_t kBitWidth = 256;    // DTE data-path width (bits/cycle)
constexpr uint64_t kPayloadBits = 768; // -> transmit = 3 cycles
constexpr uint64_t kGammaCycles = 4;   // fixed launch overhead component
constexpr uint64_t kTauCycles = 2;     // fixed launch overhead component

const std::vector<uint32_t> kChannelCounts = {1, 2, 3, 4, 6, 8, 12, 16};

long long CycleOf(const sc_time &t) {
    return static_cast<long long>(t.value() / sc_time(CYCLE, SC_NS).value());
}

// Issues kNumTransfers identical transfers back-to-back at t=0 against one
// DTEUnit configured with a given channel_count, then records completion.
struct BurstProbe : sc_module {
    std::unique_ptr<DTEUnit> dte;
    std::vector<DteTransferContext *> contexts;

    SC_HAS_PROCESS(BurstProbe);
    BurstProbe(sc_module_name name, const DTEConfig &config)
        : sc_module(name) {
        dte = std::make_unique<DTEUnit>("dte", config);
        SC_THREAD(drive);
    }

    void drive() {
        for (int i = 0; i < kNumTransfers; ++i) {
            DteTransferContext &ctx =
                dte->Issue(kPayloadBits, DteDir::SPM_TO_REMOTE);
            contexts.push_back(&ctx);
        }
    }
};

} // namespace

int sc_main(int, char *[]) {
    std::vector<std::unique_ptr<BurstProbe>> probes;
    for (uint32_t channels : kChannelCounts) {
        DTEConfig config{channels, kBitWidth, kGammaCycles, kTauCycles};
        auto probe = std::make_unique<BurstProbe>(
            sc_module_name(("probe_c" + std::to_string(channels)).c_str()),
            config);
        probes.push_back(std::move(probe));
    }

    // Long enough for the slowest (channel_count=1, fully serial) config to
    // drain: N * (launch + transmit) cycles, doubled for margin.
    const uint64_t transmit_cycles =
        (kPayloadBits + kBitWidth - 1) / kBitWidth;
    const uint64_t launch_cycles = kGammaCycles + kTauCycles;
    sc_start(static_cast<double>(2 * kNumTransfers *
                                  (launch_cycles + transmit_cycles) * CYCLE),
              SC_NS);

    std::cout << "channel_count,transfer_index,completion_cycle,"
                 "max_active_count\n";
    for (auto &probe : probes) {
        const uint32_t channels = probe->dte->config().channel_count;
        for (size_t i = 0; i < probe->contexts.size(); ++i) {
            const DteTransferContext *ctx = probe->contexts[i];
            const bool completed =
                ctx->state == DteTransferState::COMPLETED;
            std::cout << channels << "," << i << ","
                      << (completed ? CycleOf(ctx->completion_time) : -1)
                      << "," << probe->dte->MaxActiveCount() << "\n";
        }
    }
    return 0;
}
