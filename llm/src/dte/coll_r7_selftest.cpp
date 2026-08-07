#include "dte/coll_config.h"

#include "nlohmann/json.hpp"

#include <iostream>
#include <string>

namespace {
int failures = 0;
int checks = 0;

void Check(bool ok, const std::string &name) {
    ++checks;
    if (!ok) ++failures;
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name
              << std::endl;
}

NocCollectiveConfig Combined() {
    return ParseNocCollectiveConfig(
        {{"collective", {{"enabled", true},
                          {"profile", "reduce_broadcast"}}}});
}
} // namespace

int RunCollR7SelfTest() {
    failures = checks = 0;
    std::cout << "==== NoC collective refactor R7 reduce+broadcast self-test ===="
              << std::endl;

    const auto config = Combined();
    Check(config.profile == NocCollProfile::REDUCE_BROADCAST &&
              config.UsesDcaOffload() && config.UsesMulticast(),
          "reduce_broadcast selects stream DCA plus multicast");

    const auto broadcast = NocCollTreeUseFor(config, CollOp::BROADCAST);
    const auto allgather = NocCollTreeUseFor(config, CollOp::ALLGATHER);
    Check(broadcast.multicast && !broadcast.reduce &&
              allgather.multicast && !allgather.reduce,
          "Broadcast and per-source AllGather use multicast only");

    const auto reduce = NocCollTreeUseFor(config, CollOp::REDUCE);
    Check(reduce.reduce && !reduce.multicast,
          "Reduce terminates at root without an orphan multicast tree");

    const auto reduce_scatter =
        NocCollTreeUseFor(config, CollOp::REDUCESCATTER);
    Check(reduce_scatter.reduce && !reduce_scatter.multicast,
          "ReduceScatter keeps result distribution on ordinary Scatter");

    const auto allreduce = NocCollTreeUseFor(config, CollOp::ALLREDUCE);
    Check(allreduce.reduce && allreduce.multicast,
          "AllReduce composes reduce tree and one result multicast tree");

    Check(NocCollProfileName(config.profile) == "reduce_broadcast" &&
              NocCollReduceWireName(config.reduce_wire) == "stream_v2",
          "combined profile cannot silently fall back to legacy reduce wire");

    std::cout << "R7 self-test: " << checks - failures << "/" << checks
              << " checks passed" << std::endl;
    return failures;
}
