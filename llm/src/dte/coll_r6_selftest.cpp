#include "dte/coll_config.h"
#include "dte/coll_refactor_contract.h"

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

NocCollectiveConfig ReduceOnly() {
    return ParseNocCollectiveConfig(
        {{"collective", {{"enabled", true},
                          {"profile", "reduce_only"}}}});
}
} // namespace

int RunCollR6SelfTest() {
    using namespace coll_refactor;
    failures = checks = 0;
    std::cout << "==== NoC collective refactor R6 reduce-only self-test ===="
              << std::endl;

    const auto config = ReduceOnly();
    Check(config.profile == NocCollProfile::REDUCE_ONLY &&
              config.UsesDcaOffload() && !config.UsesMulticast(),
          "reduce_only selects stream DCA without multicast");

    const auto broadcast = NocCollTreeUseFor(config, CollOp::BROADCAST);
    const auto allgather = NocCollTreeUseFor(config, CollOp::ALLGATHER);
    Check(!broadcast.multicast && !broadcast.reduce &&
              !allgather.multicast && !allgather.reduce,
          "Broadcast and AllGather remain ordinary unicast");

    const auto reduce = NocCollTreeUseFor(config, CollOp::REDUCE);
    const auto reduce_scatter =
        NocCollTreeUseFor(config, CollOp::REDUCESCATTER);
    Check(reduce.reduce && !reduce.multicast &&
              reduce_scatter.reduce && !reduce_scatter.multicast,
          "Reduce and ReduceScatter program only a reduce tree");

    const auto allreduce = NocCollTreeUseFor(config, CollOp::ALLREDUCE);
    Check(allreduce.reduce && !allreduce.multicast,
          "AllReduce result distribution cannot enter multicast");

    const auto singleton =
        ComputeVectorWork(67, 1, 512, CollDType::UINT8);
    const auto four_rank =
        ComputeVectorWork(67, 4, 512, CollDType::UINT8);
    Check(singleton.total_issues == 0 && singleton.vector_beats == 2 &&
              four_rank.total_issues == 6 &&
              four_rank.pairwise_issues_per_beat == 3,
          "N=1 bypass and N=4 vector issue accounting are explicit");

    Check(CollRankCountOffset(67, 4, 0) ==
                  std::make_pair<uint64_t, uint64_t>(17, 0) &&
              CollRankCountOffset(67, 4, 1) ==
                  std::make_pair<uint64_t, uint64_t>(17, 17) &&
              CollRankCountOffset(67, 4, 2) ==
                  std::make_pair<uint64_t, uint64_t>(17, 34) &&
              CollRankCountOffset(67, 4, 3) ==
                  std::make_pair<uint64_t, uint64_t>(16, 51),
          "ReduceScatter quotient/remainder slices cover the whole tensor");

    Check(NocCollProfileName(config.profile) == "reduce_only" &&
              NocCollReduceBackendName(config.reduce_backend) ==
                  "dca_offload" &&
              NocCollBroadcastBackendName(config.broadcast_backend) ==
                  "unicast",
          "profile observability reports the selected backends");

    std::cout << "R6 self-test: " << checks - failures << "/" << checks
              << " checks passed" << std::endl;
    return failures;
}
