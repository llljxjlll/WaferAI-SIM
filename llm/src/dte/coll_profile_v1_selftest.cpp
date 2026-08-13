#include "dte/coll_profile_v1_selftest.h"

#include "dte/coll_profile_v1.h"

#include <array>
#include <functional>
#include <iostream>
#include <stdexcept>
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

bool ThrowsInvalid(const std::function<void()> &fn) {
    try {
        fn();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

constexpr std::array<CollOp, 9> kOps{{
    CollOp::P2P,          CollOp::SCATTER, CollOp::GATHER,
    CollOp::BROADCAST,    CollOp::ALLTOALL,
    CollOp::ALLGATHER,    CollOp::REDUCE,
    CollOp::REDUCESCATTER, CollOp::ALLREDUCE}};

constexpr std::array<NocCollProfile, 4> kProfiles{{
    NocCollProfile::BASELINE, NocCollProfile::BROADCAST_ONLY,
    NocCollProfile::REDUCE_ONLY,
    NocCollProfile::REDUCE_BROADCAST}};

struct MatrixRow {
    CollOp op;
    std::array<IsaV1ProfileBroadcastBackend, 4> broadcast;
    std::array<IsaV1ProfileReduceBackend, 4> reduce;
    std::array<size_t, 4> multicast_trees;
    std::array<size_t, 4> dca_trees;
};

constexpr IsaV1ProfileBroadcastBackend NA_B =
    IsaV1ProfileBroadcastBackend::NOT_APPLICABLE;
constexpr IsaV1ProfileBroadcastBackend U =
    IsaV1ProfileBroadcastBackend::UNICAST;
constexpr IsaV1ProfileBroadcastBackend M =
    IsaV1ProfileBroadcastBackend::MULTICAST;
constexpr IsaV1ProfileReduceBackend NA_R =
    IsaV1ProfileReduceBackend::NOT_APPLICABLE;
constexpr IsaV1ProfileReduceBackend E =
    IsaV1ProfileReduceBackend::ENDPOINT;
constexpr IsaV1ProfileReduceBackend D =
    IsaV1ProfileReduceBackend::DCA_OFFLOAD;

constexpr std::array<MatrixRow, 9> kMatrix{{
    {CollOp::P2P, {NA_B, NA_B, NA_B, NA_B}, {NA_R, NA_R, NA_R, NA_R},
     {0, 0, 0, 0}, {0, 0, 0, 0}},
    {CollOp::SCATTER, {NA_B, NA_B, NA_B, NA_B},
     {NA_R, NA_R, NA_R, NA_R}, {0, 0, 0, 0}, {0, 0, 0, 0}},
    {CollOp::GATHER, {NA_B, NA_B, NA_B, NA_B},
     {NA_R, NA_R, NA_R, NA_R}, {0, 0, 0, 0}, {0, 0, 0, 0}},
    {CollOp::BROADCAST, {U, M, U, M}, {NA_R, NA_R, NA_R, NA_R},
     {0, 1, 0, 1}, {0, 0, 0, 0}},
    {CollOp::ALLTOALL, {NA_B, NA_B, NA_B, NA_B},
     {NA_R, NA_R, NA_R, NA_R}, {0, 0, 0, 0}, {0, 0, 0, 0}},
    {CollOp::ALLGATHER, {U, M, U, M}, {NA_R, NA_R, NA_R, NA_R},
     {0, 4, 0, 4}, {0, 0, 0, 0}},
    {CollOp::REDUCE, {NA_B, NA_B, NA_B, NA_B}, {E, E, D, D},
     {0, 0, 0, 0}, {0, 0, 1, 1}},
    {CollOp::REDUCESCATTER, {NA_B, NA_B, NA_B, NA_B}, {E, E, D, D},
     {0, 0, 0, 0}, {0, 0, 4, 4}},
    {CollOp::ALLREDUCE, {U, M, U, M}, {E, E, D, D},
     {0, 4, 0, 4}, {0, 0, 4, 4}},
}};

IsaV1CollectiveProfileRequest Request(
    CollOp op, NocCollProfile profile,
    IsaV1CollectiveProfileCapabilities caps = {true, true, true},
    size_t group_size = 4) {
    return {op, group_size, profile, caps};
}

void TestFullMatrix() {
    for (const auto &row : kMatrix) {
        for (size_t profile_index = 0; profile_index < kProfiles.size();
             ++profile_index) {
            const auto result = PlanIsaV1CollectiveProfile(
                Request(row.op, kProfiles[profile_index]));
            const std::string cell =
                "matrix op=" + std::to_string(static_cast<int>(row.op)) +
                " profile=" + std::to_string(profile_index);
            Check(result.accepted &&
                      result.reject_reason ==
                          IsaV1ProfileRejectReason::NONE,
                  cell + " accepted");
            Check(result.broadcast_backend == row.broadcast[profile_index] &&
                      result.reduce_backend == row.reduce[profile_index],
                  cell + " selects exact backends");
            Check(result.endpoint_reduce_compute ==
                      (row.reduce[profile_index] == E),
                  cell + " endpoint compute is DCA-exclusive");
            Check(result.multicast_tree_count ==
                          row.multicast_trees[profile_index] &&
                      result.dca_reduce_tree_count ==
                          row.dca_trees[profile_index],
                  cell + " reports exact tree demand");
            Check(result.requires_multicast ==
                          (row.broadcast[profile_index] == M) &&
                      result.requires_dca ==
                          (row.reduce[profile_index] == D),
                  cell + " reports exact capabilities");
            Check(result.trace.find("status=accepted") !=
                          std::string::npos &&
                      result.trace.find("reason=accepted") !=
                          std::string::npos,
                  cell + " trace is explicit");
        }
    }
}

void TestSingleMemberBypass() {
    const IsaV1CollectiveProfileCapabilities no_caps{};
    for (CollOp op : kOps) {
        for (NocCollProfile profile : kProfiles) {
            const auto result = PlanIsaV1CollectiveProfile(
                Request(op, profile, no_caps, 1));
            Check(result.accepted &&
                      result.broadcast_backend ==
                          IsaV1ProfileBroadcastBackend::BYPASS &&
                      result.reduce_backend ==
                          IsaV1ProfileReduceBackend::BYPASS,
                  "N=1 bypass is accepted without capabilities");
            Check(!result.endpoint_reduce_compute &&
                      !result.requires_multicast && !result.requires_dca &&
                      result.multicast_tree_count == 0 &&
                      result.dca_reduce_tree_count == 0,
                  "N=1 bypass consumes no backend resource");
            Check(result.trace.find("profile=") == 0 &&
                      result.trace.find("broadcast=bypass") !=
                          std::string::npos &&
                      result.trace.find("reduce=bypass") !=
                          std::string::npos &&
                      result.trace.find("reason=group_size_one_bypass") !=
                          std::string::npos,
                  "N=1 trace preserves requested profile and actual bypass");
        }
    }
}

void TestCapabilityFailures() {
    const IsaV1CollectiveProfileCapabilities no_caps{};
    auto result = PlanIsaV1CollectiveProfile(Request(
        CollOp::BROADCAST, NocCollProfile::BROADCAST_ONLY, no_caps));
    Check(!result.accepted &&
              result.broadcast_backend ==
                  IsaV1ProfileBroadcastBackend::MULTICAST &&
              result.reject_reason ==
                  IsaV1ProfileRejectReason::MULTICAST_UNAVAILABLE,
          "missing multicast rejects without unicast fallback");

    result = PlanIsaV1CollectiveProfile(
        Request(CollOp::REDUCE, NocCollProfile::REDUCE_ONLY, no_caps));
    Check(!result.accepted &&
              result.reduce_backend ==
                  IsaV1ProfileReduceBackend::DCA_OFFLOAD &&
              !result.endpoint_reduce_compute &&
              result.reject_reason ==
                  IsaV1ProfileRejectReason::DCA_UNAVAILABLE,
          "missing DCA rejects without endpoint fallback");

    result = PlanIsaV1CollectiveProfile(Request(
        CollOp::REDUCESCATTER, NocCollProfile::REDUCE_ONLY, no_caps));
    Check(!result.accepted &&
              result.reject_reason ==
                  IsaV1ProfileRejectReason::REDUCE_SCATTER_DCA_UNSUPPORTED,
          "closed D-11 gate has an exact rejection reason");

    result = PlanIsaV1CollectiveProfile(Request(
        CollOp::REDUCESCATTER, NocCollProfile::REDUCE_ONLY,
        {false, false, true}));
    Check(!result.accepted &&
              result.reject_reason ==
                  IsaV1ProfileRejectReason::DCA_UNAVAILABLE,
          "open D-11 gate still requires generic DCA capability");

    result = PlanIsaV1CollectiveProfile(Request(
        CollOp::ALLREDUCE, NocCollProfile::REDUCE_BROADCAST, no_caps));
    Check(!result.accepted &&
              result.reject_reason ==
                  IsaV1ProfileRejectReason::MULTICAST_UNAVAILABLE,
          "combined missing capabilities use deterministic multicast-first reason");
    result = PlanIsaV1CollectiveProfile(Request(
        CollOp::ALLREDUCE, NocCollProfile::REDUCE_BROADCAST,
        {true, false, false}));
    Check(!result.accepted &&
              result.reject_reason ==
                  IsaV1ProfileRejectReason::DCA_UNAVAILABLE,
          "combined profile reports DCA after multicast is available");
    Check(result.trace.find("status=rejected") != std::string::npos &&
              result.trace.find("reduce=dca_offload") !=
                  std::string::npos &&
              result.trace.find("reason=dca_unavailable") !=
                  std::string::npos,
          "rejected trace exposes requested backend and exact reason");
}

void TestForbiddenAccelerationsAndDeterminism() {
    const IsaV1CollectiveProfileCapabilities no_caps{};
    for (CollOp op : {CollOp::P2P, CollOp::SCATTER, CollOp::GATHER,
                      CollOp::ALLTOALL}) {
        const auto result = PlanIsaV1CollectiveProfile(Request(
            op, NocCollProfile::REDUCE_BROADCAST, no_caps));
        Check(result.accepted &&
                  result.broadcast_backend ==
                      IsaV1ProfileBroadcastBackend::NOT_APPLICABLE &&
                  result.reduce_backend ==
                      IsaV1ProfileReduceBackend::NOT_APPLICABLE &&
                  result.multicast_tree_count == 0 &&
                  result.dca_reduce_tree_count == 0,
              "P2P/Scatter/Gather/AllToAll forbid multicast and DCA");
    }

    const auto scatter = PlanIsaV1CollectiveProfile(Request(
        CollOp::SCATTER, NocCollProfile::BROADCAST_ONLY,
        {true, true, true}));
    Check(scatter.broadcast_backend ==
                  IsaV1ProfileBroadcastBackend::NOT_APPLICABLE &&
              !scatter.requires_multicast,
          "Scatter never enters multicast even when available");

    const auto first = PlanIsaV1CollectiveProfile(Request(
        CollOp::ALLREDUCE, NocCollProfile::REDUCE_BROADCAST,
        {true, true, true}, 7));
    const auto second = PlanIsaV1CollectiveProfile(Request(
        CollOp::ALLREDUCE, NocCollProfile::REDUCE_BROADCAST,
        {true, true, true}, 7));
    Check(first == second && first.trace == second.trace &&
              first.multicast_tree_count == 7 &&
              first.dca_reduce_tree_count == 7,
          "identical planner input is deterministic");
}

void TestMalformedContractsAndNames() {
    Check(ThrowsInvalid([] {
              PlanIsaV1CollectiveProfile(Request(
                  CollOp::P2P, NocCollProfile::BASELINE,
                  {true, true, true}, 0));
          }),
          "zero group size is rejected");
    Check(ThrowsInvalid([] {
              PlanIsaV1CollectiveProfile(Request(
                  static_cast<CollOp>(255), NocCollProfile::BASELINE));
          }),
          "invalid operation enum is rejected");
    Check(ThrowsInvalid([] {
              PlanIsaV1CollectiveProfile(Request(
                  CollOp::P2P, static_cast<NocCollProfile>(255)));
          }),
          "invalid profile enum is rejected");
    Check(ThrowsInvalid([] {
              IsaV1ProfileBroadcastBackendName(
                  static_cast<IsaV1ProfileBroadcastBackend>(255));
          }) &&
              ThrowsInvalid([] {
                  IsaV1ProfileReduceBackendName(
                      static_cast<IsaV1ProfileReduceBackend>(255));
              }) &&
              ThrowsInvalid([] {
                  IsaV1ProfileRejectReasonName(
                      static_cast<IsaV1ProfileRejectReason>(255));
              }),
          "trace enum names reject invalid values");
}

} // namespace

int RunIsaV1CollectiveProfileSelfTest() {
    failures = checks = 0;
    std::cout << "==== ISA-v1 collective profile planner self-test ===="
              << std::endl;
    TestFullMatrix();
    TestSingleMemberBypass();
    TestCapabilityFailures();
    TestForbiddenAccelerationsAndDeterminism();
    TestMalformedContractsAndNames();
    std::cout << "ISA-v1 collective profile planner self-test: "
              << checks - failures << "/" << checks << " checks passed"
              << std::endl;
    return failures;
}

#ifdef ISA_V1_COLL_PROFILE_SELFTEST_MAIN
int main() { return RunIsaV1CollectiveProfileSelfTest(); }
#endif
