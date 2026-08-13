#include "dte/dte_async.h"

#include "defs/spec.h"
#include "prims/norm_prims.h"
#include "utils/config_utils.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
template <typename Exception, typename F> bool ThrowsExpected(F &&fn) {
    try {
        fn();
    } catch (const Exception &) {
        return true;
    } catch (...) {
    }
    return false;
}

class DteV3Probe : public sc_module {
public:
    SC_HAS_PROCESS(DteV3Probe);

    DteV3Probe(const sc_module_name &name)
        : sc_module(name), overlap_unit("v3_overlap_unit", Config(2)),
          overlap("v3_overlap_tracker", overlap_unit),
          select_unit("v3_select_unit", Config(2)),
          select("v3_select_tracker", select_unit),
          hazard_unit("v3_hazard_unit", Config(2)),
          hazard("v3_hazard_tracker", hazard_unit),
          cancel_unit("v3_cancel_unit", Config(1)),
          cancel("v3_cancel_tracker", cancel_unit),
          scan1_unit("v3_scan1_unit", Config(1)),
          scan1("v3_scan1_tracker", scan1_unit),
          scan2_unit("v3_scan2_unit", Config(2)),
          scan2("v3_scan2_tracker", scan2_unit),
          scan4_unit("v3_scan4_unit", Config(4)),
          scan4("v3_scan4_tracker", scan4_unit) {
        SC_THREAD(run);
    }

    int failures = 0;
    int checks = 0;

private:
    static DTEConfig Config(uint32_t channels) {
        DTEConfig config;
        config.channel_count = channels;
        config.bit_width_bits = 128;
        config.gamma_cycles = 2;
        config.tau_launch_cycles = 1;
        return config;
    }

    void Check(bool condition, const std::string &name) {
        ++checks;
        std::cout << "  [" << (condition ? " ok " : "FAIL") << "] "
                  << name << "\n";
        if (!condition)
            ++failures;
    }

    void run() {
        const sc_time overlap_issue = sc_time_stamp();
        const uint64_t first_xfer = overlap.IssueToken(
            10, 1024, DteDir::SPM_TO_REMOTE, 0, 128);
        Check(overlap.OutstandingCount() == 1 &&
                  overlap.Record(10).xfer_id == first_xfer,
              "async issue returns a stable logical-token mapping");
        Check(!overlap.PollToken(10),
              "poll is non-blocking before completion");
        const sc_time compute_begin = sc_time_stamp();
        wait(8, SC_NS); // independent compute window
        const sc_time compute_end = sc_time_stamp();
        Check(compute_begin < overlap.Record(10).context->completion_time ||
                  overlap.Record(10).context->state !=
                      DteTransferState::COMPLETED,
              "independent compute executes while DMA is outstanding");
        Check(!overlap.PollToken(10),
              "poll still observes an incomplete long transfer");
        overlap.WaitToken(10);
        Check(sc_time_stamp() == overlap_issue + sc_time(22, SC_NS),
              "dependency wait starts compute-independent DMA at exact time");
        Check(compute_end < sc_time_stamp() && overlap.OutstandingCount() == 0,
              "wait consumes only its completed token");

        const uint64_t reuse0 = overlap.IssueToken(
            70, 128, DteDir::SPM_TO_REMOTE, 4096, 16);
        overlap.WaitToken(70);
        const uint64_t reuse1 = overlap.IssueToken(
            70, 128, DteDir::SPM_TO_REMOTE, 4096, 16);
        overlap.WaitToken(70);
        Check(reuse1 > reuse0,
              "logical token reuse receives a fresh physical xfer id");

        select.IssueToken(20, 128, DteDir::SPM_TO_REMOTE, 0, 16);
        select.IssueToken(21, 4096, DteDir::SPM_TO_REMOTE, 256, 512);
        select.WaitToken(20);
        Check(select.HasToken(21) && !select.PollToken(21),
              "waiting one token does not wait or consume another token");
        const sc_time selective_compute_begin = sc_time_stamp();
        wait(4, SC_NS);
        Check(selective_compute_begin < sc_time_stamp() &&
                  select.HasToken(21),
              "compute continues after a selective wait");
        select.Fence();
        Check(select.OutstandingCount() == 0 &&
                  sc_time_stamp() >= selective_compute_begin + sc_time(4, SC_NS),
              "fence drains every prior outstanding token");

        const sc_time read_pair_issue = sc_time_stamp();
        hazard.IssueToken(30, 1024, DteDir::SPM_TO_REMOTE, 1000, 128);
        hazard.IssueToken(31, 1024, DteDir::SPM_TO_REMOTE, 1000, 128);
        Check(sc_time_stamp() == read_pair_issue &&
                  hazard.OutstandingCount() == 2,
              "overlapping read/read ranges remain concurrent");
        hazard.Fence();

        const sc_time raw_begin = sc_time_stamp();
        hazard.IssueToken(40, 2048, DteDir::SPM_TO_REMOTE, 2000, 256);
        hazard.IssueToken(41, 2048, DteDir::REMOTE_TO_SPM, 2000, 256);
        Check(sc_time_stamp() > raw_begin && hazard.HasToken(40) &&
                  hazard.HasToken(41),
              "RAW hazard waits without consuming either logical token");
        hazard.Fence();

        const sc_time war_begin = sc_time_stamp();
        hazard.IssueToken(42, 2048, DteDir::REMOTE_TO_SPM, 3000, 256);
        hazard.IssueToken(43, 2048, DteDir::SPM_TO_REMOTE, 3000, 256);
        Check(sc_time_stamp() > war_begin && hazard.HasToken(42) &&
                  hazard.HasToken(43),
              "WAR hazard waits without consuming either logical token");
        hazard.Fence();

        const sc_time waw_begin = sc_time_stamp();
        hazard.IssueToken(44, 2048, DteDir::REMOTE_TO_SPM, 4000, 256);
        hazard.IssueToken(45, 2048, DteDir::REMOTE_TO_SPM, 4000, 256);
        Check(sc_time_stamp() > waw_begin && hazard.HasToken(44) &&
                  hazard.HasToken(45),
              "WAW hazard serializes writes without consuming their tokens");
        hazard.Fence();

        cancel.IssueToken(50, 4096, DteDir::SPM_TO_REMOTE, 0, 512);
        cancel.IssueToken(51, 128, DteDir::SPM_TO_REMOTE, 1024, 16);
        cancel.CancelToken(51);
        Check(!cancel.HasToken(51) && cancel.OutstandingCount() == 1,
              "pending cancellation removes exactly one descriptor");
        cancel.WaitToken(50);
        Check(cancel.OutstandingCount() == 0 &&
                  cancel_unit.PendingCount() == 0 &&
                  cancel_unit.ActiveCount() == 0,
              "cancel plus wait leaves no DTE context active");

        cancel.IssueToken(52, 1024, DteDir::SPM_TO_REMOTE, 2048, 128);
        while (cancel.Record(52).context->state ==
               DteTransferState::PENDING)
            wait(SC_ZERO_TIME);
        Check(ThrowsExpected<std::runtime_error>(
                  [&]() { cancel.CancelToken(52); }),
              "active cancellation is rejected explicitly");
        cancel.WaitToken(52);

        overlap.IssueToken(60, 128, DteDir::SPM_TO_REMOTE, 8192, 16);
        Check(ThrowsExpected<std::invalid_argument>([&]() {
                  overlap.IssueToken(60, 128, DteDir::SPM_TO_REMOTE,
                                     8448, 16);
              }),
              "duplicate logical token is rejected before issuing");
        overlap.WaitToken(60);
        Check(ThrowsExpected<std::invalid_argument>(
                  [&]() { overlap.WaitToken(999); }) &&
                  ThrowsExpected<std::invalid_argument>([&]() {
                      overlap.IssueToken(61, 129, DteDir::SPM_TO_REMOTE,
                                         9000, 16);
                  }),
              "invalid dependency and undersized ranges fail without leaks");
        Check(overlap.OutstandingCount() == 0 &&
                  overlap_unit.ActiveCount() == 0,
              "exception paths preserve an empty token/context table");

        for (uint32_t token = 0; token < 4; ++token) {
            const uint64_t addr = uint64_t(token) * 1024;
            scan1.IssueToken(token, 2048, DteDir::SPM_TO_REMOTE, addr, 256);
            scan2.IssueToken(token, 2048, DteDir::SPM_TO_REMOTE, addr, 256);
            scan4.IssueToken(token, 2048, DteDir::SPM_TO_REMOTE, addr, 256);
        }
        while (scan1_unit.ActiveCount() != 1 ||
               scan2_unit.ActiveCount() != 2 ||
               scan4_unit.ActiveCount() != 4)
            wait(SC_ZERO_TIME);
        Check(scan1_unit.ActiveCount() == 1 &&
                  scan2_unit.ActiveCount() == 2 &&
                  scan4_unit.ActiveCount() == 4,
              "channel=1/2/4 bounds active outstanding transfers exactly");
        scan1.Fence();
        scan2.Fence();
        scan4.Fence();
        Check(scan1_unit.MaxActiveCount() == 1 &&
                  scan2_unit.MaxActiveCount() == 2 &&
                  scan4_unit.MaxActiveCount() == 4,
              "channel scan records exact maximum active counts");
        Check(scan1.OutstandingCount() == 0 &&
                  scan2.OutstandingCount() == 0 &&
                  scan4.OutstandingCount() == 0 &&
                  scan1_unit.ActiveCount() == 0 &&
                  scan2_unit.ActiveCount() == 0 &&
                  scan4_unit.ActiveCount() == 0,
              "all V3a self-test trackers drain at simulation end");

        sc_stop();
    }

    DTEUnit overlap_unit;
    DteAsyncTracker overlap;
    DTEUnit select_unit;
    DteAsyncTracker select;
    DTEUnit hazard_unit;
    DteAsyncTracker hazard;
    DTEUnit cancel_unit;
    DteAsyncTracker cancel;
    DTEUnit scan1_unit;
    DteAsyncTracker scan1;
    DTEUnit scan2_unit;
    DteAsyncTracker scan2;
    DTEUnit scan4_unit;
    DteAsyncTracker scan4;
};
} // namespace

int RunDTEV3SelfTest() {
    std::cout << "==== DTE V3a async self-test ====\n";
    int failures = 0;
    int checks = 0;
    auto check = [&](bool condition, const std::string &name) {
        ++checks;
        std::cout << "  [" << (condition ? " ok " : "FAIL") << "] "
                  << name << "\n";
        if (!condition)
            ++failures;
    };

    Dte_async_prim issue;
    issue.op = DteAsyncOp::ISSUE;
    issue.token = 0xfedcba98U;
    issue.payload_bits = 0x123456789abcdef0ULL;
    issue.direction = DteDir::REMOTE_TO_SPM;
    issue.spm_addr = 0x23456789abcdef01ULL;
    issue.spm_size = 0x0123456789abcdefULL;
    const auto issue_segments = issue.serialize();
    Dte_async_prim issue_wire;
    issue_wire.deserialize(issue_segments);
    check(issue_wire.op == issue.op && issue_wire.token == issue.token &&
              issue_wire.payload_bits == issue.payload_bits &&
              issue_wire.direction == issue.direction &&
              issue_wire.spm_addr == issue.spm_addr &&
              issue_wire.spm_size == issue.spm_size &&
              issue_wire.remote_peer == DTE_ASYNC_INVALID_REMOTE_PEER &&
              issue_segments.size() == 4,
          "Dte_async issue uses stable PrimId framing and round-trips");

    Dte_async_prim fence;
    fence.op = DteAsyncOp::FENCE;
    Dte_async_prim fence_wire;
    fence_wire.deserialize(fence.serialize());
    check(fence_wire.op == DteAsyncOp::FENCE && fence_wire.token == 0 &&
              fence_wire.payload_bits == 0 && fence_wire.spm_size == 0 &&
              fence_wire.remote_peer == DTE_ASYNC_INVALID_REMOTE_PEER &&
              fence_wire.remote_addr == 0 && fence_wire.address_block == 0,
          "Dte_async fence has a canonical zero-payload wire encoding");
    check(ThrowsExpected<std::invalid_argument>([&]() {
              Dte_async_prim malformed;
              malformed.deserialize({sc_bv<128>(0)});
          }),
          "Dte_async rejects a truncated wire encoding");

    const bool old_use_dte = SPEC_USE_BEHA_DTE;
    const bool old_streaming = SPEC_DTE_STREAMING;
    const bool old_async = SPEC_DTE_ASYNC;
    const bool old_fine_grained = SPEC_DTE_V4_RESOURCES;
    const bool old_parallel = SPEC_SEND_RECV_PARALLEL;
    const SIM_MODE old_mode = SYSTEM_MODE;

    SPEC_USE_BEHA_DTE = false;
    SPEC_DTE_STREAMING = false;
    SPEC_DTE_ASYNC = false;
    SPEC_SEND_RECV_PARALLEL = false;
    SYSTEM_MODE = SIM_DATAFLOW;
    bool accepted = true;
    try {
        ParseSimulationConfig(nlohmann::json{
            {"noc", {{"send_recv_parallel", false}}},
            {"dte", {{"use_beha_dte", true}, {"streaming", false},
                     {"async", true}}}});
    } catch (...) {
        accepted = false;
    }
    check(accepted && SPEC_USE_BEHA_DTE && SPEC_DTE_ASYNC,
          "DTE V3a accepts sequential dataflow async mode");

    SPEC_USE_BEHA_DTE = false;
    SPEC_DTE_STREAMING = false;
    SPEC_DTE_ASYNC = false;
    SPEC_SEND_RECV_PARALLEL = false;
    SYSTEM_MODE = SIM_DATAFLOW;
    check(ThrowsExpected<std::invalid_argument>([]() {
              ParseSimulationConfig(nlohmann::json{
                  {"dte", {{"use_beha_dte", false}, {"async", true}}}});
          }),
          "DTE V3a rejects async mode while DTE is disabled");

    SPEC_USE_BEHA_DTE = false;
    SPEC_DTE_STREAMING = false;
    SPEC_DTE_ASYNC = false;
    SPEC_SEND_RECV_PARALLEL = false;
    SYSTEM_MODE = SIM_PD;
    check(ThrowsExpected<std::invalid_argument>([]() {
              ParseSimulationConfig(nlohmann::json{
                  {"dte", {{"use_beha_dte", true}, {"async", true}}}});
          }),
          "DTE V3a rejects async mode outside dataflow");

    SPEC_USE_BEHA_DTE = false;
    SPEC_DTE_STREAMING = false;
    SPEC_DTE_ASYNC = false;
    SPEC_SEND_RECV_PARALLEL = false;
    SYSTEM_MODE = SIM_DATAFLOW;
    check(ThrowsExpected<std::invalid_argument>([]() {
              ParseSimulationConfig(nlohmann::json{
                  {"dte", {{"use_beha_dte", true}, {"streaming", true},
                           {"async", true}}}});
          }),
          "DTE V3a rejects async plus streaming");

    SPEC_USE_BEHA_DTE = false;
    SPEC_DTE_STREAMING = false;
    SPEC_DTE_ASYNC = false;
    SPEC_SEND_RECV_PARALLEL = false;
    SYSTEM_MODE = SIM_DATAFLOW;
    check(ThrowsExpected<std::invalid_argument>([]() {
              ParseSimulationConfig(nlohmann::json{
                  {"noc", {{"send_recv_parallel", true}}},
                  {"dte", {{"use_beha_dte", true}, {"async", true}}}});
          }),
          "DTE V3a rejects the parallel dispatcher");

    SPEC_USE_BEHA_DTE = false;
    SPEC_DTE_STREAMING = false;
    SPEC_DTE_ASYNC = false;
    SPEC_DTE_V4_RESOURCES = false;
    SPEC_SEND_RECV_PARALLEL = false;
    SYSTEM_MODE = SIM_DATAFLOW;
    accepted = true;
    try {
        ParseSimulationConfig(nlohmann::json{
            {"noc", {{"send_recv_parallel", false}}},
            {"dte", {{"use_beha_dte", true}, {"streaming", true},
                     {"async", false}, {"aggregation", false},
                     {"fine_grained_resources", false}}}});
    } catch (...) {
        accepted = false;
    }
    check(accepted && SPEC_USE_BEHA_DTE && SPEC_DTE_STREAMING &&
              !SPEC_DTE_ASYNC && !SPEC_DTE_V4_RESOURCES,
          "DTE V2b accepts streaming with async and V4 resources disabled");

    SPEC_USE_BEHA_DTE = false;
    SPEC_DTE_STREAMING = false;
    SPEC_DTE_ASYNC = false;
    SPEC_DTE_V4_RESOURCES = false;
    SPEC_SEND_RECV_PARALLEL = false;
    SYSTEM_MODE = SIM_DATAFLOW;
    check(ThrowsExpected<std::invalid_argument>([]() {
              ParseSimulationConfig(nlohmann::json{
                  {"noc", {{"send_recv_parallel", false}}},
                  {"dte", {{"use_beha_dte", true}, {"streaming", true},
                           {"async", false},
                           {"fine_grained_resources", true}}}});
          }),
          "DTE V2b rejects V4 fine-grained resources without async");

    SYSTEM_MODE = old_mode;
    SPEC_SEND_RECV_PARALLEL = old_parallel;
    SPEC_DTE_ASYNC = old_async;
    SPEC_DTE_V4_RESOURCES = old_fine_grained;
    SPEC_DTE_STREAMING = old_streaming;
    SPEC_USE_BEHA_DTE = old_use_dte;

    DteV3Probe probe("dte_v3_probe");
    sc_start();
    failures += probe.failures;
    checks += probe.checks;
    std::cout << "DTE V3a self-test: "
              << (failures == 0 ? "PASS" : "FAIL") << " ("
              << checks - failures << "/" << checks << " checks)\n";
    return failures;
}
