#include "dte/dte_async.h"

#include "prims/norm_prims.h"

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

DTEConfig UnitConfig(uint32_t channels = 16) {
    DTEConfig config;
    config.channel_count = channels;
    config.bit_width_bits = 128;
    config.gamma_cycles = 2;
    config.tau_launch_cycles = 1;
    return config;
}

DteAggregationConfig Agg(uint32_t max_descriptors = 16,
                         uint64_t max_bytes = 1024,
                         uint64_t timeout_cycles = 50) {
    DteAggregationConfig config;
    config.enabled = true;
    config.max_descriptors = max_descriptors;
    config.max_payload_bytes = max_bytes;
    config.timeout_cycles = timeout_cycles;
    config.address_block_bytes = 65536;
    return config;
}

class DteV3bProbe : public sc_module {
public:
    SC_HAS_PROCESS(DteV3bProbe);

    DteV3bProbe(const sc_module_name &name)
        : sc_module(name),
          pair_unit("v3b_pair_unit", UnitConfig(2)),
          pair("v3b_pair_tracker", pair_unit, Agg(2, 16)),
          noncontig_unit("v3b_noncontig_unit", UnitConfig()),
          noncontig("v3b_noncontig_tracker", noncontig_unit, Agg()),
          different_unit("v3b_different_unit", UnitConfig()),
          different("v3b_different_tracker", different_unit, Agg()),
          timeout_unit("v3b_timeout_unit", UnitConfig()),
          timeout("v3b_timeout_tracker", timeout_unit, Agg(16, 64, 4)),
          disabled_unit("v3b_disabled_unit", UnitConfig()),
          disabled("v3b_disabled_tracker", disabled_unit),
          cancel_unit("v3b_cancel_unit", UnitConfig()),
          cancel("v3b_cancel_tracker", cancel_unit, Agg(4, 64)),
          trend1_unit("v3b_trend1_unit", UnitConfig()),
          trend1("v3b_trend1_tracker", trend1_unit),
          trend2_unit("v3b_trend2_unit", UnitConfig()),
          trend2("v3b_trend2_tracker", trend2_unit, Agg(2)),
          trend4_unit("v3b_trend4_unit", UnitConfig()),
          trend4("v3b_trend4_tracker", trend4_unit, Agg(4)),
          trend8_unit("v3b_trend8_unit", UnitConfig()),
          trend8("v3b_trend8_tracker", trend8_unit, Agg(8)),
          trend16_unit("v3b_trend16_unit", UnitConfig()),
          trend16("v3b_trend16_tracker", trend16_unit, Agg(16)) {
        SC_THREAD(run);
    }

    int failures = 0;
    int checks = 0;

private:
    void Check(bool condition, const std::string &name) {
        ++checks;
        std::cout << "  [" << (condition ? " ok " : "FAIL") << "] "
                  << name << "\n";
        if (!condition)
            ++failures;
    }

    static void IssueSmall(DteAsyncTracker &tracker, uint32_t token,
                           uint64_t offset, uint32_t peer = 1,
                           uint32_t block = 0) {
        tracker.IssueToken(token, 64, DteDir::SPM_TO_REMOTE,
                           offset, 8, peer, offset, block);
    }

    void run() {
        IssueSmall(pair, 1, 0);
        Check(pair.Record(1).staged && pair.OpenGroupCount() == 1 &&
                  pair.AggregationMetrics().physical_transfers == 0,
              "first small request waits in an open aggregation group");
        IssueSmall(pair, 2, 8);
        Check(!pair.Record(1).staged && !pair.Record(2).staged &&
                  pair.Record(1).xfer_id == pair.Record(2).xfer_id &&
                  pair.AggregationMetrics().physical_transfers == 1,
              "two contiguous requests share one physical transfer");
        pair.WaitToken(1);
        Check(pair.HasToken(2) && pair.PollToken(2),
              "one logical completion does not consume its group peers");
        pair.WaitToken(2);
        Check(pair.OutstandingCount() == 0 &&
                  pair.AggregationMetrics().launch_savings == 1 &&
                  pair.AggregationMetrics().BandwidthUtilizationPpm() ==
                      1000000,
              "completion fan-out releases the shared context exactly once");

        IssueSmall(noncontig, 10, 0);
        IssueSmall(noncontig, 11, 16);
        Check(!noncontig.Record(10).staged &&
                  noncontig.Record(11).staged &&
                  noncontig.AggregationMetrics().physical_transfers == 1,
              "non-contiguous address flushes the old group");
        noncontig.Fence();
        Check(noncontig.AggregationMetrics().physical_transfers == 2,
              "non-contiguous descriptors remain separate transfers");

        IssueSmall(different, 20, 0, 1, 0);
        IssueSmall(different, 21, 8, 2, 0);
        different.IssueToken(22, 64, DteDir::REMOTE_TO_SPM,
                             16, 8, 1, 16, 0);
        different.IssueToken(23, 64, DteDir::SPM_TO_REMOTE,
                             24, 8, 1, 65536, 1);
        Check(different.OpenGroupCount() == 4,
              "direction, peer and address block partition groups");
        different.Fence();
        Check(different.AggregationMetrics().physical_transfers == 4 &&
                  different.AggregationMetrics().launch_savings == 0,
              "incompatible descriptors are never coalesced");

        const sc_time timeout_issue = sc_time_stamp();
        IssueSmall(timeout, 30, 0);
        wait(6, SC_NS);
        Check(timeout.Record(30).staged,
              "partial group remains staged before its timeout");
        wait(3, SC_NS);
        Check(!timeout.Record(30).staged &&
                  timeout.Record(30).context->issue_time ==
                      timeout_issue + sc_time(8, SC_NS),
              "partial group flushes at the exact configured timeout");
        timeout.WaitToken(30);

        disabled.IssueToken(40, 64, DteDir::SPM_TO_REMOTE, 0, 8);
        disabled.IssueToken(41, 64, DteDir::SPM_TO_REMOTE, 8, 8);
        Check(disabled.AggregationMetrics().physical_transfers == 2 &&
                  disabled.Record(40).xfer_id != disabled.Record(41).xfer_id,
              "aggregation disabled preserves V3a one-token-per-transfer");
        disabled.Fence();
        Check(disabled.AggregationMetrics().BandwidthUtilizationPpm() ==
                  500000,
              "small uncoalesced requests expose half-word bus utilization");

        IssueSmall(cancel, 50, 0);
        IssueSmall(cancel, 51, 8);
        cancel.CancelToken(51);
        Check(cancel.HasToken(50) && !cancel.HasToken(51) &&
                  cancel.OpenGroupCount() == 1,
              "tail cancellation updates an open group without a leak");
        cancel.Fence();
        IssueSmall(cancel, 52, 32);
        IssueSmall(cancel, 53, 40);
        Check(ThrowsExpected<std::runtime_error>(
                  [&]() { cancel.CancelToken(52); }),
              "non-tail staged cancellation is rejected explicitly");
        cancel.Fence();

        IssueSmall(pair, 3, 32);
        IssueSmall(pair, 4, 40);
        Check(ThrowsExpected<std::runtime_error>(
                  [&]() { pair.CancelToken(3); }),
              "one member of an issued compound descriptor cannot cancel");
        pair.Fence();

        for (uint32_t i = 0; i < 16; ++i) {
            const uint64_t offset = uint64_t(i) * 8;
            trend1.IssueToken(i, 64, DteDir::SPM_TO_REMOTE,
                              offset, 8);
            IssueSmall(trend2, i, offset);
            IssueSmall(trend4, i, offset);
            IssueSmall(trend8, i, offset);
            IssueSmall(trend16, i, offset);
        }
        Check(trend1.AggregationMetrics().physical_transfers == 16 &&
                  trend2.AggregationMetrics().physical_transfers == 8 &&
                  trend4.AggregationMetrics().physical_transfers == 4 &&
                  trend8.AggregationMetrics().physical_transfers == 2 &&
                  trend16.AggregationMetrics().physical_transfers == 1,
              "fixed total data follows the COMET 1/2/4/8/16 launch trend");
        Check(trend1.AggregationMetrics().launch_savings == 0 &&
                  trend2.AggregationMetrics().launch_savings == 8 &&
                  trend4.AggregationMetrics().launch_savings == 12 &&
                  trend8.AggregationMetrics().launch_savings == 14 &&
                  trend16.AggregationMetrics().launch_savings == 15,
              "launch savings equal logical minus physical descriptors");
        Check(trend1.AggregationMetrics().BandwidthUtilizationPpm() ==
                  500000 &&
                  trend2.AggregationMetrics().BandwidthUtilizationPpm() ==
                      1000000 &&
                  trend4.AggregationMetrics().BandwidthUtilizationPpm() ==
                      1000000 &&
                  trend8.AggregationMetrics().BandwidthUtilizationPpm() ==
                      1000000 &&
                  trend16.AggregationMetrics().BandwidthUtilizationPpm() ==
                      1000000,
              "coalescing improves shared-bus useful-bit utilization");
        trend1.Fence();
        trend2.Fence();
        trend4.Fence();
        trend8.Fence();
        trend16.Fence();
        Check(trend1.OutstandingCount() == 0 &&
                  trend2.OutstandingCount() == 0 &&
                  trend4.OutstandingCount() == 0 &&
                  trend8.OutstandingCount() == 0 &&
                  trend16.OutstandingCount() == 0,
              "all trend groups fan out and drain at simulation end");

        sc_stop();
    }

    DTEUnit pair_unit;
    DteAsyncTracker pair;
    DTEUnit noncontig_unit;
    DteAsyncTracker noncontig;
    DTEUnit different_unit;
    DteAsyncTracker different;
    DTEUnit timeout_unit;
    DteAsyncTracker timeout;
    DTEUnit disabled_unit;
    DteAsyncTracker disabled;
    DTEUnit cancel_unit;
    DteAsyncTracker cancel;
    DTEUnit trend1_unit;
    DteAsyncTracker trend1;
    DTEUnit trend2_unit;
    DteAsyncTracker trend2;
    DTEUnit trend4_unit;
    DteAsyncTracker trend4;
    DTEUnit trend8_unit;
    DteAsyncTracker trend8;
    DTEUnit trend16_unit;
    DteAsyncTracker trend16;
};
} // namespace

int RunDTEV3bSelfTest() {
    std::cout << "==== DTE V3b aggregation self-test ====\n";
    int failures = 0;
    int checks = 0;
    auto check = [&](bool condition, const std::string &name) {
        ++checks;
        std::cout << "  [" << (condition ? " ok " : "FAIL") << "] "
                  << name << "\n";
        if (!condition)
            ++failures;
    };

    DteAggregationConfig valid = Agg();
    check(!ThrowsExpected<std::invalid_argument>(
              [&]() { ValidateDteAggregationConfig(valid); }) &&
              ThrowsExpected<std::invalid_argument>([&]() {
                  auto bad = valid;
                  bad.max_descriptors = 1;
                  ValidateDteAggregationConfig(bad);
              }) &&
              ThrowsExpected<std::invalid_argument>([&]() {
                  auto bad = valid;
                  bad.timeout_cycles = 0;
                  ValidateDteAggregationConfig(bad);
              }),
          "aggregation config validates group size and timeout");

    Dte_async_prim issue;
    issue.op = DteAsyncOp::ISSUE;
    issue.token = 0xfedcba98U;
    issue.payload_bits = 0x123456789abcdef0ULL;
    issue.direction = DteDir::REMOTE_TO_SPM;
    issue.spm_addr = 0x23456789abcdef01ULL;
    issue.spm_size = 0x0123456789abcdefULL;
    issue.remote_peer = 0x89abcdefU;
    issue.remote_addr = 0x3456789abcdef012ULL;
    issue.address_block = 0x76543210U;
    Dte_async_prim decoded;
    decoded.deserialize(issue.serialize());
    check(decoded.token == issue.token &&
              decoded.payload_bits == issue.payload_bits &&
              decoded.direction == issue.direction &&
              decoded.spm_addr == issue.spm_addr &&
              decoded.spm_size == issue.spm_size &&
              decoded.remote_peer == issue.remote_peer &&
              decoded.remote_addr == issue.remote_addr &&
              decoded.address_block == issue.address_block,
          "V3b address metadata survives the three-segment wire");

    DteV3bProbe probe("dte_v3b_probe");
    sc_start();
    failures += probe.failures;
    checks += probe.checks;
    std::cout << "DTE V3b self-test: "
              << (failures == 0 ? "PASS" : "FAIL") << " ("
              << checks - failures << "/" << checks << " checks)\n";
    return failures;
}
