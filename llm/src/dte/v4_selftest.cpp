#include "dte/dte_async.h"
#include "dte/dte_unit.h"

#include "prims/norm_prims.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

DTEConfig FineConfig(uint32_t channels = 1) {
    DTEConfig config;
    config.channel_count = channels;
    config.bit_width_bits = 128;
    config.gamma_cycles = 1;
    config.tau_launch_cycles = 0;
    config.fine_grained_resources = true;
    config.command_slots_per_channel = 2;
    config.pending_queue_depth = 1;
    config.spm_read_width_bits = 64;
    config.spm_write_width_bits = 32;
    config.axi_read_width_bits = 16;
    config.axi_write_width_bits = 128;
    config.launch_energy_pj = 10.0;
    config.spm_energy_pj_per_bit = 0.01;
    config.axi_energy_pj_per_bit = 0.02;
    config.base_area_um2 = 1000.0;
    config.channel_area_um2 = 100.0;
    config.command_slot_area_um2 = 10.0;
    config.port_bit_area_um2 = 0.5;
    return config;
}

long long Cycles(const sc_time &duration) {
    return static_cast<long long>(duration.value() /
                                  sc_time(CYCLE, SC_NS).value());
}

void WaitCompleted(DteTransferContext *context) {
    if (context->state != DteTransferState::COMPLETED)
        wait(context->done);
}

class DteV4Probe : public sc_module {
public:
    SC_HAS_PROCESS(DteV4Probe);

    DteV4Probe(const sc_module_name &name)
        : sc_module(name), directions("v4_directions", FineConfig(3)),
          duplex("v4_duplex", FineConfig()),
          contention("v4_contention", FineConfig()),
          credit("v4_credit", FineConfig()) {
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

    void run() {
        const DteDir dirs[] = {
            DteDir::SPM_TO_REMOTE, DteDir::REMOTE_TO_SPM,
            DteDir::SPM_TO_SPM, DteDir::SPM_TO_DRAM,
            DteDir::DRAM_TO_SPM, DteDir::DRAM_TO_REMOTE};
        std::vector<DteTransferContext *> direction_contexts;
        for (DteDir dir : dirs)
            direction_contexts.push_back(&directions.Issue(256, dir));

        DteTransferContext &read =
            duplex.Issue(256, DteDir::SPM_TO_REMOTE);
        DteTransferContext &write =
            duplex.Issue(256, DteDir::REMOTE_TO_SPM);
        DteTransferContext &read1 =
            contention.Issue(256, DteDir::SPM_TO_REMOTE);
        DteTransferContext &read2 =
            contention.Issue(256, DteDir::SPM_TO_REMOTE);

        std::vector<DteTransferContext *> credit_contexts;
        credit_contexts.push_back(
            credit.TryIssue(256, DteDir::SPM_TO_REMOTE));
        credit_contexts.push_back(
            credit.TryIssue(256, DteDir::SPM_TO_REMOTE));
        credit_contexts.push_back(
            credit.TryIssue(256, DteDir::SPM_TO_REMOTE));
        DteTransferContext *rejected =
            credit.TryIssue(256, DteDir::SPM_TO_REMOTE);
        Check(rejected == nullptr && !credit.CanAccept(),
              "finite descriptor credits reject a fourth immediate issue");
        credit.WaitForCredit();
        credit_contexts.push_back(
            &credit.Issue(256, DteDir::SPM_TO_REMOTE));

        for (DteTransferContext *ctx : direction_contexts)
            WaitCompleted(ctx);
        WaitCompleted(&read);
        WaitCompleted(&write);
        WaitCompleted(&read1);
        WaitCompleted(&read2);
        for (DteTransferContext *ctx : credit_contexts)
            WaitCompleted(ctx);

        Check(directions.CompletedCount() == 6,
              "all six endpoint directions complete");
        const long long expected[] = {4, 8, 8, 4, 16, 16};
        bool exact_service = true;
        for (size_t i = 0; i < direction_contexts.size(); ++i)
            exact_service &=
                Cycles(direction_contexts[i]->completion_time -
                       direction_contexts[i]->transmit_start_time) ==
                expected[i];
        Check(exact_service,
              "every direction follows its closed-form slowest-port service");
        Check(read.transmit_start_time == write.transmit_start_time &&
                  read.completion_time < write.completion_time,
              "independent SPM read/write ports overlap in full duplex");
        Check(read2.transmit_start_time >= read1.completion_time,
              "two SPM readers serialize on the shared read port");
        Check(duplex.MaxActiveCount() == 2 &&
                  credit.MaxActiveCount() == 2,
              "one channel exposes exactly two command slots");
        Check(credit.statistics().backpressure_stalls == 1 &&
                  credit.statistics().physical_issued == 4 &&
                  credit.statistics().completed == 4,
              "credit wait resumes once and all accepted commands complete");
        Check(std::abs(directions.statistics().TotalDynamicEnergyPj() -
                       90.72) < 1e-9,
              "launch plus per-port dynamic energy is exact");
        Check(std::abs(directions.statistics().area_um2 - 1480.0) < 1e-9,
              "base/channel/slot/port area formula is exact");
        Check(directions.AveragePowerMw() > 0.0,
              "average dynamic power is derived from energy and elapsed time");
        Check(directions.ActiveCount() == 0 &&
                  directions.InflightCount() == 0 &&
                  !directions.BusBusy(),
              "V4 resources and credits fully drain");

        for (DteTransferContext *ctx : direction_contexts)
            directions.Release(ctx->xfer_id);
        duplex.Release(read.xfer_id);
        duplex.Release(write.xfer_id);
        contention.Release(read1.xfer_id);
        contention.Release(read2.xfer_id);
        for (DteTransferContext *ctx : credit_contexts)
            credit.Release(ctx->xfer_id);
        sc_stop();
    }

    DTEUnit directions;
    DTEUnit duplex;
    DTEUnit contention;
    DTEUnit credit;
};
} // namespace

int RunDTEV4SelfTest() {
    std::cout << "==== DTE V4 resource self-test ====\n";
    int failures = 0;
    int checks = 0;
    auto check = [&](bool condition, const std::string &name) {
        ++checks;
        std::cout << "  [" << (condition ? " ok " : "FAIL") << "] "
                  << name << "\n";
        if (!condition)
            ++failures;
    };

    DTEConfig valid = FineConfig();
    check(!ThrowsExpected<std::invalid_argument>(
              [&]() { DTEUnit::ValidateConfig(valid); }),
          "valid V4 resource configuration is accepted");
    check(ThrowsExpected<std::invalid_argument>([&]() {
              auto bad = valid;
              bad.command_slots_per_channel = 1;
              DTEUnit::ValidateConfig(bad);
          }),
          "V4 rejects a command depth other than two");
    check(ThrowsExpected<std::invalid_argument>([&]() {
              auto bad = valid;
              bad.pending_queue_depth = 0;
              DTEUnit::ValidateConfig(bad);
          }),
          "V4 requires finite pending credit capacity");
    check(ThrowsExpected<std::invalid_argument>([&]() {
              auto bad = valid;
              bad.axi_read_width_bits = 0;
              DTEUnit::ValidateConfig(bad);
          }),
          "V4 rejects a zero-width endpoint port");
    check(ThrowsExpected<std::invalid_argument>([&]() {
              auto bad = valid;
              bad.spm_energy_pj_per_bit = -0.1;
              DTEUnit::ValidateConfig(bad);
          }),
          "V4 rejects negative power/area coefficients");

    const uint32_t read = DtePortBit(DtePort::SPM_READ);
    const uint32_t write = DtePortBit(DtePort::SPM_WRITE);
    const uint32_t axi_read = DtePortBit(DtePort::AXI_READ);
    const uint32_t axi_write = DtePortBit(DtePort::AXI_WRITE);
    check(DTEUnit::RequiredPorts(DteDir::SPM_TO_REMOTE, true) == read &&
              DTEUnit::RequiredPorts(DteDir::REMOTE_TO_SPM, true) == write &&
              DTEUnit::RequiredPorts(DteDir::SPM_TO_SPM, true) ==
                  (read | write) &&
              DTEUnit::RequiredPorts(DteDir::SPM_TO_DRAM, true) ==
                  (read | axi_write) &&
              DTEUnit::RequiredPorts(DteDir::DRAM_TO_SPM, true) ==
                  (axi_read | write) &&
              DTEUnit::RequiredPorts(DteDir::DRAM_TO_REMOTE, true) ==
                  axi_read,
          "six directions map to the frozen endpoint-port masks");

    Dte_async_prim local_copy;
    local_copy.op = DteAsyncOp::ISSUE;
    local_copy.token = 1;
    local_copy.payload_bits = 256;
    local_copy.direction = DteDir::SPM_TO_SPM;
    local_copy.spm_addr = 0x1000;
    local_copy.spm_size = 32;
    local_copy.remote_addr = 0x2000;
    Dte_async_prim decoded_copy;
    decoded_copy.deserialize(local_copy.serialize());
    check(decoded_copy.direction == DteDir::SPM_TO_SPM &&
              decoded_copy.spm_addr == 0x1000 &&
              decoded_copy.remote_addr == 0x2000,
          "SPM_TO_SPM source and destination survive the wire");

    Dte_async_prim dram_remote;
    dram_remote.op = DteAsyncOp::ISSUE;
    dram_remote.token = 2;
    dram_remote.payload_bits = 256;
    dram_remote.direction = DteDir::DRAM_TO_REMOTE;
    dram_remote.remote_peer = 7;
    dram_remote.remote_addr = 0x3000;
    Dte_async_prim decoded_remote;
    decoded_remote.deserialize(dram_remote.serialize());
    check(decoded_remote.direction == DteDir::DRAM_TO_REMOTE &&
              decoded_remote.spm_size == 0 &&
              decoded_remote.remote_peer == 7,
          "DRAM_TO_REMOTE uses a canonical zero-SPM wire");

    DteV4Probe probe("dte_v4_probe");
    sc_start();
    failures += probe.failures;
    checks += probe.checks;
    std::cout << "DTE V4 self-test: "
              << (failures == 0 ? "PASS" : "FAIL") << " ("
              << checks - failures << "/" << checks << " checks)\n";
    return failures;
}
