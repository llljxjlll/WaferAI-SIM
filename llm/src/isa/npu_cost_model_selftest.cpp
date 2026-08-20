#include "isa/npu_cost_model_selftest.h"

#include "common/config.h"
#include "defs/global.h"
#include "isa/npu_cost_model.h"
#include "isa/published_npu_ops.h"
#include "isa/published_npu_ops_selftest.h"
#include "memory/sram/sram_region.h"
#include "prims/comp_prims.h"
#include "prims/exact_stage2_prims.h"
#include "prims/norm_prims.h"
#include "utils/prim_utils.h"

#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace {
void Check(IsaV1SelfTestResult &result, bool condition,
           const std::string &message) {
    ++result.checks;
    if (!condition) {
        result.failures.push_back(message);
    }
}

template <typename Function>
void Reject(IsaV1SelfTestResult &result, const std::string &message,
            Function &&function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &) {
        rejected = true;
    }
    Check(result, rejected, message);
}

NpuCostHardware Hardware() {
    NpuCostHardware hardware;
    hardware.exu_x_dims = 4;
    hardware.exu_count = 1;
    hardware.sfu_x_dims = 4;
    hardware.vec_x_dims = 4;
    hardware.vec_count = 1;
    hardware.compute_utilization = 1.0F;
    hardware.cycle_ns = 2;
    return hardware;
}

template <typename Primitive>
class ImplicitMemoryProbe : public Primitive {
public:
    bool LegacyAccessesEnabled(const TaskCoreContext &context) const {
        return this->usesLegacyImplicitMemory(context);
    }
};

class ScopedCostHardware {
public:
    ScopedCostHardware() : previous_(g_core_hw_config) {
        hardware_ = new CoreHWConfig(
            0, new ExuConfig(MAC_Array, 4, 1),
            new SfuConfig(Linear, 4), new VectorConfig(4, 1), "", 0, 128);
        g_core_hw_config = {{0, hardware_}};
    }

    ~ScopedCostHardware() {
        g_core_hw_config.clear();
        delete hardware_;
        g_core_hw_config = std::move(previous_);
    }

    ScopedCostHardware(const ScopedCostHardware &) = delete;
    ScopedCostHardware &operator=(const ScopedCostHardware &) = delete;

private:
    std::vector<std::pair<int, CoreHWConfig *>> previous_;
    CoreHWConfig *hardware_ = nullptr;
};

class ScopedStrictPrimWire {
public:
    ScopedStrictPrimWire()
        : previous_(prim_wire::LegacyCompatibilityEnabled()) {
        prim_wire::SetLegacyCompatibility(false);
    }

    ~ScopedStrictPrimWire() {
        prim_wire::SetLegacyCompatibility(previous_);
    }

    ScopedStrictPrimWire(const ScopedStrictPrimWire &) = delete;
    ScopedStrictPrimWire &operator=(const ScopedStrictPrimWire &) = delete;

private:
    bool previous_;
};

NpuCostHardware CostHardware(PublishedNpuHardwareView hardware) {
    NpuCostHardware result;
    result.exu_x_dims = hardware.exu_x_dims;
    result.exu_count = hardware.exu_count;
    result.sfu_x_dims = hardware.sfu_x_dims;
    result.vec_x_dims = hardware.vec_x_dims;
    result.vec_count = hardware.vec_count;
    result.compute_utilization = hardware.compute_utilization;
    result.cycle_ns = hardware.cycle_ns;
    return result;
}

bool SameSnapshot(const NpuCostSnapshot &lhs, const NpuCostSnapshot &rhs) {
    return lhs.ops.exu == rhs.ops.exu && lhs.ops.sfu == rhs.ops.sfu &&
           lhs.ops.vec == rhs.ops.vec &&
           lhs.exu_cycle_ns == rhs.exu_cycle_ns &&
           lhs.sfu_cycle_ns == rhs.sfu_cycle_ns &&
           lhs.vec_cycle_ns == rhs.vec_cycle_ns &&
           lhs.compute_cycle_ns == rhs.compute_cycle_ns &&
           lhs.dram_time_ns == rhs.dram_time_ns &&
           lhs.overlap_delay_ns == rhs.overlap_delay_ns;
}

class SkipOutputCostProbe final : public NpuBase {
public:
    uint64_t forced_dram_time = 0;
    size_t calls = 0;

    SkipOutputCostProbe() {
        name = "SkipOutputCostProbe";
        skip_input = true;
        skip_output = true;
    }

    void initialize() override {
        data_size_input = {1};
        data_chunk = {{"output", 1}};
    }

    void taskCore(TaskCoreContext &, std::string, uint64_t &dram_time,
                  uint64_t &exu_ops, uint64_t &sfu_ops,
                  uint64_t &vec_ops) override {
        ++calls;
        dram_time = forced_dram_time;
        exu_ops = 1U << 20;
        sfu_ops = 1U << 19;
        vec_ops = 1U << 18;
    }
};

NpuOps EvaluateProductionOps(
    NpuBase &prim, TaskCoreContext &context,
    std::unordered_map<std::string, int> parameters) {
    prim.param_value = std::move(parameters);
    prim.initialize();
    uint64_t dram_time = 0;
    NpuOps ops;
    prim.taskCore(context, "cost_oracle", dram_time, ops.exu, ops.sfu, ops.vec);
    if (dram_time != 0)
        throw std::logic_error(
            "side-effect-free compute cost oracle touched memory");
    return ops;
}
} // namespace

IsaV1SelfTestResult CheckNpuCostModelSelfTest() {
    IsaV1SelfTestResult result;
    IsaV1SelfTestResult published = CheckPublishedNpuOpsSelfTest();
    result.checks += published.checks;
    for (std::string &failure : published.failures)
        result.failures.push_back("published ops: " + std::move(failure));
    const NpuCostHardware hardware = Hardware();

    int sram_address = 0;
#if USE_NB_DRAMSYS == 1
    TaskCoreContext context(nullptr, nullptr, nullptr, nullptr, &sram_address,
                            nullptr, nullptr, nullptr, nullptr, 0, 16);
#else
    TaskCoreContext context(nullptr, nullptr, nullptr, nullptr, nullptr,
                            &sram_address, nullptr, nullptr, nullptr, 0, 0, 16);
#endif
    {
        Cross_entropy_forward_prim prim;
        prim.operands.logits.kind = SramAddressKind::ABSOLUTE;
        prim.operands.logits.absolute_address_bytes = 0x1000;
        prim.operands.labels.kind = SramAddressKind::ABSOLUTE;
        prim.operands.labels.absolute_address_bytes = 0x2000;
        prim.operands.loss.kind = SramAddressKind::ABSOLUTE;
        prim.operands.loss.absolute_address_bytes = 0x3000;
        prim.operands.logical_rows = 8;
        prim.operands.rank_rows = 4;
        prim.operands.tp_degree = 2;
        prim.operands.vocab_size = 32;
        const int previous_cid = context.cid;
        context.cid = 17;
        uint64_t dram_time = 0;
        NpuOps ops;
        std::ostringstream marker;
        std::streambuf *const previous = std::cout.rdbuf(marker.rdbuf());
        try {
            prim.taskCore(context, "ce_marker", dram_time,
                          ops.exu, ops.sfu, ops.vec);
        } catch (...) {
            std::cout.rdbuf(previous);
            context.cid = previous_cid;
            throw;
        }
        std::cout.rdbuf(previous);
        context.cid = previous_cid;
        Check(result,
              marker.str() ==
                  "[TRAIN_CE] core=17 invocations=1 rank_rows=4 "
                  "label_read_bytes=16 loss_write_bytes=16\n",
              "CE runtime marker is exact and derived from taskCore work");
        Check(result,
              dram_time == 0 && ops.exu == 0 && ops.sfu == 132 &&
                  ops.vec == 260,
              "CE marker path preserves the exact published compute work");
    }
    auto check_ops = [&](const std::string &label, NpuBase &prim,
                         std::unordered_map<std::string, int> parameters,
                         NpuOps expected) {
        const NpuOps actual =
            EvaluateProductionOps(prim, context, std::move(parameters));
        Check(result, actual.exu == expected.exu &&
                          actual.sfu == expected.sfu &&
                          actual.vec == expected.vec,
              label + " production ops match the frozen oracle");
    };

    sram::Config manual_config;
    manual_config.capacity_bytes = 64;
    manual_config.manual_memory_schedule = true;
    sram::RegionTable manual_regions(manual_config);
    auto check_implicit_memory_policy = [&](const std::string &label,
                                            const auto &prim) {
        context.sram_regions = nullptr;
        Check(result, prim.LegacyAccessesEnabled(context),
              label + " preserves legacy implicit memory accesses");
        context.sram_regions = &manual_regions;
        Check(result, !prim.LegacyAccessesEnabled(context),
              label + " disables implicit memory under a manual schedule");
    };
    check_implicit_memory_policy("MATMUL", ImplicitMemoryProbe<Matmul_f>{});
    check_implicit_memory_policy("ATTENTION",
                                 ImplicitMemoryProbe<Attention_f>{});
    check_implicit_memory_policy("RMSNORM",
                                 ImplicitMemoryProbe<rmsnorm_forward>{});

    // These direct production calls would enter the legacy address-zero/static
    // paths without the manual-memory gate. In manual mode they publish only
    // their compute cost and leave dram_time at zero.
    context.sram_regions = &manual_regions;
    {
        Attention_f prim;
        check_ops("ATTENTION manual memory", prim,
                  {{"B", 1}, {"T", 2}, {"C", 4}, {"NH", 2}, {"R", 3}},
                  {64, 8, 16});
    }
    {
        rmsnorm_forward prim;
        check_ops("RMSNORM manual memory", prim,
                  {{"B", 1}, {"T", 2}, {"C", 4}}, {0, 0, 34});
    }
    context.sram_regions = nullptr;
    {
        gate_forward prim;
        check_ops("GATE", prim,
                  {{"B", 2}, {"T", 3}, {"C", 4}, {"E_N", 5}, {"K", 2}},
                  {120, 0, 0});
    }
    {
        Max_pool prim;
        check_ops("MAXPOOL", prim,
                  {{"B", 1}, {"W", 4}, {"H", 4}, {"C", 2},
                   {"pX", 0}, {"pY", 0}, {"sX", 2}, {"sY", 2},
                   {"kX", 2}, {"kY", 2}},
                  {0, 32, 0});
    }
    {
        Gelu_f prim;
        check_ops("GELU", prim, {{"N", 7}}, {0, 7, 28});
    }
    {
        silu_forward prim;
        check_ops("SILU", prim, {{"N", 7}}, {0, 7, 21});
    }
    {
        swiglu_forward prim;
        check_ops("SWIGLU", prim, {{"N", 7}}, {0, 7, 28});
        Check(result,
              prim.data_size_input == std::vector<int>{14} &&
                  prim.data_chunk ==
                      std::vector<std::pair<std::string, int>>{{"output", 7}},
              "SWIGLU uses one concat input of 2N and one output of N");
    }

    ScopedCostHardware scoped_cost_hardware;
    const PublishedNpuHardwareView runtime_hardware =
        PublishedNpuHardwareForCore(0);
    const NpuCostHardware runtime_cost_hardware =
        CostHardware(runtime_hardware);
    auto manual_core = std::make_shared<PrimCoreContext>();
    manual_core->cid = 0;
    manual_core->loop_cnt = 0;
    manual_core->auto_pd_ = 0;
    context.sram_regions = &manual_regions;

    auto check_manual_default =
        [&](const std::string &label, Opcode opcode, NpuBase &prim,
            PublishedNpuParameters parameters) {
            prim.param_value = parameters;
            prim.datatype = FP16;
            prim.initialize();
            static_cast<CompBase &>(prim).initializeDefault();
            prim.prim_context = manual_core;

            Sram_bind_oneshot bind;
            bind.input_count =
                static_cast<uint32_t>(prim.data_size_input.size());
            for (uint32_t index = 0; index < bind.input_count; ++index)
                bind.datapass_label.indata[index] =
                    label + "_input_" + std::to_string(index);
            bind.datapass_label.outdata = label + "_output";
            bind.prim_context = manual_core;
            Check(result, bind.taskCoreDefault(context) == 0,
                  label + " installs a one-shot SRAM binding");

            const NpuOps ops =
                EvaluatePublishedNpuOps(opcode, parameters, runtime_hardware);
            const NpuCostSnapshot expected =
                CalculateNpuCost(ops, runtime_cost_hardware, 0);
            const size_t allocations_before =
                manual_regions.AllocationCount();
            const int sram_address_before = sram_address;
            const int actual_delay = prim.taskCoreDefault(context);

            Check(result, expected.overlap_delay_ns > 0 &&
                              actual_delay ==
                                  static_cast<int>(expected.overlap_delay_ns),
                  label + " manual taskCoreDefault returns compute delay once");
            Check(result,
                  SameSnapshot(prim.lastCostSnapshot(), expected),
                  label + " publishes the exact compute-only cost snapshot");
            Check(result,
                  manual_regions.AllocationCount() == allocations_before &&
                      sram_address == sram_address_before,
                  label + " performs no implicit SRAM allocation or write");
            Check(result, !manual_core->sram_bind_pending_,
                  label + " consumes its one-shot binding exactly once");
        };

    {
        Matmul_f prim;
        check_manual_default(
            "MATMUL", Opcode::MATMUL, prim,
            {{"B", 1}, {"T", 128}, {"C", 128}, {"OC", 128}});
    }
    {
        ScopedStrictPrimWire strict_wire;
        Matmul_f source;
        source.param_value =
            {{"B", 2}, {"T", 3}, {"C", 4}, {"OC", 5}};
        source.datatype = FP16;
        source.initialize();
        static_cast<CompBase &>(source).initializeDefault();
        const auto wire = source.serialize();

        Matmul_f decoded;
        decoded.deserialize(wire);
        Check(result,
              decoded.serialize() == wire &&
                  decoded.data_size_input == std::vector<int>{24},
              "MATMUL strict roundtrip retains the legacy one-input profile");
        decoded.prim_context = manual_core;

        Sram_bind_oneshot bind;
        bind.input_count = 2;
        bind.datapass_label.indata[0] = "matmul_wgrad_activation";
        bind.datapass_label.indata[1] = "matmul_wgrad_gradient";
        bind.datapass_label.outdata = "matmul_wgrad_output";
        bind.prim_context = manual_core;
        Check(result, bind.taskCoreDefault(context) == 0,
              "MATMUL WGRAD installs an explicit two-input SRAM binding");

        (void)decoded.taskCoreDefault(context);
        Check(result,
              decoded.data_size_input == std::vector<int>({24, 20}) &&
                  !manual_core->sram_bind_pending_,
              "MATMUL WGRAD derives exact dual-input sizes after strict roundtrip");
    }
    {
        rmsnorm_forward prim;
        check_manual_default(
            "RMSNORM", Opcode::RMSNORM, prim,
            {{"B", 1}, {"T", 128}, {"C", 128}});
    }

    // Legacy skip_output still suppresses only the implicit write. It must not
    // skip the compute charge.
    context.sram_regions = nullptr;
    auto legacy_core = std::make_shared<PrimCoreContext>();
    legacy_core->cid = 0;
    legacy_core->loop_cnt = 0;
    legacy_core->auto_pd_ = 0;
    legacy_core->datapass_label_->indata[0] = "_legacy_cost_input";
    legacy_core->datapass_label_->outdata = "legacy_cost_output";
    SkipOutputCostProbe legacy_probe;
    legacy_probe.datatype = FP16;
    legacy_probe.prim_context = legacy_core;
    legacy_probe.initialize();
    static_cast<CompBase &>(legacy_probe).initializeDefault();
    const NpuOps probe_ops{1U << 20, 1U << 19, 1U << 18};
    const NpuCostSnapshot probe_expected =
        CalculateNpuCost(probe_ops, runtime_cost_hardware, 0);
    const int legacy_sram_address_before = sram_address;
    const int legacy_delay = legacy_probe.taskCoreDefault(context);
    Check(result, probe_expected.overlap_delay_ns > 0 &&
                      legacy_delay ==
                          static_cast<int>(probe_expected.overlap_delay_ns) &&
                      legacy_probe.calls == 1,
          "legacy skip_output charges the compute cost exactly once");
    Check(result, SameSnapshot(legacy_probe.lastCostSnapshot(), probe_expected),
          "legacy skip_output publishes the exact compute snapshot");
    Check(result, sram_address == legacy_sram_address_before,
          "legacy skip_output performs no implicit output write");

    // A manual primitive that reports any implicit memory time is rejected
    // before compute accounting can be published.
    context.sram_regions = &manual_regions;
    SkipOutputCostProbe bad_manual_probe;
    bad_manual_probe.forced_dram_time = 1;
    bad_manual_probe.datatype = FP16;
    bad_manual_probe.prim_context = manual_core;
    bad_manual_probe.initialize();
    static_cast<CompBase &>(bad_manual_probe).initializeDefault();
    Sram_bind_oneshot bad_bind;
    bad_bind.input_count = 1;
    bad_bind.datapass_label.indata[0] = "bad_manual_input";
    bad_bind.datapass_label.outdata = "bad_manual_output";
    bad_bind.prim_context = manual_core;
    bad_bind.taskCoreDefault(context);
    Reject(result, "manual compute rejects non-zero implicit DRAM time", [&] {
        (void)bad_manual_probe.taskCoreDefault(context);
    });
    Check(result, bad_manual_probe.calls == 1 &&
                      !manual_core->sram_bind_pending_ &&
                      SameSnapshot(bad_manual_probe.lastCostSnapshot(), {}),
          "rejected manual compute consumes its binding without charging");
    context.sram_regions = nullptr;

    {
        Relu_f prim;
        check_ops("RELU", prim, {{"N", 7}}, {7, 0, 0});
    }
    {
        Residual_f prim;
        check_ops("RESIDUAL", prim, {{"N", 7}}, {0, 0, 7});
    }
    {
        Split_matmul prim;
        check_ops("SPLIT_MATMUL", prim,
                  {{"B", 2}, {"T", 3}, {"C", 4}, {"dim", 1}, {"slice", 2}},
                  {0, 0, 0});
    }
    {
        Merge_matmul prim;
        check_ops("MERGE_MATMUL", prim,
                  {{"B", 2}, {"T", 3}, {"C", 4}, {"dim", 1}, {"slice", 2}},
                  {24, 0, 0});
    }
    {
        Dummy_p prim;
        check_ops("DUMMY", prim, {}, {10, 0, 0});
    }

    NpuCostSnapshot cost =
        CalculateNpuCost(NpuOps{64, 4, 4}, hardware, 0);
    Check(result, cost.exu_cycle_ns == 4 && cost.sfu_cycle_ns == 2 &&
                      cost.vec_cycle_ns == 2 && cost.compute_cycle_ns == 4,
          "EXU dominates and max cycle is selected");
    Check(result, cost.overlap_delay_ns == 4,
          "compute-only delay equals compute cycle");

    cost = CalculateNpuCost(NpuOps{32, 20, 4}, hardware, 3);
    Check(result, cost.exu_cycle_ns == 2 && cost.sfu_cycle_ns == 10 &&
                      cost.compute_cycle_ns == 10,
          "SFU dominates and max cycle is selected");
    Check(result, cost.overlap_delay_ns == 7,
          "DRAM shorter than compute overlaps exactly once");

    cost = CalculateNpuCost(NpuOps{32, 4, 24}, hardware, 12);
    Check(result, cost.vec_cycle_ns == 12 && cost.compute_cycle_ns == 12,
          "vector unit dominates and max cycle is selected");
    Check(result, cost.overlap_delay_ns == 0,
          "DRAM equal to compute leaves no extra delay");

    cost = CalculateNpuCost(NpuOps{32, 4, 24}, hardware, 13);
    Check(result, cost.overlap_delay_ns == 0,
          "DRAM longer than compute leaves no extra delay");

    cost = CalculateNpuCost(NpuOps{31, 3, 3}, hardware, 0);
    Check(result, cost.exu_cycle_ns == 1 && cost.sfu_cycle_ns == 0 &&
                      cost.vec_cycle_ns == 0,
          "historical cost divisions truncate instead of rounding up");

    NpuCostHardware bad = hardware;
    bad.exu_x_dims = 0;
    Reject(result, "zero EXU dimension is rejected", [&] {
        (void)CalculateNpuCost({}, bad, 0);
    });
    bad = hardware;
    bad.compute_utilization = 0.0F;
    Reject(result, "zero compute utilization is rejected", [&] {
        (void)CalculateNpuCost({}, bad, 0);
    });
    bad = hardware;
    bad.vec_x_dims = std::numeric_limits<uint64_t>::max();
    bad.vec_count = 2;
    Reject(result, "vector width multiplication overflow is rejected", [&] {
        (void)CalculateNpuCost({}, bad, 0);
    });
    bad = hardware;
    bad.sfu_x_dims = 1;
    bad.cycle_ns = 2;
    Reject(result, "SFU cycle multiplication overflow is rejected", [&] {
        (void)CalculateNpuCost(
            NpuOps{0, std::numeric_limits<uint64_t>::max(), 0}, bad, 0);
    });
    return result;
}
