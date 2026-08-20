#include "isa/prim_wire_selftest.h"

#include "common/config.h"
#include "common/memory.h"
#include "defs/global.h"
#include "dte/coll_codec.h"
#include "prims/collective_data_v1_prim.h"
#include "prims/collective_launch_v1_prim_selftest.h"
#include "prims/collective_phase_barrier_v1_prim.h"
#include "prims/dte_endpoint_prims.h"
#include "prims/comp_prims.h"
#include "prims/gpu_prims.h"
#include "prims/norm_prims.h"
#include "prims/sram_lifecycle_prim.h"
#include "prims/sync_prims.h"
#include "utils/prim_utils.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using Wire = std::vector<sc_bv<128>>;

class TestState {
public:
    void Check(bool condition, const std::string &what) {
        ++checks_;
        if (condition) return;
        ++failures_;
        std::cerr << "Prim wire self-test failure: " << what << '\n';
    }

    template <typename Function>
    void Throws(const std::string &what, Function function) {
        ++checks_;
        try {
            function();
        } catch (const std::exception &) {
            return;
        }
        ++failures_;
        std::cerr << "Prim wire self-test failure: " << what
                  << " did not reject invalid input\n";
    }

    int checks() const { return checks_; }
    int failures() const { return failures_; }

private:
    int checks_ = 0;
    int failures_ = 0;
};

bool SameWire(const Wire &left, const Wire &right) {
    if (left.size() != right.size()) return false;
    for (size_t i = 0; i < left.size(); ++i)
        if (left[i] != right[i]) return false;
    return true;
}

class LegacyModeGuard {
public:
    explicit LegacyModeGuard(bool enabled)
        : previous_(prim_wire::LegacyCompatibilityEnabled()) {
        prim_wire::SetLegacyCompatibility(enabled);
    }
    ~LegacyModeGuard() {
        prim_wire::SetLegacyCompatibility(previous_);
    }

    LegacyModeGuard(const LegacyModeGuard &) = delete;
    LegacyModeGuard &operator=(const LegacyModeGuard &) = delete;

private:
    bool previous_;
};

class PrimWireCostHardwareGuard {
public:
    PrimWireCostHardwareGuard() : previous_(g_core_hw_config) {
        hardware_ = new CoreHWConfig(
            0, new ExuConfig(MAC_Array, 4, 1),
            new SfuConfig(Linear, 4), new VectorConfig(4, 1), "", 0, 128);
        g_core_hw_config = {{0, hardware_}};
    }

    ~PrimWireCostHardwareGuard() {
        g_core_hw_config.clear();
        delete hardware_;
        g_core_hw_config = std::move(previous_);
    }

    PrimWireCostHardwareGuard(const PrimWireCostHardwareGuard &) = delete;
    PrimWireCostHardwareGuard &
    operator=(const PrimWireCostHardwareGuard &) = delete;

private:
    std::vector<std::pair<int, CoreHWConfig *>> previous_;
    CoreHWConfig *hardware_ = nullptr;
};

template <typename Prim>
void CheckLegacyWrappedTransport(TestState &state,
                                 const std::string &name,
                                 const Wire &strict) {
    Prim prototype;
    const Wire legacy = prim_wire::LegacyTransportSegments(
        strict, prototype.name);
    state.Check(legacy.size() + 2 == strict.size(),
                name + " legacy transport removes carry and trailer");
    {
        LegacyModeGuard guard(true);
        Prim decoded;
        decoded.deserialize(legacy);
        state.Check(SameWire(strict, decoded.serialize()),
                    name + " legacy transport preserves strict semantics");
    }
    state.Check(!prim_wire::LegacyCompatibilityEnabled(),
                name + " legacy test restores strict mode");
}


class BindingProbeNpu final : public NpuBase {
public:
    explicit BindingProbeNpu(size_t input_count) {
        name = "BindingProbeNpu";
        data_size_input.assign(input_count, 1);
        data_chunk = {{"output", 1}};
        data_byte = 1;
        out_size = 1;
        data_chunk_addr["output"] = 0;
        skip_input = true;
        skip_output = true;
    }

    void initialize() override {}

    void taskCore(TaskCoreContext &, string, u_int64_t &, u_int64_t &,
                  u_int64_t &, u_int64_t &) override {
        ++calls;
        saw_pending_during_task = prim_context->sram_bind_pending_;
        seen_inputs.clear();
        for (size_t index = 0; index < data_size_input.size(); ++index)
            seen_inputs.push_back(
                prim_context->datapass_label_->indata[index]);
        seen_output = prim_context->datapass_label_->outdata;
        if (throw_from_task)
            throw std::runtime_error("BindingProbeNpu injected failure");
    }

    size_t calls = 0;
    bool saw_pending_during_task = true;
    bool throw_from_task = false;
    std::vector<std::string> seen_inputs;
    std::string seen_output;
};

class PassThroughPrim final : public PrimBase {
public:
    explicit PassThroughPrim(int category) {
        name = "PassThroughPrim";
        setPrimMainCategory(category);
    }
    int taskCoreDefault(TaskCoreContext &) override { return 0; }
    Wire serialize() override { return {}; }
    void deserialize(Wire) override {}
    void printSelf() override {}
};

uint8_t DifferentId(const Wire &wire) {
    const uint8_t id = static_cast<uint8_t>(wire.front().range(7, 0).to_uint());
    return id == 1 ? 2 : 1;
}

template <typename Prim>
void CheckRoundTrip(TestState &state, const std::string &name, Prim &source) {
    const Wire first = source.serialize();
    state.Check(SameWire(first, source.serialize()),
                name + " serialization is deterministic");
    Prim decoded;
    decoded.deserialize(first);
    state.Check(SameWire(first, decoded.serialize()),
                name + " round-trip preserves its wire");
}

template <typename Prim>
void CheckNativeMultiFraming(TestState &state, const std::string &name,
                             const Wire &valid) {
    state.Check(valid.size() > 1, name + " fixture is multi-segment");
    state.Throws(name + " short segment list", [&] {
        Wire bad = valid;
        bad.pop_back();
        Prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws(name + " long segment list", [&] {
        Wire bad = valid;
        sc_bv<128> extra = 0;
        extra.range(7, 0) = valid.front().range(7, 0);
        bad.push_back(extra);
        Prim decoded;
        decoded.deserialize(std::move(bad));
    });
    for (size_t i = 0; i < valid.size(); ++i) {
        state.Throws(name + " wrong PrimId in segment " + std::to_string(i),
                     [&, i] {
            Wire bad = valid;
            bad[i].range(7, 0) = DifferentId(valid);
            Prim decoded;
            decoded.deserialize(std::move(bad));
        });
    }
}

template <typename Prim>
void CheckSingleFraming(TestState &state, const std::string &name,
                        const Wire &valid, int reserved_bit) {
    state.Check(valid.size() == 1, name + " fixture is single-segment");
    state.Throws(name + " empty segment list", [&] {
        Prim decoded;
        decoded.deserialize({});
    });
    state.Throws(name + " long segment list", [&] {
        Wire bad = valid;
        bad.push_back(valid.front());
        Prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws(name + " wrong PrimId", [&] {
        Wire bad = valid;
        bad[0].range(7, 0) = DifferentId(valid);
        Prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws(name + " reserved bit", [&] {
        Wire bad = valid;
        bad[0][reserved_bit] = sc_dt::SC_LOGIC_1;
        Prim decoded;
        decoded.deserialize(std::move(bad));
    });
}

template <typename Prim>
void CheckWrappedFraming(TestState &state, const std::string &name,
                         const Wire &valid) {
    state.Check(valid.size() >= 4, name + " fixture has framing");
    auto decode = [](Wire wire) {
        Prim decoded;
        decoded.deserialize(std::move(wire));
    };
    state.Throws(name + " short/trailer-missing wire", [&] {
        Wire bad = valid;
        bad.pop_back();
        decode(std::move(bad));
    });
    state.Throws(name + " long wire", [&] {
        Wire bad = valid;
        sc_bv<128> extra = 0;
        extra.range(7, 0) = valid.front().range(7, 0);
        bad.insert(bad.end() - 1, extra);
        decode(std::move(bad));
    });
    for (size_t i = 0; i < valid.size(); ++i) {
        state.Throws(name + " wrong PrimId in segment " + std::to_string(i),
                     [&, i] {
            Wire bad = valid;
            bad[i].range(7, 0) = DifferentId(valid);
            decode(std::move(bad));
        });
    }
    state.Throws(name + " trailer magic", [&] {
        Wire bad = valid;
        bad.back()[8] = bad.back()[8].to_bool() ? sc_dt::SC_LOGIC_0
                                                : sc_dt::SC_LOGIC_1;
        decode(std::move(bad));
    });
    state.Throws(name + " trailer version", [&] {
        Wire bad = valid;
        bad.back().range(47, 40) = prim_wire::kTrailerVersion + 1;
        decode(std::move(bad));
    });
    state.Throws(name + " trailer original count", [&] {
        Wire bad = valid;
        bad.back().range(63, 48) =
            bad.back().range(63, 48).to_uint() + 1;
        decode(std::move(bad));
    });
    state.Throws(name + " trailer carry count", [&] {
        Wire bad = valid;
        bad.back().range(79, 64) =
            bad.back().range(79, 64).to_uint() + 1;
        decode(std::move(bad));
    });
    state.Throws(name + " trailer reserved bit", [&] {
        Wire bad = valid;
        bad.back()[80] = sc_dt::SC_LOGIC_1;
        decode(std::move(bad));
    });

    const size_t original = valid.back().range(63, 48).to_uint();
    const size_t carries = valid.back().range(79, 64).to_uint();
    const size_t used = (original - 1) % prim_wire::kCarryBytesPerSegment;
    state.Check(carries != 0 && used != 0,
                name + " fixture exposes carry padding");
    if (carries != 0 && used != 0) {
        state.Throws(name + " carry padding", [&] {
            Wire bad = valid;
            bad[original + carries - 1]
                [static_cast<int>(8 + used * 8)] = sc_dt::SC_LOGIC_1;
            decode(std::move(bad));
        });
    }
}

template <typename Prim>
void CheckWrappedBitRejected(TestState &state, const std::string &name,
                             const Wire &valid, size_t segment, int bit) {
    state.Throws(name, [&] {
        Wire bad = valid;
        bad.at(segment)[bit] = sc_dt::SC_LOGIC_1;
        Prim decoded;
        decoded.deserialize(std::move(bad));
    });
}

CollDescriptor MakeDescriptor(bool reduction) {
    CollDescriptor descriptor;
    descriptor.op = reduction ? CollOp::REDUCE : CollOp::P2P;
    descriptor.algorithm = CollAlgorithm::DIRECT;
    descriptor.dtype = reduction ? CollDType::INT32 : CollDType::UINT8;
    descriptor.reduce_op = reduction ? CollReduceOp::SUM : CollReduceOp::NONE;
    descriptor.key.group_id = 0x12345678;
    descriptor.key.collective_id = 0x23456789;
    descriptor.key.epoch = 0x3456789a;
    descriptor.group = {2, 7, 11};
    descriptor.root_rank = 0;
    descriptor.self_rank = 1;
    descriptor.count = 8;
    descriptor.chunk_bits = reduction ? 256 : 64;
    descriptor.stride_bits = descriptor.chunk_bits;
    descriptor.src_addr = 0x1000;
    descriptor.dst_addr = 0x2000;
    descriptor.gather_reorder_depth = 5;
    return descriptor;
}

template <typename Prim>
void CheckCollectiveDescriptorWire(TestState &state, const std::string &name,
                                   const Wire &valid) {
    state.Throws(name + " descriptor magic", [&] {
        Wire bad = valid;
        bad[1][8] = bad[1][8].to_bool() ? sc_dt::SC_LOGIC_0
                                        : sc_dt::SC_LOGIC_1;
        Prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws(name + " descriptor version", [&] {
        Wire bad = valid;
        bad[1].range(23, 16) = COLL_WIRE_VERSION + 1;
        Prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws(name + " descriptor invalid enum", [&] {
        Wire bad = valid;
        bad[1].range(31, 24) = 0xff;
        Prim decoded;
        decoded.deserialize(std::move(bad));
    });
    CheckWrappedBitRejected<Prim>(state, name + " descriptor reserved bit",
                                  valid, 1, 120);
    state.Throws(name + " descriptor encoded segment count", [&] {
        Wire bad = valid;
        bad[1].range(119, 104) =
            bad[1].range(119, 104).to_uint() + 1;
        Prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws(name + " descriptor group padding", [&] {
        Wire bad = valid;
        // Fixture has three group IDs; byte 6 begins padding in its group
        // segment. Marker + five fixed descriptor segments => index six.
        bad[6][48] = sc_dt::SC_LOGIC_1;
        Prim decoded;
        decoded.deserialize(std::move(bad));
    });
}

void TestNpuBase(TestState &state) {
    Layernorm_f source;
    source.datatype = FP16;
    source.inp_offset = std::numeric_limits<int32_t>::min();
    source.data_offset = 17;
    source.out_offset = std::numeric_limits<int32_t>::max();
    source.param_value = {{"B", 1}, {"T", 2}, {"C", 3}};
    CheckRoundTrip(state, "NpuBase", source);
    const Wire wire = source.serialize();
    CheckNativeMultiFraming<Layernorm_f>(state, "NpuBase", wire);
    state.Throws("NpuBase metadata reserved bit", [&] {
        Wire bad = wire;
        bad[0][106] = sc_dt::SC_LOGIC_1;
        Layernorm_f decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("NpuBase invalid datatype decode", [&] {
        Wire bad = wire;
        bad[0].range(9, 8) = 2;
        Layernorm_f decoded;
        decoded.deserialize(std::move(bad));
    });
    Layernorm_f invalid = source;
    invalid.datatype = static_cast<DATATYPE>(2);
    state.Throws("NpuBase invalid datatype encode", [&] { invalid.serialize(); });
    Layernorm_f maximum = source;
    maximum.param_value["C"] = 0x3fffffff;
    state.Check(!maximum.serialize().empty(), "NpuBase 30-bit parameter maximum");
    maximum.param_value["C"] = 0x40000000;
    state.Throws("NpuBase 30-bit parameter maximum plus one",
                 [&] { maximum.serialize(); });
    state.Throws("NpuBase unused parameter padding", [&] {
        Wire bad = wire;
        bad.back()[98] = sc_dt::SC_LOGIC_1;
        Layernorm_f decoded;
        decoded.deserialize(std::move(bad));
    });
    Layernorm_f reused;
    reused.param_value["STALE"] = 9;
    reused.deserialize(wire);
    state.Check(reused.param_value.count("STALE") == 0,
                "NpuBase repeated deserialize clears old parameters");

    Layernorm_f legacy_source;
    legacy_source.datatype = FP16;
    legacy_source.inp_offset = 1;
    legacy_source.data_offset = 2;
    legacy_source.out_offset = 3;
    legacy_source.param_value = {{"B", 4}, {"T", 5}, {"C", 6}};
    const Wire legacy_strict = legacy_source.serialize();
    const Wire legacy = prim_wire::LegacyTransportSegments(
        legacy_strict, legacy_source.name);
    Wire expected_legacy = legacy_strict;
    sc_bv<128> expected_metadata = 0;
    expected_metadata.range(7, 0) = legacy_strict[0].range(7, 0);
    expected_metadata.range(8, 8) = FP16;
    expected_metadata.range(24, 9) = 1;
    expected_metadata.range(40, 25) = 2;
    expected_metadata.range(56, 41) = 3;
    expected_legacy[0] = expected_metadata;
    state.Check(SameWire(legacy, expected_legacy),
                "NpuBase legacy metadata golden");
    {
        LegacyModeGuard guard(true);
        Layernorm_f decoded;
        decoded.deserialize(legacy);
        state.Check(decoded.inp_offset == 1 && decoded.data_offset == 2 &&
                        decoded.out_offset == 3 &&
                        SameWire(decoded.serialize(), legacy_strict),
                    "NpuBase legacy decode preserves semantics");
    }

    Conv_f collision;
    collision.datatype = INT8;
    collision.param_value = {{"B", 1}, {"C", 2}, {"F", 3}, {"H", 4},
                             {"W", 5}, {"kX", 6}, {"kY", 7}, {"pX", 8},
                             {"pY", static_cast<int>(prim_wire::kTrailerMagic)},
                             {"sX", 4}, {"sY", 9}};
    const Wire collision_strict = collision.serialize();
    state.Check(prim_wire::HasStrictTrailer(collision_strict),
                "NpuBase collision fixture mimics trailer payload");
    state.Check(prim_wire::LegacyTransportSegments(
                    collision_strict, collision.name).size() ==
                    collision_strict.size(),
                "NpuBase legacy dispatch ignores trailer-like payload");
}

void TestGpuBase(TestState &state) {
    Matmul_f_gpu source;
    source.datatype = INT8;
    source.fetch_index = 19;
    source.req_sm = 31;
    source.param_value = {{"B", 1}, {"T", 2}, {"C", 3}, {"OC", 4},
                          {"slice_x", 5}, {"slice_y", 6}};
    CheckRoundTrip(state, "GpuBase", source);
    const Wire wire = source.serialize();
    CheckNativeMultiFraming<Matmul_f_gpu>(state, "GpuBase", wire);
    state.Throws("GpuBase metadata reserved bit", [&] {
        Wire bad = wire;
        bad[0][74] = sc_dt::SC_LOGIC_1;
        Matmul_f_gpu decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("GpuBase invalid datatype decode", [&] {
        Wire bad = wire;
        bad[0].range(9, 8) = 2;
        Matmul_f_gpu decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("GpuBase runtime index overflow", [&] {
        Wire bad = wire;
        bad[0].range(41, 10) = 0xffffffffU;
        Matmul_f_gpu decoded;
        decoded.deserialize(std::move(bad));
    });
    Matmul_f_gpu maximum = source;
    maximum.param_value["C"] = 0x3fffffff;
    state.Check(!maximum.serialize().empty(), "GpuBase 30-bit parameter maximum");
    maximum.param_value["C"] = 0x40000000;
    state.Throws("GpuBase 30-bit parameter maximum plus one",
                 [&] { maximum.serialize(); });
    state.Throws("GpuBase unused parameter padding", [&] {
        Wire bad = wire;
        bad.back()[68] = sc_dt::SC_LOGIC_1;
        Matmul_f_gpu decoded;
        decoded.deserialize(std::move(bad));
    });
    Matmul_f_gpu reused;
    reused.param_value["STALE"] = 9;
    reused.deserialize(wire);
    state.Check(reused.param_value.count("STALE") == 0,
                "GpuBase repeated deserialize clears old parameters");

    Matmul_f_gpu legacy_source;
    legacy_source.datatype = FP16;
    legacy_source.fetch_index = 19;
    legacy_source.req_sm = 31;
    legacy_source.param_value = {{"B", 1}, {"T", 2}, {"C", 3}, {"OC", 4},
                                 {"slice_x", 5}, {"slice_y", 6}};
    const Wire legacy_strict = legacy_source.serialize();
    const Wire legacy = prim_wire::LegacyTransportSegments(
        legacy_strict, legacy_source.name);
    Wire expected_legacy = legacy_strict;
    sc_bv<128> expected_metadata = 0;
    expected_metadata.range(7, 0) = legacy_strict[0].range(7, 0);
    expected_metadata.range(8, 8) = FP16;
    expected_metadata.range(24, 9) = 19;
    expected_metadata.range(56, 41) = 31;
    expected_legacy[0] = expected_metadata;
    state.Check(SameWire(legacy, expected_legacy),
                "GpuBase legacy metadata golden");
    {
        LegacyModeGuard guard(true);
        Matmul_f_gpu decoded;
        decoded.deserialize(legacy);
        state.Check(decoded.fetch_index == 19 && decoded.req_sm == 31 &&
                        SameWire(decoded.serialize(), legacy_strict),
                    "GpuBase legacy decode preserves semantics");
    }

    legacy_source.param_value["C"] = 0x3fffffff;
    const Wire high_strict = legacy_source.serialize();
    const Wire high_legacy = prim_wire::LegacyTransportSegments(
        high_strict, legacy_source.name);
    state.Check(high_legacy[1] == high_strict[1],
                "GpuBase legacy transport preserves parameter bits");
    {
        LegacyModeGuard guard(true);
        Matmul_f_gpu decoded;
        decoded.deserialize(high_legacy);
        state.Check(decoded.param_value.at("C") == 0x3fffff,
                    "GpuBase legacy decoder preserves historical 22-bit window");
    }
}

void TestCollectiveDataV1Wire(TestState &state) {
    Collective_data_v1_prim prim;
    prim.mode = CollectiveDataV1PrimMode::REDUCE;
    prim.key = {0x10203040u, 0x50607080u, 0x90a0b0c0u};
    prim.phase_id = 0xd0e0u;
    prim.source_address_bytes = UINT64_C(0x1122334455667788);
    prim.destination_address_bytes = UINT64_C(0x8877665544332211);
    prim.length_bytes = 32;
    prim.input_count = 4;
    prim.dtype = CollDType::INT32;
    prim.reduce_op = CollReduceOp::MAX;
    CheckRoundTrip(state, "Collective_data_v1_prim", prim);
    const Wire wire = prim.serialize();
    CheckNativeMultiFraming<Collective_data_v1_prim>(
        state, "Collective_data_v1_prim", wire);
    bool headers = wire.size() == kCollectiveDataV1PrimWireSegments;
    for (size_t index = 0; index < wire.size(); ++index)
        headers = headers &&
            wire[index].range(7, 0).to_uint() ==
                PrimIdValue(PrimId::COLLECTIVE_DATA_V1) &&
            wire[index].range(15, 8).to_uint() == index;
    state.Check(headers &&
                    wire[0].range(23, 16).to_uint() ==
                        kCollectiveDataV1PrimWireVersion &&
                    wire[0].range(31, 24).to_uint() ==
                        kCollectiveDataV1PrimWireSegments &&
                    wire[1].range(47, 16).to_uint64() == prim.key.group_id &&
                    wire[1].range(127, 112).to_uint64() == prim.phase_id &&
                    wire[2].range(79, 16).to_uint64() ==
                        prim.source_address_bytes &&
                    wire[3].range(79, 16).to_uint64() ==
                        prim.destination_address_bytes &&
                    wire[4].range(79, 16).to_uint64() == prim.length_bytes &&
                    wire[5].range(31, 16).to_uint64() == prim.input_count,
                "Collective_data_v1_prim strict fixed wire golden");
    state.Throws("Collective_data_v1_prim wrong ordinal", [&] {
        Wire bad = wire;
        bad[3].range(15, 8) = 2;
        Collective_data_v1_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Collective_data_v1_prim reserved bits", [&] {
        Wire bad = wire;
        bad[5][32] = sc_dt::SC_LOGIC_1;
        Collective_data_v1_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    Collective_data_v1_prim local;
    local.mode = CollectiveDataV1PrimMode::REDUCE;
    local.key = {};
    local.phase_id = 0;
    local.source_address_bytes = 0x100;
    local.destination_address_bytes = 0x200;
    local.length_bytes = 32;
    local.input_count = 4;
    local.dtype = CollDType::FP16;
    local.reduce_op = CollReduceOp::SUM;
    CheckRoundTrip(state, "Collective_data_v1_prim local FP16", local);
    const Wire local_wire = local.serialize();
    Collective_data_v1_prim local_decoded;
    local_decoded.deserialize(local_wire);
    state.Check(local_wire[0].range(34, 33).to_uint() == 3 &&
                    local_decoded.dtype == CollDType::FP16 &&
                    local_decoded.key == CollectiveKey{} &&
                    local_decoded.phase_id == 0,
                "Collective_data_v1_prim wire dtype code 3 explicitly maps FP16");
    state.Throws("Collective_data_v1_prim FP32 rejected", [&] {
        Collective_data_v1_prim bad = local;
        bad.dtype = CollDType::FP32;
        (void)bad.serialize();
    });
    state.Throws("Collective_data_v1_prim FP8 rejected", [&] {
        Collective_data_v1_prim bad = local;
        bad.dtype = CollDType::FP8;
        (void)bad.serialize();
    });
    state.Throws("Collective_data_v1_prim key-zero phase rejected", [&] {
        Collective_data_v1_prim bad = local;
        bad.phase_id = 1;
        (void)bad.serialize();
    });
    state.Throws("Collective_data_v1_prim key-zero non-FP16 rejected", [&] {
        Collective_data_v1_prim bad = local;
        bad.dtype = CollDType::INT32;
        (void)bad.serialize();
    });
    state.Throws("Collective_data_v1_prim zero length", [&] {
        Collective_data_v1_prim bad = prim;
        bad.length_bytes = 0;
        (void)bad.serialize();
    });
    state.Throws("Collective_data_v1_prim zero input count", [&] {
        Collective_data_v1_prim bad = prim;
        bad.input_count = 0;
        (void)bad.serialize();
    });
    state.Throws("Collective_data_v1_prim source multiplication overflow", [&] {
        Collective_data_v1_prim bad = prim;
        bad.length_bytes = std::numeric_limits<uint64_t>::max();
        bad.input_count = 2;
        (void)bad.serialize();
    });
    state.Throws("Collective_data_v1_prim destination span overflow", [&] {
        Collective_data_v1_prim bad = prim;
        bad.destination_address_bytes =
            std::numeric_limits<uint64_t>::max() - 15;
        (void)bad.serialize();
    });
    state.Throws("Collective_data_v1_prim noncanonical local copy", [&] {
        Collective_data_v1_prim bad = prim;
        bad.mode = CollectiveDataV1PrimMode::LOCAL_COPY;
        (void)bad.serialize();
    });
    state.Throws("Collective_data_v1_prim legacy transport", [&] {
        (void)prim_wire::LegacyTransportSegments(wire, prim.name);
    });
    {
        LegacyModeGuard guard(true);
        state.Throws("Collective_data_v1_prim legacy serialize", [&] {
            (void)prim.serialize();
        });
        state.Throws("Collective_data_v1_prim legacy deserialize", [&] {
            Collective_data_v1_prim decoded;
            decoded.deserialize(wire);
        });
    }
}

void TestCollectivePhaseBarrierV1Wire(TestState &state) {
    Collective_phase_barrier_v1_prim prim;
    prim.key = {0x10203040u, 0x50607080u, 0x90a0b0c0u};
    prim.phase_id = 0xd0e0u;
    prim.rank = 3;
    prim.group_size = 4;
    prim.release_tree_id = std::numeric_limits<uint16_t>::max();
    CheckRoundTrip(state, "Collective_phase_barrier_v1_prim", prim);
    const Wire wire = prim.serialize();
    CheckNativeMultiFraming<Collective_phase_barrier_v1_prim>(
        state, "Collective_phase_barrier_v1_prim", wire);
    bool headers = wire.size() == kCollectivePhaseBarrierV1WireSegments;
    for (size_t index = 0; index < wire.size(); ++index)
        headers = headers &&
            wire[index].range(7, 0).to_uint() ==
                PrimIdValue(PrimId::COLLECTIVE_PHASE_BARRIER_V1) &&
            wire[index].range(15, 8).to_uint() == index;
    state.Check(headers &&
                    wire[0].range(23, 16).to_uint() ==
                        kCollectivePhaseBarrierV1WireVersion &&
                    wire[0].range(31, 24).to_uint() ==
                        kCollectivePhaseBarrierV1WireSegments &&
                    wire[0].range(47, 32).to_uint() == prim.phase_id &&
                    wire[0].range(63, 48).to_uint() == prim.rank &&
                    wire[0].range(79, 64).to_uint() == prim.group_size &&
                    wire[0].range(95, 80).to_uint() ==
                        prim.release_tree_id &&
                    wire[1].range(47, 16).to_uint64() ==
                        prim.key.group_id &&
                    wire[1].range(79, 48).to_uint64() ==
                        prim.key.collective_id &&
                    wire[1].range(111, 80).to_uint64() == prim.key.epoch,
                "Collective_phase_barrier_v1_prim strict fixed wire golden");
    state.Throws("Collective_phase_barrier_v1_prim wrong ordinal", [&] {
        Wire bad = wire;
        bad[1].range(15, 8) = 0;
        Collective_phase_barrier_v1_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Collective_phase_barrier_v1_prim wrong version", [&] {
        Wire bad = wire;
        bad[0].range(23, 16) = 2;
        Collective_phase_barrier_v1_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Collective_phase_barrier_v1_prim wrong count field", [&] {
        Wire bad = wire;
        bad[0].range(31, 24) = 3;
        Collective_phase_barrier_v1_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Collective_phase_barrier_v1_prim reserved bits", [&] {
        Wire bad = wire;
        bad[1][112] = sc_dt::SC_LOGIC_1;
        Collective_phase_barrier_v1_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Collective_phase_barrier_v1_prim zero group key", [&] {
        Collective_phase_barrier_v1_prim bad = prim;
        bad.key.group_id = 0;
        (void)bad.serialize();
    });
    state.Throws("Collective_phase_barrier_v1_prim GROUP_SYNC key", [&] {
        Collective_phase_barrier_v1_prim bad = prim;
        bad.key.collective_id = std::numeric_limits<uint32_t>::max();
        (void)bad.serialize();
    });
    state.Throws("Collective_phase_barrier_v1_prim zero group", [&] {
        Collective_phase_barrier_v1_prim bad = prim;
        bad.group_size = 0;
        (void)bad.serialize();
    });
    state.Throws("Collective_phase_barrier_v1_prim rank range", [&] {
        Collective_phase_barrier_v1_prim bad = prim;
        bad.rank = bad.group_size;
        (void)bad.serialize();
    });
    state.Throws("Collective_phase_barrier_v1_prim legacy transport", [&] {
        (void)prim_wire::LegacyTransportSegments(wire, prim.name);
    });
    {
        LegacyModeGuard guard(true);
        state.Throws("Collective_phase_barrier_v1_prim legacy serialize", [&] {
            (void)prim.serialize();
        });
        state.Throws("Collective_phase_barrier_v1_prim legacy deserialize", [&] {
            Collective_phase_barrier_v1_prim decoded;
            decoded.deserialize(wire);
        });
    }
}

Dte_send_endpoint_prim MakeEndpointSendRegion() {
    Dte_send_endpoint_prim prim;
    prim.mode = DteEndpointSendMode::P2P;
    prim.completion = DteEndpointCompletion::ASYNC;
    prim.datatype = DteEndpointDataType::UINT8;
    prim.reduce_op = DteEndpointReduceOp::NONE;
    prim.fsm_id = 0x10203040u;
    prim.token = 0x50607080u;
    prim.length_bytes = kDteEndpointP2pMaxBytes;
    prim.source.kind = DteEndpointAddressKind::REGION;
    prim.source.region = "endpoint/source";
    prim.source.region_offset_bytes = UINT64_C(0x8877665544332211);
    prim.peer_core = 0xabcd;
    return prim;
}

void TestDteEndpointWires(TestState &state) {
    Dte_send_endpoint_prim send = MakeEndpointSendRegion();
    CheckRoundTrip(state, "Dte_send_endpoint_prim", send);
    const Wire send_wire = send.serialize();
    CheckNativeMultiFraming<Dte_send_endpoint_prim>(
        state, "Dte_send_endpoint_prim", send_wire);
    state.Check(send_wire.size() == 7,
                "DTE send region uses exact variable segment count");
    bool send_headers = true;
    for (size_t index = 0; index < send_wire.size(); ++index) {
        send_headers = send_headers &&
            send_wire[index].range(7, 0).to_uint() ==
                PrimIdValue(PrimId::DTE_SEND_ENDPOINT) &&
            send_wire[index].range(15, 8).to_uint() == index;
    }
    state.Check(send_headers,
                "DTE send writes ID and ordinal into every segment");
    state.Check(
        send_wire[0].range(23, 16).to_uint() ==
                kDteEndpointPrimWireVersion &&
            send_wire[0].range(31, 24).to_uint() == send_wire.size() &&
            send_wire[0].range(33, 32).to_uint() ==
                static_cast<uint8_t>(DteEndpointSendMode::P2P) &&
            send_wire[0].range(34, 34).to_uint() ==
                static_cast<uint8_t>(DteEndpointCompletion::ASYNC) &&
            send_wire[0].range(36, 35).to_uint() ==
                static_cast<uint8_t>(DteEndpointDataType::UINT8) &&
            send_wire[0].range(38, 37).to_uint() ==
                static_cast<uint8_t>(DteEndpointReduceOp::NONE) &&
            send_wire[0].range(40, 39).to_uint() ==
                static_cast<uint8_t>(DteEndpointAddressKind::REGION) &&
            send_wire[0].range(42, 41).to_uint() ==
                static_cast<uint8_t>(DteEndpointSourceSpace::SRAM) &&
            !send_wire[0].range(127, 43).or_reduce(),
        "DTE send metadata strict wire golden");
    state.Check(
        send_wire[1].range(47, 16).to_uint64() == send.fsm_id &&
            send_wire[1].range(79, 48).to_uint64() == send.token &&
            send_wire[1].range(95, 80).to_uint64() == send.peer_core &&
            send_wire[1].range(111, 96).to_uint64() == 0 &&
            send_wire[1].range(127, 112).to_uint64() == 0 &&
            send_wire[2].range(79, 16).to_uint64() == send.length_bytes &&
            send_wire[3].range(79, 16).to_uint64() ==
                send.source.region_offset_bytes &&
            send_wire[4].range(119, 112).to_uint64() ==
                send.source.region.size(),
        "DTE send control/address strict wire golden");

    state.Throws("DTE send unsupported wire version", [&] {
        Wire bad = send_wire;
        bad[0].range(23, 16) = 2;
        Dte_send_endpoint_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("DTE send explicit segment count mismatch", [&] {
        Wire bad = send_wire;
        bad[0].range(31, 24) = send_wire.size() - 1;
        Dte_send_endpoint_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("DTE send segment ordinal mismatch", [&] {
        Wire bad = send_wire;
        bad[3].range(15, 8) = 2;
        Dte_send_endpoint_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    for (const auto &location :
         {std::pair<size_t, int>{0, 43}, {2, 80}, {3, 80}, {4, 120}}) {
        state.Throws("DTE send reserved bit in segment " +
                         std::to_string(location.first),
                     [&, location] {
            Wire bad = send_wire;
            bad[location.first][location.second] = sc_dt::SC_LOGIC_1;
            Dte_send_endpoint_prim decoded;
            decoded.deserialize(std::move(bad));
        });
    }
    state.Throws("DTE send invalid mode decode", [&] {
        Wire bad = send_wire;
        bad[0].range(33, 32) = 3;
        Dte_send_endpoint_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("DTE send invalid datatype decode", [&] {
        Wire bad = send_wire;
        bad[0].range(36, 35) = 3;
        Dte_send_endpoint_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("DTE send invalid reduce enum decode", [&] {
        Wire bad = send_wire;
        bad[0].range(38, 37) = 3;
        Dte_send_endpoint_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("DTE send invalid address kind decode", [&] {
        Wire bad = send_wire;
        bad[0].range(40, 39) = 0;
        Dte_send_endpoint_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("DTE send invalid source space decode", [&] {
        Wire bad = send_wire;
        bad[0].range(42, 41) = 2;
        Dte_send_endpoint_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("DTE send HBM region decode", [&] {
        Wire bad = send_wire;
        bad[0].range(42, 41) =
            static_cast<uint8_t>(DteEndpointSourceSpace::HBM);
        Dte_send_endpoint_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("DTE send region NUL decode", [&] {
        Wire bad = send_wire;
        bad[5].range(23, 16) = 0;
        Dte_send_endpoint_prim decoded;
        decoded.deserialize(std::move(bad));
    });

    Dte_send_endpoint_prim max_send;
    max_send.mode = DteEndpointSendMode::BROADCAST;
    max_send.completion = DteEndpointCompletion::ASYNC;
    max_send.fsm_id = UINT32_MAX;
    max_send.token = UINT32_MAX;
    max_send.length_bytes = kDteEndpointP2pMaxBytes;
    max_send.source.kind = DteEndpointAddressKind::REGION;
    max_send.source.region.assign(kDteEndpointPrimRegionMaxBytes, 'R');
    max_send.source.region_offset_bytes = UINT64_MAX;
    max_send.tree_id = UINT16_MAX;
    max_send.group_id = UINT32_MAX;
    max_send.collective_id = UINT32_MAX - 1;
    max_send.epoch = UINT32_MAX;
    CheckRoundTrip(state, "DTE send maximum", max_send);
    const Wire max_send_wire = max_send.serialize();
    state.Check(max_send_wire.size() == 10,
                "DTE send maximum region occupies five text segments");
    state.Throws("DTE send region tail padding", [&] {
        Wire bad = max_send_wire;
        bad.back()[80] = sc_dt::SC_LOGIC_1;
        Dte_send_endpoint_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("DTE send region length above maximum", [&] {
        Wire bad = max_send_wire;
        bad[4].range(119, 112) = 65;
        Dte_send_endpoint_prim decoded;
        decoded.deserialize(std::move(bad));
    });

    Dte_send_endpoint_prim invalid_send = send;
    invalid_send.fsm_id = 0;
    state.Throws("DTE send zero fsm encode",
                 [&] { invalid_send.serialize(); });
    invalid_send = send;
    invalid_send.token = 0;
    state.Throws("DTE send async zero token encode",
                 [&] { invalid_send.serialize(); });
    invalid_send = send;
    invalid_send.length_bytes = 0;
    state.Throws("DTE send zero length encode",
                 [&] { invalid_send.serialize(); });
    invalid_send = send;
    invalid_send.length_bytes = kDteEndpointP2pMaxBytes + 1;
    state.Throws("DTE send P2P transport max+1 encode",
                 [&] { invalid_send.serialize(); });
    invalid_send = send;
    invalid_send.source_space = static_cast<DteEndpointSourceSpace>(2);
    state.Throws("DTE send source space enum max+1 encode",
                 [&] { invalid_send.serialize(); });
    invalid_send = send;
    invalid_send.source_space = DteEndpointSourceSpace::HBM;
    state.Throws("DTE send HBM region encode",
                 [&] { invalid_send.serialize(); });
    invalid_send = send;
    invalid_send.source.absolute_address_bytes = 1;
    state.Throws("DTE send region/absolute XOR encode",
                 [&] { invalid_send.serialize(); });
    invalid_send = send;
    invalid_send.source.region.assign(65, 'x');
    state.Throws("DTE send oversized region encode",
                 [&] { invalid_send.serialize(); });
    invalid_send = send;
    invalid_send.expected_sources = 1;
    state.Throws("DTE send receive-only metadata encode",
                 [&] { invalid_send.serialize(); });
    invalid_send = send;
    invalid_send.group_id = 1;
    state.Throws("DTE send P2P collective metadata encode",
                 [&] { invalid_send.serialize(); });
    invalid_send = max_send;
    invalid_send.tree_id = 0;
    state.Throws("DTE send broadcast zero tree encode",
                 [&] { invalid_send.serialize(); });
    invalid_send = max_send;
    invalid_send.collective_id = UINT32_MAX;
    state.Throws("DTE send reserved collective ID encode",
                 [&] { invalid_send.serialize(); });

    Dte_send_endpoint_prim scatter_send;
    scatter_send.mode = DteEndpointSendMode::SCATTER;
    scatter_send.fsm_id = 3;
    scatter_send.token = 4;
    scatter_send.length_bytes = 5;
    scatter_send.source.absolute_address_bytes = 6;
    scatter_send.group_id = 7;
    scatter_send.collective_id = 8;
    scatter_send.epoch = 9;
    CheckRoundTrip(state, "DTE send SCATTER", scatter_send);
    invalid_send = scatter_send;
    invalid_send.peer_core = 1;
    state.Throws("DTE send scatter peer encode",
                 [&] { invalid_send.serialize(); });

    Dte_send_endpoint_prim absolute_send;
    absolute_send.fsm_id = 7;
    absolute_send.token = 8;
    absolute_send.length_bytes = 9;
    absolute_send.peer_core = 10;
    absolute_send.source.absolute_address_bytes = UINT64_MAX;
    const Wire absolute_send_wire = absolute_send.serialize();
    state.Check(absolute_send_wire.size() ==
                    kDteEndpointPrimBaseSegments &&
                    absolute_send_wire[0].range(42, 41).to_uint() ==
                        static_cast<uint8_t>(
                            DteEndpointSourceSpace::SRAM) &&
                    absolute_send_wire[3].range(79, 16).to_uint64() ==
                        UINT64_MAX &&
                    absolute_send_wire[4].range(119, 112).to_uint() == 0,
                "DTE send SRAM absolute address golden");

    Dte_send_endpoint_prim hbm_send = absolute_send;
    hbm_send.source_space = DteEndpointSourceSpace::HBM;
    hbm_send.length_bytes = kDteEndpointP2pMaxBytes;
    const Wire hbm_send_wire = hbm_send.serialize();
    CheckRoundTrip(state, "DTE send HBM absolute maximum", hbm_send);
    state.Check(hbm_send_wire[0].range(42, 41).to_uint() ==
                        static_cast<uint8_t>(DteEndpointSourceSpace::HBM) &&
                    hbm_send_wire[3].range(79, 16).to_uint64() == UINT64_MAX,
                "DTE send HBM source-space/address maximum wire");
    Dte_send_endpoint_prim hbm_min = hbm_send;
    hbm_min.source.absolute_address_bytes = 0;
    const Wire hbm_min_wire = hbm_min.serialize();
    Dte_send_endpoint_prim hbm_min_decoded;
    hbm_min_decoded.deserialize(hbm_min_wire);
    state.Check(hbm_min_decoded.source_space ==
                        DteEndpointSourceSpace::HBM &&
                    hbm_min_decoded.source.absolute_address_bytes == 0,
                "DTE send HBM absolute minimum roundtrip");

    Dte_send_endpoint_prim reused_send;
    reused_send.deserialize(send_wire);
    reused_send.deserialize(hbm_send_wire);
    state.Check(reused_send.source_space == DteEndpointSourceSpace::HBM,
                "DTE send repeated decode installs HBM source space");
    reused_send.deserialize(absolute_send_wire);
    state.Check(reused_send.source_space == DteEndpointSourceSpace::SRAM &&
                    reused_send.source.kind == DteEndpointAddressKind::ABSOLUTE &&
                    reused_send.source.region.empty() &&
                    reused_send.source.region_offset_bytes == 0 &&
                    SameWire(reused_send.serialize(), absolute_send_wire),
                "DTE send repeated decode clears HBM/region state");
    const Wire reused_before_failure = reused_send.serialize();
    state.Throws("DTE send failed decode is transactional", [&] {
        Wire bad = send_wire;
        bad[0][43] = sc_dt::SC_LOGIC_1;
        reused_send.deserialize(std::move(bad));
    });
    state.Check(SameWire(reused_send.serialize(), reused_before_failure),
                "DTE send failed decode preserves prior state");

    Dte_recv_endpoint_prim recv;
    recv.mode = DteEndpointRecvMode::REDUCE;
    recv.completion = DteEndpointCompletion::SYNC;
    recv.datatype = DteEndpointDataType::INT64;
    recv.reduce_op = DteEndpointReduceOp::MAX;
    recv.fsm_id = UINT32_MAX;
    recv.token = 0;
    recv.length_bytes = kDteEndpointP2pMaxBytes;
    recv.destination.kind = DteEndpointAddressKind::REGION;
    recv.destination.region.assign(kDteEndpointPrimRegionMaxBytes, 'D');
    recv.destination.region_offset_bytes = UINT64_MAX;
    recv.expected_sources = UINT16_MAX;
    recv.group_id = UINT32_MAX;
    recv.collective_id = UINT32_MAX - 1;
    recv.epoch = UINT32_MAX;
    CheckRoundTrip(state, "Dte_recv_endpoint_prim maximum", recv);
    const Wire recv_wire = recv.serialize();
    CheckNativeMultiFraming<Dte_recv_endpoint_prim>(
        state, "Dte_recv_endpoint_prim", recv_wire);
    state.Check(
        recv_wire[0].range(7, 0).to_uint() ==
                PrimIdValue(PrimId::DTE_RECV_ENDPOINT) &&
            recv_wire[0].range(33, 32).to_uint() ==
                static_cast<uint8_t>(DteEndpointRecvMode::REDUCE) &&
            recv_wire[0].range(34, 34).to_uint() ==
                static_cast<uint8_t>(DteEndpointCompletion::SYNC) &&
            recv_wire[0].range(36, 35).to_uint() ==
                static_cast<uint8_t>(DteEndpointDataType::INT64) &&
            recv_wire[0].range(38, 37).to_uint() ==
                static_cast<uint8_t>(DteEndpointReduceOp::MAX) &&
            recv_wire[1].range(111, 96).to_uint64() == UINT16_MAX &&
            recv_wire[4].range(47, 16).to_uint64() == UINT32_MAX &&
            recv_wire[4].range(79, 48).to_uint64() == UINT32_MAX - 1 &&
            recv_wire[4].range(111, 80).to_uint64() == UINT32_MAX,
        "DTE recv collective strict wire golden/maxima");

    Dte_recv_endpoint_prim invalid_recv = recv;
    invalid_recv.tree_id = 1;
    state.Throws("DTE recv tree metadata encode",
                 [&] { invalid_recv.serialize(); });
    invalid_recv = recv;
    invalid_recv.expected_sources = 0;
    state.Throws("DTE recv reduce zero sources encode",
                 [&] { invalid_recv.serialize(); });
    invalid_recv = recv;
    invalid_recv.reduce_op = DteEndpointReduceOp::NONE;
    state.Throws("DTE recv reduce missing operator encode",
                 [&] { invalid_recv.serialize(); });
    invalid_recv = recv;
    invalid_recv.peer_core = 1;
    state.Throws("DTE recv collective peer encode",
                 [&] { invalid_recv.serialize(); });
    invalid_recv = recv;
    invalid_recv.token = 1;
    state.Throws("DTE recv sync token encode",
                 [&] { invalid_recv.serialize(); });
    invalid_recv = recv;
    invalid_recv.mode = static_cast<DteEndpointRecvMode>(3);
    state.Throws("DTE recv invalid mode encode",
                 [&] { invalid_recv.serialize(); });

    Dte_recv_endpoint_prim gather_recv;
    gather_recv.mode = DteEndpointRecvMode::GATHER;
    gather_recv.fsm_id = 15;
    gather_recv.token = 16;
    gather_recv.length_bytes = 17;
    gather_recv.destination.absolute_address_bytes = 18;
    gather_recv.expected_sources = 2;
    gather_recv.group_id = 19;
    gather_recv.collective_id = 20;
    gather_recv.epoch = 21;
    CheckRoundTrip(state, "DTE recv GATHER", gather_recv);
    invalid_recv = gather_recv;
    invalid_recv.datatype = DteEndpointDataType::INT32;
    state.Throws("DTE recv gather non-byte datatype encode",
                 [&] { invalid_recv.serialize(); });

    Dte_recv_endpoint_prim p2p_recv;
    p2p_recv.fsm_id = 11;
    p2p_recv.token = 12;
    p2p_recv.length_bytes = kDteEndpointP2pMaxBytes;
    p2p_recv.destination.absolute_address_bytes = 14;
    p2p_recv.peer_core = UINT16_MAX;
    const Wire p2p_recv_wire = p2p_recv.serialize();
    CheckRoundTrip(state, "DTE recv P2P transport maximum", p2p_recv);
    invalid_recv = p2p_recv;
    invalid_recv.length_bytes = kDteEndpointP2pMaxBytes + 1;
    state.Throws("DTE recv P2P transport max+1 encode",
                 [&] { invalid_recv.serialize(); });
    state.Throws("DTE recv P2P transport max+1 decode", [&] {
        Wire bad = p2p_recv_wire;
        bad[2].range(79, 16) = kDteEndpointP2pMaxBytes + 1;
        Dte_recv_endpoint_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    Dte_recv_endpoint_prim reused_recv;
    reused_recv.deserialize(recv_wire);
    reused_recv.deserialize(p2p_recv_wire);
    state.Check(reused_recv.mode == DteEndpointRecvMode::P2P &&
                    reused_recv.destination.region.empty() &&
                    reused_recv.expected_sources == 0 &&
                    reused_recv.group_id == 0 &&
                    SameWire(reused_recv.serialize(), p2p_recv_wire),
                "DTE recv repeated decode clears collective/region state");

    state.Throws("DTE send strict wire rejects legacy helper", [&] {
        (void)prim_wire::LegacyTransportSegments(send_wire, send.name);
    });
    state.Throws("DTE recv strict wire rejects legacy helper", [&] {
        (void)prim_wire::LegacyTransportSegments(recv_wire, recv.name);
    });
    {
        LegacyModeGuard guard(true);
        state.Throws("DTE send legacy mode encode", [&] { send.serialize(); });
        state.Throws("DTE send HBM legacy mode encode",
                     [&] { hbm_send.serialize(); });
        state.Throws("DTE send HBM legacy mode decode", [&] {
            Dte_send_endpoint_prim decoded;
            decoded.deserialize(hbm_send_wire);
        });
        state.Throws("DTE send legacy mode decode", [&] {
            Dte_send_endpoint_prim decoded;
            decoded.deserialize(send_wire);
        });
        state.Throws("DTE recv legacy mode encode", [&] { recv.serialize(); });
        state.Throws("DTE recv legacy mode decode", [&] {
            Dte_recv_endpoint_prim decoded;
            decoded.deserialize(recv_wire);
        });
    }
}

void TestSynchronizationWires(TestState &state) {
    Group_sync_prim group;
    group.group_id = UINT32_MAX;
    group.sync_seq = 0xfedcba98u;
    CheckRoundTrip(state, "Group_sync_prim", group);
    const Wire group_wire = group.serialize();
    CheckSingleFraming<Group_sync_prim>(
        state, "Group_sync_prim", group_wire, 72);
    state.Check(group_wire[0].range(7, 0).to_uint() ==
                        PrimIdValue(PrimId::GROUP_SYNC) &&
                    group_wire[0].range(39, 8).to_uint64() == UINT32_MAX &&
                    group_wire[0].range(71, 40).to_uint64() == 0xfedcba98u,
                "Group_sync_prim strict wire golden fields");
    Group_sync_prim invalid_group = group;
    invalid_group.group_id = 0;
    state.Throws("Group_sync_prim zero group encode",
                 [&] { invalid_group.serialize(); });
    state.Throws("Group_sync_prim zero group decode", [&] {
        Wire bad = group_wire;
        bad[0].range(39, 8) = 0;
        Group_sync_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Group_sync_prim legacy transport helper", [&] {
        (void)prim_wire::LegacyTransportSegments(group_wire, group.name);
    });
    {
        LegacyModeGuard guard(true);
        state.Throws("Group_sync_prim legacy encode",
                     [&] { group.serialize(); });
        state.Throws("Group_sync_prim legacy decode", [&] {
            Group_sync_prim decoded;
            decoded.deserialize(group_wire);
        });
    }

    Event_control_prim set;
    set.op = EventControlOp::SET;
    set.source_core = UINT16_MAX;
    set.destination_core = 0;
    set.tag = UINT32_MAX;
    set.count = 1;
    CheckRoundTrip(state, "Event_control_prim SET", set);
    const Wire set_wire = set.serialize();
    CheckSingleFraming<Event_control_prim>(
        state, "Event_control_prim", set_wire, 105);
    state.Check(set_wire[0].range(7, 0).to_uint() ==
                        PrimIdValue(PrimId::EVENT_CONTROL) &&
                    set_wire[0].range(8, 8).to_uint() == 0 &&
                    set_wire[0].range(24, 9).to_uint() == UINT16_MAX &&
                    set_wire[0].range(40, 25).to_uint() == 0 &&
                    set_wire[0].range(72, 41).to_uint64() == UINT32_MAX &&
                    set_wire[0].range(104, 73).to_uint64() == 1,
                "Event_control_prim SET strict wire golden fields");

    Event_control_prim wait_prim;
    wait_prim.op = EventControlOp::WAIT;
    wait_prim.source_core = 1;
    wait_prim.destination_core = UINT16_MAX;
    wait_prim.tag = 0x80000001u;
    wait_prim.count = UINT32_MAX;
    CheckRoundTrip(state, "Event_control_prim WAIT", wait_prim);
    const Wire wait_wire = wait_prim.serialize();
    state.Check(wait_wire[0].range(8, 8).to_uint() == 1 &&
                    wait_wire[0].range(104, 73).to_uint64() == UINT32_MAX,
                "Event_control_prim WAIT preserves full count/tag ranges");

    Event_control_prim invalid_event = set;
    invalid_event.count = 2;
    state.Throws("Event_control_prim SET noncanonical count encode",
                 [&] { invalid_event.serialize(); });
    invalid_event = wait_prim;
    invalid_event.count = 0;
    state.Throws("Event_control_prim WAIT zero count encode",
                 [&] { invalid_event.serialize(); });
    state.Throws("Event_control_prim SET noncanonical count decode", [&] {
        Wire bad = set_wire;
        bad[0].range(104, 73) = 2;
        Event_control_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Event_control_prim WAIT zero count decode", [&] {
        Wire bad = wait_wire;
        bad[0].range(104, 73) = 0;
        Event_control_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Event_control_prim legacy transport helper", [&] {
        (void)prim_wire::LegacyTransportSegments(set_wire, set.name);
    });
    {
        LegacyModeGuard guard(true);
        state.Throws("Event_control_prim legacy encode",
                     [&] { set.serialize(); });
        state.Throws("Event_control_prim legacy decode", [&] {
            Event_control_prim decoded;
            decoded.deserialize(set_wire);
        });
    }
}

Sram_lifecycle MakeSramLifecycleAlloc() {
    Sram_lifecycle source;
    const std::string prefix = "__prim_wire_sram_lifecycle_" +
                               std::to_string(g_addr_label_table.table.size());
    source.op = SramLifecycleOp::ALLOC;
    source.region_name = prefix + "_region";
    source.label = prefix + "_label";
    source.size_bytes = 0x12345;
    source.alignment_bytes = 64;
    source.lifetime = sram::AllocationLifetime::kLayer;
    source.spillable = true;
    return source;
}

void TestSramLifecycle(TestState &state) {
    Sram_lifecycle source = MakeSramLifecycleAlloc();
    CheckRoundTrip(state, "Sram_lifecycle ALLOC", source);
    const Wire wire = source.serialize();
    CheckNativeMultiFraming<Sram_lifecycle>(
        state, "Sram_lifecycle", wire);

    const uint8_t id = PrimIdValue(PrimId::SRAM_LIFECYCLE);
    bool all_ids = wire.size() == 4;
    for (const auto &segment : wire)
        all_ids = all_ids && segment.range(7, 0).to_uint() == id;
    state.Check(all_ids,
                "Sram_lifecycle has four segments all carrying PrimId 53");
    state.Check(wire[0].range(10, 8).to_uint() ==
                        static_cast<uint8_t>(SramLifecycleOp::ALLOC) &&
                    wire[0].range(12, 11).to_uint() ==
                        static_cast<uint8_t>(
                            sram::AllocationLifetime::kLayer) &&
                    wire[0].range(13, 13).to_uint() == 1 &&
                    !wire[0].range(127, 14).or_reduce() &&
                    wire[2].range(71, 8).to_uint64() == source.size_bytes &&
                    wire[3].range(71, 8).to_uint64() ==
                        source.alignment_bytes,
                "Sram_lifecycle ALLOC wire golden fields");

    Sram_lifecycle alloc_at = source;
    alloc_at.op = SramLifecycleOp::ALLOC_AT;
    alloc_at.region_offset_bytes = 0x23456;
    CheckRoundTrip(state, "Sram_lifecycle ALLOC_AT", alloc_at);
    const Wire alloc_at_wire = alloc_at.serialize();
    state.Check(alloc_at_wire.size() == 5 &&
                    alloc_at_wire[0].range(10, 8).to_uint() ==
                        static_cast<uint8_t>(SramLifecycleOp::ALLOC_AT) &&
                    alloc_at_wire[4].range(71, 8).to_uint64() ==
                        alloc_at.region_offset_bytes &&
                    !alloc_at_wire[4].range(127, 72).or_reduce(),
                "Sram_lifecycle ALLOC_AT has a strict fifth offset segment");

    Sram_lifecycle free_prim;
    free_prim.op = SramLifecycleOp::FREE;
    free_prim.label = source.label;
    CheckRoundTrip(state, "Sram_lifecycle FREE", free_prim);
    Sram_lifecycle resize_prim = free_prim;
    resize_prim.op = SramLifecycleOp::RESIZE;
    resize_prim.size_bytes = 777;
    CheckRoundTrip(state, "Sram_lifecycle RESIZE", resize_prim);
    Sram_lifecycle rename_prim = free_prim;
    rename_prim.op = SramLifecycleOp::RENAME;
    rename_prim.new_label = source.label + "_renamed";
    CheckRoundTrip(state, "Sram_lifecycle RENAME", rename_prim);
    Sram_lifecycle clear_prim = free_prim;
    clear_prim.op = SramLifecycleOp::CLEAR_TARGETED;
    CheckRoundTrip(state, "Sram_lifecycle CLEAR_TARGETED", clear_prim);

    const size_t table_before_invalid = g_addr_label_table.table.size();
    Sram_lifecycle invalid = source;
    invalid.alignment_bytes = 3;
    state.Throws("Sram_lifecycle non-power-of-two alignment",
                 [&] { invalid.serialize(); });
    invalid = source;
    invalid.new_label = "inactive";
    state.Throws("Sram_lifecycle ALLOC noncanonical new label",
                 [&] { invalid.serialize(); });
    invalid = source;
    invalid.region_offset_bytes = 64;
    state.Throws("Sram_lifecycle ALLOC preserves zero offset ABI",
                 [&] { invalid.serialize(); });
    invalid = free_prim;
    invalid.region_name = "inactive";
    state.Throws("Sram_lifecycle FREE noncanonical region",
                 [&] { invalid.serialize(); });
    invalid = rename_prim;
    invalid.new_label = invalid.label;
    state.Throws("Sram_lifecycle RENAME identical labels",
                 [&] { invalid.serialize(); });
    invalid = source;
    invalid.label.assign(256, static_cast<char>(0x61));
    state.Throws("Sram_lifecycle oversized label",
                 [&] { invalid.serialize(); });
    state.Check(g_addr_label_table.table.size() == table_before_invalid,
                "invalid Sram_lifecycle serialization is label-table atomic");

    state.Throws("Sram_lifecycle truncated wire", [&] {
        Wire bad = wire;
        bad.pop_back();
        Sram_lifecycle decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_lifecycle extra wire segment", [&] {
        Wire bad = wire;
        bad.push_back(wire.back());
        Sram_lifecycle decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_lifecycle ALLOC_AT truncated offset segment", [&] {
        Wire bad = alloc_at_wire;
        bad.pop_back();
        Sram_lifecycle decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_lifecycle ALLOC_AT offset padding", [&] {
        Wire bad = alloc_at_wire;
        bad[4].range(72, 72) = sc_bv<1>(1);
        Sram_lifecycle decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_lifecycle inconsistent segment ID", [&] {
        Wire bad = wire;
        bad[2].range(7, 0) = sc_bv<8>(PrimIdValue(PrimId::SET_ADDR));
        Sram_lifecycle decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_lifecycle unknown operation", [&] {
        Wire bad = wire;
        bad[0].range(10, 8) = sc_bv<3>(7);
        Sram_lifecycle decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_lifecycle unknown lifetime", [&] {
        Wire bad = wire;
        bad[0].range(12, 11) = sc_bv<2>(3);
        Sram_lifecycle decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_lifecycle metadata reserved bit", [&] {
        Wire bad = wire;
        bad[0].range(14, 14) = sc_bv<1>(1);
        Sram_lifecycle decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_lifecycle label reserved bit", [&] {
        Wire bad = wire;
        bad[1].range(104, 104) = sc_bv<1>(1);
        Sram_lifecycle decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_lifecycle numeric reserved bit", [&] {
        Wire bad = wire;
        bad[3].range(72, 72) = sc_bv<1>(1);
        Sram_lifecycle decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_lifecycle unknown label ID", [&] {
        Wire bad = wire;
        bad[1].range(71, 40) = sc_bv<32>(
            g_addr_label_table.table.size() + 1);
        Sram_lifecycle decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_lifecycle decoded missing region", [&] {
        Wire bad = wire;
        bad[1].range(39, 8) = 0;
        Sram_lifecycle decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_lifecycle decoded zero alignment", [&] {
        Wire bad = wire;
        bad[3].range(71, 8) = 0;
        Sram_lifecycle decoded;
        decoded.deserialize(std::move(bad));
    });

    {
        LegacyModeGuard guard(true);
        state.Throws("Sram_lifecycle legacy serialize",
                     [&] { source.serialize(); });
        state.Throws("Sram_lifecycle legacy deserialize", [&] {
            Sram_lifecycle decoded;
            decoded.deserialize(wire);
        });
    }
    state.Throws("Sram_lifecycle legacy conversion", [&] {
        (void)prim_wire::LegacyTransportSegments(wire, source.name);
    });
}

Sram_bind_oneshot MakeSramBindOneShot() {
    Sram_bind_oneshot source;
    source.input_count = 4;
    const std::string prefix = "__prim_wire_sram_bind_" +
                               std::to_string(g_addr_label_table.table.size());
    for (uint32_t index = 0; index < source.input_count; ++index)
        source.datapass_label.indata[index] =
            prefix + "_in_" + std::to_string(index);
    source.datapass_label.outdata = prefix + "_out";
    return source;
}

void TestSramBindOneShot(TestState &state) {
    Sram_bind_oneshot source = MakeSramBindOneShot();
    CheckRoundTrip(state, "Sram_bind_oneshot", source);
    const Wire wire = source.serialize();
    CheckNativeMultiFraming<Sram_bind_oneshot>(
        state, "Sram_bind_oneshot", wire);

    const uint8_t id = PrimIdValue(PrimId::SRAM_BIND_ONESHOT);
    state.Check(wire.size() == 8,
                "Sram_bind_oneshot has fixed eight-segment wire");
    bool all_ids = true;
    for (const auto &segment : wire)
        all_ids = all_ids && segment.range(7, 0).to_uint() == id;
    state.Check(all_ids,
                "Sram_bind_oneshot writes PrimId into every segment");
    state.Check(wire[0].range(15, 8).to_uint() == source.input_count &&
                    !wire[0].range(127, 16).or_reduce(),
                "Sram_bind_oneshot metadata golden");
    bool unused_zero = true;
    for (size_t index = source.input_count; index < 18; ++index) {
        const size_t segment = 1 + index / 3;
        const int low = static_cast<int>(8 + (index % 3) * 32);
        unused_zero = unused_zero &&
                      wire[segment].range(low + 31, low).to_uint64() == 0;
    }
    state.Check(unused_zero,
                "Sram_bind_oneshot unused and physical label slots are zero");

    {
        LegacyModeGuard guard(true);
        const Wire strict_in_legacy_mode = source.serialize();
        Sram_bind_oneshot decoded;
        decoded.deserialize(strict_in_legacy_mode);
        state.Check(SameWire(strict_in_legacy_mode, decoded.serialize()),
                    "Sram_bind_oneshot remains strict in legacy mode");
    }
    state.Throws("Sram_bind_oneshot rejects legacy transport", [&] {
        (void)prim_wire::LegacyTransportSegments(wire, source.name);
    });

    const size_t table_before_invalid = g_addr_label_table.table.size();
    Sram_bind_oneshot invalid_count = source;
    invalid_count.input_count = 0;
    state.Throws("Sram_bind_oneshot zero input count",
                 [&] { invalid_count.serialize(); });
    invalid_count.input_count = MAX_SPLIT_NUM + 1;
    state.Throws("Sram_bind_oneshot oversized input count",
                 [&] { invalid_count.serialize(); });
    Sram_bind_oneshot unset_input = source;
    unset_input.datapass_label.indata[1] = UNSET_LABEL;
    state.Throws("Sram_bind_oneshot unset used input",
                 [&] { unset_input.serialize(); });
    Sram_bind_oneshot set_unused = source;
    set_unused.datapass_label.indata[source.input_count] = "unexpected";
    state.Throws("Sram_bind_oneshot set unused input",
                 [&] { set_unused.serialize(); });
    Sram_bind_oneshot unset_output = source;
    unset_output.datapass_label.outdata = UNSET_LABEL;
    state.Throws("Sram_bind_oneshot unset output",
                 [&] { unset_output.serialize(); });
    state.Check(g_addr_label_table.table.size() == table_before_invalid,
                "Sram_bind_oneshot invalid serialization is table-atomic");

    state.Throws("Sram_bind_oneshot zero decoded input count", [&] {
        Wire bad = wire;
        bad[0].range(15, 8) = 0;
        Sram_bind_oneshot decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_bind_oneshot oversized decoded input count", [&] {
        Wire bad = wire;
        bad[0].range(15, 8) = MAX_SPLIT_NUM + 1;
        Sram_bind_oneshot decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_bind_oneshot metadata padding", [&] {
        Wire bad = wire;
        bad[0][16] = sc_dt::SC_LOGIC_1;
        Sram_bind_oneshot decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_bind_oneshot input padding", [&] {
        Wire bad = wire;
        bad[1][104] = sc_dt::SC_LOGIC_1;
        Sram_bind_oneshot decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_bind_oneshot unused input slot", [&] {
        Wire bad = wire;
        bad[2].range(71, 40) = 1;
        Sram_bind_oneshot decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_bind_oneshot physical input slot", [&] {
        Wire bad = wire;
        bad[6].range(71, 40) = 1;
        Sram_bind_oneshot decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_bind_oneshot unknown input label", [&] {
        Wire bad = wire;
        bad[1].range(39, 8) = g_addr_label_table.table.size() + 1;
        Sram_bind_oneshot decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_bind_oneshot output padding", [&] {
        Wire bad = wire;
        bad.back()[40] = sc_dt::SC_LOGIC_1;
        Sram_bind_oneshot decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_bind_oneshot missing output label", [&] {
        Wire bad = wire;
        bad.back().range(39, 8) = 0;
        Sram_bind_oneshot decoded;
        decoded.deserialize(std::move(bad));
    });
}


void TestSramBindOneShotLifecycle(TestState &state) {
#if USE_NB_DRAMSYS == 1
    TaskCoreContext task_context(nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, nullptr, nullptr, nullptr, 0, 0);
#else
    TaskCoreContext task_context(nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, nullptr, nullptr, nullptr, 0, 0, 0);
#endif
    task_context.cid = 0;

    PrimWireCostHardwareGuard cost_hardware;
    auto core = std::make_shared<PrimCoreContext>();
    core->cid = 0;
    core->loop_cnt = 0;
    core->auto_pd_ = 0;
    core->datapass_label_->indata[0] = "_persistent_in_0";
    core->datapass_label_->indata[1] = "_persistent_in_1";
    core->datapass_label_->outdata = "persistent_out";
    const AddrDatapassLabel persistent = *core->datapass_label_;

    auto make_bind = [&](const std::string &prefix) {
        Sram_bind_oneshot bind;
        bind.input_count = 2;
        bind.datapass_label.indata[0] = "_" + prefix + "_in_0";
        bind.datapass_label.indata[1] = "_" + prefix + "_in_1";
        bind.datapass_label.outdata = prefix + "_out";
        bind.prim_context = core;
        return bind;
    };

    Sram_bind_oneshot bind = make_bind("run");
    state.Check(bind.taskCoreDefault(task_context) == 0 &&
                    core->program_mode_ && core->sram_bind_pending_ &&
                    core->sram_bind_input_count_ == 2,
                "Sram_bind_oneshot installs independent pending state");
    Sram_bind_oneshot duplicate = make_bind("duplicate");
    state.Throws("Sram_bind_oneshot rejects double bind", [&] {
        duplicate.taskCoreDefault(task_context);
    });
    state.Check(core->sram_bind_pending_labels_.outdata == "run_out",
                "Sram_bind_oneshot double bind preserves first binding");

    for (int category : {MEM_PRIM, SYNC_PRIM, COMM_PRIM}) {
        PassThroughPrim intervening(category);
        intervening.prim_context = core;
        intervening.taskCoreDefault(task_context);
    }
    state.Check(core->sram_bind_pending_ &&
                    core->sram_bind_pending_labels_.outdata == "run_out",
                "one-shot binding crosses MEM/SYNC/COMM Prims");

    BindingProbeNpu compute(2);
    compute.prim_context = core;
    state.Check(compute.taskCoreDefault(task_context) == 0 &&
                    compute.calls == 1 &&
                    compute.seen_inputs ==
                        std::vector<std::string>({"run_in_0", "run_in_1"}) &&
                    compute.seen_output == "run_out" &&
                    !compute.saw_pending_during_task,
                "next NPU compute consumes and observes one-shot labels");
    state.Check(!core->sram_bind_pending_ &&
                    core->sram_bind_input_count_ == 0 &&
                    core->sram_bind_pending_labels_.indata[0] == UNSET_LABEL &&
                    core->sram_bind_pending_labels_.outdata == UNSET_LABEL,
                "successful NPU start immediately clears pending binding");
    state.Check(core->datapass_label_->indata[0] ==
                        persistent.indata[0] &&
                    core->datapass_label_->indata[1] ==
                        persistent.indata[1] &&
                    core->datapass_label_->outdata == persistent.outdata,
                "NPU completion restores legacy persistent labels");

    const size_t calls_after_first = compute.calls;
    state.Throws("program NPU compute without a new bind", [&] {
        compute.taskCoreDefault(task_context);
    });
    state.Check(compute.calls == calls_after_first,
                "missing bind fails before NPU task starts");

    Sram_bind_oneshot throwing_bind = make_bind("throw");
    throwing_bind.taskCoreDefault(task_context);
    compute.throw_from_task = true;
    state.Throws("throwing NPU compute consumes one-shot binding", [&] {
        compute.taskCoreDefault(task_context);
    });
    state.Check(!core->sram_bind_pending_ &&
                    core->datapass_label_->indata[0] ==
                        persistent.indata[0] &&
                    core->datapass_label_->outdata == persistent.outdata,
                "NPU exception consumes binding and restores persistent labels");
    compute.throw_from_task = false;

    Sram_bind_oneshot mismatch_bind = make_bind("mismatch");
    mismatch_bind.taskCoreDefault(task_context);
    BindingProbeNpu wrong_arity(1);
    wrong_arity.prim_context = core;
    state.Throws("NPU input arity mismatches pending SRAM_BIND", [&] {
        wrong_arity.taskCoreDefault(task_context);
    });
    state.Check(core->sram_bind_pending_ && wrong_arity.calls == 0,
                "rejected NPU start does not consume pending binding");
    core->sram_bind_pending_ = false;
    core->sram_bind_input_count_ = 0;
    core->sram_bind_pending_labels_ = AddrDatapassLabel();
}

Set_addr MakeSetAddr() {
    Set_addr source;
    source.sram_addr = 0xffffff;
    source.datatype = FP16;
    source.prim_context = std::make_shared<PrimCoreContext>();
    const std::string prefix = "__prim_wire_set_addr_" +
                               std::to_string(g_addr_label_table.table.size());
    for (int i = 0; i < MAX_SPLIT_NUM; ++i)
        source.prim_context->datapass_label_->indata[i] =
            prefix + "_in_" + std::to_string(i);
    source.prim_context->datapass_label_->outdata = prefix + "_out";
    return source;
}

void TestSetAddr(TestState &state) {
    Set_addr source = MakeSetAddr();
    CheckRoundTrip(state, "Set_addr", source);
    const Wire wire = source.serialize();
    CheckNativeMultiFraming<Set_addr>(state, "Set_addr", wire);
    state.Throws("Set_addr metadata reserved bit", [&] {
        Wire bad = wire;
        bad[0][34] = sc_dt::SC_LOGIC_1;
        Set_addr decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Set_addr invalid datatype", [&] {
        Wire bad = wire;
        bad[0].range(33, 32) = 2;
        Set_addr decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Set_addr input padding", [&] {
        Wire bad = wire;
        bad[6][104] = sc_dt::SC_LOGIC_1;
        Set_addr decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Set_addr output padding", [&] {
        Wire bad = wire;
        bad.back()[40] = sc_dt::SC_LOGIC_1;
        Set_addr decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Set_addr unknown label ID", [&] {
        Wire bad = wire;
        bad[1].range(39, 8) = g_addr_label_table.table.size() + 1;
        Set_addr decoded;
        decoded.deserialize(std::move(bad));
    });
    Set_addr overflow = MakeSetAddr();
    overflow.sram_addr = 0x1000000;
    state.Throws("Set_addr address maximum plus one",
                 [&] { overflow.serialize(); });

    const Wire legacy = prim_wire::LegacyTransportSegments(wire, source.name);
    Wire expected_legacy(6);
    for (auto &segment : expected_legacy)
        segment = 0;
    expected_legacy[0] = wire[0];
    for (size_t index = 0; index < MAX_SPLIT_NUM; ++index) {
        const size_t segment = 1 + index / 4;
        const int low = static_cast<int>((index % 4) * 32);
        const uint32_t label_id = static_cast<uint32_t>(
            g_addr_label_table.addRecord(
                source.prim_context->datapass_label_->indata[index]));
        expected_legacy[segment].range(low + 31, low) =
            sc_bv<32>(label_id);
    }
    expected_legacy.back().range(31, 0) = sc_bv<32>(
        g_addr_label_table.addRecord(
            source.prim_context->datapass_label_->outdata));
    state.Check(legacy.size() == 6 && SameWire(legacy, expected_legacy),
                "Set_addr legacy six-segment golden");
    Set_addr decoded;
    {
        LegacyModeGuard guard(true);
        decoded.deserialize(legacy);
    }
    state.Check(SameWire(decoded.serialize(), wire),
                "Set_addr legacy decode preserves strict semantics");
}

void TestSetBatch(TestState &state) {
    Set_batch source({Stage(0xff, PD_DONE, 0xfff), Stage(7, DECODE, 3),
                      Stage(9, PREFILL, 11)},
                     0xffff);
    CheckRoundTrip(state, "Set_batch", source);
    const Wire wire = source.serialize();
    CheckNativeMultiFraming<Set_batch>(state, "Set_batch", wire);
    state.Throws("Set_batch metadata reserved bit", [&] {
        Wire bad = wire;
        bad[0][40] = sc_dt::SC_LOGIC_1;
        Set_batch decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Set_batch invalid phase", [&] {
        Wire bad = wire;
        bad[1].range(17, 16) = 3;
        bad[1].range(39, 38) = 3;
        bad[1].range(61, 60) = 3;
        // PD_DONE is valid (3); force a synthetic fourth enum is impossible
        // in two bits, so semantic invalidity is exercised at encode below.
        Set_batch decoded;
        decoded.deserialize(std::move(bad));
        decoded.batch_info[0].type = static_cast<PD_PHASE>(4);
        decoded.serialize();
    });
    state.Throws("Set_batch trailing stage padding", [&] {
        Wire bad = wire;
        bad.back()[74] = sc_dt::SC_LOGIC_1;
        Set_batch decoded;
        decoded.deserialize(std::move(bad));
    });
    Set_batch overflow = source;
    overflow.auto_pd = 0x10000;
    state.Throws("Set_batch auto_pd maximum plus one",
                 [&] { overflow.serialize(); });
    overflow = source;
    overflow.batch_info[0].req_id = 0x100;
    state.Throws("Set_batch req_id maximum plus one",
                 [&] { overflow.serialize(); });
    overflow = source;
    overflow.batch_info[0].token_num = 0x1000;
    state.Throws("Set_batch token maximum plus one",
                 [&] { overflow.serialize(); });
    Set_batch reused({Stage(1, PREFILL, 1), Stage(2, DECODE, 2)});
    reused.deserialize(wire);
    state.Check(reused.batch_info.size() == source.batch_info.size(),
                "Set_batch repeated deserialize clears old stages");

    const Wire legacy = prim_wire::LegacyTransportSegments(wire, source.name);
    Wire expected_legacy(wire.size());
    for (auto &segment : expected_legacy)
        segment = 0;
    expected_legacy[0] = wire[0];
    for (size_t index = 0; index < source.batch_info.size(); ++index) {
        const size_t segment = 1 + index / 5;
        const int low = static_cast<int>((index % 5) * 22);
        expected_legacy[segment].range(low + 7, low) =
            sc_bv<8>(source.batch_info[index].req_id);
        expected_legacy[segment].range(low + 9, low + 8) =
            sc_bv<2>(source.batch_info[index].type);
        expected_legacy[segment].range(low + 21, low + 10) =
            sc_bv<12>(source.batch_info[index].token_num);
    }
    state.Check(SameWire(legacy, expected_legacy),
                "Set_batch legacy payload golden");
    Set_batch decoded;
    {
        LegacyModeGuard guard(true);
        decoded.deserialize(legacy);
    }
    state.Check(decoded.auto_pd == source.auto_pd &&
                    decoded.batch_info.size() == source.batch_info.size() &&
                    SameWire(decoded.serialize(), wire),
                "Set_batch legacy decode preserves strict semantics");
}

void TestDteAsync(TestState &state) {
    Dte_async_prim source;
    source.op = DteAsyncOp::ISSUE;
    source.token = std::numeric_limits<uint32_t>::max();
    source.payload_bits = 4096;
    source.direction = DteDir::DRAM_TO_SPM;
    source.spm_size = 512;
    source.sram_region = "wire_region";
    source.sram_offset = 17;
    source.remote_addr = 0x123456789abcdef0ULL;
    CheckRoundTrip(state, "Dte_async", source);
    const Wire wire = source.serialize();
    CheckWrappedFraming<Dte_async_prim>(state, "Dte_async", wire);
    CheckLegacyWrappedTransport<Dte_async_prim>(state, "Dte_async", wire);
    CheckWrappedBitRejected<Dte_async_prim>(
        state, "Dte_async metadata reserved bit", wire, 0, 110);
    CheckWrappedBitRejected<Dte_async_prim>(
        state, "Dte_async region reserved bit", wire, 3, 80);
    state.Throws("Dte_async invalid op", [&] {
        Wire bad = wire;
        bad[0].range(10, 8) = 7;
        Dte_async_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Dte_async invalid direction", [&] {
        Wire bad = wire;
        bad[0].range(13, 11) = 7;
        Dte_async_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Dte_async string tail padding", [&] {
        Wire bad = wire;
        bad[4][8 * 11] = sc_dt::SC_LOGIC_1;
        Dte_async_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    Dte_async_prim invalid = source;
    invalid.spm_addr = 1;
    state.Throws("Dte_async absolute/region XOR", [&] { invalid.serialize(); });
    invalid = source;
    invalid.sram_region.assign(65, 'r');
    state.Throws("Dte_async region length maximum plus one",
                 [&] { invalid.serialize(); });
    Dte_async_prim absolute;
    absolute.op = DteAsyncOp::ISSUE;
    absolute.token = 9;
    absolute.payload_bits = 64;
    absolute.direction = DteDir::SPM_TO_SPM;
    absolute.spm_addr = 0x800;
    absolute.spm_size = 8;
    const Wire absolute_wire = absolute.serialize();
    Dte_async_prim reused;
    reused.deserialize(wire);
    reused.deserialize(absolute_wire);
    state.Check(reused.sram_region.empty() && reused.sram_offset == 0 &&
                    SameWire(reused.serialize(), absolute_wire),
                "Dte_async repeated deserialize clears old region state");
}

void TestLsu(TestState &state) {
    Lsu_mem_prim source;
    source.op = LsuMemOp::ISSUE;
    source.token = std::numeric_limits<uint64_t>::max();
    source.direction = sram::LsuDirection::kHbmToSram;
    source.hbm_addr = 0xffffffffffffffffULL;
    source.sram_offset = 7;
    source.size_bytes = 4096;
    source.sram_region = "lsu_region";
    source.absolute_sram = false;
    CheckRoundTrip(state, "Lsu_mem", source);
    const Wire wire = source.serialize();
    CheckWrappedFraming<Lsu_mem_prim>(state, "Lsu_mem", wire);
    CheckLegacyWrappedTransport<Lsu_mem_prim>(state, "Lsu_mem", wire);
    CheckWrappedBitRejected<Lsu_mem_prim>(
        state, "Lsu_mem metadata reserved bit", wire, 0, 77);
    CheckWrappedBitRejected<Lsu_mem_prim>(
        state, "Lsu_mem details reserved bit", wire, 2, 80);
    state.Throws("Lsu_mem invalid op", [&] {
        Wire bad = wire;
        bad[0].range(10, 8) = 7;
        Lsu_mem_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Lsu_mem string tail padding", [&] {
        Wire bad = wire;
        bad[3][8 * 10] = sc_dt::SC_LOGIC_1;
        Lsu_mem_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    Lsu_mem_prim invalid = source;
    invalid.absolute_sram = true;
    invalid.sram_addr = 5;
    state.Throws("Lsu_mem absolute/region XOR", [&] { invalid.serialize(); });
    invalid = source;
    invalid.sram_region.assign(65, 'r');
    state.Throws("Lsu_mem region length maximum plus one",
                 [&] { invalid.serialize(); });
    Lsu_mem_prim absolute;
    absolute.op = LsuMemOp::LOAD_BLOCKING;
    absolute.direction = sram::LsuDirection::kHbmToSram;
    absolute.hbm_addr = 1;
    absolute.sram_addr = 2;
    absolute.size_bytes = 3;
    absolute.absolute_sram = true;
    const Wire absolute_wire = absolute.serialize();
    Lsu_mem_prim reused;
    reused.deserialize(wire);
    reused.deserialize(absolute_wire);
    state.Check(reused.sram_region.empty() && reused.sram_offset == 0 &&
                    SameWire(reused.serialize(), absolute_wire),
                "Lsu_mem repeated deserialize clears old region state");
}

void TestSramPipeline(TestState &state) {
    Sram_pipeline_prim source;
    source.engine = SramPipelineEngine::kDte;
    source.double_buffer = true;
    source.tile_count = 4;
    source.tile_bytes = 128;
    source.compute_cycles = std::numeric_limits<uint64_t>::max();
    source.token_base = std::numeric_limits<uint32_t>::max();
    source.region_a = "pipe_a";
    source.region_b = "pipe_b";
    CheckRoundTrip(state, "Sram_pipeline", source);
    const Wire wire = source.serialize();
    CheckWrappedFraming<Sram_pipeline_prim>(state, "Sram_pipeline", wire);
    CheckLegacyWrappedTransport<Sram_pipeline_prim>(state, "Sram_pipeline", wire);
    CheckWrappedBitRejected<Sram_pipeline_prim>(
        state, "Sram_pipeline metadata reserved bit", wire, 0, 114);
    CheckWrappedBitRejected<Sram_pipeline_prim>(
        state, "Sram_pipeline size reserved bit", wire, 1, 110);
    state.Throws("Sram_pipeline string A tail padding", [&] {
        Wire bad = wire;
        bad[3][8 * 6] = sc_dt::SC_LOGIC_1;
        Sram_pipeline_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Sram_pipeline string B tail padding", [&] {
        Wire bad = wire;
        bad[4][8 * 6] = sc_dt::SC_LOGIC_1;
        Sram_pipeline_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    Sram_pipeline_prim invalid = source;
    invalid.engine = static_cast<SramPipelineEngine>(2);
    state.Throws("Sram_pipeline invalid engine", [&] { invalid.serialize(); });
    invalid = source;
    invalid.region_a.assign(65, 'a');
    state.Throws("Sram_pipeline region length maximum plus one",
                 [&] { invalid.serialize(); });
    invalid = source;
    invalid.tile_count = std::numeric_limits<uint32_t>::max();
    invalid.tile_bytes = 1;
    invalid.input_hbm_base = 0;
    invalid.output_hbm_base = 0;
    state.Check(!invalid.serialize().empty(),
                "Sram_pipeline tile_count field maximum");
    Sram_pipeline_prim reused;
    reused.region_a = "stale_a";
    reused.region_b = "stale_b";
    reused.deserialize(wire);
    state.Check(reused.region_a == source.region_a &&
                    reused.region_b == source.region_b,
                "Sram_pipeline repeated deserialize replaces strings");
}

void TestCollectiveData(TestState &state) {
    Collective_data_prim source;
    source.descriptor = MakeDescriptor(false);
    source.tree_id = std::numeric_limits<uint16_t>::max();
    source.mode = Collective_data_prim::Mode::CORE_VECTOR_START;
    source.core_vector_beats = std::numeric_limits<uint32_t>::max();
    CheckRoundTrip(state, "Collective_data", source);
    const Wire wire = source.serialize();
    CheckWrappedFraming<Collective_data_prim>(state, "Collective_data", wire);
    CheckLegacyWrappedTransport<Collective_data_prim>(state, "Collective_data", wire);
    CheckWrappedBitRejected<Collective_data_prim>(
        state, "Collective_data marker reserved bit", wire, 0, 64);
    state.Throws("Collective_data invalid mode", [&] {
        Wire bad = wire;
        bad[0].range(31, 24) = 0xff;
        Collective_data_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    CheckCollectiveDescriptorWire<Collective_data_prim>(
        state, "Collective_data", wire);
    Collective_data_prim invalid = source;
    invalid.tree_id = 0;
    state.Throws("Collective_data tree ID minimum", [&] { invalid.serialize(); });
    invalid = source;
    invalid.mode = Collective_data_prim::Mode::BROADCAST_RX;
    state.Throws("Collective_data mode/count mismatch",
                 [&] { invalid.serialize(); });
}

void TestCollective(TestState &state) {
    Collective_prim source;
    source.descriptor = MakeDescriptor(false);
    source.phase_id = std::numeric_limits<uint16_t>::max();
    source.marker_kind = Collective_prim::MarkerKind::BARRIER;
    source.release_tree_id = std::numeric_limits<uint16_t>::max();
    CheckRoundTrip(state, "Collective", source);
    const Wire wire = source.serialize();
    CheckWrappedFraming<Collective_prim>(state, "Collective", wire);
    CheckLegacyWrappedTransport<Collective_prim>(state, "Collective", wire);
    CheckWrappedBitRejected<Collective_prim>(
        state, "Collective marker reserved bit", wire, 0, 64);
    state.Throws("Collective invalid marker kind", [&] {
        Wire bad = wire;
        bad[0].range(47, 40) = 0xff;
        Collective_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Collective marker group mismatch", [&] {
        Wire bad = wire;
        bad[0].range(39, 24) = 4;
        Collective_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    CheckCollectiveDescriptorWire<Collective_prim>(state, "Collective", wire);
    Collective_prim invalid = source;
    invalid.marker_kind = Collective_prim::MarkerKind::GATHER_ARRIVAL;
    state.Throws("Collective non-barrier tree release",
                 [&] { invalid.deserialize(invalid.serialize()); });
}

void TestReduceCompute(TestState &state) {
    Reduce_compute_prim source;
    source.descriptor = MakeDescriptor(true);
    CheckRoundTrip(state, "Reduce_compute", source);
    const Wire wire = source.serialize();
    CheckWrappedFraming<Reduce_compute_prim>(state, "Reduce_compute", wire);
    CheckLegacyWrappedTransport<Reduce_compute_prim>(state, "Reduce_compute", wire);
    CheckWrappedBitRejected<Reduce_compute_prim>(
        state, "Reduce_compute marker reserved bit", wire, 0, 24);
    state.Throws("Reduce_compute group mismatch", [&] {
        Wire bad = wire;
        bad[0].range(23, 8) = 4;
        Reduce_compute_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    CheckCollectiveDescriptorWire<Reduce_compute_prim>(
        state, "Reduce_compute", wire);
    Reduce_compute_prim invalid = source;
    invalid.descriptor.dtype = CollDType::FP16;
    invalid.descriptor.chunk_bits = 128;
    state.Throws("Reduce_compute unsupported dtype",
                 [&] { invalid.deserialize(invalid.serialize()); });
}

template <typename Prim>
void TestLoadStore(TestState &state, const std::string &name) {
    Prim source;
    source.dram_addr = 0xffff;
    source.sram_addr = 0xffff;
    source.size = 0xffff;
    source.datatype = FP16;
    CheckRoundTrip(state, name, source);
    const Wire wire = source.serialize();
    CheckSingleFraming<Prim>(state, name, wire, 58);
    Prim invalid = source;
    invalid.dram_addr = 0x10000;
    state.Throws(name + " address maximum plus one",
                 [&] { invalid.serialize(); });
    invalid = source;
    invalid.size = 0x10000;
    state.Throws(name + " size maximum plus one", [&] { invalid.serialize(); });
    invalid = source;
    invalid.datatype = static_cast<DATATYPE>(2);
    state.Throws(name + " invalid datatype encode", [&] { invalid.serialize(); });
    state.Throws(name + " invalid datatype decode", [&] {
        Wire bad = wire;
        bad[0].range(57, 56) = 2;
        Prim decoded;
        decoded.deserialize(std::move(bad));
    });
}

void TestClearLoadStore(TestState &state) {
    Clear_sram clear;
    CheckRoundTrip(state, "Clear_sram", clear);
    CheckSingleFraming<Clear_sram>(state, "Clear_sram", clear.serialize(), 8);
    TestLoadStore<Load_prim>(state, "Load_prim");
    TestLoadStore<Store_prim>(state, "Store_prim");
}

void TestRecv(TestState &state) {
    Recv_prim source;
    source.type = RECV_START;
    source.tag_id = 0xffff;
    source.recv_cnt = 0xff;
    source.datatype = FP16;
    source.stripe_count = 4;
    CheckRoundTrip(state, "Recv_prim", source);
    const Wire wire = source.serialize();
    CheckSingleFraming<Recv_prim>(state, "Recv_prim", wire, 41);
    Recv_prim invalid = source;
    invalid.tag_id = 0x10000;
    state.Throws("Recv_prim tag maximum plus one",
                 [&] { invalid.serialize(); });
    invalid = source;
    invalid.recv_cnt = 0x100;
    state.Throws("Recv_prim count maximum plus one",
                 [&] { invalid.serialize(); });
    invalid = source;
    invalid.type = static_cast<RECV_TYPE>(7);
    state.Throws("Recv_prim invalid enum encode", [&] { invalid.serialize(); });
    state.Throws("Recv_prim invalid enum decode", [&] {
        Wire bad = wire;
        bad[0].range(11, 8) = 7;
        Recv_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Recv_prim invalid stripe", [&] {
        Wire bad = wire;
        bad[0].range(40, 38) = 3;
        Recv_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    Wire legacy = wire;
    legacy[0].range(40, 38) = 0;
    Recv_prim decoded;
    decoded.deserialize(legacy);
    state.Check(decoded.stripe_count == 1,
                "Recv_prim legacy zero stripe decodes as one");
}

void TestSend(TestState &state) {
    Send_prim source;
    source.type = SEND_DATA;
    source.des_id = 0xffff;
    source.output_label = "__prim_wire_send_" +
                          std::to_string(g_addr_label_table.table.size());
    source.packet_scale = 0xff;
    source.packets_in_last_group = 0xff;
    source.max_packet = std::numeric_limits<int>::max();
    source.tag_id = 0xfffff;
    source.end_length = 0xff;
    source.datatype = FP16;
    source.stripe_count = 4;
    CheckRoundTrip(state, "Send_prim", source);
    const Wire wire = source.serialize();
    CheckSingleFraming<Send_prim>(state, "Send_prim", wire, 52);
    Send_prim invalid = source;
    invalid.des_id = 0x10000;
    state.Throws("Send_prim destination maximum plus one",
                 [&] { invalid.serialize(); });
    invalid = source;
    invalid.packet_scale = 0x100;
    state.Throws("Send_prim packet scale maximum plus one",
                 [&] { invalid.serialize(); });
    invalid = source;
    invalid.tag_id = 0x100000;
    state.Throws("Send_prim tag maximum plus one",
                 [&] { invalid.serialize(); });
    invalid = source;
    invalid.end_length = 0x100;
    state.Throws("Send_prim end length maximum plus one",
                 [&] { invalid.serialize(); });
    invalid = source;
    invalid.type = static_cast<SEND_TYPE>(5);
    state.Throws("Send_prim invalid enum encode", [&] { invalid.serialize(); });
    state.Throws("Send_prim invalid enum decode", [&] {
        Wire bad = wire;
        bad[0].range(59, 56) = 5;
        Send_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Send_prim invalid stripe", [&] {
        Wire bad = wire;
        bad[0].range(124, 122) = 3;
        Send_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Send_prim unknown DATA label", [&] {
        Wire bad = wire;
        bad[0].range(35, 24) = g_addr_label_table.table.size() + 1;
        Send_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    state.Throws("Send_prim half-zero packet metadata", [&] {
        Wire bad = wire;
        bad[0].range(43, 36) = 0;
        Send_prim decoded;
        decoded.deserialize(std::move(bad));
    });
    Wire legacy = wire;
    legacy[0].range(43, 36) = 0;
    legacy[0].range(51, 44) = 0;
    legacy[0].range(124, 122) = 0;
    Send_prim decoded;
    decoded.deserialize(legacy);
    state.Check(decoded.packet_scale == 1 &&
                    decoded.packets_in_last_group == 1 &&
                    decoded.stripe_count == 1,
                "Send_prim legacy zero packet/stripe fields decode as one");
    Send_prim sentinel = source;
    sentinel.des_id = -1;
    state.Check(sentinel.serialize()[0].range(23, 8).to_uint() == 0xffff,
                "Send_prim -1 destination uses legacy 0xffff sentinel");
    Send_prim control;
    control.type = SEND_DONE;
    control.des_id = -1;
    control.tag_id = 0;
    control.max_packet = 0;
    control.end_length = 0;
    control.output_label = UNSET_LABEL;
    const Wire control_wire = control.serialize();
    state.Throws("Send_prim control label bits", [&] {
        Wire bad = control_wire;
        bad[0].range(35, 24) = 1;
        Send_prim target;
        target.deserialize(std::move(bad));
    });
    state.Throws("Send_prim control packet bits", [&] {
        Wire bad = control_wire;
        bad[0].range(43, 36) = 1;
        Send_prim target;
        target.deserialize(std::move(bad));
    });
    state.Throws("Send_prim control payload size bits", [&] {
        Wire bad = control_wire;
        bad[0].range(91, 60) = 1;
        Send_prim target;
        target.deserialize(std::move(bad));
    });
    state.Throws("Send_prim SEND_DONE tag", [&] {
        Wire bad = control_wire;
        bad[0].range(111, 92) = 1;
        Send_prim target;
        target.deserialize(std::move(bad));
    });
    Send_prim reused;
    reused.d2d_exit_port = 7;
    reused.d2d_exit_selected = true;
    reused.stripe_packets = {1, 2};
    reused.stripe_sent = {1};
    reused.stripe_exit_ports = {3};
    reused.next_subflow = 8;
    reused.stripe_saf_reserved = true;
    reused.deserialize(wire);
    state.Check(reused.d2d_exit_port == -1 && !reused.d2d_exit_selected &&
                    reused.stripe_packets.empty() &&
                    reused.stripe_sent.empty() &&
                    reused.stripe_exit_ports.empty() &&
                    reused.next_subflow == 0 && !reused.stripe_saf_reserved,
                "Send_prim deserialize clears execution-only state");
}

} // namespace

int RunPrimWireSelfTest() {
    TestState state;
    TestNpuBase(state);
    TestGpuBase(state);
    TestDteEndpointWires(state);
    TestCollectiveDataV1Wire(state);
    TestCollectivePhaseBarrierV1Wire(state);
    state.Check(RunCollectiveLaunchV1PrimSelfTest() == 0,
                "Collective_launch_v1_prim strict wire harness passes");
    TestSynchronizationWires(state);
    TestSetAddr(state);
    TestSramBindOneShot(state);
    TestSramLifecycle(state);
    TestSramBindOneShotLifecycle(state);
    TestSetBatch(state);
    TestDteAsync(state);
    TestLsu(state);
    TestSramPipeline(state);
    TestCollectiveData(state);
    TestCollective(state);
    TestReduceCompute(state);
    TestClearLoadStore(state);
    TestRecv(state);
    TestSend(state);

    if (state.failures() == 0) {
        std::cout << "Prim wire codec self-test: PASS (" << state.checks()
                  << " checks)\n";
    } else {
        std::cerr << "Prim wire codec self-test: FAIL (" << state.failures()
                  << "/" << state.checks() << " checks failed)\n";
    }
    return state.failures();
}
