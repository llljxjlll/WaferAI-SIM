#include "prims/gemm_input_dx_npu_prim.h"
#include "memory/sram/sram_access_unit.h"
#include "memory/sram/sram_storage.h"
#include "utils/prim_utils.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

REGISTER_PRIM(gemm_input_dx_timing, PrimId::GEMM_INPUT_DX_TIMING);

namespace {
uint64_t Profile(const NpuBase &prim, const char *field) {
    const auto it = prim.param_value.find(field);
    if (it == prim.param_value.end() || it->second <= 0)
        throw std::invalid_argument(prim.name + " requires positive profile " + field);
    return static_cast<uint64_t>(it->second);
}

int HalfBytes(uint64_t bytes, const char *field) {
    if (bytes % 2 || bytes / 2 > std::numeric_limits<int>::max())
        throw std::invalid_argument(std::string(field) + " exceeds NpuBase data ABI");
    return static_cast<int>(bytes / 2);
}
} // namespace

gemm_input_dx_timing::gemm_input_dx_timing() {
    name = "gemm_input_dx_timing";
    datatype = FP16; // W, dY, and independent activation dX are FP16.
    skip_input = true;
    skip_output = true;
    param_name = {"M", "N", "K"};
}

GemmInputDxTimingWork gemm_input_dx_timing::work() const {
    if (datatype != FP16 || param_value.size() != 3)
        throw std::invalid_argument(name + " requires exact FP16 source profile");
    const auto k = Profile(*this, "K");
    const auto m = Profile(*this, "M");
    const auto n = Profile(*this, "N");
    if (inp_offset < 0 || data_offset < 0 || out_offset < 0)
        throw std::invalid_argument(name + " needs nonnegative SRAM offsets");
    const GemmInputDxTimingTile tile{
        k, m, n,
        {static_cast<uint32_t>(inp_offset), 2 * m * n, GemmInputDxDType::FP16},
        {static_cast<uint32_t>(data_offset), 2 * k * n, GemmInputDxDType::FP16},
        {static_cast<uint32_t>(out_offset), 2 * k * m, GemmInputDxDType::FP16},
    };
    return BuildGemmInputDxPhysicalWork(tile);
}

void gemm_input_dx_timing::initialize() {
    const auto profile = work();
    data_size_input = {
        HalfBytes(profile.tile.weight.bytes, "GEMM dX weight"),
        HalfBytes(profile.tile.upstream.bytes, "GEMM dX upstream"),
    };
    data_chunk = {{"upstream", HalfBytes(profile.tile.upstream.bytes,
                                          "GEMM dX upstream")},
                  {"output", HalfBytes(profile.tile.output.bytes,
                                        "GEMM dX FP16 output")}};
}

void gemm_input_dx_timing::taskCore(TaskCoreContext &context, string,
                                    u_int64_t &dram, u_int64_t &exu,
                                    u_int64_t &sfu, u_int64_t &vec) {
    const auto profile = work();
    dram = sfu = 0;
    exu = profile.exu_flops;
    vec = profile.fp16_output_vec_ops;
    // A timing kernel has no numerical derivative. Still, its full typed dX
    // span must become valid before a later DTE/LSU consumer reads it.
    if (context.sram_storage != nullptr &&
        context.sram_storage->payload_mode()) {
        if (context.sram_access == nullptr)
            throw std::logic_error("timing GEMM dX lacks program SRAM output");
        sram::Request write;
        write.initiator = sram::Initiator::kCompute;
        write.command = sram::Command::kWrite;
        write.address = profile.tile.output.byte_address;
        write.size_bytes = profile.tile.output.bytes;
        write.payload = std::vector<uint8_t>(
            static_cast<size_t>(profile.tile.output.bytes));
        context.sram_access->Access(write);
    }
}

vector<sc_bv<128>> gemm_input_dx_timing::serialize() {
    if (prim_wire::LegacyCompatibilityEnabled())
        throw std::invalid_argument(name + " requires strict Prim wire");
    work();
    return NpuBase::serialize();
}

void gemm_input_dx_timing::deserialize(vector<sc_bv<128>> wire) {
    if (prim_wire::LegacyCompatibilityEnabled())
        throw std::invalid_argument(name + " requires strict Prim wire");
    NpuBase::deserialize(std::move(wire));
    work();
}
