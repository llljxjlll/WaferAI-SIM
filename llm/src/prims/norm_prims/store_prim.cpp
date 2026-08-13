#include "systemc.h"

#include "prims/base.h"
#include "prims/norm_prims.h"
#include "utils/prim_utils.h"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

REGISTER_PRIM(Store_prim, PrimId::LEGACY_STORE);

namespace {
constexpr uint8_t kStorePrimId = PrimIdValue(PrimId::LEGACY_STORE);

uint8_t ExpectedPrimId() {
    const int registered =
        PrimFactory::getInstance().getPrimId("Store_prim");
    if (registered != static_cast<int>(kStorePrimId))
        throw std::invalid_argument(
            "Store_prim factory ID does not match its fixed PrimId");
    return kStorePrimId;
}

uint16_t EncodeU16(int value, const char *field) {
    if (value < 0)
        throw std::invalid_argument(std::string("Store_prim ") + field +
                                    " must be non-negative");
    if (static_cast<uint64_t>(value) >
        std::numeric_limits<uint16_t>::max())
        throw std::overflow_error(std::string("Store_prim ") + field +
                                  " exceeds the 16-bit Prim wire");
    return static_cast<uint16_t>(value);
}

uint8_t EncodeDatatype(DATATYPE value) {
    if (value != INT8 && value != FP16)
        throw std::invalid_argument("Store_prim datatype is invalid");
    return static_cast<uint8_t>(value);
}

DATATYPE DecodeDatatype(uint64_t value) {
    if (value > static_cast<uint64_t>(FP16))
        throw std::invalid_argument(
            "Store_prim Prim wire datatype is invalid");
    return static_cast<DATATYPE>(value);
}

const sc_bv<128> &ValidateWire(const vector<sc_bv<128>> &segments) {
    if (segments.size() != 1)
        throw std::invalid_argument(
            "Store_prim Prim wire requires exactly one segment");
    const auto &wire = segments.front();
    if (wire.range(7, 0).to_uint64() != ExpectedPrimId())
        throw std::invalid_argument("Store_prim Prim wire ID mismatch");
    if (wire.range(127, 58).or_reduce())
        throw std::invalid_argument(
            "Store_prim Prim wire reserved bits are non-zero");
    return wire;
}
} // namespace

void Store_prim::printSelf() {  }

void Store_prim::deserialize(vector<sc_bv<128>> segments) {
    const auto &buffer = ValidateWire(segments);
    const int decoded_dram_addr =
        static_cast<int>(buffer.range(23, 8).to_uint64());
    const int decoded_sram_addr =
        static_cast<int>(buffer.range(39, 24).to_uint64());
    const int decoded_size =
        static_cast<int>(buffer.range(55, 40).to_uint64());
    const DATATYPE decoded_datatype =
        DecodeDatatype(buffer.range(57, 56).to_uint64());

    dram_addr = decoded_dram_addr;
    sram_addr = decoded_sram_addr;
    size = decoded_size;
    datatype = decoded_datatype;
}

vector<sc_bv<128>> Store_prim::serialize() {
    const uint16_t encoded_dram_addr = EncodeU16(dram_addr, "dram_addr");
    const uint16_t encoded_sram_addr = EncodeU16(sram_addr, "sram_addr");
    const uint16_t encoded_size = EncodeU16(size, "size");
    const uint8_t encoded_datatype = EncodeDatatype(datatype);

    vector<sc_bv<128>> segments;

    sc_bv<128> d = 0;
    d.range(7, 0) = sc_bv<8>(ExpectedPrimId());
    d.range(23, 8) = sc_bv<16>(encoded_dram_addr);
    d.range(39, 24) = sc_bv<16>(encoded_sram_addr);
    d.range(55, 40) = sc_bv<16>(encoded_size);
    d.range(57, 56) = sc_bv<2>(encoded_datatype);
    segments.push_back(d);

    return segments;
}
int Store_prim::taskCoreDefault(TaskCoreContext &context) {
    if (size == 0) return 0;
    if (!context.lsu_memory)
        throw std::runtime_error(
            "Store_prim transfer requires memory.sram.real_data_path=true");
    context.lsu_memory->Store(sram_addr, dram_addr, size);
    return 0;
}
