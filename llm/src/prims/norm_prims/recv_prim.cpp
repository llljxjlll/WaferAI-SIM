#include "systemc.h"

#include "defs/enums.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "prims/norm_prims.h"
#include "utils/memory_utils.h"
#include "utils/prim_utils.h"

#include <cstdint>
#include <stdexcept>
#include <string>

REGISTER_PRIM(Recv_prim, PrimId::RECV);

namespace {
constexpr uint8_t kRecvPrimId = PrimIdValue(PrimId::RECV);

uint8_t ExpectedPrimId() {
    const int registered = PrimFactory::getInstance().getPrimId("Recv_prim");
    if (registered != static_cast<int>(kRecvPrimId))
        throw std::invalid_argument(
            "Recv_prim factory ID does not match its fixed PrimId");
    return kRecvPrimId;
}

uint64_t EncodeUnsigned(int value, uint64_t maximum, int bits,
                        const char *field) {
    if (value < 0)
        throw std::invalid_argument(std::string("Recv_prim ") + field +
                                    " must be non-negative");
    if (static_cast<uint64_t>(value) > maximum)
        throw std::overflow_error(std::string("Recv_prim ") + field +
                                  " exceeds the " + std::to_string(bits) +
                                  "-bit Prim wire");
    return static_cast<uint64_t>(value);
}

uint8_t EncodeType(RECV_TYPE value) {
    const int raw = static_cast<int>(value);
    if (raw < 0)
        throw std::invalid_argument("Recv_prim type is invalid");
    if (raw > 0xf)
        throw std::overflow_error(
            "Recv_prim type exceeds the 4-bit Prim wire");
    switch (value) {
    case RECV_CONF:
    case RECV_ACK:
    case RECV_FLAG:
    case RECV_DATA:
    case RECV_SRAM:
    case RECV_WEIGHT:
    case RECV_START:
        return static_cast<uint8_t>(raw);
    }
    throw std::invalid_argument("Recv_prim type is unknown");
}

RECV_TYPE DecodeType(uint64_t raw) {
    if (raw > static_cast<uint64_t>(RECV_START))
        throw std::invalid_argument("Recv_prim Prim wire type is unknown");
    return static_cast<RECV_TYPE>(raw);
}

uint8_t EncodeDatatype(DATATYPE value) {
    const int raw = static_cast<int>(value);
    if (raw < 0)
        throw std::invalid_argument("Recv_prim datatype is invalid");
    if (raw > 0x3)
        throw std::overflow_error(
            "Recv_prim datatype exceeds the 2-bit Prim wire");
    if (value != INT8 && value != FP16)
        throw std::invalid_argument("Recv_prim datatype is unknown");
    return static_cast<uint8_t>(raw);
}

DATATYPE DecodeDatatype(uint64_t raw) {
    if (raw > static_cast<uint64_t>(FP16))
        throw std::invalid_argument(
            "Recv_prim Prim wire datatype is unknown");
    return static_cast<DATATYPE>(raw);
}

uint8_t EncodeStripeCount(int value) {
    if (value < 0)
        throw std::invalid_argument(
            "Recv_prim stripe_count must be non-negative");
    if (value > 0x7)
        throw std::overflow_error(
            "Recv_prim stripe_count exceeds the 3-bit Prim wire");
    if (value != 1 && value != 2 && value != 4)
        throw std::invalid_argument(
            "Recv_prim stripe_count must be 1, 2, or 4");
    return static_cast<uint8_t>(value);
}

int DecodeStripeCount(uint64_t raw) {
    // Pre-V5 legacy wires left the newly allocated stripe bits at zero;
    // zero is the sole documented compatibility encoding for one stripe.
    if (raw == 0)
        return 1;
    if (raw != 1 && raw != 2 && raw != 4)
        throw std::invalid_argument(
            "Recv_prim Prim wire stripe_count must be 0, 1, 2, or 4");
    return static_cast<int>(raw);
}

const sc_bv<128> &ValidateWire(const vector<sc_bv<128>> &segments) {
    if (segments.size() != 1)
        throw std::invalid_argument(
            "Recv_prim Prim wire requires exactly one segment");
    const auto &wire = segments.front();
    if (wire.range(7, 0).to_uint64() != ExpectedPrimId())
        throw std::invalid_argument("Recv_prim Prim wire ID mismatch");
    if (wire.range(127, 41).or_reduce())
        throw std::invalid_argument(
            "Recv_prim Prim wire reserved bits are non-zero");
    return wire;
}
} // namespace

void Recv_prim::printSelf() {
}

void Recv_prim::deserialize(vector<sc_bv<128>> segments) {
    const auto &buffer = ValidateWire(segments);
    const RECV_TYPE decoded_type =
        DecodeType(buffer.range(11, 8).to_uint64());
    const int decoded_tag =
        static_cast<int>(buffer.range(27, 12).to_uint64());
    const int decoded_recv_count =
        static_cast<int>(buffer.range(35, 28).to_uint64());
    const DATATYPE decoded_datatype =
        DecodeDatatype(buffer.range(37, 36).to_uint64());
    const int decoded_stripe_count =
        DecodeStripeCount(buffer.range(40, 38).to_uint64());

    type = decoded_type;
    tag_id = decoded_tag;
    recv_cnt = decoded_recv_count;
    datatype = decoded_datatype;
    stripe_count = decoded_stripe_count;
}

vector<sc_bv<128>> Recv_prim::serialize() {
    const uint8_t encoded_type = EncodeType(type);
    const uint64_t encoded_tag = EncodeUnsigned(tag_id, 0xffff, 16, "tag_id");
    const uint64_t encoded_recv_count =
        EncodeUnsigned(recv_cnt, 0xff, 8, "recv_cnt");
    const uint8_t encoded_datatype = EncodeDatatype(datatype);
    const uint8_t encoded_stripe_count = EncodeStripeCount(stripe_count);

    vector<sc_bv<128>> segments;

    sc_bv<128> d = 0;
    d.range(7, 0) = sc_bv<8>(ExpectedPrimId());
    d.range(11, 8) = sc_bv<4>(encoded_type);
    d.range(27, 12) = sc_bv<16>(encoded_tag);
    d.range(35, 28) = sc_bv<8>(encoded_recv_count);
    d.range(37, 36) = sc_bv<2>(encoded_datatype);
    d.range(40, 38) = sc_bv<3>(encoded_stripe_count);
    segments.push_back(d);

    return segments;
}

int Recv_prim::taskCoreDefault(TaskCoreContext &context) {
    u_int64_t elapsed_time;
    // sram_write_append_generic(context, M_D_DATA, elapsed_time);

    return 0;
}
