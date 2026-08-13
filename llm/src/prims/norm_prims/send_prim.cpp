#include "systemc.h"

#include "defs/enums.h"
#include "prims/base.h"
#include "prims/norm_prims.h"
#include "utils/prim_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

REGISTER_PRIM(Send_prim, PrimId::SEND);

namespace {
constexpr uint8_t kSendPrimId = PrimIdValue(PrimId::SEND);
constexpr uint64_t kMaxLabelId = 0xfff;

uint8_t ExpectedPrimId() {
    const int registered = PrimFactory::getInstance().getPrimId("Send_prim");
    if (registered != static_cast<int>(kSendPrimId))
        throw std::invalid_argument(
            "Send_prim factory ID does not match its fixed PrimId");
    return kSendPrimId;
}

uint64_t EncodeUnsigned(int value, uint64_t maximum, int bits,
                        const char *field) {
    if (value < 0)
        throw std::invalid_argument(std::string("Send_prim ") + field +
                                    " must be non-negative");
    if (static_cast<uint64_t>(value) > maximum)
        throw std::overflow_error(std::string("Send_prim ") + field +
                                  " exceeds the " + std::to_string(bits) +
                                  "-bit Prim wire");
    return static_cast<uint64_t>(value);
}

uint16_t EncodeDestination(int value) {
    // Legacy constructors use -1 as an explicit unbound/host sentinel. The
    // historical wire representation is 0xffff; make that conversion
    // explicit instead of relying on sc_bv signed truncation.
    if (value == -1)
        return std::numeric_limits<uint16_t>::max();
    return static_cast<uint16_t>(
        EncodeUnsigned(value, 0xffff, 16, "des_id"));
}

uint8_t EncodeType(SEND_TYPE value) {
    const int raw = static_cast<int>(value);
    if (raw < 0)
        throw std::invalid_argument("Send_prim type is invalid");
    if (raw > 0xf)
        throw std::overflow_error(
            "Send_prim type exceeds the 4-bit Prim wire");
    switch (value) {
    case SEND_ACK:
    case SEND_REQ:
    case SEND_DATA:
    case SEND_SRAM:
    case SEND_DONE:
        return static_cast<uint8_t>(raw);
    }
    throw std::invalid_argument("Send_prim type is unknown");
}

SEND_TYPE DecodeType(uint64_t raw) {
    if (raw > static_cast<uint64_t>(SEND_DONE))
        throw std::invalid_argument("Send_prim Prim wire type is unknown");
    return static_cast<SEND_TYPE>(raw);
}

uint8_t EncodeDatatype(DATATYPE value) {
    const int raw = static_cast<int>(value);
    if (raw < 0)
        throw std::invalid_argument("Send_prim datatype is invalid");
    if (raw > 0x3)
        throw std::overflow_error(
            "Send_prim datatype exceeds the 2-bit Prim wire");
    if (value != INT8 && value != FP16)
        throw std::invalid_argument("Send_prim datatype is unknown");
    return static_cast<uint8_t>(raw);
}

DATATYPE DecodeDatatype(uint64_t raw) {
    if (raw > static_cast<uint64_t>(FP16))
        throw std::invalid_argument(
            "Send_prim Prim wire datatype is unknown");
    return static_cast<DATATYPE>(raw);
}

uint8_t EncodeStripeCount(int value) {
    if (value < 0)
        throw std::invalid_argument(
            "Send_prim stripe_count must be non-negative");
    if (value > 0x7)
        throw std::overflow_error(
            "Send_prim stripe_count exceeds the 3-bit Prim wire");
    if (value != 1 && value != 2 && value != 4)
        throw std::invalid_argument(
            "Send_prim stripe_count must be 1, 2, or 4");
    return static_cast<uint8_t>(value);
}

int DecodeStripeCount(uint64_t raw) {
    // Pre-V5 legacy wires left the newly allocated stripe bits at zero;
    // zero is the sole documented compatibility encoding for one stripe.
    if (raw == 0)
        return 1;
    if (raw != 1 && raw != 2 && raw != 4)
        throw std::invalid_argument(
            "Send_prim Prim wire stripe_count must be 0, 1, 2, or 4");
    return static_cast<int>(raw);
}

bool CarriesPacketMetadata(SEND_TYPE value) {
    return value == SEND_DATA || value == SEND_REQ;
}

std::pair<uint8_t, uint8_t> EncodePacketMetadata(
    SEND_TYPE send_type, int scale, int last_group) {
    const uint64_t encoded_scale =
        EncodeUnsigned(scale, 0xff, 8, "packet_scale");
    const uint64_t encoded_last = EncodeUnsigned(
        last_group, 0xff, 8, "packets_in_last_group");
    if (!CarriesPacketMetadata(send_type)) {
        if (scale != 1 || last_group != 1)
            throw std::invalid_argument(
                "Send_prim control type has non-default packet metadata");
        return {0, 0};
    }
    if (encoded_scale == 0 || encoded_last == 0 ||
        encoded_last > encoded_scale)
        throw std::invalid_argument(
            "Send_prim packet aggregation metadata is invalid");
    return {static_cast<uint8_t>(encoded_scale),
            static_cast<uint8_t>(encoded_last)};
}

std::pair<int, int> DecodePacketMetadata(const sc_bv<128> &wire,
                                         SEND_TYPE send_type) {
    const uint64_t raw_scale = wire.range(43, 36).to_uint64();
    const uint64_t raw_last = wire.range(51, 44).to_uint64();
    if (!CarriesPacketMetadata(send_type)) {
        if (raw_scale != 0 || raw_last != 0)
            throw std::invalid_argument(
                "Send_prim Prim wire has packet metadata for a control type");
        return {1, 1};
    }
    // Pre-aggregation wires left both fields zero. Only the exact 0/0 pair
    // is compatible and means the historical one-packet-per-group form.
    if (raw_scale == 0 && raw_last == 0)
        return {1, 1};
    if (raw_scale == 0 || raw_last == 0 || raw_last > raw_scale)
        throw std::invalid_argument(
            "Send_prim Prim wire packet metadata is invalid");
    return {static_cast<int>(raw_scale), static_cast<int>(raw_last)};
}

uint16_t AddLabelId(const std::string &label) {
    if (label.empty() || label == UNSET_LABEL)
        throw std::invalid_argument(
            "SEND_DATA must have a non-empty output_label");

    for (size_t i = 0; i < g_addr_label_table.table.size(); ++i) {
        if (g_addr_label_table.table[i] != label)
            continue;
        const uint64_t id = static_cast<uint64_t>(i) + 1;
        if (id > kMaxLabelId)
            throw std::overflow_error(
                "Send_prim label table ID exceeds the 12-bit Prim wire");
        return static_cast<uint16_t>(id);
    }

    if (g_addr_label_table.table.size() >= kMaxLabelId)
        throw std::overflow_error(
            "Send_prim label table has exhausted its 12-bit IDs");
    const int raw = g_addr_label_table.addRecord(label);
    if (raw <= 0)
        throw std::invalid_argument(
            "Send_prim label table returned an invalid ID");
    if (static_cast<uint64_t>(raw) > kMaxLabelId)
        throw std::overflow_error(
            "Send_prim label table ID exceeds the 12-bit Prim wire");
    return static_cast<uint16_t>(raw);
}

std::string DecodeLabel(uint64_t raw) {
    if (raw == 0 || raw > g_addr_label_table.table.size())
        throw std::invalid_argument(
            "Send_prim Prim wire output label ID is unknown");
    const std::string label =
        g_addr_label_table.findRecord(static_cast<int>(raw));
    if (label.empty() || label == UNSET_LABEL)
        throw std::invalid_argument(
            "Send_prim Prim wire output label is missing");
    return label;
}

const sc_bv<128> &ValidateWire(const vector<sc_bv<128>> &segments) {
    if (segments.size() != 1)
        throw std::invalid_argument(
            "Send_prim Prim wire requires exactly one segment");
    const auto &wire = segments.front();
    if (wire.range(7, 0).to_uint64() != ExpectedPrimId())
        throw std::invalid_argument("Send_prim Prim wire ID mismatch");
    if (wire.range(55, 52).or_reduce() ||
        wire.range(127, 125).or_reduce())
        throw std::invalid_argument(
            "Send_prim Prim wire reserved bits are non-zero");
    return wire;
}
} // namespace

void Send_prim::printSelf() {}

void Send_prim::deserialize(vector<sc_bv<128>> segments) {
    const auto &buffer = ValidateWire(segments);
    const SEND_TYPE decoded_type =
        DecodeType(buffer.range(59, 56).to_uint64());
    const uint64_t raw_label = buffer.range(35, 24).to_uint64();
    std::string decoded_label = UNSET_LABEL;
    if (decoded_type == SEND_DATA)
        decoded_label = DecodeLabel(raw_label);
    else if (raw_label != 0)
        throw std::invalid_argument(
            "Send_prim Prim wire has an output label for a non-DATA type");

    const auto decoded_packet =
        DecodePacketMetadata(buffer, decoded_type);
    const uint64_t raw_max_packet = buffer.range(91, 60).to_uint64();
    if (raw_max_packet >
        static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error(
            "Send_prim Prim wire max_packet cannot fit its runtime field");
    const int decoded_max_packet = static_cast<int>(raw_max_packet);
    const int decoded_end_length =
        static_cast<int>(buffer.range(119, 112).to_uint64());
    if (!CarriesPacketMetadata(decoded_type) &&
        (decoded_max_packet != 0 || decoded_end_length != 0))
        throw std::invalid_argument(
            "Send_prim Prim wire has payload size fields for a control type");

    const int decoded_tag =
        static_cast<int>(buffer.range(111, 92).to_uint64());
    if (decoded_type == SEND_DONE && decoded_tag != 0)
        throw std::invalid_argument(
            "Send_prim Prim wire SEND_DONE tag must be zero");
    const DATATYPE decoded_datatype =
        DecodeDatatype(buffer.range(121, 120).to_uint64());
    const int decoded_stripe_count =
        DecodeStripeCount(buffer.range(124, 122).to_uint64());

    des_id = static_cast<int>(buffer.range(23, 8).to_uint64());
    type = decoded_type;
    output_label = std::move(decoded_label);
    packet_scale = decoded_packet.first;
    packets_in_last_group = decoded_packet.second;
    max_packet = decoded_max_packet;
    tag_id = decoded_tag;
    end_length = decoded_end_length;
    datatype = decoded_datatype;
    stripe_count = decoded_stripe_count;

    // Execution-only state must never survive a successful decode.
    d2d_exit_port = -1;
    d2d_exit_selected = false;
    stripe_packets.clear();
    stripe_sent.clear();
    stripe_exit_ports.clear();
    next_subflow = 0;
    stripe_saf_reserved = false;
}

vector<sc_bv<128>> Send_prim::serialize() {
    const uint8_t encoded_type = EncodeType(type);
    const uint16_t encoded_destination = EncodeDestination(des_id);
    const uint64_t encoded_max_packet = EncodeUnsigned(
        max_packet, std::numeric_limits<uint32_t>::max(), 32, "max_packet");
    const uint64_t encoded_tag = EncodeUnsigned(tag_id, 0xfffff, 20, "tag_id");
    const uint64_t encoded_end_length =
        EncodeUnsigned(end_length, 0xff, 8, "end_length");
    const uint8_t encoded_datatype = EncodeDatatype(datatype);
    const uint8_t encoded_stripe_count = EncodeStripeCount(stripe_count);
    const auto encoded_packet =
        EncodePacketMetadata(type, packet_scale, packets_in_last_group);

    if (!CarriesPacketMetadata(type) &&
        (max_packet != 0 || end_length != 0))
        throw std::invalid_argument(
            "Send_prim control type has payload size fields");
    if (type == SEND_DONE && tag_id != 0)
        throw std::invalid_argument("Send_prim SEND_DONE tag must be zero");
    if (type != SEND_DATA && !output_label.empty() &&
        output_label != UNSET_LABEL)
        throw std::invalid_argument(
            "Send_prim non-DATA type must not carry an output_label");

    uint16_t encoded_label = 0;
    if (type == SEND_DATA)
        encoded_label = AddLabelId(output_label);

    vector<sc_bv<128>> segments;

    sc_bv<128> d = 0;
    d.range(7, 0) = sc_bv<8>(ExpectedPrimId());
    d.range(23, 8) = sc_bv<16>(encoded_destination);
    d.range(35, 24) = sc_bv<12>(encoded_label);
    d.range(43, 36) = sc_bv<8>(encoded_packet.first);
    d.range(51, 44) = sc_bv<8>(encoded_packet.second);
    d.range(59, 56) = sc_bv<4>(encoded_type);
    d.range(91, 60) = sc_bv<32>(encoded_max_packet);
    d.range(111, 92) = sc_bv<20>(encoded_tag);
    d.range(119, 112) = sc_bv<8>(encoded_end_length);
    d.range(121, 120) = sc_bv<2>(encoded_datatype);
    d.range(124, 122) = sc_bv<3>(encoded_stripe_count);
    segments.push_back(d);

    return segments;
}
int Send_prim::taskCoreDefault(TaskCoreContext &context) {
#if USE_NB_DRAMSYS == 0
    auto wc = context.wc;
#endif
    auto mau = context.mau;
    auto hmau = context.hmau;
    sc_bv<128> msg_data = 0;
    sc_time elapsed_time;

    // 找到output_label对应的数据块
    if (type == SEND_DATA) {
        bool need_delete = false;

        std::size_t pos = output_label.find("DEL_");
        if (pos != std::string::npos) {
            output_label = output_label.substr(pos + 4);
            need_delete = true;
        }

        AddrPosKey sc_key;
        int flag =
            prim_context->sram_pos_locator_->findPair(output_label, sc_key);
        if (context.sram_access) {
            const int sram_bits = GetCoreHWConfig(context.cid)->sram_bitwidth;
            if ((sram_bits % 8) != 0)
                throw std::runtime_error(
                    "Send_prim requires byte-addressable SRAM bitwidth");
            sram::Request request;
            request.initiator = sram::Initiator::kCompute;
            request.command = sram::Command::kRead;
            request.address =
                static_cast<uint64_t>(sc_key.pos) * (sram_bits / 8);
            request.size_bytes = 16;
            context.sram_access->Access(request);
        } else {
#if USE_SRAM_MANAGER == 1
            mau->mem_read_port->read(0, msg_data, elapsed_time);
#else
            wait(CYCLE, SC_NS);
#endif
        }
        if (need_delete)
            prim_context->sram_pos_locator_->deletePair(output_label);
    }

    msg_data = 0b1;
    return 0;
}
