#include "systemc.h"

#include "prims/norm_prims.h"
#include "utils/prim_utils.h"

#include <algorithm>
#include <stdexcept>
#include <string>

REGISTER_PRIM(Dte_async_prim, PrimId::DTE_ASYNC);

namespace {
DteAsyncOp ParseOp(const std::string &value) {
    if (value == "issue") return DteAsyncOp::ISSUE;
    if (value == "wait") return DteAsyncOp::WAIT;
    if (value == "poll") return DteAsyncOp::POLL;
    if (value == "fence") return DteAsyncOp::FENCE;
    if (value == "cancel") return DteAsyncOp::CANCEL;
    throw std::invalid_argument("Dte_async has an unknown op: " + value);
}

DteDir ParseDirection(const std::string &value) {
    if (value == "SPM_TO_REMOTE") return DteDir::SPM_TO_REMOTE;
    if (value == "REMOTE_TO_SPM") return DteDir::REMOTE_TO_SPM;
    if (value == "SPM_TO_SPM") return DteDir::SPM_TO_SPM;
    if (value == "SPM_TO_DRAM") return DteDir::SPM_TO_DRAM;
    if (value == "DRAM_TO_SPM") return DteDir::DRAM_TO_SPM;
    if (value == "DRAM_TO_REMOTE") return DteDir::DRAM_TO_REMOTE;
    throw std::invalid_argument("DTE async direction is unknown: " + value);
}

constexpr uint32_t kLocalRegionMarker = 0x5352414dU;
constexpr uint32_t kDestinationRegionMarker = 0x44535452U;

void AppendRegionBlock(vector<sc_bv<128>> &segments, uint32_t marker,
                       uint64_t offset, const std::string &name) {
    if (name.empty()) return;
    sc_bv<128> region_metadata = 0;
    region_metadata.range(63, 0) = sc_bv<64>(offset);
    region_metadata.range(79, 64) = sc_bv<16>(name.size());
    region_metadata.range(127, 96) = sc_bv<32>(marker);
    segments.push_back(region_metadata);
    for (size_t base = 0; base < name.size(); base += 16) {
        sc_bv<128> text = 0;
        const size_t count = std::min<size_t>(16, name.size() - base);
        for (size_t i = 0; i < count; ++i)
            text.range(static_cast<int>(8 * i + 7),
                       static_cast<int>(8 * i)) =
                sc_bv<8>(static_cast<uint8_t>(name[base + i]));
        segments.push_back(text);
    }
}

size_t DecodeRegionBlock(const vector<sc_bv<128>> &segments, size_t index,
                         std::string &name, uint64_t &offset) {
    if (index >= segments.size())
        throw std::invalid_argument(
            "Dte_async wire region metadata is truncated");
    if (segments[index].range(95, 80).or_reduce())
        throw std::invalid_argument(
            "Dte_async wire region reserved bits are non-zero");
    offset = segments[index].range(63, 0).to_uint64();
    const size_t name_size =
        segments[index].range(79, 64).to_uint64();
    const size_t text_segments = (name_size + 15) / 16;
    if (name_size == 0 || name_size > 64 ||
        text_segments > segments.size() - index - 1)
        throw std::invalid_argument(
            "Dte_async wire region name length is inconsistent");
    name.clear();
    name.reserve(name_size);
    for (size_t name_index = 0; name_index < name_size; ++name_index) {
        const auto &text =
            segments[index + 1 + name_index / 16];
        const int lo = static_cast<int>(8 * (name_index % 16));
        name.push_back(static_cast<char>(
            text.range(lo + 7, lo).to_uint()));
    }
    const size_t used = name_size % 16;
    if (used != 0 &&
        segments[index + text_segments]
            .range(127, static_cast<int>(used * 8)).or_reduce())
        throw std::invalid_argument(
            "Dte_async wire region-name padding is non-zero");
    return index + 1 + text_segments;
}

void Validate(const Dte_async_prim &prim) {
    const auto raw_op = static_cast<uint8_t>(prim.op);
    if (raw_op > static_cast<uint8_t>(DteAsyncOp::CANCEL))
        throw std::invalid_argument("Dte_async op encoding is invalid");
    if (prim.op == DteAsyncOp::ISSUE) {
        if (prim.payload_bits == 0)
            throw std::invalid_argument(
                "Dte_async issue payload_bits must be > 0");
        if (prim.direction < DteDir::SPM_TO_REMOTE ||
            prim.direction > DteDir::DRAM_TO_REMOTE)
            throw std::invalid_argument(
                "Dte_async issue direction is unsupported");
        if (prim.direction == DteDir::DRAM_TO_REMOTE) {
            if (prim.spm_addr != 0 || prim.spm_size != 0 ||
                !prim.sram_region.empty() || prim.sram_offset != 0)
                throw std::invalid_argument(
                    "DRAM_TO_REMOTE must not carry an SPM range");
            if (prim.remote_peer == DTE_ASYNC_INVALID_REMOTE_PEER)
                throw std::invalid_argument(
                    "DRAM_TO_REMOTE requires remote_peer");
        } else if (prim.spm_size == 0) {
            throw std::invalid_argument(
                "Dte_async issue spm_size must be > 0");
        }
        if (!prim.sram_region.empty()) {
            if (prim.sram_region.size() > 64)
                throw std::invalid_argument(
                    "Dte_async sram_region exceeds the 64-byte wire limit");
            if (prim.direction != DteDir::DRAM_TO_SPM &&
                prim.direction != DteDir::SPM_TO_DRAM &&
                prim.direction != DteDir::SPM_TO_SPM)
                throw std::invalid_argument(
                    "Dte_async named source/local SRAM region is unsupported "
                    "for this direction");
            if (prim.spm_addr != 0)
                throw std::invalid_argument(
                    "Dte_async source region form must not carry spm_addr");
        } else if (prim.sram_offset != 0) {
            throw std::invalid_argument(
                "Dte_async absolute source form must not carry sram_offset");
        }
        if (!prim.destination_sram_region.empty()) {
            if (prim.destination_sram_region.size() > 64)
                throw std::invalid_argument(
                    "Dte_async destination_sram_region exceeds the 64-byte "
                    "wire limit");
            if (prim.direction != DteDir::SPM_TO_SPM)
                throw std::invalid_argument(
                    "Dte_async named destination SRAM region requires SPM_TO_SPM");
            if (prim.remote_addr != 0)
                throw std::invalid_argument(
                    "Dte_async destination region form must not carry remote_addr");
        } else if (prim.destination_sram_offset != 0) {
            throw std::invalid_argument(
                "Dte_async absolute destination form must not carry destination_sram_offset");
        }
        return;
    }
    if (prim.payload_bits != 0 || prim.spm_addr != 0 || prim.spm_size != 0 ||
        !prim.sram_region.empty() || prim.sram_offset != 0 ||
        !prim.destination_sram_region.empty() ||
        prim.destination_sram_offset != 0 ||
        prim.remote_peer != DTE_ASYNC_INVALID_REMOTE_PEER ||
        prim.remote_addr != 0 || prim.address_block != 0)
        throw std::invalid_argument(
            "Dte_async non-issue op must not carry payload or address metadata");
    if (prim.op == DteAsyncOp::FENCE && prim.token != 0)
        throw std::invalid_argument(
            "Dte_async fence must not carry a logical token");
}
} // namespace
void Dte_async_prim::refreshPrimType() {
    if (op != DteAsyncOp::ISSUE) {
        setPrimMainCategory(SYNC_PRIM);
        return;
    }
    switch (direction) {
    case DteDir::SPM_TO_REMOTE:
    case DteDir::REMOTE_TO_SPM:
    case DteDir::DRAM_TO_REMOTE:
        setPrimMainCategory(COMM_PRIM);
        return;
    case DteDir::SPM_TO_SPM:
    case DteDir::SPM_TO_DRAM:
    case DteDir::DRAM_TO_SPM:
        setPrimMainCategory(MEM_PRIM);
        return;
    }
}


void Dte_async_prim::printSelf() {}

void Dte_async_prim::parseJson(json j) {
    if (!j.contains("op"))
        throw std::invalid_argument("Dte_async requires an op field");
    op = ParseOp(j.at("op").get<std::string>());
    token = j.value("token", uint32_t(0));
    payload_bits = 0;
    poll_complete = false;
    direction = DteDir::SPM_TO_REMOTE;
    spm_addr = 0;
    spm_size = 0;
    sram_region.clear();
    sram_offset = 0;
    destination_sram_region.clear();
    destination_sram_offset = 0;
    remote_peer = DTE_ASYNC_INVALID_REMOTE_PEER;
    remote_addr = 0;
    address_block = 0;
    if (op == DteAsyncOp::ISSUE) {
        if (!j.contains("payload_bits") || !j.contains("direction"))
            throw std::invalid_argument(
                "Dte_async issue requires payload_bits and direction");
        payload_bits = j.at("payload_bits").get<uint64_t>();
        direction = ParseDirection(j.at("direction").get<std::string>());
        if (direction != DteDir::DRAM_TO_REMOTE) {
            if (!j.contains("spm_size"))
                throw std::invalid_argument(
                    "Dte_async direction requires spm_size");
            const bool has_absolute = j.contains("spm_addr");
            const bool has_region = j.contains("sram_region");
            if (has_absolute == has_region)
                throw std::invalid_argument(
                    "Dte_async requires exactly one of spm_addr or sram_region");
            if (has_region && direction != DteDir::DRAM_TO_SPM &&
                direction != DteDir::SPM_TO_DRAM &&
                direction != DteDir::SPM_TO_SPM)
                throw std::invalid_argument(
                    "named source/local SRAM region is unsupported for this "
                    "direction");
            spm_addr = j.value("spm_addr", uint64_t(0));
            sram_region = j.value("sram_region", std::string{});
            sram_offset = j.value("sram_offset", uint64_t(0));
            if (sram_region.size() > 64)
                throw std::invalid_argument(
                    "Dte_async sram_region exceeds the 64-byte wire limit");
        }
        spm_size = j.value("spm_size", uint64_t(0));
        remote_peer = j.value("remote_peer", DTE_ASYNC_INVALID_REMOTE_PEER);
        remote_addr = j.value("remote_addr", uint64_t(0));
        if (j.contains("destination_sram_region")) {
            if (direction != DteDir::SPM_TO_SPM)
                throw std::invalid_argument(
                    "Dte_async destination_sram_region requires SPM_TO_SPM");
            if (j.contains("remote_addr"))
                throw std::invalid_argument(
                    "Dte_async SPM_TO_SPM destination requires exactly one "
                    "of remote_addr or destination_sram_region");
            destination_sram_region =
                j.at("destination_sram_region").get<std::string>();
            destination_sram_offset =
                j.value("destination_sram_offset", uint64_t(0));
        } else if (j.contains("destination_sram_offset")) {
            throw std::invalid_argument(
                "Dte_async destination_sram_offset requires destination_sram_region");
        }
        if (j.contains("hbm_addr")) {
            if (direction != DteDir::DRAM_TO_SPM &&
                direction != DteDir::SPM_TO_DRAM &&
                direction != DteDir::DRAM_TO_REMOTE)
                throw std::invalid_argument(
                    "Dte_async hbm_addr is valid only for DRAM directions");
            const uint64_t hbm_addr = j.at("hbm_addr").get<uint64_t>();
            if (j.contains("remote_addr") && remote_addr != hbm_addr)
                throw std::invalid_argument(
                    "Dte_async hbm_addr and remote_addr disagree");
            remote_addr = hbm_addr;
        }
        address_block = j.value("address_block", uint32_t(0));
    }
    Validate(*this);
    refreshPrimType();
}

vector<sc_bv<128>> Dte_async_prim::serialize() {
    Validate(*this);
    refreshPrimType();
    sc_bv<128> metadata = 0;
    metadata.range(7, 0) =
        sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    metadata.range(10, 8) = sc_bv<3>(static_cast<uint8_t>(op));
    metadata.range(13, 11) =
        sc_bv<3>(static_cast<uint8_t>(direction));
    metadata.range(45, 14) = sc_bv<32>(token);
    metadata.range(109, 46) = sc_bv<64>(payload_bits);

    sc_bv<128> local_range = 0;
    local_range.range(63, 0) = sc_bv<64>(spm_addr);
    local_range.range(127, 64) = sc_bv<64>(spm_size);

    sc_bv<128> remote_range = 0;
    remote_range.range(31, 0) = sc_bv<32>(remote_peer);
    remote_range.range(95, 32) = sc_bv<64>(remote_addr);
    remote_range.range(127, 96) = sc_bv<32>(address_block);

    vector<sc_bv<128>> result = {metadata, local_range};
    const bool needs_remote_range =
        remote_peer != DTE_ASYNC_INVALID_REMOTE_PEER || remote_addr != 0 ||
        address_block != 0 || !sram_region.empty() ||
        !destination_sram_region.empty();
    if (needs_remote_range) result.push_back(remote_range);
    AppendRegionBlock(result, kLocalRegionMarker, sram_offset, sram_region);
    AppendRegionBlock(result, kDestinationRegionMarker,
                      destination_sram_offset, destination_sram_region);
    return prim_wire::WrapSegments(std::move(result), name);
}

void Dte_async_prim::deserialize(vector<sc_bv<128>> segments) {
    segments = prim_wire::UnwrapSegments(segments, name);
    if (segments.size() < 2)
        throw std::invalid_argument(
            "Dte_async wire encoding requires at least two segments");
    const uint64_t raw_op = segments[0].range(10, 8).to_uint64();
    const uint64_t raw_direction = segments[0].range(13, 11).to_uint64();
    if (raw_op > static_cast<uint8_t>(DteAsyncOp::CANCEL))
        throw std::invalid_argument("Dte_async wire op is invalid");
    if (raw_direction > static_cast<uint8_t>(DteDir::DRAM_TO_REMOTE))
        throw std::invalid_argument("Dte_async wire direction is invalid");
    if (segments[0].range(127, 110).or_reduce())
        throw std::invalid_argument(
            "Dte_async wire metadata reserved bits are non-zero");
    op = static_cast<DteAsyncOp>(raw_op);
    poll_complete = false;
    direction = static_cast<DteDir>(raw_direction);
    token = segments[0].range(45, 14).to_uint64();
    payload_bits = segments[0].range(109, 46).to_uint64();
    spm_addr = segments[1].range(63, 0).to_uint64();
    spm_size = segments[1].range(127, 64).to_uint64();
    remote_peer = DTE_ASYNC_INVALID_REMOTE_PEER;
    remote_addr = 0;
    address_block = 0;
    sram_region.clear();
    sram_offset = 0;
    destination_sram_region.clear();
    destination_sram_offset = 0;
    if (segments.size() >= 3) {
        remote_peer = segments[2].range(31, 0).to_uint64();
        remote_addr = segments[2].range(95, 32).to_uint64();
        address_block = segments[2].range(127, 96).to_uint64();
    }
    size_t index = std::min<size_t>(segments.size(), 3);
    while (index < segments.size()) {
        const uint32_t marker = static_cast<uint32_t>(
            segments[index].range(127, 96).to_uint64());
        if (marker == kLocalRegionMarker) {
            if (!sram_region.empty())
                throw std::invalid_argument(
                    "Dte_async wire has duplicate source region metadata");
            index = DecodeRegionBlock(segments, index, sram_region,
                                      sram_offset);
            spm_addr = 0;
        } else if (marker == kDestinationRegionMarker) {
            if (!destination_sram_region.empty())
                throw std::invalid_argument(
                    "Dte_async wire has duplicate destination region metadata");
            index = DecodeRegionBlock(segments, index,
                                      destination_sram_region,
                                      destination_sram_offset);
            remote_addr = 0;
        } else {
            throw std::invalid_argument(
                "Dte_async region metadata marker is invalid");
        }
    }
    Validate(*this);
    refreshPrimType();
}

int Dte_async_prim::taskCoreDefault(TaskCoreContext &context) {
    refreshPrimType();
    (void)context;
    return 0;
}
