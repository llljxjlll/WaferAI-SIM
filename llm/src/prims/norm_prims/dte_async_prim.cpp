#include "systemc.h"

#include "prims/norm_prims.h"
#include "utils/prim_utils.h"

#include <algorithm>
#include <stdexcept>
#include <string>

REGISTER_PRIM(Dte_async_prim);

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
        return;
    }
    if (prim.payload_bits != 0 || prim.spm_addr != 0 || prim.spm_size != 0 ||
        !prim.sram_region.empty() || prim.sram_offset != 0 ||
        prim.remote_peer != DTE_ASYNC_INVALID_REMOTE_PEER ||
        prim.remote_addr != 0 || prim.address_block != 0)
        throw std::invalid_argument(
            "Dte_async non-issue op must not carry payload or address metadata");
    if (prim.op == DteAsyncOp::FENCE && prim.token != 0)
        throw std::invalid_argument(
            "Dte_async fence must not carry a logical token");
}
} // namespace

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
                direction != DteDir::SPM_TO_DRAM)
                throw std::invalid_argument(
                    "named SRAM region is supported only for DRAM directions");
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
}

vector<sc_bv<128>> Dte_async_prim::serialize() {
    Validate(*this);
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
    if (sram_region.empty()) {
        if (remote_peer == DTE_ASYNC_INVALID_REMOTE_PEER &&
            remote_addr == 0 && address_block == 0)
            return {metadata, local_range};
        return {metadata, local_range, remote_range};
    }

    sc_bv<128> region_metadata = 0;
    region_metadata.range(63, 0) = sc_bv<64>(sram_offset);
    region_metadata.range(79, 64) = sc_bv<16>(sram_region.size());
    region_metadata.range(127, 96) = sc_bv<32>(0x5352414dU);
    vector<sc_bv<128>> result = {metadata, local_range, remote_range,
                                 region_metadata};
    for (size_t base = 0; base < sram_region.size(); base += 16) {
        sc_bv<128> text = 0;
        const size_t count = std::min<size_t>(16, sram_region.size() - base);
        for (size_t i = 0; i < count; ++i)
            text.range(static_cast<int>(8 * i + 7),
                       static_cast<int>(8 * i)) =
                sc_bv<8>(static_cast<uint8_t>(sram_region[base + i]));
        result.push_back(text);
    }
    return result;
}

void Dte_async_prim::deserialize(vector<sc_bv<128>> segments) {
    if (segments.size() < 2)
        throw std::invalid_argument(
            "Dte_async wire encoding requires at least two segments");
    const uint64_t raw_op = segments[0].range(10, 8).to_uint64();
    const uint64_t raw_direction = segments[0].range(13, 11).to_uint64();
    if (raw_op > static_cast<uint8_t>(DteAsyncOp::CANCEL))
        throw std::invalid_argument("Dte_async wire op is invalid");
    if (raw_direction > static_cast<uint8_t>(DteDir::DRAM_TO_REMOTE))
        throw std::invalid_argument("Dte_async wire direction is invalid");
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
    if (segments.size() >= 3) {
        remote_peer = segments[2].range(31, 0).to_uint64();
        remote_addr = segments[2].range(95, 32).to_uint64();
        address_block = segments[2].range(127, 96).to_uint64();
    }
    if (segments.size() > 3) {
        if (segments[3].range(127, 96).to_uint64() != 0x5352414dU)
            throw std::invalid_argument(
                "Dte_async region metadata marker is invalid");
        sram_offset = segments[3].range(63, 0).to_uint64();
        const size_t name_size = segments[3].range(79, 64).to_uint64();
        const size_t expected = 4 + (name_size + 15) / 16;
        if (name_size == 0 || name_size > 64 || segments.size() != expected)
            throw std::invalid_argument(
                "Dte_async wire region name length is inconsistent");
        sram_region.reserve(name_size);
        for (size_t index = 0; index < name_size; ++index) {
            const auto &text = segments[4 + index / 16];
            const int lo = static_cast<int>(8 * (index % 16));
            sram_region.push_back(
                static_cast<char>(text.range(lo + 7, lo).to_uint()));
        }
        spm_addr = 0;
    } else if (segments.size() != 2 && segments.size() != 3) {
        throw std::invalid_argument(
            "Dte_async legacy wire encoding requires two or three segments");
    }
    Validate(*this);
}

int Dte_async_prim::taskCoreDefault(TaskCoreContext &context) {
    (void)context;
    return 0;
}
