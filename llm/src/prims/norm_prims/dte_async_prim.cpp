#include "systemc.h"

#include "prims/norm_prims.h"
#include "utils/prim_utils.h"

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
            if (prim.spm_addr != 0 || prim.spm_size != 0)
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
    remote_peer = DTE_ASYNC_INVALID_REMOTE_PEER;
    remote_addr = 0;
    address_block = 0;
    if (op == DteAsyncOp::ISSUE) {
        if (!j.contains("payload_bits") || !j.contains("direction"))
            throw std::invalid_argument(
                "Dte_async issue requires payload_bits and direction");
        payload_bits = j.at("payload_bits").get<uint64_t>();
        direction = ParseDirection(j.at("direction").get<std::string>());
        if (direction != DteDir::DRAM_TO_REMOTE &&
            (!j.contains("spm_addr") || !j.contains("spm_size")))
            throw std::invalid_argument(
                "Dte_async direction requires spm_addr and spm_size");
        spm_addr = j.value("spm_addr", uint64_t(0));
        spm_size = j.value("spm_size", uint64_t(0));
        remote_peer = j.value("remote_peer", DTE_ASYNC_INVALID_REMOTE_PEER);
        remote_addr = j.value("remote_addr", uint64_t(0));
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
    if (remote_peer == DTE_ASYNC_INVALID_REMOTE_PEER &&
        remote_addr == 0 && address_block == 0)
        return {metadata, local_range};
    return {metadata, local_range, remote_range};
}

void Dte_async_prim::deserialize(vector<sc_bv<128>> segments) {
    if (segments.size() != 2 && segments.size() != 3)
        throw std::invalid_argument(
            "Dte_async wire encoding requires two or three segments");
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
    if (segments.size() == 3) {
        remote_peer = segments[2].range(31, 0).to_uint64();
        remote_addr = segments[2].range(95, 32).to_uint64();
        address_block = segments[2].range(127, 96).to_uint64();
    }
    Validate(*this);
}

int Dte_async_prim::taskCoreDefault(TaskCoreContext &context) {
    (void)context;
    return 0;
}
