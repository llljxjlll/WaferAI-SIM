#include "prims/norm_prims.h"

#include "utils/prim_utils.h"
#include <stdexcept>

REGISTER_PRIM(Lsu_mem_prim);

namespace {

LsuMemOp ParseLsuOp(const std::string &value) {
    if (value == "issue") return LsuMemOp::ISSUE;
    if (value == "wait") return LsuMemOp::WAIT;
    if (value == "poll") return LsuMemOp::POLL;
    if (value == "fence") return LsuMemOp::FENCE;
    if (value == "cancel") return LsuMemOp::CANCEL;
    if (value == "load_blocking") return LsuMemOp::LOAD_BLOCKING;
    if (value == "store_blocking") return LsuMemOp::STORE_BLOCKING;
    throw std::invalid_argument("Lsu_mem has an unknown op: " + value);
}

sram::LsuDirection ParseLsuDirection(const std::string &value) {
    if (value == "HBM_TO_SRAM") return sram::LsuDirection::kHbmToSram;
    if (value == "SRAM_TO_HBM") return sram::LsuDirection::kSramToHbm;
    throw std::invalid_argument("Lsu_mem has an unknown direction: " + value);
}

bool IsTransfer(LsuMemOp op) {
    return op == LsuMemOp::ISSUE || op == LsuMemOp::LOAD_BLOCKING ||
           op == LsuMemOp::STORE_BLOCKING;
}

void ValidateLsuPrim(const Lsu_mem_prim &prim) {
    if (static_cast<uint8_t>(prim.op) >
        static_cast<uint8_t>(LsuMemOp::STORE_BLOCKING))
        throw std::invalid_argument("Lsu_mem op encoding is invalid");
    if (IsTransfer(prim.op)) {
        if (prim.size_bytes == 0)
            throw std::invalid_argument(
                "Lsu_mem transfer size_bytes must be non-zero");
        if (!prim.absolute_sram && prim.sram_region.empty())
            throw std::invalid_argument(
                "Lsu_mem transfer requires sram_region or sram_addr");
        if (prim.sram_region.size() > 64)
            throw std::invalid_argument(
                "Lsu_mem sram_region exceeds the 64-byte wire limit");
        if (prim.op == LsuMemOp::ISSUE && prim.token == 0)
            throw std::invalid_argument(
                "Lsu_mem asynchronous issue requires a non-zero token");
        if (prim.op == LsuMemOp::LOAD_BLOCKING &&
            prim.direction != sram::LsuDirection::kHbmToSram)
            throw std::invalid_argument(
                "Lsu_mem load_blocking requires HBM_TO_SRAM");
        if (prim.op == LsuMemOp::STORE_BLOCKING &&
            prim.direction != sram::LsuDirection::kSramToHbm)
            throw std::invalid_argument(
                "Lsu_mem store_blocking requires SRAM_TO_HBM");
        return;
    }
    if (prim.op == LsuMemOp::FENCE) {
        if (prim.token != 0)
            throw std::invalid_argument("Lsu_mem fence token must be zero");
    } else if (prim.token == 0) {
        throw std::invalid_argument(
            "Lsu_mem wait/poll/cancel requires a non-zero token");
    }
    if (prim.hbm_addr != 0 || prim.sram_addr != 0 ||
        prim.sram_offset != 0 || prim.size_bytes != 0 ||
        !prim.sram_region.empty())
        throw std::invalid_argument(
            "Lsu_mem non-transfer op carries address metadata");
}

} // namespace

void Lsu_mem_prim::parseJson(json j) {
    if (!j.contains("op"))
        throw std::invalid_argument("Lsu_mem requires an op field");
    op = ParseLsuOp(j.at("op").get<std::string>());
    token = j.value("token", uint64_t{0});
    poll_complete = false;
    hbm_addr = 0;
    sram_addr = 0;
    sram_offset = 0;
    size_bytes = 0;
    sram_region.clear();
    absolute_sram = false;
    direction = sram::LsuDirection::kHbmToSram;
    if (IsTransfer(op)) {
        if (!j.contains("hbm_addr") || !j.contains("size_bytes"))
            throw std::invalid_argument(
                "Lsu_mem transfer requires hbm_addr and size_bytes");
        hbm_addr = j.at("hbm_addr").get<uint64_t>();
        size_bytes = j.at("size_bytes").get<uint64_t>();
        if (j.contains("sram_region")) {
            sram_region = j.at("sram_region").get<std::string>();
            sram_offset = j.value("sram_offset", uint64_t{0});
        } else if (j.contains("sram_addr")) {
            absolute_sram = true;
            sram_addr = j.at("sram_addr").get<uint64_t>();
        } else {
            throw std::invalid_argument(
                "Lsu_mem transfer requires sram_region or sram_addr");
        }
        if (op == LsuMemOp::LOAD_BLOCKING) {
            direction = sram::LsuDirection::kHbmToSram;
        } else if (op == LsuMemOp::STORE_BLOCKING) {
            direction = sram::LsuDirection::kSramToHbm;
        } else {
            if (!j.contains("direction"))
                throw std::invalid_argument(
                    "Lsu_mem issue requires direction");
            direction =
                ParseLsuDirection(j.at("direction").get<std::string>());
        }
    }
    ValidateLsuPrim(*this);
}

std::vector<sc_bv<128>> Lsu_mem_prim::serialize() {
    ValidateLsuPrim(*this);
    sc_bv<128> metadata = 0;
    metadata.range(7, 0) =
        sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    metadata.range(10, 8) = sc_bv<3>(static_cast<uint8_t>(op));
    metadata[11] =
        direction == sram::LsuDirection::kSramToHbm ? sc_dt::SC_LOGIC_1
                                                     : sc_dt::SC_LOGIC_0;
    metadata.range(75, 12) = sc_bv<64>(token);
    metadata[76] = absolute_sram ? sc_dt::SC_LOGIC_1 : sc_dt::SC_LOGIC_0;

    sc_bv<128> addresses = 0;
    addresses.range(63, 0) = sc_bv<64>(hbm_addr);
    addresses.range(127, 64) =
        sc_bv<64>(absolute_sram ? sram_addr : sram_offset);

    sc_bv<128> details = 0;
    details.range(63, 0) = sc_bv<64>(size_bytes);
    details.range(79, 64) = sc_bv<16>(sram_region.size());

    std::vector<sc_bv<128>> result = {metadata, addresses, details};
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

void Lsu_mem_prim::deserialize(std::vector<sc_bv<128>> segments) {
    if (segments.size() < 3)
        throw std::invalid_argument(
            "Lsu_mem wire encoding requires at least three segments");
    const uint64_t raw_op = segments[0].range(10, 8).to_uint64();
    if (raw_op > static_cast<uint8_t>(LsuMemOp::STORE_BLOCKING))
        throw std::invalid_argument("Lsu_mem wire op is invalid");
    op = static_cast<LsuMemOp>(raw_op);
    direction = segments[0][11].to_bool()
                    ? sram::LsuDirection::kSramToHbm
                    : sram::LsuDirection::kHbmToSram;
    token = segments[0].range(75, 12).to_uint64();
    absolute_sram = segments[0][76].to_bool();
    hbm_addr = segments[1].range(63, 0).to_uint64();
    const uint64_t local = segments[1].range(127, 64).to_uint64();
    sram_addr = absolute_sram ? local : 0;
    sram_offset = absolute_sram ? 0 : local;
    size_bytes = segments[2].range(63, 0).to_uint64();
    const size_t name_size = segments[2].range(79, 64).to_uint64();
    const size_t expected = 3 + (name_size + 15) / 16;
    if (name_size > 64 || segments.size() != expected)
        throw std::invalid_argument(
            "Lsu_mem wire region name length is inconsistent");
    sram_region.clear();
    sram_region.reserve(name_size);
    for (size_t index = 0; index < name_size; ++index) {
        const auto &text = segments[3 + index / 16];
        const int lo = static_cast<int>(8 * (index % 16));
        sram_region.push_back(
            static_cast<char>(text.range(lo + 7, lo).to_uint()));
    }
    poll_complete = false;
    ValidateLsuPrim(*this);
}

int Lsu_mem_prim::taskCoreDefault(TaskCoreContext &context) {
    if (!context.lsu_memory || !context.sram_regions)
        throw std::runtime_error(
            "Lsu_mem requires memory.sram.real_data_path=true");
    auto resolve = [&]() {
        if (absolute_sram)
            return sram::ResolvedRange{sram_addr, size_bytes, -1};
        const auto command =
            direction == sram::LsuDirection::kHbmToSram
                ? sram::Command::kWrite
                : sram::Command::kRead;
        return context.sram_regions->Resolve(
            sram_region, sram_offset, size_bytes, sram::Initiator::kLsu,
            command);
    };

    switch (op) {
    case LsuMemOp::ISSUE: {
        const auto range = resolve();
        context.lsu_memory->Issue(
            {direction, hbm_addr, range.address, size_bytes, {}}, token);
        break;
    }
    case LsuMemOp::WAIT: context.lsu_memory->Wait(token); break;
    case LsuMemOp::POLL:
        poll_complete = context.lsu_memory->Poll(token);
        break;
    case LsuMemOp::FENCE: context.lsu_memory->Fence(); break;
    case LsuMemOp::CANCEL: context.lsu_memory->Cancel(token); break;
    case LsuMemOp::LOAD_BLOCKING: {
        const auto range = resolve();
        context.lsu_memory->Load(hbm_addr, range.address, size_bytes);
        break;
    }
    case LsuMemOp::STORE_BLOCKING: {
        const auto range = resolve();
        context.lsu_memory->Store(range.address, hbm_addr, size_bytes);
        break;
    }
    }
    return 0;
}

void Lsu_mem_prim::printSelf() {}
