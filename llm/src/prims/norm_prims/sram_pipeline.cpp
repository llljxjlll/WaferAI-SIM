#include "prims/norm_prims.h"

#include "common/memory.h"

#include "dte/dte_async.h"
#include "memory/sram/compute_timeline.h"
#include "utils/prim_utils.h"
#include "utils/memory_utils.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

REGISTER_PRIM(Sram_pipeline_prim);

namespace {
constexpr size_t kMaxRegionName = 64;

void Validate(const Sram_pipeline_prim &p) {
    if (p.tile_count == 0 || p.tile_bytes == 0)
        throw std::invalid_argument(
            "Sram_pipeline requires non-zero tile_count and tile_bytes");
    if (p.double_buffer && p.tile_count < 2)
        throw std::invalid_argument(
            "Sram_pipeline double_buffer requires at least two tiles");
    if (p.region_a.empty() || p.region_b.empty() ||
        p.region_a.size() > kMaxRegionName ||
        p.region_b.size() > kMaxRegionName)
        throw std::invalid_argument(
            "Sram_pipeline region names must contain 1..64 bytes");
    if (p.tile_count > UINT64_MAX / p.tile_bytes ||
        p.input_hbm_base > UINT64_MAX - p.tile_count * p.tile_bytes ||
        p.output_hbm_base > UINT64_MAX - p.tile_count * p.tile_bytes)
        throw std::invalid_argument("Sram_pipeline HBM range overflows");
    if (p.engine == SramPipelineEngine::kDte && p.token_base == 0)
        throw std::invalid_argument(
            "Sram_pipeline DTE token_base must be non-zero");
}

std::vector<uint8_t> Pattern(uint32_t tile, uint64_t bytes) {
    std::vector<uint8_t> result(bytes);
    for (uint64_t i = 0; i < bytes; ++i)
        result[i] = static_cast<uint8_t>(0x31u + tile * 17u + i * 3u);
    return result;
}

void AppendText(std::vector<sc_bv<128>> &wire, const std::string &value) {
    for (size_t base = 0; base < value.size(); base += 16) {
        sc_bv<128> text = 0;
        const size_t count = std::min<size_t>(16, value.size() - base);
        for (size_t i = 0; i < count; ++i)
            text.range(static_cast<int>(8 * i + 7),
                       static_cast<int>(8 * i)) =
                sc_bv<8>(static_cast<uint8_t>(value[base + i]));
        wire.push_back(text);
    }
}

std::string ReadText(const std::vector<sc_bv<128>> &wire, size_t &segment,
                     size_t length) {
    std::string result;
    result.reserve(length);
    for (size_t index = 0; index < length; ++index) {
        const auto &text = wire.at(segment + index / 16);
        const int lo = static_cast<int>(8 * (index % 16));
        result.push_back(
            static_cast<char>(text.range(lo + 7, lo).to_uint()));
    }
    segment += (length + 15) / 16;
    return result;
}
} // namespace

void Sram_pipeline_prim::parseJson(json j) {
    const std::string engine_name = j.value("engine", std::string("lsu"));
    if (engine_name == "lsu")
        engine = SramPipelineEngine::kLsu;
    else if (engine_name == "dte")
        engine = SramPipelineEngine::kDte;
    else
        throw std::invalid_argument(
            "Sram_pipeline engine must be 'lsu' or 'dte'");
    const std::string schedule =
        j.value("schedule", std::string("double_buffer"));
    if (schedule == "double_buffer")
        double_buffer = true;
    else if (schedule == "blocking")
        double_buffer = false;
    else
        throw std::invalid_argument(
            "Sram_pipeline schedule must be 'blocking' or 'double_buffer'");
    tile_count = j.value("tile_count", uint32_t{4});
    tile_bytes = j.value("tile_bytes", uint64_t{256});
    compute_cycles = j.value("compute_cycles", uint64_t{32});
    input_hbm_base = j.value("input_hbm_base", uint64_t{0x10000});
    output_hbm_base = j.value("output_hbm_base", uint64_t{0x20000});
    token_base = j.value("token_base", uint32_t{1000});
    transform_xor = j.value("transform_xor", uint8_t{0x5a});
    region_a = j.value("region_a", std::string("double_a"));
    region_b = j.value("region_b", std::string("double_b"));
    Validate(*this);
}

std::vector<sc_bv<128>> Sram_pipeline_prim::serialize() {
    Validate(*this);
    sc_bv<128> metadata = 0;
    metadata.range(7, 0) =
        sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    metadata[8] = engine == SramPipelineEngine::kDte ? sc_dt::SC_LOGIC_1
                                                     : sc_dt::SC_LOGIC_0;
    metadata[9] = double_buffer ? sc_dt::SC_LOGIC_1 : sc_dt::SC_LOGIC_0;
    metadata.range(17, 10) = sc_bv<8>(transform_xor);
    metadata.range(49, 18) = sc_bv<32>(tile_count);
    metadata.range(113, 50) = sc_bv<64>(compute_cycles);
    sc_bv<128> sizes = 0;
    sizes.range(63, 0) = sc_bv<64>(tile_bytes);
    sizes.range(95, 64) = sc_bv<32>(token_base);
    sizes.range(102, 96) = sc_bv<7>(region_a.size());
    sizes.range(109, 103) = sc_bv<7>(region_b.size());
    sc_bv<128> addresses = 0;
    addresses.range(63, 0) = sc_bv<64>(input_hbm_base);
    addresses.range(127, 64) = sc_bv<64>(output_hbm_base);
    std::vector<sc_bv<128>> wire = {metadata, sizes, addresses};
    AppendText(wire, region_a);
    AppendText(wire, region_b);
    return wire;
}

void Sram_pipeline_prim::deserialize(std::vector<sc_bv<128>> wire) {
    if (wire.size() < 3)
        throw std::invalid_argument(
            "Sram_pipeline wire encoding requires at least three segments");
    engine = wire[0][8].to_bool() ? SramPipelineEngine::kDte
                                  : SramPipelineEngine::kLsu;
    double_buffer = wire[0][9].to_bool();
    transform_xor = wire[0].range(17, 10).to_uint();
    tile_count = wire[0].range(49, 18).to_uint64();
    compute_cycles = wire[0].range(113, 50).to_uint64();
    tile_bytes = wire[1].range(63, 0).to_uint64();
    token_base = wire[1].range(95, 64).to_uint64();
    const size_t a_size = wire[1].range(102, 96).to_uint64();
    const size_t b_size = wire[1].range(109, 103).to_uint64();
    input_hbm_base = wire[2].range(63, 0).to_uint64();
    output_hbm_base = wire[2].range(127, 64).to_uint64();
    const size_t expected =
        3 + (a_size + 15) / 16 + (b_size + 15) / 16;
    if (a_size > kMaxRegionName || b_size > kMaxRegionName ||
        wire.size() != expected)
        throw std::invalid_argument(
            "Sram_pipeline wire region-name lengths are inconsistent");
    size_t segment = 3;
    region_a = ReadText(wire, segment, a_size);
    region_b = ReadText(wire, segment, b_size);
    Validate(*this);
}

int Sram_pipeline_prim::taskCoreDefault(TaskCoreContext &context) {
    Validate(*this);
    if (!context.sram_regions || !context.sram_access ||
        !context.compute_timeline)
        throw std::runtime_error(
            "Sram_pipeline requires memory.sram.real_data_path=true");
    if (engine == SramPipelineEngine::kLsu && !context.lsu_memory)
        throw std::runtime_error("Sram_pipeline LSU engine is unavailable");
    if (engine == SramPipelineEngine::kDte && !context.dte_memory)
        throw std::runtime_error("Sram_pipeline DTE engine is unavailable");

    if (prim_context == nullptr ||
        prim_context->sram_pos_locator_ == nullptr)
        throw std::runtime_error(
            "Sram_pipeline requires a production SRAM label locator");

    // Exercise production label binding outside the measured pipeline: a
    // compatibility-word payload is relocated into each role region, read
    // back through the unified access unit, and released through the label
    // lifetime path. This keeps the performance oracle focused on tiling.
    const uint64_t word_bytes = LegacySramWordBytes(context);
    uint64_t probe_word = 0;
    for (const char *role : {"input", "intermediate", "comm"}) {
        std::vector<uint8_t> expected(word_bytes);
        for (uint64_t i = 0; i < word_bytes; ++i)
            expected[i] = static_cast<uint8_t>(0x70u + probe_word + i);
        sram::Request seed;
        seed.initiator = sram::Initiator::kCompute;
        seed.command = sram::Command::kWrite;
        seed.address = probe_word * word_bytes;
        seed.size_bytes = word_bytes;
        seed.payload = expected;
        context.sram_access->Access(seed);

        std::string label = "__sram_pipeline_" + std::string(role);
        AddrPosKey key(static_cast<int>(probe_word),
                       static_cast<int>(word_bytes));
        key.preferred_region = role;
        uint64_t label_time = 0;
        prim_context->sram_pos_locator_->addPair(
            label, key, context, label_time);
        if (prim_context->sram_pos_locator_->findPair(label, key) < 0 ||
            key.region_allocation_id == 0 ||
            context.sram_regions->Region(key.region_id).name != role)
            throw std::runtime_error(
                "Sram_pipeline production label region binding failed");

        sram::Request verify;
        verify.initiator = sram::Initiator::kCompute;
        verify.command = sram::Command::kRead;
        verify.address =
            static_cast<uint64_t>(key.pos) * word_bytes;
        verify.size_bytes = word_bytes;
        if (context.sram_access->Access(verify).payload != expected)
            throw std::runtime_error(
                "Sram_pipeline production label relocation failed");
        prim_context->sram_pos_locator_->deletePair(label);
        ++probe_word;
    }
    std::cout << "[SRAM_REGION_BIND_DONE] roles=input,intermediate,comm"
              << std::endl;

    auto slot_name = [&](uint32_t tile) -> const std::string & {
        return (double_buffer && (tile & 1u)) ? region_b : region_a;
    };
    auto slot = [&](uint32_t tile, sram::Initiator initiator,
                    sram::Command command) {
        return context.sram_regions->Resolve(slot_name(tile), 0, tile_bytes,
                                             initiator, command);
    };
    uint32_t next_dte_token = token_base;
    auto issue = [&](bool load, uint32_t tile, uint64_t hbm_addr) -> uint64_t {
        const auto range = slot(
            tile,
            engine == SramPipelineEngine::kDte ? sram::Initiator::kDte
                                                : sram::Initiator::kLsu,
            load ? sram::Command::kWrite : sram::Command::kRead);
        if (engine == SramPipelineEngine::kLsu)
            return load ? context.lsu_memory->IssueLoad(
                              hbm_addr, range.address, tile_bytes)
                        : context.lsu_memory->IssueStore(
                              range.address, hbm_addr, tile_bytes);
        const uint32_t token = next_dte_token++;
        context.dte_memory->IssueToken(
            token, tile_bytes * 8,
            load ? DteDir::DRAM_TO_SPM : DteDir::SPM_TO_DRAM,
            range.address, tile_bytes, DTE_ASYNC_INVALID_REMOTE_PEER,
            hbm_addr, 0);
        return token;
    };
    auto wait_token = [&](uint64_t token) {
        if (engine == SramPipelineEngine::kLsu)
            context.lsu_memory->Wait(token);
        else
            context.dte_memory->WaitToken(static_cast<uint32_t>(token));
    };
    auto write_slot = [&](uint32_t tile,
                          const std::vector<uint8_t> &payload) {
        const auto range = slot(tile, sram::Initiator::kCompute,
                                sram::Command::kWrite);
        sram::Request request;
        request.initiator = sram::Initiator::kCompute;
        request.command = sram::Command::kWrite;
        request.address = range.address;
        request.size_bytes = tile_bytes;
        request.payload = payload;
        context.sram_access->Access(request);
    };
    auto read_slot = [&](uint32_t tile) {
        const auto range = slot(tile, sram::Initiator::kCompute,
                                sram::Command::kRead);
        sram::Request request;
        request.initiator = sram::Initiator::kCompute;
        request.command = sram::Command::kRead;
        request.address = range.address;
        request.size_bytes = tile_bytes;
        return context.sram_access->Access(request).payload;
    };

    // Seed HBM through the selected production mover, not a fake-memory API.
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        write_slot(tile, Pattern(tile, tile_bytes));
        const auto token =
            issue(false, tile, input_hbm_base + tile * tile_bytes);
        wait_token(token);
    }

    const sc_time measured_begin = sc_time_stamp();
    uint64_t checksum = 0;
    if (!double_buffer) {
        for (uint32_t tile = 0; tile < tile_count; ++tile) {
            auto token =
                issue(true, tile, input_hbm_base + tile * tile_bytes);
            wait_token(token);
            const auto range = slot(tile, sram::Initiator::kCompute,
                                    sram::Command::kRead);
            auto payload =
                context.compute_timeline->RunTile(range, compute_cycles);
            if (payload != Pattern(tile, tile_bytes))
                throw std::runtime_error(
                    "Sram_pipeline blocking load payload mismatch");
            for (auto &byte : payload) {
                byte ^= transform_xor;
                checksum += byte;
            }
            write_slot(tile, payload);
            token = issue(false, tile,
                          output_hbm_base + tile * tile_bytes);
            wait_token(token);
        }
    } else {
        std::vector<uint64_t> stores(2, 0);
        uint64_t load = issue(true, 0, input_hbm_base);
        for (uint32_t tile = 0; tile < tile_count; ++tile) {
            wait_token(load);
            if (tile + 1 < tile_count) {
                const uint32_t next_slot = (tile + 1) & 1u;
                if (stores[next_slot] != 0) {
                    wait_token(stores[next_slot]);
                    stores[next_slot] = 0;
                }
                load = issue(true, tile + 1,
                             input_hbm_base + (tile + 1) * tile_bytes);
            }
            const auto range = slot(tile, sram::Initiator::kCompute,
                                    sram::Command::kRead);
            auto payload =
                context.compute_timeline->RunTile(range, compute_cycles);
            if (payload != Pattern(tile, tile_bytes))
                throw std::runtime_error(
                    "Sram_pipeline double-buffer load payload mismatch");
            for (auto &byte : payload) {
                byte ^= transform_xor;
                checksum += byte;
            }
            write_slot(tile, payload);
            stores[tile & 1u] = issue(
                false, tile, output_hbm_base + tile * tile_bytes);
        }
        for (auto token : stores)
            if (token != 0) wait_token(token);
    }
    const sc_time measured_time = sc_time_stamp() - measured_begin;

    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        const auto token =
            issue(true, tile, output_hbm_base + tile * tile_bytes);
        wait_token(token);
        auto expected = Pattern(tile, tile_bytes);
        for (auto &byte : expected) byte ^= transform_xor;
        if (read_slot(tile) != expected)
            throw std::runtime_error(
                "Sram_pipeline HBM writeback verification failed");
    }
    std::cout << "[SRAM_PIPELINE_DONE] engine="
              << (engine == SramPipelineEngine::kLsu ? "lsu" : "dte")
              << " schedule="
              << (double_buffer ? "double_buffer" : "blocking")
              << " tiles=" << tile_count << " bytes=" << tile_bytes
              << " checksum=" << checksum
              << " measured_ns=" << measured_time.to_seconds() * 1.0e9
              << std::endl;
    return 0;
}

void Sram_pipeline_prim::printSelf() {}
