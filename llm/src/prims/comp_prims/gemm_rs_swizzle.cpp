#include "prims/comp_prims.h"

#include "memory/sram/compute_timeline.h"
#include "memory/sram/sram_access_unit.h"
#include "memory/sram/sram_region.h"
#include "utils/memory_utils.h"
#include "utils/prim_utils.h"

#include <iostream>
#include <numeric>
#include <stdexcept>

REGISTER_PRIM(Gemm_rs_swizzle);

namespace {
double NowNs() { return sc_time_stamp().to_seconds() * 1.0e9; }

std::vector<uint8_t> InputPattern(int core, uint32_t chunk, uint64_t bytes) {
    std::vector<uint8_t> value(bytes);
    for (uint64_t i = 0; i < bytes; ++i)
        value[i] = static_cast<uint8_t>(0x21u + core * 19u + chunk * 13u +
                                        i * 7u);
    return value;
}

std::vector<uint8_t> WeightPattern(int core, uint64_t bytes) {
    std::vector<uint8_t> value(bytes);
    for (uint64_t i = 0; i < bytes; ++i)
        value[i] = static_cast<uint8_t>(0x43u + core * 11u + i * 5u);
    return value;
}

uint64_t Checksum(const std::vector<uint8_t> &value) {
    return std::accumulate(value.begin(), value.end(), uint64_t{0});
}

void Write(sram::AccessUnit &unit, uint64_t address,
           const std::vector<uint8_t> &payload) {
    sram::Request request;
    request.initiator = sram::Initiator::kCompute;
    request.command = sram::Command::kWrite;
    request.address = address;
    request.size_bytes = payload.size();
    request.payload = payload;
    unit.Access(request);
}

std::vector<uint8_t> Read(sram::AccessUnit &unit, uint64_t address,
                          uint64_t bytes) {
    sram::Request request;
    request.initiator = sram::Initiator::kCompute;
    request.command = sram::Command::kRead;
    request.address = address;
    request.size_bytes = bytes;
    return unit.Access(request).payload;
}
} // namespace

void Gemm_rs_swizzle::initialize() {
    const auto &p = param_value;
    const int tile = p.at("tile_bytes");
    const int comm = p.at("comm_bytes");
    if ((p.at("mode") < 0 || p.at("mode") > 3) || tile <= 0 ||
        comm <= 0 || comm > tile || p.at("participants") < 2)
        throw std::invalid_argument(
            "Gemm_rs_swizzle has invalid mode/tile/comm/participants");
    data_size_input = {tile};
    data_chunk = {{"output", comm}};
}

void Gemm_rs_swizzle::taskCore(TaskCoreContext &context, string,
                               u_int64_t &dram_time, u_int64_t &exu_ops,
                               u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    (void)dram_time;
    exu_ops = sfu_ops = vec_ops = 0;
    if (!context.sram_regions || !context.sram_access ||
        !context.compute_timeline || !context.lsu_memory)
        throw std::runtime_error(
            "Gemm_rs_swizzle requires real SRAM and the production LSU");
    if (!context.sram_regions->config().manual_memory_schedule ||
        !context.sram_regions->config().manual_regions)
        throw std::runtime_error(
            "Gemm_rs_swizzle requires manual_regions and manual_memory_schedule");
    if (!prim_context || !prim_context->sram_pos_locator_)
        throw std::runtime_error(
            "Gemm_rs_swizzle requires the production SRAM label locator");

    const auto &p = param_value;
    const uint32_t chunk = static_cast<uint32_t>(p.at("chunk"));
    const uint64_t tile_bytes = static_cast<uint64_t>(p.at("tile_bytes"));
    const uint64_t comm_bytes = static_cast<uint64_t>(p.at("comm_bytes"));
    const uint32_t participants = static_cast<uint32_t>(p.at("participants"));
    if (chunk >= participants)
        throw std::invalid_argument("Gemm_rs_swizzle chunk is out of range");
    if (p.at("mode") == 3) {
        const double begin = NowNs();
        context.compute_timeline->RunCycles(p.at("compute_cycles"));
        std::cout << "[GEMM_RS_OWNER_WINDOW] core=" << context.cid
                  << " chunk=" << chunk << " wait_ns=" << (NowNs() - begin)
                  << std::endl;
        return;
    }
    auto resolve = [&](const char *region, uint64_t offset, uint64_t bytes,
                       sram::Command command) {
        return context.sram_regions->Resolve(region, offset, bytes,
                                             sram::Initiator::kCompute,
                                             command);
    };

    if (p.at("mode") == 2) {
        auto terminal = resolve("scratch", 0, comm_bytes,
                                sram::Command::kWrite);
        std::vector<uint8_t> payload(comm_bytes);
        for (uint64_t i = 0; i < comm_bytes; ++i)
            payload[i] = static_cast<uint8_t>(0xd0u + context.cid + i);
        Write(*context.sram_access, terminal.address, payload);
        const uint64_t word_bytes = LegacySramWordBytes(context);
        if (terminal.address % word_bytes != 0)
            throw std::runtime_error(
                "Gemm_rs_swizzle terminal slot is not word aligned");
        std::string label = prim_context->datapass_label_->outdata;
        AddrPosKey key(static_cast<int>(terminal.address / word_bytes),
                       static_cast<int>(comm_bytes));
        uint64_t label_time = 0;
        prim_context->sram_pos_locator_->addPair(label, key, context,
                                                 label_time);
        std::cout << "[GEMM_RS_TERMINAL] core=" << context.cid
                  << " checksum=" << Checksum(payload)
                  << " done_ns=" << NowNs() << std::endl;
        return;
    }

    if (p.at("mode") == 1) {
        const uint64_t remote_bytes = (participants - 1) * comm_bytes;
        auto recv = resolve("rs_recv", 0, remote_bytes,
                            sram::Command::kWrite);
        std::vector<uint8_t> remote(remote_bytes);
        for (uint64_t i = 0; i < remote_bytes; ++i)
            remote[i] = static_cast<uint8_t>(0x91u + context.cid * 7u + i);
        const double begin = NowNs();
        Write(*context.sram_access, recv.address, remote);
        recv = resolve("rs_recv", 0, remote_bytes, sram::Command::kRead);
        auto reduced =
            context.compute_timeline->RunTile(recv, p.at("reduce_cycles"));
        auto local = resolve("gemm_accum", chunk * tile_bytes, comm_bytes,
                             sram::Command::kRead);
        const auto local_value =
            Read(*context.sram_access, local.address, comm_bytes);
        std::vector<uint8_t> final(comm_bytes, 0);
        for (uint64_t i = 0; i < comm_bytes; ++i) {
            uint32_t sum = local_value[i];
            for (uint32_t rank = 0; rank + 1 < participants; ++rank)
                sum += reduced[rank * comm_bytes + i];
            final[i] = static_cast<uint8_t>(sum);
        }
        auto scratch = resolve("scratch", 0, comm_bytes,
                               sram::Command::kWrite);
        Write(*context.sram_access, scratch.address, final);
        const double done = NowNs();
        std::cout << "[GEMM_RS_REDUCE] core=" << context.cid
                  << " chunk=" << chunk << " contributors=" << participants
                  << " reduce_ns=" << (done - begin)
                  << " checksum=" << Checksum(final)
                  << " done_ns=" << done << std::endl;
        return;
    }

    const uint64_t hbm_base = static_cast<uint64_t>(p.at("hbm_base"));
    const char *input_name = (chunk & 1u) ? "gemm_input_b" : "gemm_input_a";
    auto input = resolve(input_name, 0, tile_bytes, sram::Command::kRead);
    const auto expected_input = InputPattern(context.cid, chunk, tile_bytes);
    double store_begin = NowNs();
    double store_done = store_begin;
    double current_load_ns = 0.0;
    uint64_t token = 0;

    if (chunk == 0) {
        store_begin = NowNs();
        for (uint32_t tile = 0; tile < participants; ++tile) {
            const char *seed_name =
                (tile & 1u) ? "gemm_input_b" : "gemm_input_a";
            auto seed = resolve(seed_name, 0, tile_bytes,
                                sram::Command::kWrite);
            Write(*context.sram_access, seed.address,
                  InputPattern(context.cid, tile, tile_bytes));
            token = context.lsu_memory->IssueStore(
                seed.address, hbm_base + tile * tile_bytes, tile_bytes);
            context.lsu_memory->Wait(token);
        }
        store_done = NowNs();
        input = resolve("gemm_input_a", 0, tile_bytes,
                        sram::Command::kWrite);
        const double load_begin = NowNs();
        token = context.lsu_memory->IssueLoad(hbm_base, input.address,
                                              tile_bytes);
        context.lsu_memory->Wait(token);
        current_load_ns = NowNs() - load_begin;
    }

    input = resolve(input_name, 0, tile_bytes, sram::Command::kRead);
    if (Read(*context.sram_access, input.address, tile_bytes) != expected_input)
        throw std::runtime_error("Gemm_rs_swizzle prefetched input mismatch");

    double weight_ns = 0.0;
    if (chunk == 0) {
        auto weight_seed = resolve("gemm_weight_b", 0, tile_bytes,
                                   sram::Command::kWrite);
        const auto expected_weight = WeightPattern(context.cid, tile_bytes);
        const double weight_begin = NowNs();
        Write(*context.sram_access, weight_seed.address, expected_weight);
        token = context.lsu_memory->IssueStore(
            weight_seed.address, hbm_base + 0x8000, tile_bytes);
        context.lsu_memory->Wait(token);
        Write(*context.sram_access, weight_seed.address,
              std::vector<uint8_t>(tile_bytes, 0));
        token = context.lsu_memory->IssueLoad(hbm_base + 0x8000,
                                              weight_seed.address, tile_bytes);
        context.lsu_memory->Wait(token);
        if (Read(*context.sram_access, weight_seed.address, tile_bytes) !=
            expected_weight)
            throw std::runtime_error(
                "Gemm_rs_swizzle weight HBM round-trip failed");
        weight_ns = NowNs() - weight_begin;
    }

    uint64_t prefetch_token = 0;
    double prefetch_begin = NowNs();
    if (chunk + 1 < participants) {
        const char *next_name =
            ((chunk + 1) & 1u) ? "gemm_input_b" : "gemm_input_a";
        auto next = resolve(next_name, 0, tile_bytes, sram::Command::kWrite);
        prefetch_begin = NowNs();
        prefetch_token = context.lsu_memory->IssueLoad(
            hbm_base + (chunk + 1) * tile_bytes, next.address, tile_bytes);
    }
    const double gemm_begin = NowNs();
    auto output =
        context.compute_timeline->RunTile(input, p.at("compute_cycles"));
    auto weight = resolve("gemm_weight_b", 0, tile_bytes,
                          sram::Command::kRead);
    const auto weights = Read(*context.sram_access, weight.address, tile_bytes);
    for (uint64_t i = 0; i < tile_bytes; ++i)
        output[i] = static_cast<uint8_t>(output[i] + weights[i] + chunk);
    auto accum = resolve("gemm_accum", chunk * tile_bytes, tile_bytes,
                         sram::Command::kWrite);
    Write(*context.sram_access, accum.address, output);
    const double gemm_done = NowNs();
    double prefetch_ns = 0.0;
    if (prefetch_token != 0) {
        context.lsu_memory->Wait(prefetch_token);
        prefetch_ns = NowNs() - prefetch_begin;
        const char *next_name =
            ((chunk + 1) & 1u) ? "gemm_input_b" : "gemm_input_a";
        auto next = resolve(next_name, 0, tile_bytes, sram::Command::kRead);
        if (Read(*context.sram_access, next.address, tile_bytes) !=
            InputPattern(context.cid, chunk + 1, tile_bytes))
            throw std::runtime_error(
                "Gemm_rs_swizzle asynchronous prefetch mismatch");
    }

    std::vector<uint8_t> swizzled(comm_bytes);
    for (uint64_t i = 0; i < comm_bytes; ++i)
        swizzled[i] = output[(i * 17u + chunk * 29u) % tile_bytes];
    const char *slot_name = (chunk & 1u) ? "rs_send_b" : "rs_send_a";
    auto slot = resolve(slot_name, 0, comm_bytes, sram::Command::kWrite);
    const double swizzle_begin = NowNs();
    Write(*context.sram_access, slot.address, swizzled);

    const uint64_t word_bytes = LegacySramWordBytes(context);
    if (slot.address % word_bytes != 0)
        throw std::runtime_error(
            "Gemm_rs_swizzle send slot is not legacy-word aligned");
    std::string output_label = prim_context->datapass_label_->outdata;
    AddrPosKey key(static_cast<int>(slot.address / word_bytes),
                   static_cast<int>(comm_bytes));
    uint64_t label_time = 0;
    prim_context->sram_pos_locator_->addPair(output_label, key, context,
                                             label_time);
    AddrPosKey check;
    if (prim_context->sram_pos_locator_->findPair(output_label, check) < 0 ||
        static_cast<uint64_t>(check.pos) * word_bytes != slot.address)
        throw std::runtime_error(
            "Gemm_rs_swizzle output label did not bind to its send slot");
    const double swizzle_done = NowNs();

    std::cout << "[GEMM_RS_TILE] core=" << context.cid
              << " chunk=" << chunk << " owner=" << chunk
              << " slot=" << slot_name
              << " hbm_store_ns=" << (store_done - store_begin)
              << " hbm_load_ns=" << (current_load_ns + prefetch_ns)
              << " weight_ns=" << weight_ns
              << " gemm_ns=" << (gemm_done - gemm_begin)
              << " swizzle_ns=" << (swizzle_done - swizzle_begin)
              << " checksum=" << Checksum(swizzled)
              << " ready_ns=" << swizzle_done << std::endl;
}
