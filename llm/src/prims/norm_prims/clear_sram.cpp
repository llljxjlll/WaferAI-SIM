#include "systemc.h"
#include <string>

#include "common/memory.h"
#include "memory/sram/sram_access_unit.h"
#include "memory/sram/sram_region.h"
#include "prims/base.h"
#include "prims/norm_prims.h"
#include "utils/prim_utils.h"
#include "utils/memory_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(Clear_sram);

void Clear_sram::printSelf() {}

void Clear_sram::deserialize(vector<sc_bv<128>> segments) {
    auto buffer = segments[0];
}

vector<sc_bv<128>> Clear_sram::serialize() {
    vector<sc_bv<128>> segments;

    sc_bv<128> d;
    d.range(7, 0) = sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    segments.push_back(d);

    return segments;
}

int Clear_sram::taskCoreDefault(TaskCoreContext &context) {
    if (prim_context == nullptr)
        throw std::runtime_error("Clear_sram requires primitive context");
    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                    << " SRAM top address: " << *(context.sram_addr);

    if (context.sram_access != nullptr &&
        context.sram_regions != nullptr &&
        context.sram_storage != nullptr) {
        if (prim_context->sram_pos_locator_ == nullptr)
            throw std::runtime_error(
                "real Clear_sram requires an SRAM label table");
        std::vector<std::string> erase_labels;
        uint64_t high_water = 0;
        for (const auto &[label, key] :
             prim_context->sram_pos_locator_->data_map) {
            if (key.size <= 0 || key.pos < 0) continue;
            const uint64_t byte_address = LegacySramByteAddress(
                context, static_cast<uint64_t>(key.pos));
            const auto resolved = context.sram_regions->LocateAbsolute(
                byte_address, static_cast<uint64_t>(key.size));
            const auto &region =
                context.sram_regions->Region(resolved.region_id);
            const bool persistent_label =
                label.rfind(ETERNAL_PREFIX, 0) == 0 ||
                key.allocation_lifetime !=
                    sram::AllocationLifetime::kTask;
            if (!region.spillable || persistent_label) {
                high_water = std::max<uint64_t>(
                    high_water, resolved.address + resolved.size_bytes);
                continue;
            }
            if (key.valid) {
                sram::Request clear;
                clear.initiator = sram::Initiator::kCompute;
                clear.command = sram::Command::kClear;
                clear.address = resolved.address;
                clear.size_bytes = resolved.size_bytes;
                context.sram_access->Access(clear);
            }
            if (key.region_allocation_id != 0)
                context.sram_regions->Free(
                    key.region_allocation_id,
                    sram::AllocationLifetime::kTask);
            erase_labels.push_back(label);
        }
        for (const auto &label : erase_labels)
            prim_context->sram_pos_locator_->data_map.erase(label);
        *(context.sram_addr) = static_cast<int>(
            LegacySramWordAddressCeil(context, high_water));
        return 0;
    }
#if USE_SRAM_MANAGER == 0
    vector<pair<string, AddrPosKey>> temp_list;

    for (auto record : prim_context->sram_pos_locator_->data_map) {
        if (!record.second.valid)
            continue;

        // clear output last layer in core and reuse input
        const bool flag = record.first.rfind(ETERNAL_PREFIX, 0) == 0;

        if (flag) {
            temp_list.push_back(record);
        }
    }

    prim_context->sram_pos_locator_->clearAll();
    int pos = 0;
    for (auto record : temp_list) {
        auto size = record.second.size;
        int dma_read_count =
            size * 8 /
            (GetCoreHWConfig(prim_context->cid)->sram_bitwidth * SRAM_BANKS);
        int byte_residue =
            size * 8 - dma_read_count *
                           (GetCoreHWConfig(prim_context->cid)->sram_bitwidth *
                            SRAM_BANKS);
        int single_read_count = CeilingDivision(
            byte_residue, GetCoreHWConfig(prim_context->cid)->sram_bitwidth);

        AddrPosKey temp_key = AddrPosKey(pos, size);
        u_int64_t temp_addr = 0;
        prim_context->sram_pos_locator_->addPair(record.first, temp_key,
                                                 context, temp_addr);

        pos += dma_read_count * SRAM_BANKS + single_read_count;
    }

    *(context.sram_addr) = pos;
    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                    << " SRAM top address: " << pos;
#endif

    return 0;
}