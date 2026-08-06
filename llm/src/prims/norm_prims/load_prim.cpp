#include "systemc.h"

#include "prims/base.h"
#include "prims/norm_prims.h"
#include "utils/prim_utils.h"
#include <stdexcept>

REGISTER_PRIM(Load_prim);

void Load_prim::printSelf() {  }

void Load_prim::deserialize(vector<sc_bv<128>> segments) {
    if (segments.empty())
        throw std::invalid_argument("Load_prim requires one segment");
    auto buffer = segments[0];
    dram_addr = buffer.range(23, 8).to_uint64();
    sram_addr = buffer.range(39, 24).to_uint64();
    size = buffer.range(55, 40).to_uint64();
    datatype = (DATATYPE)buffer.range(57, 56).to_uint64();
}

vector<sc_bv<128>> Load_prim::serialize() {
    vector<sc_bv<128>> segments;

    sc_bv<128> d;
    d.range(7, 0) = sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    d.range(23, 8) = sc_bv<16>(dram_addr);
    d.range(39, 24) = sc_bv<16>(sram_addr);
    d.range(55, 40) = sc_bv<16>(size);
    d.range(57, 56) = sc_bv<2>(datatype);
    segments.push_back(d);

    return segments;
}
int Load_prim::taskCoreDefault(TaskCoreContext &context) {
    if (size == 0) return 0;
    if (!context.lsu_memory)
        throw std::runtime_error(
            "Load_prim transfer requires memory.sram.real_data_path=true");
    context.lsu_memory->Load(dram_addr, sram_addr, size);
    return 0;
}