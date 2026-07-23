#include "systemc.h"

#include "defs/enums.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "prims/norm_prims.h"
#include "utils/memory_utils.h"
#include "utils/prim_utils.h"

REGISTER_PRIM(Recv_prim);

void Recv_prim::printSelf() {
}

void Recv_prim::deserialize(vector<sc_bv<128>> segments) {
    auto buffer = segments[0];
    
    type = RECV_TYPE(buffer.range(11, 8).to_uint64());
    tag_id = buffer.range(27, 12).to_uint64();
    recv_cnt = buffer.range(35, 28).to_uint64();
    datatype = DATATYPE(buffer.range(37, 36).to_uint64());
    stripe_count = buffer.range(40, 38).to_uint64();
    if (stripe_count == 0)
        stripe_count = 1;
    if (stripe_count != 1 && stripe_count != 2 && stripe_count != 4)
        throw std::runtime_error("Recv_prim stripe_count must be 1, 2, or 4");
}

vector<sc_bv<128>> Recv_prim::serialize() {
    vector<sc_bv<128>> segments;

    sc_bv<128> d;
    d.range(7, 0) = sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    d.range(11, 8) = sc_bv<4>(type);
    d.range(27, 12) = sc_bv<16>(tag_id);
    d.range(35, 28) = sc_bv<8>(recv_cnt);
    d.range(37, 36) = sc_bv<2>(datatype);
    if (stripe_count != 1 && stripe_count != 2 && stripe_count != 4)
        throw std::runtime_error("Recv_prim stripe_count must be 1, 2, or 4");
    d.range(40, 38) = sc_bv<3>(stripe_count);
    segments.push_back(d);

    return segments;
}

int Recv_prim::taskCoreDefault(TaskCoreContext &context) {
    u_int64_t elapsed_time;
    // sram_write_append_generic(context, M_D_DATA, elapsed_time);

    return 0;
}