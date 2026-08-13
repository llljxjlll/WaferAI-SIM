#include "prims/norm_prims.h"

#include "defs/const.h"
#include "dte/coll_codec.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"

#include <iostream>
#include <limits>
#include <stdexcept>

REGISTER_PRIM(Reduce_compute_prim, PrimId::REDUCE_COMPUTE);

vector<sc_bv<128>> Reduce_compute_prim::serialize() {
    auto wire = SerializeCollDescriptor(descriptor);
    sc_bv<128> marker = 0;
    marker.range(7, 0) =
        sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    marker.range(23, 8) = static_cast<uint16_t>(descriptor.group.size());
    wire.insert(wire.begin(), marker);
    return prim_wire::WrapSegments(std::move(wire), name);
}

void Reduce_compute_prim::deserialize(vector<sc_bv<128>> segments) {
    segments = prim_wire::UnwrapSegments(segments, name);
    if (segments.size() < 2)
        throw std::invalid_argument("Reduce_compute_prim wire is truncated");
    if (segments.front().range(127, 24).or_reduce())
        throw std::invalid_argument(
            "Reduce_compute_prim marker reserved bits are non-zero");
    const uint16_t group_size = segments.front().range(23, 8).to_uint();
    segments.erase(segments.begin());
    descriptor = DeserializeCollDescriptor(segments);
    if (group_size != descriptor.group.size())
        throw std::invalid_argument("Reduce_compute_prim group size mismatch");
    ValidateTier0ReductionDescriptor(descriptor);
}

int Reduce_compute_prim::taskCoreDefault(TaskCoreContext &context) {
    ValidateTier0ReductionDescriptor(descriptor);
    if (descriptor.self_rank != descriptor.root_rank)
        throw std::runtime_error("Reduce compute can only execute at root");
    const CoreHWConfig *hardware = GetCoreHWConfig(context.cid);
    if (!hardware || !hardware->vec || hardware->vec->x_dims <= 0 ||
        hardware->vec->count <= 0)
        throw std::runtime_error("Reduce compute requires a valid vector unit");
    const uint64_t lanes = uint64_t(hardware->vec->x_dims) *
                           uint64_t(hardware->vec->count);
    const uint64_t peers = descriptor.group.size() - 1;
    const uint64_t cycles = CollTier0ReduceComputeCycles(descriptor, lanes);
    const uint64_t operations = descriptor.count * peers;
    if (cycles > uint64_t(std::numeric_limits<int>::max()) / CYCLE)
        throw std::overflow_error("Reduce compute delay overflows int");
    std::cout << "[COLL_REDUCE_COMPUTE] op="
              << static_cast<unsigned>(descriptor.reduce_op)
              << " dtype=" << static_cast<unsigned>(descriptor.dtype)
              << " elements=" << descriptor.count
              << " operations=" << operations << " cycles=" << cycles
              << " lanes=" << lanes
              << std::endl;
    return static_cast<int>(cycles * CYCLE);
}

void Reduce_compute_prim::printSelf() {}
