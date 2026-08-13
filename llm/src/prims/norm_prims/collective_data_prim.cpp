#include "prims/norm_prims.h"
#include "dte/coll_codec.h"
#include "utils/prim_utils.h"
#include <stdexcept>

REGISTER_PRIM(Collective_data_prim, PrimId::COLLECTIVE_DATA);

std::vector<sc_bv<128>> Collective_data_prim::serialize() {
    if (tree_id == 0) throw std::invalid_argument("collective data tree_id is zero");
    const bool core_mode = mode == Mode::CORE_VECTOR_START ||
                           mode == Mode::CORE_VECTOR_WAIT;
    if (core_mode != (core_vector_beats != 0))
        throw std::invalid_argument(
            "collective core-vector marker/count mismatch");
    auto encoded = SerializeCollDescriptor(descriptor);
    sc_bv<128> marker = 0;
    marker.range(7, 0) = PrimFactory::getInstance().getPrimId(name);
    marker.range(23, 8) = tree_id;
    marker.range(31, 24) = static_cast<uint8_t>(mode);
    marker.range(63, 32) = core_vector_beats;
    encoded.insert(encoded.begin(), marker);
    return prim_wire::WrapSegments(std::move(encoded), name);
}

void Collective_data_prim::deserialize(std::vector<sc_bv<128>> segments) {
    segments = prim_wire::UnwrapSegments(segments, name);
    if (segments.size() < 2) throw std::invalid_argument("collective data primitive is truncated");
    if (segments.front().range(127, 64).or_reduce())
        throw std::invalid_argument(
            "collective data marker reserved bits are non-zero");
    tree_id = segments.front().range(23, 8).to_uint();
    const unsigned encoded_mode = segments.front().range(31, 24).to_uint();
    core_vector_beats = segments.front().range(63, 32).to_uint64();
    if (tree_id == 0 || encoded_mode >
            static_cast<unsigned>(Mode::CORE_VECTOR_WAIT))
        throw std::invalid_argument("collective data primitive marker is invalid");
    mode = static_cast<Mode>(encoded_mode);
    const bool core_mode = mode == Mode::CORE_VECTOR_START ||
                           mode == Mode::CORE_VECTOR_WAIT;
    if (core_mode != (core_vector_beats != 0))
        throw std::invalid_argument(
            "collective core-vector marker/count mismatch");
    segments.erase(segments.begin());
    descriptor = DeserializeCollDescriptor(segments);
}

int Collective_data_prim::taskCoreDefault(TaskCoreContext &) { return 0; }
void Collective_data_prim::printSelf() {}
