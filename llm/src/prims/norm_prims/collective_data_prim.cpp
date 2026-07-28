#include "prims/norm_prims.h"
#include "dte/coll_codec.h"
#include "utils/prim_utils.h"
#include <stdexcept>

REGISTER_PRIM(Collective_data_prim);

std::vector<sc_bv<128>> Collective_data_prim::serialize() {
    if (tree_id == 0) throw std::invalid_argument("collective data tree_id is zero");
    auto encoded = SerializeCollDescriptor(descriptor);
    sc_bv<128> marker = 0;
    marker.range(7, 0) = PrimFactory::getInstance().getPrimId(name);
    marker.range(23, 8) = tree_id;
    marker.range(31, 24) = static_cast<uint8_t>(mode);
    encoded.insert(encoded.begin(), marker);
    return encoded;
}

void Collective_data_prim::deserialize(std::vector<sc_bv<128>> segments) {
    if (segments.size() < 2) throw std::invalid_argument("collective data primitive is truncated");
    tree_id = segments.front().range(23, 8).to_uint();
    const unsigned encoded_mode = segments.front().range(31, 24).to_uint();
    if (tree_id == 0 || encoded_mode > static_cast<unsigned>(Mode::REDUCE_RX))
        throw std::invalid_argument("collective data primitive marker is invalid");
    mode = static_cast<Mode>(encoded_mode);
    segments.erase(segments.begin());
    descriptor = DeserializeCollDescriptor(segments);
}

int Collective_data_prim::taskCoreDefault(TaskCoreContext &) { return 0; }
void Collective_data_prim::printSelf() {}
