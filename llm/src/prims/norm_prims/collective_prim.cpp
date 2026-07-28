#include "prims/norm_prims.h"
#include "dte/coll_codec.h"
#include "dte/coll_runtime.h"
#include "utils/prim_utils.h"

#include <stdexcept>
#include <string>

REGISTER_PRIM(Collective_prim);

namespace {
CollOp ParseCollOp(const std::string &v) {
    if (v == "p2p") return CollOp::P2P;
    if (v == "scatter") return CollOp::SCATTER;
    if (v == "gather") return CollOp::GATHER;
    if (v == "broadcast") return CollOp::BROADCAST;
    if (v == "alltoall") return CollOp::ALLTOALL;
    if (v == "allgather") return CollOp::ALLGATHER;
    if (v == "reduce") return CollOp::REDUCE;
    if (v == "reducescatter") return CollOp::REDUCESCATTER;
    if (v == "allreduce") return CollOp::ALLREDUCE;
    throw std::invalid_argument("unknown collective op: " + v);
}
CollDType ParseCollDType(const std::string &v) {
    if (v == "uint8") return CollDType::UINT8;
    if (v == "int32") return CollDType::INT32;
    if (v == "int64") return CollDType::INT64;
    if (v == "fp32") return CollDType::FP32;
    throw std::invalid_argument("unknown collective dtype: " + v);
}
CollReduceOp ParseReduceOp(const std::string &v) {
    if (v == "none") return CollReduceOp::NONE;
    if (v == "sum") return CollReduceOp::SUM;
    if (v == "max") return CollReduceOp::MAX;
    throw std::invalid_argument("unknown collective reduce_op: " + v);
}
CollAlgorithm DefaultAlgorithm(CollOp op) {
    if (op == CollOp::REDUCESCATTER) return CollAlgorithm::REDUCE_ROOT_SCATTER;
    if (op == CollOp::ALLREDUCE) return CollAlgorithm::REDUCE_ROOT_BROADCAST;
    return CollAlgorithm::DIRECT;
}
}

void Collective_prim::parseJson(json j) {
    descriptor.op = ParseCollOp(j.at("op").get<std::string>());
    descriptor.algorithm = DefaultAlgorithm(descriptor.op);
    descriptor.dtype = ParseCollDType(j.value("dtype", std::string("uint8")));
    descriptor.reduce_op = ParseReduceOp(j.value("reduce_op", std::string("none")));
    descriptor.key.group_id = j.at("group_id").get<uint32_t>();
    descriptor.key.collective_id = j.at("collective_id").get<uint32_t>();
    descriptor.key.epoch = j.value("epoch", uint32_t(0));
    descriptor.group = j.at("group").get<std::vector<uint16_t>>();
    descriptor.root_rank = j.value("root_rank", uint16_t(0));
    descriptor.self_rank = j.at("self_rank").get<uint16_t>();
    descriptor.count = j.at("count").get<uint64_t>();
    descriptor.chunk_bits = j.at("chunk_bits").get<uint64_t>();
    descriptor.stride_bits = j.value("stride_bits", descriptor.chunk_bits);
    descriptor.src_addr = j.value("src_addr", uint64_t(0));
    descriptor.dst_addr = j.value("dst_addr", uint64_t(0));
    descriptor.gather_reorder_depth = j.value("gather_reorder_depth", uint32_t(0));
    phase_id = j.value("phase_id", uint16_t(0));
    ValidateCollDescriptor(descriptor);
}

vector<sc_bv<128>> Collective_prim::serialize() {
    auto encoded = SerializeCollDescriptor(descriptor);
    sc_bv<128> marker = 0;
    marker.range(7, 0) = sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    marker.range(23, 8) = phase_id;
    marker.range(39, 24) = static_cast<uint16_t>(descriptor.group.size());
    marker.range(47, 40) = static_cast<uint8_t>(marker_kind);
    marker.range(63, 48) = release_tree_id;
    encoded.insert(encoded.begin(), marker);
    return encoded;
}

void Collective_prim::deserialize(vector<sc_bv<128>> segments) {
    if (segments.size() < 2) throw std::invalid_argument("Collective_prim wire is truncated");
    phase_id = segments.front().range(23, 8).to_uint();
    const uint16_t encoded_group_size = segments.front().range(39, 24).to_uint();
    const uint8_t encoded_kind = segments.front().range(47, 40).to_uint();
    release_tree_id = segments.front().range(63, 48).to_uint();
    if (encoded_kind > static_cast<uint8_t>(MarkerKind::REDUCE_ARRIVAL))
        throw std::invalid_argument("Collective_prim marker kind is invalid");
    marker_kind = static_cast<MarkerKind>(encoded_kind);
    if (marker_kind != MarkerKind::BARRIER && release_tree_id != 0)
        throw std::invalid_argument(
            "only a collective barrier may release a tree");
    segments.erase(segments.begin());
    descriptor = DeserializeCollDescriptor(segments);
    if (encoded_group_size != descriptor.group.size())
        throw std::invalid_argument("Collective_prim marker group size mismatch");
}

int Collective_prim::taskCoreDefault(TaskCoreContext &) {
    if (marker_kind == MarkerKind::GATHER_ARRIVAL)
        ProcessGatherReorderArrival(descriptor, phase_id);
    else if (marker_kind == MarkerKind::REDUCE_ARRIVAL)
        ProcessReduceRxArrival(descriptor, phase_id);
    else
        WaitCollectiveBarrier(descriptor.key, phase_id, descriptor.self_rank,
                              static_cast<uint16_t>(descriptor.group.size()),
                              release_tree_id);
    return 0;
}
void Collective_prim::printSelf() {}
