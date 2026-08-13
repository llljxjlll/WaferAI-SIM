#include "dte/coll_config.h"

#include "nlohmann/json.hpp"

#include <limits>
#include <optional>
#include <set>
#include <stdexcept>

namespace {

using json = nlohmann::json;

uint64_t UnsignedInteger(const json &value, const std::string &name) {
    if (value.is_number_unsigned()) {
        return value.get<uint64_t>();
    } else if (value.is_number_integer()) {
        const int64_t signed_value = value.get<int64_t>();
        if (signed_value >= 0)
            return static_cast<uint64_t>(signed_value);
    }
    throw std::invalid_argument(name + " must be a non-negative integer");
}

void RejectUnknownKeys(const json &object, const std::set<std::string> &keys,
                       const std::string &name) {
    if (!object.is_object())
        throw std::invalid_argument(name + " must be an object");
    for (auto it = object.begin(); it != object.end(); ++it)
        if (!keys.count(it.key()))
            throw std::invalid_argument("unsupported " + name + " field: " +
                                        it.key());
}

NocCollProfile ParseProfile(const std::string &value) {
    if (value == "baseline") return NocCollProfile::BASELINE;
    if (value == "broadcast_only") return NocCollProfile::BROADCAST_ONLY;
    if (value == "reduce_only") return NocCollProfile::REDUCE_ONLY;
    if (value == "reduce_broadcast")
        return NocCollProfile::REDUCE_BROADCAST;
    throw std::invalid_argument("unsupported noc.collective.profile: " + value);
}

NocCollBroadcastBackend ParseBroadcastBackend(const std::string &value) {
    if (value == "unicast") return NocCollBroadcastBackend::UNICAST;
    if (value == "multicast") return NocCollBroadcastBackend::MULTICAST;
    throw std::invalid_argument(
        "unsupported noc.collective.broadcast_backend: " + value);
}

NocCollReduceBackend ParseReduceBackend(const std::string &value) {
    if (value == "endpoint") return NocCollReduceBackend::ENDPOINT;
    if (value == "dca_offload") return NocCollReduceBackend::DCA_OFFLOAD;
    if (value == "legacy_router_alu")
        return NocCollReduceBackend::LEGACY_ROUTER_ALU;
    throw std::invalid_argument(
        "unsupported noc.collective.reduce_backend: " + value);
}

NocCollReduceWire ParseReduceWire(const std::string &value) {
    if (value == "stream_v2") return NocCollReduceWire::STREAM_V2;
    if (value == "legacy_two_segment")
        return NocCollReduceWire::LEGACY_TWO_SEGMENT;
    throw std::invalid_argument(
        "unsupported noc.collective.reduce_wire: " + value);
}

NocCollDcaArbitration ParseArbitration(const std::string &value) {
    if (value == "round_robin")
        return NocCollDcaArbitration::ROUND_ROBIN;
    if (value == "core_priority")
        return NocCollDcaArbitration::CORE_PRIORITY;
    if (value == "dca_priority")
        return NocCollDcaArbitration::DCA_PRIORITY;
    throw std::invalid_argument(
        "unsupported noc.collective.dca.arbitration: " + value);
}

NocCollValueMode ParseValueMode(const std::string &value) {
    if (value == "integer_exact") return NocCollValueMode::INTEGER_EXACT;
    if (value == "fp_exact") return NocCollValueMode::FP_EXACT;
    if (value == "timing_only") return NocCollValueMode::TIMING_ONLY;
    throw std::invalid_argument(
        "unsupported noc.collective.dca.value_mode: " + value);
}

std::pair<NocCollBroadcastBackend, NocCollReduceBackend>
BackendsForProfile(NocCollProfile profile) {
    switch (profile) {
    case NocCollProfile::BASELINE:
        return {NocCollBroadcastBackend::UNICAST,
                NocCollReduceBackend::ENDPOINT};
    case NocCollProfile::BROADCAST_ONLY:
        return {NocCollBroadcastBackend::MULTICAST,
                NocCollReduceBackend::ENDPOINT};
    case NocCollProfile::REDUCE_ONLY:
        return {NocCollBroadcastBackend::UNICAST,
                NocCollReduceBackend::DCA_OFFLOAD};
    case NocCollProfile::REDUCE_BROADCAST:
        return {NocCollBroadcastBackend::MULTICAST,
                NocCollReduceBackend::DCA_OFFLOAD};
    }
    throw std::invalid_argument("invalid NoC collective profile enum");
}

NocCollProfile ProfileForTier(int tier) {
    if (tier == 0) return NocCollProfile::BASELINE;
    if (tier == 1) return NocCollProfile::BROADCAST_ONLY;
    if (tier == 2) return NocCollProfile::REDUCE_BROADCAST;
    throw std::invalid_argument("noc.collective.tier must be in [0,2]");
}

NocCollProfile ProfileForBackends(NocCollBroadcastBackend broadcast,
                                  NocCollReduceBackend reduce) {
    if (reduce == NocCollReduceBackend::LEGACY_ROUTER_ALU) {
        if (broadcast != NocCollBroadcastBackend::MULTICAST)
            throw std::invalid_argument(
                "legacy_router_alu requires multicast broadcast backend");
        return NocCollProfile::REDUCE_BROADCAST;
    }
    for (NocCollProfile profile : {
             NocCollProfile::BASELINE, NocCollProfile::BROADCAST_ONLY,
             NocCollProfile::REDUCE_ONLY,
             NocCollProfile::REDUCE_BROADCAST}) {
        if (BackendsForProfile(profile) ==
            std::make_pair(broadcast, reduce))
            return profile;
    }
    throw std::invalid_argument("unsupported collective backend combination");
}

size_t DtypeIndex(const std::string &dtype) {
    if (dtype == "uint8") return 0;
    if (dtype == "int32") return 1;
    if (dtype == "int64") return 2;
    if (dtype == "fp32") return 3;
    if (dtype == "fp16") return 4;
    if (dtype == "fp8") return 5;
    throw std::invalid_argument("unsupported DCA timing dtype: " + dtype);
}

size_t OpIndex(const std::string &op) {
    if (op == "sum") return 0;
    if (op == "max") return 1;
    throw std::invalid_argument("unsupported DCA timing operation: " + op);
}

void ParseTimingField(const json &object, const std::string &field,
                      NocCollDcaConfig &config) {
    if (!object.contains(field)) return;
    const json &by_dtype = object.at(field);
    if (!by_dtype.is_object())
        throw std::invalid_argument("noc.collective.dca." + field +
                                    " must be an object");
    for (auto dtype = by_dtype.begin(); dtype != by_dtype.end(); ++dtype) {
        const size_t di = DtypeIndex(dtype.key());
        if (!dtype.value().is_object())
            throw std::invalid_argument("DCA dtype timing must be an object");
        for (auto op = dtype.value().begin(); op != dtype.value().end(); ++op) {
            const size_t oi = OpIndex(op.key());
            const uint64_t value = UnsignedInteger(
                op.value(), "noc.collective.dca." + field);
            if (field == "latency")
                config.timing[di][oi].latency = value;
            else
                config.timing[di][oi].initiation_interval = value;
        }
    }
}

NocCollDcaConfig ParseDcaConfig(const json &object) {
    RejectUnknownKeys(
        object,
        {"vector_bits", "slice_bits", "slices_per_tile", "latency",
         "initiation_interval", "header_fifo_depth", "operand_fifo_depth",
         "result_fifo_depth", "arbitration", "value_mode"},
        "noc.collective.dca");
    NocCollDcaConfig result;
    if (object.contains("vector_bits"))
        result.vector_bits = UnsignedInteger(
            object.at("vector_bits"), "noc.collective.dca.vector_bits");
    if (object.contains("slice_bits"))
        result.slice_bits = UnsignedInteger(
            object.at("slice_bits"), "noc.collective.dca.slice_bits");
    if (object.contains("slices_per_tile"))
        result.slices_per_tile = UnsignedInteger(
            object.at("slices_per_tile"),
            "noc.collective.dca.slices_per_tile");
    if (object.contains("header_fifo_depth"))
        result.header_fifo_depth = UnsignedInteger(
            object.at("header_fifo_depth"),
            "noc.collective.dca.header_fifo_depth");
    if (object.contains("operand_fifo_depth"))
        result.operand_fifo_depth = UnsignedInteger(
            object.at("operand_fifo_depth"),
            "noc.collective.dca.operand_fifo_depth");
    if (object.contains("result_fifo_depth"))
        result.result_fifo_depth = UnsignedInteger(
            object.at("result_fifo_depth"),
            "noc.collective.dca.result_fifo_depth");
    if (object.contains("arbitration"))
        result.arbitration =
            ParseArbitration(object.at("arbitration").get<std::string>());
    if (object.contains("value_mode"))
        result.value_mode =
            ParseValueMode(object.at("value_mode").get<std::string>());
    ParseTimingField(object, "latency", result);
    ParseTimingField(object, "initiation_interval", result);
    return result;
}

} // namespace

void NocCollDcaConfig::Validate() const {
    if (vector_bits == 0 || slice_bits == 0 || slices_per_tile == 0 ||
        slice_bits > std::numeric_limits<uint64_t>::max() / slices_per_tile ||
        vector_bits != slice_bits * slices_per_tile)
        throw std::invalid_argument(
            "DCA vector_bits must equal slice_bits*slices_per_tile");
    if (header_fifo_depth == 0 || operand_fifo_depth == 0 ||
        result_fifo_depth == 0)
        throw std::invalid_argument("DCA FIFO depths must be positive");
    for (const auto &dtype : timing)
        for (const auto &entry : dtype)
            if (entry.latency == 0 || entry.initiation_interval == 0)
                throw std::invalid_argument(
                    "DCA latency and initiation interval must be positive");
}

void NocCollDcaConfig::ValidateForDtype(CollDType dtype) const {
    Validate();
    const uint64_t dtype_bits = CollDTypeBits(dtype);
    if (vector_bits % dtype_bits != 0)
        throw std::invalid_argument(
            "DCA vector_bits must contain whole lanes for workload dtype");
    const bool floating = dtype == CollDType::FP32 ||
                          dtype == CollDType::FP16 ||
                          dtype == CollDType::FP8;
    if (value_mode == NocCollValueMode::INTEGER_EXACT && floating)
        throw std::invalid_argument(
            "integer_exact value mode does not support floating dtype");
    if (value_mode == NocCollValueMode::FP_EXACT &&
        dtype != CollDType::FP32)
        throw std::invalid_argument(
            "fp_exact value mode requires FP32 workload dtype");
}

NocCollectiveConfig ParseNocCollectiveConfig(const json &noc_config) {
    if (!noc_config.is_object())
        throw std::invalid_argument("noc must be an object");
    NocCollectiveConfig result;
    if (noc_config.contains("transport"))
        result.transport = noc_config.at("transport").get<std::string>();
    if (result.transport != "conventional") {
        if (result.transport == "smart")
            throw std::invalid_argument(
                "noc.transport=smart is not implemented in this refactor");
        throw std::invalid_argument("unsupported noc.transport: " +
                                    result.transport);
    }
    if (!noc_config.contains("collective")) return result;

    const json &coll = noc_config.at("collective");
    RejectUnknownKeys(
        coll,
        {"enabled", "tier", "profile", "broadcast_backend",
         "reduce_backend", "reduce_wire", "allow_legacy_backend",
         "max_trees_per_batch", "dca"},
        "noc.collective");
    if (coll.contains("enabled"))
        result.enabled = coll.at("enabled").get<bool>();
    if (coll.contains("allow_legacy_backend"))
        result.allow_legacy_backend =
            coll.at("allow_legacy_backend").get<bool>();
    if (coll.contains("max_trees_per_batch")) {
        const uint64_t value =
            coll.at("max_trees_per_batch").get<uint64_t>();
        if (value == 0 || value > kNocCollMaxTreesPerBatch)
            throw std::invalid_argument(
                "noc.collective.max_trees_per_batch must be in [1,64]");
        result.max_trees_per_batch = static_cast<uint16_t>(value);
    }

    std::optional<NocCollProfile> selected_profile;
    if (coll.contains("profile"))
        selected_profile = ParseProfile(coll.at("profile").get<std::string>());
    if (coll.contains("tier")) {
        const NocCollProfile tier_profile =
            ProfileForTier(coll.at("tier").get<int>());
        result.used_tier_alias = true;
        if (selected_profile && *selected_profile != tier_profile)
            throw std::invalid_argument(
                "noc.collective.tier conflicts with profile");
        selected_profile = tier_profile;
    }

    std::optional<NocCollBroadcastBackend> explicit_broadcast;
    std::optional<NocCollReduceBackend> explicit_reduce;
    if (coll.contains("broadcast_backend"))
        explicit_broadcast = ParseBroadcastBackend(
            coll.at("broadcast_backend").get<std::string>());
    if (coll.contains("reduce_backend"))
        explicit_reduce =
            ParseReduceBackend(coll.at("reduce_backend").get<std::string>());

    if (selected_profile) {
        const auto expected = BackendsForProfile(*selected_profile);
        if ((explicit_broadcast && *explicit_broadcast != expected.first) ||
            (explicit_reduce && *explicit_reduce != expected.second))
            throw std::invalid_argument(
                "noc.collective profile/tier conflicts with explicit backend");
        result.profile = *selected_profile;
        result.broadcast_backend = expected.first;
        result.reduce_backend = expected.second;
    } else {
        result.broadcast_backend = explicit_broadcast.value_or(
            NocCollBroadcastBackend::UNICAST);
        result.reduce_backend = explicit_reduce.value_or(
            NocCollReduceBackend::ENDPOINT);
        result.profile = ProfileForBackends(result.broadcast_backend,
                                            result.reduce_backend);
    }

    const NocCollReduceWire expected_wire = result.UsesLegacyReduce()
        ? NocCollReduceWire::LEGACY_TWO_SEGMENT
        : NocCollReduceWire::STREAM_V2;
    result.reduce_wire = expected_wire;
    if (coll.contains("reduce_wire")) {
        const NocCollReduceWire explicit_wire =
            ParseReduceWire(coll.at("reduce_wire").get<std::string>());
        if (explicit_wire != expected_wire)
            throw std::invalid_argument(
                "noc.collective reduce backend conflicts with reduce_wire");
        result.reduce_wire = explicit_wire;
    }
    if (result.UsesLegacyReduce()) {
        if (selected_profile || result.used_tier_alias)
            throw std::invalid_argument(
                "legacy backend must not be combined with profile or tier");
        if (!result.allow_legacy_backend)
            throw std::invalid_argument(
                "legacy_router_alu requires allow_legacy_backend=true");
    }

    if (coll.contains("dca"))
        result.dca = ParseDcaConfig(coll.at("dca"));
    // Dormant DCA fields remain syntactically parsed, but their resource and
    // value-mode semantics apply only when this configuration instantiates
    // the DCA backend.
    if (result.enabled && result.UsesDcaOffload())
        result.dca.Validate();
    return result;
}

NocCollTreeUse NocCollTreeUseFor(const NocCollectiveConfig &config,
                                CollOp op) {
    NocCollTreeUse use;
    if (op == CollOp::BROADCAST || op == CollOp::ALLGATHER)
        use.multicast = config.UsesMulticast();
    if (CollIsReduction(op)) {
        use.reduce = config.UsesDcaOffload() || config.UsesLegacyReduce();
        if (config.UsesLegacyReduce())
            use.multicast = true;
        else if (op == CollOp::ALLREDUCE && use.reduce)
            use.multicast = config.UsesMulticast();
    }
    return use;
}

std::string NocCollProfileName(NocCollProfile profile) {
    switch (profile) {
    case NocCollProfile::BASELINE: return "baseline";
    case NocCollProfile::BROADCAST_ONLY: return "broadcast_only";
    case NocCollProfile::REDUCE_ONLY: return "reduce_only";
    case NocCollProfile::REDUCE_BROADCAST: return "reduce_broadcast";
    }
    throw std::invalid_argument("invalid NoC collective profile enum");
}

std::string NocCollBroadcastBackendName(NocCollBroadcastBackend backend) {
    return backend == NocCollBroadcastBackend::UNICAST ? "unicast" :
                                                         "multicast";
}

std::string NocCollReduceBackendName(NocCollReduceBackend backend) {
    switch (backend) {
    case NocCollReduceBackend::ENDPOINT: return "endpoint";
    case NocCollReduceBackend::DCA_OFFLOAD: return "dca_offload";
    case NocCollReduceBackend::LEGACY_ROUTER_ALU:
        return "legacy_router_alu";
    }
    throw std::invalid_argument("invalid NoC collective reduce backend enum");
}

std::string NocCollReduceWireName(NocCollReduceWire wire) {
    return wire == NocCollReduceWire::STREAM_V2 ? "stream_v2" :
                                                  "legacy_two_segment";
}

std::string NocCollDcaArbitrationName(
    NocCollDcaArbitration arbitration) {
    switch (arbitration) {
    case NocCollDcaArbitration::ROUND_ROBIN: return "round_robin";
    case NocCollDcaArbitration::CORE_PRIORITY: return "core_priority";
    case NocCollDcaArbitration::DCA_PRIORITY: return "dca_priority";
    }
    throw std::invalid_argument("invalid DCA arbitration enum");
}

std::string NocCollValueModeName(NocCollValueMode mode) {
    switch (mode) {
    case NocCollValueMode::INTEGER_EXACT: return "integer_exact";
    case NocCollValueMode::FP_EXACT: return "fp_exact";
    case NocCollValueMode::TIMING_ONLY: return "timing_only";
    }
    throw std::invalid_argument("invalid DCA value mode enum");
}
