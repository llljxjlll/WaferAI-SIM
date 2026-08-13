#pragma once

#include "dte/coll_types.h"
#include "dte/coll_wire.h"
#include "nlohmann/json_fwd.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>

inline constexpr uint16_t kNocCollMaxTreesPerBatch = 64;

enum class NocCollProfile : uint8_t {
    BASELINE = 0,
    BROADCAST_ONLY = 1,
    REDUCE_ONLY = 2,
    REDUCE_BROADCAST = 3,
};

enum class NocCollBroadcastBackend : uint8_t {
    UNICAST = 0,
    MULTICAST = 1,
};

enum class NocCollReduceBackend : uint8_t {
    ENDPOINT = 0,
    DCA_OFFLOAD = 1,
    LEGACY_ROUTER_ALU = 2,
};

enum class NocCollDcaArbitration : uint8_t {
    ROUND_ROBIN = 0,
    CORE_PRIORITY = 1,
    DCA_PRIORITY = 2,
};

enum class NocCollValueMode : uint8_t {
    INTEGER_EXACT = 0,
    FP_EXACT = 1,
    TIMING_ONLY = 2,
};

struct NocCollDcaTiming {
    uint64_t latency = 1;
    uint64_t initiation_interval = 1;
};

struct NocCollDcaConfig {
    uint64_t vector_bits = 512;
    uint64_t slice_bits = 64;
    uint64_t slices_per_tile = 8;
    uint64_t header_fifo_depth = 8;
    uint64_t operand_fifo_depth = 8;
    uint64_t result_fifo_depth = 8;
    NocCollDcaArbitration arbitration =
        NocCollDcaArbitration::ROUND_ROBIN;
    NocCollValueMode value_mode = NocCollValueMode::INTEGER_EXACT;
    // dtype order: uint8, int32, int64, fp32, fp16, fp8;
    // op order: sum, max.
    std::array<std::array<NocCollDcaTiming, 2>, 6> timing{};

    void Validate() const;
    void ValidateForDtype(CollDType dtype) const;
};

struct NocCollectiveConfig {
    bool enabled = false;
    NocCollProfile profile = NocCollProfile::BASELINE;
    NocCollBroadcastBackend broadcast_backend =
        NocCollBroadcastBackend::UNICAST;
    NocCollReduceBackend reduce_backend = NocCollReduceBackend::ENDPOINT;
    NocCollReduceWire reduce_wire = NocCollReduceWire::STREAM_V2;
    bool allow_legacy_backend = false;
    bool used_tier_alias = false;
    std::string transport = "conventional";
    // Absent preserves the scheduler's historical unbounded K.  An explicit
    // production value is bounded so it is representable and testable.
    std::optional<uint16_t> max_trees_per_batch;
    NocCollDcaConfig dca;

    bool UsesMulticast() const {
        return broadcast_backend == NocCollBroadcastBackend::MULTICAST;
    }
    bool UsesDcaOffload() const {
        return reduce_backend == NocCollReduceBackend::DCA_OFFLOAD;
    }
    bool UsesLegacyReduce() const {
        return reduce_backend == NocCollReduceBackend::LEGACY_ROUTER_ALU;
    }
};

struct NocCollTreeUse {
    bool multicast = false;
    bool reduce = false;
};

NocCollectiveConfig ParseNocCollectiveConfig(
    const nlohmann::json &noc_config);

NocCollTreeUse NocCollTreeUseFor(const NocCollectiveConfig &config,
                                CollOp op);
std::string NocCollProfileName(NocCollProfile profile);
std::string NocCollBroadcastBackendName(NocCollBroadcastBackend backend);
std::string NocCollReduceBackendName(NocCollReduceBackend backend);
std::string NocCollReduceWireName(NocCollReduceWire wire);
std::string NocCollDcaArbitrationName(NocCollDcaArbitration arbitration);
std::string NocCollValueModeName(NocCollValueMode mode);
