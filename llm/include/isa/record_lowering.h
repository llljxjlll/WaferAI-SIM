#pragma once

#include "isa/record_codec.h"
#include "isa/program_format.h"
#include "isa/collective_graph_v1.h"
#include "prims/base.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

class RecordLoweringError : public std::runtime_error {
public:
    explicit RecordLoweringError(const std::string &message)
        : std::runtime_error(message) {}
};

class LoweringUnavailableError : public RecordLoweringError {
public:
    explicit LoweringUnavailableError(const std::string &message)
        : RecordLoweringError(message) {}
};

struct LoweringContext {
    uint64_t enabled_capabilities = 0;

    // Resolvers return nullopt for an unknown table index. Lowering never
    // interns either result into a process-global table.
    std::function<std::optional<std::string>(uint32_t)> resolve_symbol;
    std::function<std::optional<std::string>(uint32_t)> resolve_string;
};

using LoweredPrimList = std::vector<std::unique_ptr<PrimBase>>;

// P6-B2 loader product.  It is deliberately not a Prim stream: public
// collective tokens complete only after all child endpoints, and reduction
// targets require the future strict byte-executing P6-D runtime.
struct IsaV1LoweredCollectiveChild {
    std::size_t plan_index = 0;
    uint32_t child_index = 0;
    uint32_t source_internal_token = 0;
    uint32_t destination_internal_token = 0;
    EndpointSourceSpace source_space = EndpointSourceSpace::SRAM;
    SramAddressOperand source;
    SramAddressOperand destination;

    bool operator==(const IsaV1LoweredCollectiveChild &other) const;
};

struct IsaV1LoweredCollectiveAction {
    std::size_t plan_index = 0;
    CollectiveKey key;
    IsaV1Action action;
    // Zero for local-copy, barrier, and reduction-compute actions.
    uint32_t internal_token = 0;
    uint32_t public_aggregate_token = 0;

    bool operator==(const IsaV1LoweredCollectiveAction &other) const;
};

struct IsaV1CoreCollectiveActionStream {
    uint16_t core_id = 0;
    std::vector<IsaV1LoweredCollectiveAction> actions;

    bool operator==(const IsaV1CoreCollectiveActionStream &other) const;
};

struct IsaV1CollectiveArtifactLowering {
    std::vector<IsaV1CollectivePlan> plans;
    std::vector<IsaV1LoweredCollectiveChild> children;
    std::vector<IsaV1CoreCollectiveActionStream> core_actions;
    // P6-B2 is a validated loader product only.  Worker byte execution and
    // aggregate-token completion remain explicit P6-D/E gates.
    bool executable = false;

    bool operator==(const IsaV1CollectiveArtifactLowering &other) const;
};

// Whole-artifact adapter.  The artifact must already have semantic
// relocations applied.  Validation and deterministic internal-token
// allocation finish before any caller-visible/global commit.
IsaV1CollectiveArtifactLowering LowerIsaV1CollectiveArtifact(
    const ProgramArtifact &artifact, uint32_t total_cores,
    uint32_t cores_per_die,
    const IsaV1PlannerCapacity &capacity = IsaV1PlannerCapacity{});

// Validates the external record first, then constructs untracked Prim objects.
// The returned ownership is suitable for a loader's validate-then-commit flow.
// This function never writes g_prim_stash, g_addr_label_table, or Router state.
LoweredPrimList LowerExternalRecord(const ExternalRecord &record,
                                    const LoweringContext &context = {});
