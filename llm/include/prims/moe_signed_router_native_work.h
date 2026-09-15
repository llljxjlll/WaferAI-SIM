#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Stage R source/physical work only. Opcode0x27/0x28 and the fifth public
// SemanticOperandId are deliberately not registered until the shared codec,
// finalizer and ProgramIO versioned bridge is installed.
struct MoeSignedRouterRoute {
    uint32_t token_index = 0;
    uint32_t source_rank = 0;
    uint32_t selected_expert = 0;
    uint32_t expert_home_rank = 0;
    uint32_t expert_slot_index = 0;
};
struct MoeSignedRouterGroup {
    uint32_t expert_home_rank = 0;
    uint32_t expert_index = 0;
    uint32_t byte_offset = 0;
    uint32_t byte_extent = 0;
};
struct MoeSignedRouterSramSpan {
    uint32_t byte_address = 0;
    uint64_t bytes = 0;
};
struct MoeSignedRouterSource {
    std::string dynamic_case_ref;
    std::string original_static_case_ref;
    std::string gate_action_ref;
    std::string combine_action_ref;
    std::string combine_backward_action_ref;
    uint32_t source_rank = 0;
    uint64_t rank_rows = 0;
    uint64_t hidden = 0;
    uint64_t experts = 0;
    std::vector<MoeSignedRouterRoute> routes;
    std::vector<MoeSignedRouterGroup> groups;
};
struct MoeSignedRouterForwardTile {
    MoeSignedRouterSramSpan route;    // INT32 K,5 source-frozen ProgramIO blob
    MoeSignedRouterSramSpan score;    // FP16 K,E
    MoeSignedRouterSramSpan returns;  // FP16 grouped K,H
    MoeSignedRouterSramSpan combined; // distinct owned FP16 K,H
};
struct MoeSignedRouterBackwardTile {
    MoeSignedRouterSramSpan route;     // INT32 K,5 source-frozen ProgramIO blob
    MoeSignedRouterSramSpan score;     // FP16 K,E
    MoeSignedRouterSramSpan returns;   // FP16 grouped K,H
    MoeSignedRouterSramSpan dcombined; // FP16 K,H
    MoeSignedRouterSramSpan dscore;    // distinct owned FP16 K,E
    MoeSignedRouterSramSpan dexpert;   // distinct owned FP16 grouped K,H
};
struct MoeSignedRouterWork {
    uint64_t route_table_read_bytes = 0;
    uint64_t fp16_read_bytes = 0;
    uint64_t fp16_write_bytes = 0;
    uint64_t exu_flops = 0;
    uint64_t vector_flops = 0;
};
struct MoeSignedRouterMultiwordPrimProfile {
    uint32_t metadata_input_offset = 0;
    uint32_t metadata_data_offset = 0;
    uint32_t metadata_output_offset = 0;
    std::vector<std::pair<std::string, uint32_t>> named_30bit_parameters;
};

struct MoeSignedRouterBackwardValues {
    std::vector<float> dscore;  // unselected expert entries are physically zero
    std::vector<float> dexpert; // grouped by expert home and slot
};

std::vector<uint8_t> SerializeMoeSignedRouterRouteTable(
    const MoeSignedRouterSource &);
void RequireSignedMoeRouterRouteTable(
    const MoeSignedRouterSource &, const std::vector<uint8_t> &);

MoeSignedRouterWork BuildMoeSignedRouterForwardWork(
    const MoeSignedRouterSource &, const MoeSignedRouterForwardTile &);
MoeSignedRouterWork BuildMoeSignedRouterBackwardWork(
    const MoeSignedRouterSource &, const MoeSignedRouterBackwardTile &);

// NpuBase::serialize already supports metadata (three addresses) plus
// multiword 30-bit named parameters. These *proposed* profiles remain
// unregistered until strict public 0x27/0x28 codec/finalizer is installed.
MoeSignedRouterMultiwordPrimProfile BuildMoeRouterForwardPrimProfile(
    const MoeSignedRouterSource &, const MoeSignedRouterForwardTile &);
MoeSignedRouterMultiwordPrimProfile BuildMoeRouterBackwardPrimProfile(
    const MoeSignedRouterSource &, const MoeSignedRouterBackwardTile &);

// Independent float reference for exact route/score math. The timing prim
// does not imply that the existing NpuSim computes numerical gradients.
std::vector<float> EvaluateMoeSignedRouterForward(
    const MoeSignedRouterSource &, const std::vector<float> &score,
    const std::vector<float> &grouped_return,
    const std::vector<uint8_t> &route_blob);
MoeSignedRouterBackwardValues EvaluateMoeSignedRouterBackward(
    const MoeSignedRouterSource &, const std::vector<float> &score,
    const std::vector<float> &grouped_return,
    const std::vector<float> &dcombined,
    const std::vector<uint8_t> &route_blob);
