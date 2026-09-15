#include "prims/moe_signed_router_native_work.h"

#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
constexpr uint64_t kMaxProfile = (uint64_t{1} << 30) - 1;
constexpr uint64_t kSramCapacity = uint64_t{1} << 16;
using GroupKey = std::pair<uint32_t, uint32_t>;

uint64_t Mul(uint64_t lhs, uint64_t rhs, const char *field) {
    if (rhs && lhs > std::numeric_limits<uint64_t>::max() / rhs)
        throw std::overflow_error(std::string(field) + " overflow");
    return lhs * rhs;
}

void CheckSpan(const MoeSignedRouterSramSpan &span, uint64_t required,
               const char *field) {
    if (!required || span.bytes != required || span.byte_address % 16 ||
        span.byte_address > kSramCapacity ||
        span.bytes > kSramCapacity - span.byte_address)
        throw std::invalid_argument(std::string(field) +
                                    " typed FP16 SRAM extent/16-bit wire differs");
}

bool Overlap(const MoeSignedRouterSramSpan &lhs,
             const MoeSignedRouterSramSpan &rhs) {
    return uint64_t{lhs.byte_address} < rhs.byte_address + rhs.bytes &&
           uint64_t{rhs.byte_address} < lhs.byte_address + lhs.bytes;
}

template <class... S>
void Independent(const S &... spans) {
    const std::vector<MoeSignedRouterSramSpan> values = {spans...};
    for (size_t i = 0; i < values.size(); ++i)
        for (size_t j = i + 1; j < values.size(); ++j)
            if (Overlap(values[i], values[j]))
                throw std::invalid_argument(
                    "router score/returned expert/upstream/two gradient SRAM spans overlap");
}

void CheckSource(const MoeSignedRouterSource &source) {
    if (source.dynamic_case_ref.empty() || source.original_static_case_ref.empty() ||
        source.dynamic_case_ref == source.original_static_case_ref ||
        source.gate_action_ref.empty() || source.combine_action_ref.empty() ||
        source.combine_backward_action_ref.empty() || !source.rank_rows ||
        !source.hidden || !source.experts || source.rank_rows > kMaxProfile ||
        source.hidden > kMaxProfile || source.experts > kMaxProfile ||
        source.routes.size() != source.rank_rows || source.groups.empty())
        throw std::invalid_argument("router dynamic source case/actions/geometry absent");
    const uint64_t row_bytes = Mul(2, source.hidden, "router return row");
    const uint64_t total_bytes = Mul(row_bytes, source.rank_rows,
                                     "router return total");
    std::map<GroupKey, std::pair<uint64_t, uint64_t>> group_extents;
    uint64_t end = 0;
    for (const auto &group : source.groups) {
        const GroupKey key{group.expert_home_rank, group.expert_index};
        if (group.expert_index >= source.experts ||
            group.byte_offset != end || !group.byte_extent ||
            group.byte_extent % row_bytes || group_extents.count(key) ||
            uint64_t{group.byte_extent} > total_bytes - end)
            throw std::invalid_argument("router expert groups must fill nonoverlapping signed home slots");
        group_extents[key] = {end, group.byte_extent / row_bytes};
        end += group.byte_extent;
    }
    if (end != total_bytes)
        throw std::invalid_argument("router expert return dropped the last token");
    std::map<GroupKey, std::vector<bool>> used;
    for (const auto &entry : group_extents)
        used[entry.first].resize(entry.second.second);
    for (size_t row = 0; row < source.routes.size(); ++row) {
        const auto &route = source.routes[row];
        const GroupKey key{route.expert_home_rank, route.selected_expert};
        const auto it = used.find(key);
        if (route.token_index != row || route.source_rank != source.source_rank ||
            route.selected_expert >= source.experts || it == used.end() ||
            route.expert_slot_index >= it->second.size() ||
            it->second[route.expert_slot_index])
            throw std::invalid_argument("router token/expert/home/slot trace differs from source");
        it->second[route.expert_slot_index] = true;
    }
    for (const auto &entry : used)
        for (bool visited : entry.second)
            if (!visited)
                throw std::invalid_argument("router expert grouped return has unused slot");
}

uint64_t ReturnOffset(const MoeSignedRouterSource &source,
                      const MoeSignedRouterRoute &route) {
    const uint64_t bytes_per_row = Mul(2, source.hidden, "router return offset");
    for (const auto &group : source.groups)
        if (group.expert_home_rank == route.expert_home_rank &&
            group.expert_index == route.selected_expert)
            return group.byte_offset + route.expert_slot_index * bytes_per_row;
    throw std::invalid_argument("router return missing signed expert home");
}

void CheckSamples(const MoeSignedRouterSource &source,
                  const std::vector<float> &score,
                  const std::vector<float> &grouped_return) {
    CheckSource(source);
    if (score.size() != Mul(source.rank_rows, source.experts, "score size") ||
        grouped_return.size() != Mul(source.rank_rows, source.hidden, "return size"))
        throw std::invalid_argument("router reference FP16 element counts differ");
}
} // namespace

std::vector<uint8_t> SerializeMoeSignedRouterRouteTable(
    const MoeSignedRouterSource &source) {
    CheckSource(source);
    std::vector<uint8_t> bytes;
    bytes.reserve(source.routes.size() * 5 * 4);
    for (const auto &route : source.routes)
        for (uint32_t field : {route.token_index, route.source_rank,
                               route.selected_expert, route.expert_home_rank,
                               route.expert_slot_index})
            for (unsigned shift = 0; shift < 32; shift += 8)
                bytes.push_back(static_cast<uint8_t>(field >> shift));
    return bytes;
}

void RequireSignedMoeRouterRouteTable(
    const MoeSignedRouterSource &source, const std::vector<uint8_t> &bytes) {
    if (bytes != SerializeMoeSignedRouterRouteTable(source))
        throw std::invalid_argument("router INT32 physical route table differs from one signed source trace");
}

MoeSignedRouterWork BuildMoeSignedRouterForwardWork(
    const MoeSignedRouterSource &source,
    const MoeSignedRouterForwardTile &tile) {
    CheckSource(source);
    const uint64_t score_bytes = Mul(2, Mul(source.rank_rows, source.experts,
                                            "score elements"), "score bytes");
    const uint64_t hidden_bytes = Mul(2, Mul(source.rank_rows, source.hidden,
                                             "returned elements"), "returned bytes");
    const uint64_t route_bytes = Mul(20, source.rank_rows, "router INT32 route bytes");
    CheckSpan(tile.route, route_bytes, "router forward route INT32");
    CheckSpan(tile.score, score_bytes, "router forward score");
    CheckSpan(tile.returns, hidden_bytes, "router forward returned expert");
    CheckSpan(tile.combined, hidden_bytes, "router forward combined");
    Independent(tile.route, tile.score, tile.returns, tile.combined);
    return {route_bytes, score_bytes + hidden_bytes, hidden_bytes,
            Mul(2, Mul(source.rank_rows, source.hidden, "weighted cells"),
                "weighted forward EXU"), 0};
}

MoeSignedRouterWork BuildMoeSignedRouterBackwardWork(
    const MoeSignedRouterSource &source,
    const MoeSignedRouterBackwardTile &tile) {
    CheckSource(source);
    const uint64_t score_bytes = Mul(2, Mul(source.rank_rows, source.experts,
                                            "score elements"), "score bytes");
    const uint64_t hidden_bytes = Mul(2, Mul(source.rank_rows, source.hidden,
                                             "returned elements"), "returned bytes");
    const uint64_t route_bytes = Mul(20, source.rank_rows, "router INT32 route bytes");
    CheckSpan(tile.route, route_bytes, "router backward route INT32");
    CheckSpan(tile.score, score_bytes, "router backward score");
    CheckSpan(tile.returns, hidden_bytes, "router backward returned expert");
    CheckSpan(tile.dcombined, hidden_bytes, "router backward upstream");
    CheckSpan(tile.dscore, score_bytes, "router backward dScore");
    CheckSpan(tile.dexpert, hidden_bytes, "router backward dExpert");
    Independent(tile.route, tile.score, tile.returns, tile.dcombined,
                tile.dscore, tile.dexpert);
    const uint64_t cells = Mul(source.rank_rows, source.hidden,
                               "router backward cells");
    return {route_bytes, score_bytes + 2 * hidden_bytes,
            score_bytes + hidden_bytes,
            Mul(2, cells, "router score dot EXU"), cells};
}

MoeSignedRouterMultiwordPrimProfile BuildMoeRouterForwardPrimProfile(
    const MoeSignedRouterSource &source,
    const MoeSignedRouterForwardTile &tile) {
    const auto work = BuildMoeSignedRouterForwardWork(source, tile);
    return {tile.score.byte_address, tile.returns.byte_address,
            tile.combined.byte_address,
            {{"E", static_cast<uint32_t>(source.experts)},
             {"H", static_cast<uint32_t>(source.hidden)},
             {"K", static_cast<uint32_t>(source.rank_rows)},
             {"ROUTE_ADDRESS", tile.route.byte_address},
             {"ROUTE_BYTES", static_cast<uint32_t>(work.route_table_read_bytes)}}};
}

MoeSignedRouterMultiwordPrimProfile BuildMoeRouterBackwardPrimProfile(
    const MoeSignedRouterSource &source,
    const MoeSignedRouterBackwardTile &tile) {
    const auto work = BuildMoeSignedRouterBackwardWork(source, tile);
    // The metadata output slot is a third INPUT dCombined. Both output
    // addresses are independent named parameters on additional strict wires.
    return {tile.score.byte_address, tile.returns.byte_address,
            tile.dcombined.byte_address,
            {{"DEXPERT_ADDRESS", tile.dexpert.byte_address},
             {"DSCORE_ADDRESS", tile.dscore.byte_address},
             {"E", static_cast<uint32_t>(source.experts)},
             {"H", static_cast<uint32_t>(source.hidden)},
             {"K", static_cast<uint32_t>(source.rank_rows)},
             {"ROUTE_ADDRESS", tile.route.byte_address},
             {"ROUTE_BYTES", static_cast<uint32_t>(work.route_table_read_bytes)}}};
}

std::vector<float> EvaluateMoeSignedRouterForward(
    const MoeSignedRouterSource &source, const std::vector<float> &score,
    const std::vector<float> &grouped_return,
    const std::vector<uint8_t> &route_blob) {
    RequireSignedMoeRouterRouteTable(source, route_blob);
    CheckSamples(source, score, grouped_return);
    std::vector<float> combined(source.rank_rows * source.hidden);
    for (size_t row = 0; row < source.rank_rows; ++row) {
        const auto &route = source.routes[row];
        const size_t grouped = ReturnOffset(source, route) / 2;
        const float weight = score[row * source.experts + route.selected_expert];
        for (size_t h = 0; h < source.hidden; ++h)
            combined[row * source.hidden + h] =
                weight * grouped_return[grouped + h];
    }
    return combined;
}

MoeSignedRouterBackwardValues EvaluateMoeSignedRouterBackward(
    const MoeSignedRouterSource &source, const std::vector<float> &score,
    const std::vector<float> &grouped_return,
    const std::vector<float> &dcombined,
    const std::vector<uint8_t> &route_blob) {
    RequireSignedMoeRouterRouteTable(source, route_blob);
    CheckSamples(source, score, grouped_return);
    if (dcombined.size() != source.rank_rows * source.hidden)
        throw std::invalid_argument("router dCombined loses an upstream row");
    MoeSignedRouterBackwardValues result;
    result.dscore.resize(source.rank_rows * source.experts);
    result.dexpert.resize(source.rank_rows * source.hidden);
    for (size_t row = 0; row < source.rank_rows; ++row) {
        const auto &route = source.routes[row];
        const size_t grouped = ReturnOffset(source, route) / 2;
        const float weight = score[row * source.experts + route.selected_expert];
        float selected_dscore = 0;
        for (size_t h = 0; h < source.hidden; ++h) {
            const float dy = dcombined[row * source.hidden + h];
            selected_dscore += dy * grouped_return[grouped + h];
            result.dexpert[grouped + h] = weight * dy;
        }
        result.dscore[row * source.experts + route.selected_expert] =
            selected_dscore;
    }
    return result;
}
