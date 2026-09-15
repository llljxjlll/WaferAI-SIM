#include "prims/moe_signed_router_native_work.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void Check(bool holds, const char *condition) {
    if (!holds)
        throw std::runtime_error(condition);
}
void Reject(const std::function<void()> &call, const char *condition) {
    try { call(); }
    catch (const std::invalid_argument &) { return; }
    throw std::runtime_error(condition);
}

MoeSignedRouterSource Source() {
    MoeSignedRouterSource source;
    source.dynamic_case_ref = "dynamic_1x2_h4_e2_top1_v1";
    source.original_static_case_ref = "legacy_static_1x2_h4_e2_top1";
    source.gate_action_ref = "gate_forward";
    source.combine_action_ref = "weighted_forward";
    source.combine_backward_action_ref = "score_backward";
    source.rank_rows = 4;
    source.hidden = 4;
    source.experts = 2;
    source.routes = {{0, 0, 0, 0, 0}, {1, 0, 1, 1, 0},
                     {2, 0, 0, 0, 1}, {3, 0, 1, 1, 1}};
    source.groups = {{0, 0, 0, 16}, {1, 1, 16, 16}};
    return source;
}
MoeSignedRouterForwardTile Forward() {
    return {{128, 80}, {0, 16}, {32, 32}, {64, 32}};
}
MoeSignedRouterBackwardTile Backward() {
    return {{160, 80}, {0, 16}, {32, 32}, {64, 32},
            {96, 16}, {112, 32}};
}
} // namespace

int main() {
    const auto source = Source();
    const auto forward = BuildMoeSignedRouterForwardWork(source, Forward());
    Check(forward.route_table_read_bytes == 80 &&
          forward.fp16_read_bytes == 48 && forward.fp16_write_bytes == 32 &&
          forward.exu_flops == 32 && forward.vector_flops == 0,
          "0x27 full selected score+return traffic/FLOPs changed");
    const auto backward = BuildMoeSignedRouterBackwardWork(source, Backward());
    Check(backward.route_table_read_bytes == 80 &&
          backward.fp16_read_bytes == 80 && backward.fp16_write_bytes == 48 &&
          backward.exu_flops == 32 && backward.vector_flops == 16,
          "0x28 five independent SRAM spans/48 logical FLOPs changed");

    const auto forward_wire = BuildMoeRouterForwardPrimProfile(source, Forward());
    const auto backward_wire = BuildMoeRouterBackwardPrimProfile(source, Backward());
    Check(forward_wire.metadata_input_offset == 0 &&
          forward_wire.metadata_data_offset == 32 &&
          forward_wire.metadata_output_offset == 64 &&
          backward_wire.metadata_output_offset == 64 &&
          forward_wire.named_30bit_parameters.size() == 5 &&
          backward_wire.named_30bit_parameters.size() == 7,
          "Prim wire metadata three addresses and multiword params changed");
    const auto find_param = [](const MoeSignedRouterMultiwordPrimProfile &profile,
                               const std::string &name) -> uint32_t {
        for (const auto &item : profile.named_30bit_parameters)
            if (item.first == name) return item.second;
        throw std::runtime_error("named strict Prim wire parameter missing");
    };
    Check(find_param(forward_wire, "ROUTE_ADDRESS") == 128 &&
          find_param(backward_wire, "ROUTE_ADDRESS") == 160 &&
          find_param(backward_wire, "DSCORE_ADDRESS") == 96 &&
          find_param(backward_wire, "DEXPERT_ADDRESS") == 112 &&
          find_param(backward_wire, "ROUTE_BYTES") == 80,
          "0x28 multiword route and two outputs cannot alias dCombined");

    const auto route_blob = SerializeMoeSignedRouterRouteTable(source);
    Check(route_blob.size() == 80 && route_blob[20] == 1 &&
          route_blob[28] == 1 && route_blob[32] == 1,
          "typed INT32 5-column trace must keep nonzero token/expert/home");
    RequireSignedMoeRouterRouteTable(source, route_blob);
    auto forged_blob = route_blob;
    forged_blob[20] = 0;
    Reject([&] { RequireSignedMoeRouterRouteTable(source, forged_blob); },
           "ProgramIO route table override unbound from source accepted");
    forged_blob = route_blob;
    forged_blob.pop_back();
    Reject([&] { RequireSignedMoeRouterRouteTable(source, forged_blob); },
           "source route table last slot bytes omitted accepted");

    // Eight score entries are physical; only the selected entry affects each
    // row.  Home/slot grouped return rows order 0,2,1,3 must be gathered via
    // actual route table, not copied as contiguous source-token rows.
    const std::vector<float> score{2, -8, 7, 3, 5, -9, 11, 4};
    const std::vector<float> returned{1, 2, 3, 4, 5, 6, 7, 8,
                                       9, 10, 11, 12, 13, 14, 15, 16};
    const auto combined = EvaluateMoeSignedRouterForward(source, score,
                                                          returned, route_blob);
    Check(combined == std::vector<float>({2, 4, 6, 8, 27, 30, 33, 36,
                                           25, 30, 35, 40, 52, 56, 60, 64}),
          "forward score must weight each selected expert home/slot");
    const std::vector<float> upstream(16, 1.0f);
    const auto gradients = EvaluateMoeSignedRouterBackward(
        source, score, returned, upstream, route_blob);
    Check(gradients.dscore == std::vector<float>({10, 0, 0, 42,
                                                  26, 0, 0, 58}),
          "dScore must equal selected expert dot and leave other expert zero");
    Check(gradients.dexpert == std::vector<float>({2, 2, 2, 2, 5, 5, 5, 5,
                                                   3, 3, 3, 3, 4, 4, 4, 4}),
          "dExpert must be score-scaled and grouped by real expert slot");

    auto wrong = Backward();
    wrong.dexpert.bytes = 16;
    Reject([&] { BuildMoeSignedRouterBackwardWork(source, wrong); },
           "dExpert shortened to selected local expert was accepted");
    wrong = Backward();
    wrong.dexpert.byte_address = wrong.dscore.byte_address;
    Reject([&] { BuildMoeSignedRouterBackwardWork(source, wrong); },
           "dScore/dExpert alias was accepted");
    wrong = Backward();
    wrong.dcombined.byte_address = wrong.returns.byte_address;
    Reject([&] { BuildMoeSignedRouterBackwardWork(source, wrong); },
           "forward RETURN/upstream alias was accepted");
    wrong = Backward();
    wrong.route.byte_address = wrong.dexpert.byte_address;
    Reject([&] { BuildMoeSignedRouterBackwardWork(source, wrong); },
           "physical INT32 route tape aliases expert gradient accepted");
    auto dropped = source;
    dropped.routes.pop_back();
    Reject([&] { BuildMoeSignedRouterForwardWork(dropped, Forward()); },
           "last selected token loss was accepted");
    dropped = source;
    dropped.routes[3].expert_slot_index = 0;
    Reject([&] { BuildMoeSignedRouterBackwardWork(dropped, Backward()); },
           "duplicate remote expert slot was accepted");
    dropped = source;
    dropped.groups[1].byte_offset = 0;
    Reject([&] { BuildMoeSignedRouterBackwardWork(dropped, Backward()); },
           "local/remote RETURN overlap was accepted");
    dropped = source;
    dropped.dynamic_case_ref = dropped.original_static_case_ref;
    Reject([&] { BuildMoeSignedRouterForwardWork(dropped, Forward()); },
           "legacy static-weight case relabeled as dynamic was accepted");
    Reject([&] { EvaluateMoeSignedRouterBackward(source, score, returned,
                                                 std::vector<float>(8), route_blob);
    forged_blob = route_blob;
    forged_blob[28] ^= 1;
    Reject([&] { EvaluateMoeSignedRouterForward(source, score, returned,
                                                forged_blob); },
           "wrong physical route bytes consumed as valid native input"); },
           "half dCombined was accepted");
    std::cout << "moe_signed_router_native_work_selftest PASS\n";
}
