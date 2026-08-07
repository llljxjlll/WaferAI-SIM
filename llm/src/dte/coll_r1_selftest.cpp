#include "dte/coll_config.h"

#include "nlohmann/json.hpp"

#include <iostream>
#include <string>

namespace {
int failures = 0;
int checks = 0;

void Check(bool ok, const std::string &name) {
    ++checks;
    if (!ok) ++failures;
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name
              << std::endl;
}

template <class F> bool Throws(F f) {
    try {
        f();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

nlohmann::json ProfileConfig(const std::string &profile) {
    return {{"transport", "conventional"},
            {"collective", {{"enabled", true}, {"profile", profile}}}};
}

nlohmann::json LegacyConfig() {
    return {{"collective",
             {{"enabled", true},
              {"broadcast_backend", "multicast"},
              {"reduce_backend", "legacy_router_alu"},
              {"reduce_wire", "legacy_two_segment"},
              {"allow_legacy_backend", true}}}};
}
} // namespace

int RunCollR1SelfTest() {
    failures = checks = 0;
    std::cout << "==== NoC collective refactor R1 config self-test ===="
              << std::endl;

    const auto disabled = ParseNocCollectiveConfig(nlohmann::json::object());
    Check(!disabled.enabled &&
              disabled.profile == NocCollProfile::BASELINE &&
              !disabled.UsesMulticast() && !disabled.UsesDcaOffload(),
          "absent collective config preserves disabled baseline");
    Check(static_cast<uint8_t>(NocCollReduceWire::LEGACY_TWO_SEGMENT) == 0 &&
              static_cast<uint8_t>(NocCollReduceWire::STREAM_V2) == 1,
          "config and protocol share frozen reduce-wire enum values");

    const auto baseline = ParseNocCollectiveConfig(ProfileConfig("baseline"));
    Check(baseline.profile == NocCollProfile::BASELINE &&
              baseline.broadcast_backend ==
                  NocCollBroadcastBackend::UNICAST &&
              baseline.reduce_backend == NocCollReduceBackend::ENDPOINT,
          "baseline profile expands to unicast plus endpoint");
    const auto broadcast =
        ParseNocCollectiveConfig(ProfileConfig("broadcast_only"));
    Check(broadcast.profile == NocCollProfile::BROADCAST_ONLY &&
              broadcast.UsesMulticast() && !broadcast.UsesDcaOffload(),
          "broadcast_only profile expands canonically");
    const auto reduce =
        ParseNocCollectiveConfig(ProfileConfig("reduce_only"));
    Check(reduce.profile == NocCollProfile::REDUCE_ONLY &&
              !reduce.UsesMulticast() && reduce.UsesDcaOffload(),
          "reduce_only profile expands canonically");
    const auto combined =
        ParseNocCollectiveConfig(ProfileConfig("reduce_broadcast"));
    Check(combined.profile == NocCollProfile::REDUCE_BROADCAST &&
              combined.UsesMulticast() && combined.UsesDcaOffload(),
          "reduce_broadcast profile expands canonically");

    const NocCollProfile tier_profiles[] = {
        NocCollProfile::BASELINE, NocCollProfile::BROADCAST_ONLY,
        NocCollProfile::REDUCE_BROADCAST};
    for (int tier = 0; tier < 3; ++tier) {
        const auto config = ParseNocCollectiveConfig(
            {{"collective", {{"enabled", true}, {"tier", tier}}}});
        Check(config.used_tier_alias &&
                  config.profile == tier_profiles[tier],
              "tier " + std::to_string(tier) +
                  " maps to its canonical profile");
    }

    Check(Throws([] {
              auto j = ProfileConfig("baseline");
              j["collective"]["tier"] = 1;
              (void)ParseNocCollectiveConfig(j);
          }),
          "tier/profile conflict is rejected");
    Check(Throws([] {
              auto j = ProfileConfig("baseline");
              j["collective"]["broadcast_backend"] = "multicast";
              (void)ParseNocCollectiveConfig(j);
          }),
          "profile/broadcast backend conflict is rejected");
    Check(Throws([] {
              auto j = ProfileConfig("reduce_only");
              j["collective"]["reduce_backend"] = "endpoint";
              (void)ParseNocCollectiveConfig(j);
          }),
          "profile/reduce backend conflict is rejected");

    const auto derived = ParseNocCollectiveConfig(
        {{"collective", {{"enabled", true},
                          {"reduce_backend", "dca_offload"}}}});
    Check(derived.profile == NocCollProfile::REDUCE_ONLY &&
              !derived.UsesMulticast(),
          "backend-only config derives reduce_only profile");

    Check(Throws([] {
              auto j = LegacyConfig();
              j["collective"].erase("allow_legacy_backend");
              (void)ParseNocCollectiveConfig(j);
          }),
          "legacy backend requires explicit debug gate");
    const auto legacy = ParseNocCollectiveConfig(LegacyConfig());
    Check(legacy.UsesLegacyReduce() && legacy.UsesMulticast() &&
              legacy.reduce_wire == NocCollReduceWire::LEGACY_TWO_SEGMENT,
          "fully explicit legacy debug backend is accepted");
    Check(Throws([] {
              auto j = LegacyConfig();
              j["collective"]["tier"] = 2;
              (void)ParseNocCollectiveConfig(j);
          }),
          "legacy backend cannot hide behind a tier alias");
    Check(Throws([] {
              auto j = LegacyConfig();
              j["collective"]["reduce_wire"] = "stream_v2";
              (void)ParseNocCollectiveConfig(j);
          }),
          "legacy backend requires legacy two-segment wire");

    Check(Throws([] {
              (void)ParseNocCollectiveConfig(
                  {{"transport", "smart"}});
          }),
          "SMART transport is explicitly rejected");
    Check(Throws([] {
              (void)ParseNocCollectiveConfig(
                  {{"transport", "teleport"}});
          }),
          "unknown transport is rejected");

    Check(Throws([] {
              auto j = ProfileConfig("reduce_only");
              j["collective"]["dca"] =
                  {{"vector_bits", 256}, {"slice_bits", 64},
                   {"slices_per_tile", 8}};
              (void)ParseNocCollectiveConfig(j);
          }),
          "DCA vector/slice product mismatch is rejected");
    Check(Throws([] {
              auto j = ProfileConfig("reduce_only");
              j["collective"]["dca"] = {{"vector_bits", 0}};
              (void)ParseNocCollectiveConfig(j);
          }),
          "zero DCA width is rejected");
    auto dtype_width_json = ProfileConfig("reduce_only");
    dtype_width_json["collective"]["dca"] =
        {{"vector_bits", 96}, {"slice_bits", 32},
         {"slices_per_tile", 3}};
    const auto dtype_width = ParseNocCollectiveConfig(dtype_width_json);
    dtype_width.dca.ValidateForDtype(CollDType::INT32);
    Check(Throws([&] {
              dtype_width.dca.ValidateForDtype(CollDType::INT64);
          }),
          "DCA lane divisibility is checked against the workload dtype");
    Check(Throws([] {
              auto j = ProfileConfig("reduce_only");
              j["collective"]["dca"] = {{"header_fifo_depth", 0}};
              (void)ParseNocCollectiveConfig(j);
          }),
          "zero DCA FIFO depth is rejected");
    Check(Throws([] {
              auto j = ProfileConfig("reduce_only");
              j["collective"]["dca"] =
                  {{"latency", {{"uint8", {{"sum", 0}}}}}};
              (void)ParseNocCollectiveConfig(j);
          }),
          "zero DCA latency is rejected");
    Check(Throws([] {
              auto j = ProfileConfig("reduce_only");
              j["collective"]["dca"] =
                  {{"initiation_interval",
                    {{"int32", {{"max", 0}}}}}};
              (void)ParseNocCollectiveConfig(j);
          }),
          "zero DCA initiation interval is rejected");
    Check(Throws([] {
              auto j = ProfileConfig("reduce_only");
              j["collective"]["dca"] = {{"arbitration", "random"}};
              (void)ParseNocCollectiveConfig(j);
          }),
          "unknown DCA arbitration is rejected");
    Check([] {
              auto j = ProfileConfig("reduce_only");
              j["collective"]["dca"] = {{"value_mode", "fp_exact"}};
              return ParseNocCollectiveConfig(j).dca.value_mode ==
                     NocCollValueMode::FP_EXACT;
          }(),
          "R5 fp_exact mode is accepted and dtype-gated at workload load");
    Check([] {
              auto baseline = ProfileConfig("baseline");
              baseline["collective"]["dca"] =
                  {{"vector_bits", 0}, {"slice_bits", 0},
                   {"slices_per_tile", 0}, {"header_fifo_depth", 0},
                   {"latency", {{"uint8", {{"sum", 0}}}}},
                   {"value_mode", "fp_exact"}};
              auto disabled_dca = ProfileConfig("reduce_only");
              disabled_dca["collective"]["enabled"] = false;
              disabled_dca["collective"]["dca"] =
                  {{"vector_bits", 0}, {"slice_bits", 0},
                   {"slices_per_tile", 0}, {"operand_fifo_depth", 0},
                   {"initiation_interval",
                    {{"int32", {{"max", 0}}}}},
                   {"value_mode", "fp_exact"}};
              try {
                  const auto endpoint =
                      ParseNocCollectiveConfig(baseline);
                  const auto disabled =
                      ParseNocCollectiveConfig(disabled_dca);
                  return endpoint.dca.value_mode ==
                             NocCollValueMode::FP_EXACT &&
                         disabled.dca.value_mode ==
                             NocCollValueMode::FP_EXACT;
              } catch (...) {
                  return false;
              }
          }(),
          "inactive DCA settings do not fail baseline or disabled startup");

    auto timing_json = ProfileConfig("reduce_only");
    timing_json["collective"]["dca"] =
        {{"value_mode", "timing_only"},
         {"arbitration", "core_priority"},
         {"latency", {{"fp32", {{"sum", 7}}}}},
         {"initiation_interval", {{"fp32", {{"sum", 2}}}}}};
    const auto timing = ParseNocCollectiveConfig(timing_json);
    Check(timing.dca.value_mode == NocCollValueMode::TIMING_ONLY &&
              timing.dca.arbitration ==
                  NocCollDcaArbitration::CORE_PRIORITY &&
              timing.dca.timing[3][0].latency == 7 &&
              timing.dca.timing[3][0].initiation_interval == 2,
          "timing-only DCA overrides parse without enabling fp_exact");
    Check(Throws([] {
              auto j = ProfileConfig("reduce_only");
              j["collective"]["dca"] =
                  {{"latency", {{"bf16", {{"sum", 3}}}}}};
              (void)ParseNocCollectiveConfig(j);
          }),
          "unknown DCA timing dtype is rejected");
    Check(Throws([] {
              auto j = ProfileConfig("reduce_only");
              j["collective"]["dca"] =
                  {{"latency", {{"uint8", {{"mul", 3}}}}}};
              (void)ParseNocCollectiveConfig(j);
          }),
          "unknown DCA timing operation is rejected");
    Check(Throws([] {
              auto j = ProfileConfig("reduce_only");
              j["collective"]["dca"] = {{"magic_speed", 3}};
              (void)ParseNocCollectiveConfig(j);
          }),
          "unknown DCA field is rejected");

    const auto base_tree = NocCollTreeUseFor(baseline, CollOp::ALLREDUCE);
    Check(!base_tree.multicast && !base_tree.reduce,
          "baseline programs no hardware collective tree");
    const auto bcast_tree =
        NocCollTreeUseFor(broadcast, CollOp::BROADCAST);
    Check(bcast_tree.multicast && !bcast_tree.reduce,
          "broadcast_only programs only multicast tree for Broadcast");
    const auto endpoint_allreduce_tree =
        NocCollTreeUseFor(broadcast, CollOp::ALLREDUCE);
    Check(!endpoint_allreduce_tree.multicast &&
              !endpoint_allreduce_tree.reduce,
          "broadcast_only endpoint AllReduce programs no orphan tree");
    const auto reduce_tree = NocCollTreeUseFor(reduce, CollOp::ALLREDUCE);
    Check(!reduce_tree.multicast && reduce_tree.reduce,
          "reduce_only AllReduce programs reduce but no multicast tree");
    const auto combined_tree =
        NocCollTreeUseFor(combined, CollOp::ALLREDUCE);
    Check(combined_tree.multicast && combined_tree.reduce,
          "reduce_broadcast AllReduce programs both trees");
    const auto reduce_broadcast_tree =
        NocCollTreeUseFor(reduce, CollOp::BROADCAST);
    Check(!reduce_broadcast_tree.multicast && !reduce_broadcast_tree.reduce,
          "reduce_only leaves standalone Broadcast on unicast");
    const auto legacy_tree = NocCollTreeUseFor(legacy, CollOp::ALLREDUCE);
    Check(legacy_tree.multicast && legacy_tree.reduce,
          "legacy debug Tier2 retains both frozen tree registries");
    Check(Throws([] {
              auto j = ProfileConfig("baseline");
              j["collective"]["typo"] = true;
              (void)ParseNocCollectiveConfig(j);
          }),
          "unknown collective field is rejected");
    Check(Throws([] {
              auto j = ProfileConfig("baseline");
              j["collective"]["reduce_wire"] = "legacy_two_segment";
              (void)ParseNocCollectiveConfig(j);
          }),
          "endpoint backend cannot select legacy reduce wire");
    Check(Throws([] {
              (void)ParseNocCollectiveConfig(
                  {{"collective", {{"tier", 3}}}});
          }),
          "out-of-range tier alias is rejected");

    std::cout << "NoC collective refactor R1 self-test: "
              << (failures == 0 ? "PASS" : "FAIL") << " (" << checks
              << " checks)" << std::endl;
    return failures;
}
