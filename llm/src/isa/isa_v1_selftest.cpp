#include "isa/isa_v1_selftest.h"

#include "dte/coll_plan_v1_selftest.h"
#include "dte/coll_byte_wire_v1_selftest.h"
#include "dte/coll_dca_payload_v1_selftest.h"
#include "dte/coll_profile_v1_selftest.h"
#include "dte/coll_accel_runtime_v1_selftest.h"
#include "dte/coll_program_profile_v1_selftest.h"
#include "dte/coll_topology_v1_selftest.h"
#include "dte/coll_tree_batch_v1_selftest.h"
#include "dte/collective_executor_v1.h"
#include "dte/collective_aggregate_v1_selftest.h"
#include "dte/collective_data_v1_selftest.h"
#include "dte/collective_final_phase_gate_v1_selftest.h"
#include "dte/collective_wave_admission_v1_selftest.h"
#include "isa/collective_child_endpoint_v1_selftest.h"
#include "isa/collective_data_lowering_v1_selftest.h"
#include "isa/collective_graph_v1_selftest.h"
#include "isa/collective_phase_lowering_v1_selftest.h"
#include "isa/collective_program_v1_selftest.h"
#include "isa/npu_cost_model_selftest.h"

#include "isa/opcode.h"
#include "isa/prim_manifest.h"
#include "isa/prim_wire_selftest.h"
#include "isa/program_format_selftest.h"
#include "isa/record_codec_selftest.h"
#include "isa/record_lowering_selftest.h"
#include "prims/collective_data_v1_prim.h"
#include "prims/collective_launch_v1_prim.h"
#include "prims/collective_phase_barrier_v1_prim.h"
#include "prims/dte_endpoint_prims.h"
#include "utils/prim_utils.h"
#include "monitor/config_helper_program_selftest.h"
#include "router/endpoint_output_flow_lock_selftest.h"
#include "workercore/serialized_wire_queue_selftest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

void Check(IsaV1SelfTestResult &result, bool condition,
           std::string description) {
    ++result.checks;
    if (!condition)
        result.failures.push_back(std::move(description));
}

void Append(IsaV1SelfTestResult &result,
            IsaV1SelfTestResult additional) {
    result.checks += additional.checks;
    result.failures.insert(result.failures.end(),
                           std::make_move_iterator(additional.failures.begin()),
                           std::make_move_iterator(additional.failures.end()));
}

template <class Error, class Function>
bool ThrowsExactly(Function function, const std::string &expected) {
    try {
        function();
    } catch (const Error &error) {
        return error.what() == expected;
    } catch (...) {
    }
    return false;
}

std::size_t CountCategory(OpcodeCategory category) {
    std::size_t count = 0;
    for (const OpcodeManifestEntry &entry : OpcodeManifest()) {
        if (entry.category == category)
            ++count;
    }
    return count;
}

void CheckContiguousRange(IsaV1SelfTestResult &result, uint8_t first,
                          uint8_t last, OpcodeCategory category) {
    for (unsigned value = first; value <= last; ++value) {
        const OpcodeManifestEntry *entry =
            LookupOpcode(static_cast<uint8_t>(value));
        Check(result, entry != nullptr,
              "assigned opcode missing from manifest: " +
                  std::to_string(value));
        if (entry != nullptr) {
            Check(result, entry->category == category,
                  "assigned opcode has wrong category: " +
                      std::to_string(value));
        }
    }
}

int CategoryBit(PrimCategory category, PrimId id) {
    switch (category) {
    case PrimCategory::COMPUTE: return COMP_PRIM;
    case PrimCategory::COMMUNICATION: return COMM_PRIM;
    case PrimCategory::MEMORY: return MEM_PRIM;
    case PrimCategory::SYNCHRONIZATION: return SYNC_PRIM;
    case PrimCategory::DYNAMIC:
        // Default-constructed DTE is remote ISSUE; default LSU is ISSUE.
        return id == PrimId::DTE_ASYNC ? COMM_PRIM : MEM_PRIM;
    }
    return 0;
}

} // namespace

IsaV1SelfTestResult CheckIsaV1OpcodeManifest() {
    IsaV1SelfTestResult result;
    std::string manifest_error;
    const bool manifest_valid = ValidateOpcodeManifest(&manifest_error);
    Check(result, manifest_valid, "manifest invariants: " + manifest_error);
    Check(result, OpcodeManifest().size() == kOpcodeManifestSize,
          "manifest has frozen entry count");

    std::set<uint8_t> values;
    std::set<std::string> names;
    for (const OpcodeManifestEntry &entry : OpcodeManifest()) {
        Check(result, values.insert(OpcodeValue(entry.opcode)).second,
              "opcode values are unique");
        Check(result, names.insert(std::string(entry.canonical_name)).second,
              "canonical names are unique");
        Check(result, entry.visibility == OpcodeVisibility::PUBLIC,
              "external manifest contains only public entries");
    }

    Check(result, CountCategory(OpcodeCategory::COMPUTE) == 25,
          "compute range contains 25 assigned opcodes");
    Check(result, CountCategory(OpcodeCategory::COMMUNICATION) == 3,
          "communication range contains 3 assigned opcodes");
    Check(result, CountCategory(OpcodeCategory::MEMORY) == 9,
          "memory range contains 9 assigned opcodes");
    Check(result, CountCategory(OpcodeCategory::SYNCHRONIZATION) == 7,
          "synchronization range contains 7 assigned opcodes");
    CheckContiguousRange(result, kComputeOpcodeFirst, kComputeOpcodeLast,
                         OpcodeCategory::COMPUTE);
    CheckContiguousRange(result, kCommunicationOpcodeFirst,
                         kCommunicationOpcodeLast,
                         OpcodeCategory::COMMUNICATION);
    CheckContiguousRange(result, kMemoryOpcodeFirst, kMemoryOpcodeLast,
                         OpcodeCategory::MEMORY);
    CheckContiguousRange(result, kSynchronizationOpcodeFirst,
                         kSynchronizationOpcodeLast,
                         OpcodeCategory::SYNCHRONIZATION);

    const OpcodeManifestEntry *poll = LookupOpcode(Opcode::DTE_POLL);
    Check(result,
          poll != nullptr && poll->lifecycle == OpcodeLifecycle::RESERVED &&
              poll->support == OpcodeSupport::UNSUPPORTED,
          "DTE_POLL is a known reserved and unsupported opcode");
    Check(result, ValidateOpcode(Opcode::DTE_POLL) ==
                      OpcodeValidation::UNSUPPORTED,
          "known DTE_POLL is rejected as unsupported");

    constexpr std::array<Opcode, 3> kUnsupportedReserved{{
        Opcode::BATCHNORM,
        Opcode::SPLIT_CONV,
        Opcode::MERGE_CONV,
    }};
    for (Opcode opcode : kUnsupportedReserved) {
        const OpcodeManifestEntry *entry = LookupOpcode(opcode);
        Check(result,
              entry != nullptr &&
                  entry->lifecycle == OpcodeLifecycle::RESERVED &&
                  entry->support == OpcodeSupport::UNSUPPORTED,
              "unimplemented compute opcode is reserved and unsupported");
        Check(result, ValidateOpcode(opcode) == OpcodeValidation::UNSUPPORTED,
              "known unimplemented opcode is rejected as unsupported");
    }

    const OpcodeManifestEntry *experimental =
        LookupOpcode(Opcode::GEMM_REDUCE_SCATTER);
    Check(result,
          experimental != nullptr &&
              experimental->lifecycle == OpcodeLifecycle::RESERVED &&
              experimental->support == OpcodeSupport::EXPERIMENTAL,
          "experimental fused opcode is reserved and default-off");
    Check(result, ValidateOpcode(Opcode::GEMM_REDUCE_SCATTER) ==
                      OpcodeValidation::RESERVED,
          "experimental reserved opcode is rejected as reserved");

    constexpr std::array<Opcode, 4> kGated{{
        Opcode::MATMUL_MLA,
        Opcode::MATMUL_PD,
        Opcode::ATTENTION_PD,
        Opcode::ROPE_PD,
    }};
    for (Opcode opcode : kGated) {
        const OpcodeManifestEntry *entry = LookupOpcode(opcode);
        Check(result,
              entry != nullptr && entry->support == OpcodeSupport::EXPERIMENTAL,
              "program-context opcode is capability gated");
        Check(result, ValidateOpcode(opcode) == OpcodeValidation::GATED,
              "gated opcode is rejected with gated status");
        Check(result,
              ValidateOpcode(OpcodeValue(opcode),
                             CapabilityBit(IsaCapability::PD_CONTEXT)) ==
                  OpcodeValidation::AVAILABLE,
              "experimental opcode is available when capability is enabled");
    }

    constexpr std::array<uint8_t, 7> kReservedEncoding{{
        0x00, 0x1a, 0x43, 0x89, 0xc7, 0xf0, 0xff,
    }};
    for (uint8_t value : kReservedEncoding) {
        Check(result, LookupOpcode(value) == nullptr,
              "unassigned opcode has no manifest entry");
        Check(result, ValidateOpcode(value) == OpcodeValidation::RESERVED,
              "unassigned 8-bit opcode is rejected as reserved");
    }
    Check(result, ValidateOpcodeValue(uint16_t{256}) ==
                      OpcodeValidation::UNKNOWN,
          "value outside the 8-bit opcode space is unknown");

    Check(result, LookupOpcode("MATMUL") == LookupOpcode(Opcode::MATMUL),
          "canonical-name lookup resolves the same manifest entry");
    Check(result, LookupOpcode("GLOBAL_LOAD") == nullptr,
          "unsupported global-memory operations own no v1 opcode");
    Check(result, ValidateOpcode(Opcode::MATMUL) ==
                      OpcodeValidation::AVAILABLE,
          "available opcode validates successfully");

    const OpcodeManifestEntry *matmul = LookupOpcode(Opcode::MATMUL);
    Check(result,
          matmul != nullptr &&
              matmul->lowering.kind == OpcodeLoweringKind::DIRECT_PRIM &&
              matmul->lowering.target == PrimId::MATMUL_F &&
              matmul->lowering.variant == OpcodeLoweringVariant::NONE,
          "direct compute lowering names its stable PrimId");

    const OpcodeManifestEntry *lsu_load = LookupOpcode(Opcode::LSU_LOAD);
    Check(result,
          lsu_load != nullptr &&
              lsu_load->lowering.kind == OpcodeLoweringKind::PRIM_VARIANT &&
              lsu_load->lowering.target == PrimId::LSU_MEM &&
              lsu_load->lowering.variant ==
                  OpcodeLoweringVariant::LSU_LOAD_BLOCKING,
          "shared LSU primitive lowering carries a blocking-load variant");

    const OpcodeManifestEntry *send = LookupOpcode(Opcode::DTE_SEND);
    Check(result,
          send != nullptr &&
              send->lowering.kind == OpcodeLoweringKind::MODE_DISPATCH &&
              send->lowering.target == PrimId::INVALID &&
              send->lowering.variant == OpcodeLoweringVariant::DTE_SEND_MODE,
          "DTE_SEND lowering explicitly delegates mode dispatch");

    const OpcodeManifestEntry *alloc = LookupOpcode(Opcode::SRAM_ALLOC);
    Check(result,
          alloc != nullptr &&
              alloc->lowering.kind == OpcodeLoweringKind::NEW_THIN_PRIM &&
              alloc->lowering.target == PrimId::INVALID &&
              alloc->lowering.variant == OpcodeLoweringVariant::NONE,
          "new SRAM control primitive has an explicit thin-primitive marker");

    return result;
}

IsaV1SelfTestResult CheckIsaV1PrimManifest() {
    IsaV1SelfTestResult result;
    std::string error;
    const bool valid = ValidatePrimManifest(&error);
    Check(result, valid, "Prim manifest invariants: " + error);
    Check(result, PrimManifest().size() == kPrimManifestSize,
          "Prim manifest has frozen 60-entry count");
    Check(result, kMaxAssignedPrimId <= UINT8_MAX,
          "all internal PrimIds fit the 8-bit wire");

    std::size_t compute = 0;
    std::size_t communication = 0;
    std::size_t memory = 0;
    std::size_t synchronization = 0;
    std::size_t dynamic = 0;
    std::size_t public_count = 0;
    std::size_t unsupported = 0;
    std::size_t experimental = 0;
    std::size_t deprecated = 0;
    std::set<uint8_t> public_ids;
    std::set<uint8_t> unsupported_ids;
    std::set<uint8_t> experimental_ids;
    std::set<uint8_t> deprecated_ids;
    for (std::size_t i = 0; i < PrimManifest().size(); ++i) {
        const PrimManifestEntry &entry = PrimManifest()[i];
        Check(result, PrimIdValue(entry.id) == i + 1,
              "Prim manifest is sorted and contiguous");
        Check(result, LookupPrim(entry.id) == &entry,
              "PrimId lookup returns canonical entry");
        Check(result, LookupPrim(entry.factory_name) == &entry,
              "factory-name lookup returns canonical entry");
        switch (entry.primary_category) {
        case PrimCategory::COMPUTE: ++compute; break;
        case PrimCategory::COMMUNICATION: ++communication; break;
        case PrimCategory::MEMORY: ++memory; break;
        case PrimCategory::SYNCHRONIZATION: ++synchronization; break;
        case PrimCategory::DYNAMIC: ++dynamic; break;
        }
        if (entry.visibility == PrimVisibility::PUBLIC) {
            ++public_count;
            public_ids.insert(PrimIdValue(entry.id));
        }
        if (entry.support == PrimSupport::UNSUPPORTED) {
            ++unsupported;
            unsupported_ids.insert(PrimIdValue(entry.id));
        }
        if (entry.support == PrimSupport::EXPERIMENTAL) {
            ++experimental;
            experimental_ids.insert(PrimIdValue(entry.id));
        }
        if (entry.lifecycle == PrimLifecycle::DEPRECATED) {
            ++deprecated;
            deprecated_ids.insert(PrimIdValue(entry.id));
        }
    }
    Check(result, compute == 32 && communication == 8 && memory == 14 &&
                      synchronization == 4 && dynamic == 2,
          "Prim primary-category counts match frozen inventory");
    Check(result, public_count == 25,
          "Prim visibility counts match frozen inventory");
    Check(result, unsupported == 5 && experimental == 6 && deprecated == 2,
          "Prim lifecycle/support counts match frozen inventory");

    const std::set<uint8_t> expected_public{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
        17, 18, 19, 20, 22, 23, 24, 25, 35, 49, 50, 51,
    };
    const std::set<uint8_t> expected_unsupported{2, 12, 16, 21, 23};
    const std::set<uint8_t> expected_experimental{7, 10, 47, 49, 50, 51};
    const std::set<uint8_t> expected_deprecated{40, 48};
    Check(result, public_ids == expected_public,
          "exact public PrimId set matches frozen inventory");
    Check(result, unsupported_ids == expected_unsupported,
          "exact unsupported PrimId set matches frozen inventory");
    Check(result, experimental_ids == expected_experimental,
          "exact experimental PrimId set matches frozen inventory");
    Check(result, deprecated_ids == expected_deprecated,
          "exact deprecated PrimId set matches frozen inventory");


    std::vector<PrimManifestEntry> forward(PrimManifest().begin(),
                                           PrimManifest().end());
    std::vector<PrimManifestEntry> reverse = forward;
    std::reverse(reverse.begin(), reverse.end());
    error.clear();
    const bool forward_valid = ValidatePrimManifestEntries(forward, &error);
    Check(result, forward_valid, "forward Prim inventory validates: " + error);
    error.clear();
    const bool reverse_valid = ValidatePrimManifestEntries(reverse, &error);
    Check(result, reverse_valid, "reverse Prim inventory validates: " + error);

    auto build_maps = [](const std::vector<PrimManifestEntry> &entries) {
        std::pair<std::map<uint8_t, std::string>,
                  std::map<std::string, uint8_t>> maps;
        for (const PrimManifestEntry &entry : entries) {
            const uint8_t raw = PrimIdValue(entry.id);
            maps.first.emplace(raw, std::string(entry.factory_name));
            maps.second.emplace(std::string(entry.factory_name), raw);
        }
        return maps;
    };
    Check(result, build_maps(forward) == build_maps(reverse),
          "forward/reverse inventory order does not change mappings");

    std::vector<PrimManifestEntry> invalid = forward;
    invalid.front().id = PrimId::INVALID;
    error.clear();
    Check(result, !ValidatePrimManifestEntries(invalid, &error) &&
                      error == "PrimId::INVALID cannot appear in manifest",
          "INVALID PrimId has stable manifest error");

    std::vector<PrimManifestEntry> duplicate_id = forward;
    duplicate_id[1].id = duplicate_id[0].id;
    error.clear();
    Check(result, !ValidatePrimManifestEntries(duplicate_id, &error) &&
                      error == "duplicate PrimId 1",
          "duplicate PrimId has stable manifest error");

    std::vector<PrimManifestEntry> duplicate_name = forward;
    duplicate_name[1].factory_name = duplicate_name[0].factory_name;
    error.clear();
    Check(result, !ValidatePrimManifestEntries(duplicate_name, &error) &&
                      error == "duplicate factory name: Attention_f",
          "duplicate factory name has stable manifest error");

    Check(result, LookupPrim(uint16_t{0}) == nullptr &&
                      LookupPrim(uint16_t{61}) == nullptr &&
                      LookupPrim(uint16_t{256}) == nullptr,
          "invalid/out-of-range/unknown PrimIds do not resolve");
    Check(result, LookupPrim("__isa_v1_unknown_prim__") == nullptr,
          "unknown factory name does not resolve");
    return result;
}

IsaV1SelfTestResult CheckIsaV1PrimFactory() {
    IsaV1SelfTestResult result;
    PrimFactory &factory = PrimFactory::getInstance();
    Check(result, factory.registeredCount() == kPrimManifestSize,
          "PrimFactory runtime count matches 60-entry manifest");
    if (factory.registeredCount() != kPrimManifestSize)
        return result;

    std::vector<int> registered = factory.registeredIds();
    std::sort(registered.begin(), registered.end());
    Check(result, registered.size() == kPrimManifestSize,
          "PrimFactory exposes all registered IDs");
    for (std::size_t i = 0; i < registered.size(); ++i) {
        Check(result, registered[i] == static_cast<int>(i + 1),
              "PrimFactory IDs are contiguous and stable");
    }

    for (const PrimManifestEntry &entry : PrimManifest()) {
        const int raw_id = PrimIdValue(entry.id);
        Check(result, factory.getPrimId(std::string(entry.factory_name)) == raw_id,
              "factory name resolves to frozen PrimId");
        Check(result, factory.getPrimType(raw_id) == entry.factory_name,
              "factory PrimId resolves to frozen name");
        PrimBase *prim = factory.createPrim(raw_id, false, false);
        Check(result, prim != nullptr,
              "default factory creator returns a primitive");
        if (prim == nullptr)
            continue;
        Check(result, prim->name == entry.factory_name,
              "default creator name matches manifest");
        Check(result, HasExactlyOnePrimMainCategory(prim->prim_type),
              "default creator has exactly one primary category");
        Check(result,
              PrimMainCategoryBits(prim->prim_type) ==
                  CategoryBit(entry.primary_category, entry.id),
              "default creator category matches manifest/current dynamic state");

        if (entry.id == PrimId::DTE_ASYNC) {
            auto *dte = dynamic_cast<Dte_async_prim *>(prim);
            Check(result, dte != nullptr,
                  "dynamic DTE manifest target has expected runtime type");
            if (dte != nullptr) {
                dte->direction = DteDir::SPM_TO_SPM;
                dte->refreshPrimType();
                Check(result, PrimMainCategoryBits(dte->prim_type) == MEM_PRIM,
                      "local DTE ISSUE dynamically classifies as memory");
                dte->op = DteAsyncOp::WAIT;
                dte->refreshPrimType();
                Check(result, PrimMainCategoryBits(dte->prim_type) == SYNC_PRIM,
                      "DTE WAIT dynamically classifies as synchronization");
            }
        }
        if (entry.id == PrimId::SRAM_BIND_ONESHOT) {
            auto *bind = dynamic_cast<Sram_bind_oneshot *>(prim);
            Check(result, bind != nullptr,
                  "SRAM_BIND_ONESHOT manifest target has expected runtime type");
            if (bind != nullptr)
                Check(result, bind->input_count == 0,
                      "SRAM_BIND_ONESHOT starts without a valid binding");
        }
        if (entry.id == PrimId::LSU_MEM) {
            auto *lsu = dynamic_cast<Lsu_mem_prim *>(prim);
            Check(result, lsu != nullptr,
                  "dynamic LSU manifest target has expected runtime type");
            if (lsu != nullptr) {
                lsu->op = LsuMemOp::WAIT;
                lsu->refreshPrimType();
                Check(result, PrimMainCategoryBits(lsu->prim_type) == SYNC_PRIM,
                      "LSU WAIT dynamically classifies as synchronization");
            }
        }
        if (entry.id == PrimId::DTE_SEND_ENDPOINT) {
            Check(result,
                  dynamic_cast<Dte_send_endpoint_prim *>(prim) != nullptr,
                  "DTE_SEND_ENDPOINT manifest target has expected runtime type");
        }
        if (entry.id == PrimId::DTE_RECV_ENDPOINT) {
            Check(result,
                  dynamic_cast<Dte_recv_endpoint_prim *>(prim) != nullptr,
                  "DTE_RECV_ENDPOINT manifest target has expected runtime type");
        }
        if (entry.id == PrimId::COLLECTIVE_DATA_V1) {
            Check(result,
                  dynamic_cast<Collective_data_v1_prim *>(prim) != nullptr,
                  "COLLECTIVE_DATA_V1 manifest target has expected runtime type");
        }
        if (entry.id == PrimId::COLLECTIVE_PHASE_BARRIER_V1) {
            Check(result,
                  dynamic_cast<Collective_phase_barrier_v1_prim *>(prim) != nullptr,
                  "COLLECTIVE_PHASE_BARRIER_V1 manifest target has expected runtime type");
        }
        if (entry.id == PrimId::COLLECTIVE_LAUNCH_V1) {
            Check(result,
                  dynamic_cast<Collective_launch_v1_prim *>(prim) != nullptr,
                  "COLLECTIVE_LAUNCH_V1 manifest target has expected runtime type");
        }
        delete prim;
    }

    const PrimFactory::CreatorFunc unused_creator =
        []() -> PrimBase * { return nullptr; };
    Check(result,
          ThrowsExactly<std::logic_error>(
              [&] {
                  factory.registerPrim("Attention_f", PrimId::ROPE_FORWARD,
                                       unused_creator);
              },
              "duplicate primitive name: Attention_f"),
          "PrimFactory duplicate name has stable error");
    Check(result,
          ThrowsExactly<std::invalid_argument>(
              [&] {
                  factory.registerPrim("__isa_v1_name_id_mismatch__",
                                       PrimId::ATTENTION_F, unused_creator);
              },
              "primitive name/ID pair disagrees with frozen manifest: "
              "__isa_v1_name_id_mismatch__/1"),
          "PrimFactory name/ID mismatch has stable error");
    Check(result,
          ThrowsExactly<std::invalid_argument>(
              [&] {
                  factory.registerPrim("__isa_v1_id_61__",
                                       static_cast<PrimId>(61), unused_creator);
              },
              "unassigned PrimId cannot be registered: 61"),
          "PrimFactory rejects first unassigned PrimId");
    Check(result,
          ThrowsExactly<std::invalid_argument>(
              [&] {
                  factory.registerPrim("__isa_v1_id_255__",
                                       static_cast<PrimId>(255), unused_creator);
              },
              "unassigned PrimId cannot be registered: 255"),
          "PrimFactory rejects high unassigned 8-bit PrimId");
    Check(result,
          ThrowsExactly<std::invalid_argument>(
              [&] {
                  factory.registerPrim("__isa_v1_invalid_id__",
                                       PrimId::INVALID, unused_creator);
              },
              "PrimId::INVALID cannot be registered"),
          "PrimFactory INVALID ID has stable error");
    Check(result,
          ThrowsExactly<std::out_of_range>(
              [&] { (void)factory.createPrim(0, false, false); },
              "primitive ID is outside the 8-bit wire"),
          "PrimFactory invalid numeric ID has stable error");
    Check(result,
          ThrowsExactly<std::invalid_argument>(
              [&] { (void)factory.createPrim(255, false, false); },
              "unregistered primitive ID 255"),
          "PrimFactory unknown numeric ID has stable error");
    Check(result, factory.getPrimId("__isa_v1_unknown_prim__") == -1 &&
                      factory.getPrimType(255).empty(),
          "PrimFactory unknown name/ID lookup has stable sentinel");
    Check(result,
          ThrowsExactly<std::invalid_argument>(
              [&] {
                  (void)factory.createPrim("__isa_v1_unknown_prim__", false,
                                           false);
              },
              "unregistered primitive type __isa_v1_unknown_prim__"),
          "PrimFactory unknown name has stable error");
    Check(result, factory.registeredCount() == kPrimManifestSize,
          "negative factory probes do not mutate registration state");
    return result;
}

int RunIsaV1SelfTest() {
    IsaV1SelfTestResult result = CheckIsaV1OpcodeManifest();
    Append(result, CheckIsaV1PrimManifest());
    Append(result, CheckIsaV1PrimFactory());
    Append(result, CheckNpuCostModelSelfTest());
    for (const std::string &failure : result.failures)
        std::cerr << "[ISA_V1] FAIL " << failure << '\n';
    std::cout << "ISA v1 manifest/factory self-test: "
              << (result.passed()
                      ? "PASS"
                      : "FAILURES=" + std::to_string(result.failures.size()))
              << " (" << result.checks << " checks)\n";
    const int record_failures = RunIsaV1RecordCodecSelfTest();
    const int wire_failures = RunPrimWireSelfTest();
    const int format_failures = RunProgramFormatV1SelfTest();
    const int lowering_failures = RunIsaV1RecordLoweringSelfTest();
    const int helper_failures = RunConfigHelperProgramSelfTest();
    const int coll_plan_failures = RunCollPlanV1SelfTest();
    const int coll_byte_wire_failures =
        RunIsaV1CollectiveByteWireSelfTest();
    const int coll_dca_payload_failures =
        RunIsaV1DcaPayloadSelfTest();
    const int coll_data_failures = RunCollectiveDataV1SelfTest();
    const int coll_graph_failures = RunCollectiveGraphV1SelfTest();
    const int coll_data_lowering_failures =
        RunIsaV1CollectiveDataLoweringSelfTest();
    const int coll_child_endpoint_failures =
        RunIsaV1CollectiveChildEndpointSelfTest();
    const int coll_aggregate_failures =
        RunCollectiveAggregateV1SelfTest();
    const int coll_wave_admission_failures =
        RunCollectiveWaveAdmissionV1SelfTest();
    const int coll_final_phase_failures =
        RunCollectiveFinalPhaseGateV1SelfTest();
    const int coll_executor_failures =
        RunCollectiveExecutorV1SelfTest();
    const int coll_profile_failures =
        RunIsaV1CollectiveProfileSelfTest();
    const int coll_accel_runtime_failures =
        RunIsaV1CollectiveAccelerationRuntimeSelfTest();
    const int coll_program_profile_failures =
        RunIsaV1CollectiveProgramProfileSelfTest();
    const int coll_topology_failures =
        RunIsaV1CollectiveTopologySelfTest();
    const int coll_tree_batch_failures =
        RunIsaV1CollectiveTreeBatchSelfTest();
    const int coll_phase_lowering_failures =
        RunIsaV1CollectivePhaseLoweringSelfTest();
    const int coll_program_image_failures =
        RunIsaV1CollectiveProgramImageSelfTest();
    const int serialized_wire_queue_failures =
        RunSerializedWireQueueSelfTest();
    const int endpoint_output_flow_lock_failures =
        RunEndpointOutputFlowLockSelfTest();
    return static_cast<int>(result.failures.size()) + record_failures +
           wire_failures + format_failures + lowering_failures +
           helper_failures + coll_plan_failures + coll_byte_wire_failures +
           coll_dca_payload_failures + coll_data_failures +
           coll_graph_failures + coll_data_lowering_failures +
           coll_child_endpoint_failures + coll_aggregate_failures +
           coll_wave_admission_failures + coll_final_phase_failures +
           coll_executor_failures + coll_topology_failures +
           coll_profile_failures +
           coll_accel_runtime_failures +
           coll_program_profile_failures +
           coll_tree_batch_failures + coll_phase_lowering_failures +
           coll_program_image_failures + serialized_wire_queue_failures +
           endpoint_output_flow_lock_failures;
}
