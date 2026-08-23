#include "frontend/program_finalizer.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Json = nlohmann::json;
using frontend::ProgramArtifactFinalizer;
using frontend::ProgramFinalizerError;
using frontend::AddressOperandBindingDto;
using frontend::CommandFragmentDto;
using frontend::FragmentKindDto;
using frontend::LinkedFragmentDto;
using frontend::StateAccessDto;
using frontend::ProgramSymbolDefinitionDto;
using frontend::StateOperandBindingDto;

std::string Sha256(const std::vector<uint8_t> &input);

std::string CanonicalDigest(const Json &value) {
    const std::string canonical = value.dump();
    return Sha256(std::vector<uint8_t>(canonical.begin(), canonical.end()));
}

std::string StableId(const std::string &kind, const std::string &schema,
                     const Json &semantic) {
    const Json identity{{"kind", kind},
                        {"schema_version", schema},
                        {"semantic_key", semantic}};
    return kind + "_" + CanonicalDigest(identity).substr(0, 16);
}

Json Without(Json value, std::initializer_list<const char *> fields) {
    for (const char *field : fields)
        value.erase(field);
    return value;
}

Json Core(uint64_t die, uint64_t local) {
    return {{"die_id", die}, {"local_core_id", local}};
}

Json Literal(const std::string &name, Json value) {
    return {{"name", name},
            {"kind", "literal"},
            {"literal_value", std::move(value)},
            {"runtime_field", nullptr},
            {"operand_id", nullptr},
            {"symbol_ref", nullptr}};
}

Json Address(const std::string &name, uint64_t operand_id,
             const std::string &symbol) {
    return {{"name", name},
            {"kind", "address_symbol"},
            {"literal_value", nullptr},
            {"runtime_field", nullptr},
            {"operand_id", operand_id},
            {"symbol_ref", symbol}};
}

Json Record(const std::string &action, uint64_t opcode, Json operands) {
    return {{"source_global_action_id", action},
            {"opcode", opcode},
            {"operands", std::move(operands)}};
}

Json Alloc(const std::string &action, const std::string &label,
           uint64_t offset) {
    return Record(action, 0x89,
                  Json::array({Address("region_name", 8, "p_region"),
                               Address("label_symbol", 9, label),
                               Literal("region_offset_bytes", offset),
                               Literal("size_bytes", 64),
                               Literal("alignment_bytes", 16),
                               Literal("lifetime", 0),
                               Literal("spillable", false)}));
}

Json Bind(const std::string &action) {
    Json operands = Json::array();
    operands.push_back(Literal("input_count", 1));
    for (uint64_t index = 0; index < 16; ++index) {
        const std::string name = "input_label_" + std::to_string(index);
        if (index == 0)
            operands.push_back(Address(name, 0x100 + index, "p_label_input"));
        else
            operands.push_back(Literal(name, 0));
    }
    operands.push_back(Address("output_label", 0x110, "p_label_output"));
    return Record(action, 0x84, std::move(operands));
}

Json Matmul(const std::string &action) {
    return Record(action, 0x01,
                  Json::array({Literal("datatype", 1),
                               Address("input_address", 1, "p_abs_input"),
                               Address("data_address", 2, "p_abs_data"),
                               Address("output_address", 3, "p_abs_output"),
                               Literal("parameters", Json::array({1, 2, 4, 8}))}));
}

Json Stage2Compute(const std::string &action, uint64_t opcode) {
    if (opcode == 0x1a)
        return Record(action, opcode, Json::array({
            Literal("datatype", 1), Literal("packed_layout", 0),
            Address("input_address", 1, "p_abs_input"),
            Address("output_address", 3, "p_abs_output"),
            Literal("logical_tokens", 1), Literal("tp_degree", 1),
            Literal("num_heads", 2), Literal("num_kv_heads", 1),
            Literal("rank_num_heads", 2), Literal("rank_num_kv_heads", 1),
            Literal("head_dim", 2), Literal("rotary_dim", 2),
            Literal("max_position_embeddings", 8), Literal("context_max", 1),
            Literal("rope_theta_f64_bits", uint64_t{0x40c3880000000000})}));
    if (opcode == 0x1b)
        return Record(action, opcode, Json::array({
            Literal("datatype", 1), Literal("mode", 0),
            Literal("packed_layout", 0), Literal("causal", true),
            Address("input_address", 1, "p_abs_input"),
            Address("output_address", 3, "p_abs_output"),
            Literal("query_tokens", 1), Literal("tp_degree", 1),
            Literal("num_heads", 2), Literal("num_kv_heads", 1),
            Literal("rank_num_heads", 2), Literal("rank_num_kv_heads", 1),
            Literal("head_dim", 2), Literal("context_sum", 1),
            Literal("context_max", 1), Literal("query_key_pairs", 1),
            Literal("rank_kv_read_bytes", 0),
            Literal("rank_kv_write_bytes", 8)}));
    if (opcode == 0x1c)
        return Record(action, opcode, Json::array({
            Literal("index_datatype", 2), Literal("table_datatype", 1),
            Literal("output_datatype", 1), Literal("placement", 0),
            Address("indices_address", 1, "p_abs_input"),
            Address("table_address", 2, "p_abs_data"),
            Address("output_address", 3, "p_abs_output"),
            Literal("logical_rows", 1), Literal("rank_rows", 1),
            Literal("tp_degree", 1), Literal("vocab_size", 8),
            Literal("hidden_size", 8)}));
    if (opcode == 0x1d)
        return Record(action, opcode, Json::array({
            Literal("logits_datatype", 1), Literal("output_datatype", 2),
            Literal("mode", 0), Literal("row_selection", 0),
            Address("logits_address", 1, "p_abs_input"),
            Address("output_address", 3, "p_abs_output"),
            Literal("tp_degree", 1), Literal("token_rows", 1),
            Literal("vocab_size", 8), Literal("sample_count", 1),
            Literal("comparisons", 7)}));
    if (opcode == 0x1e)
        return Record(action, opcode, Json::array({
            Literal("logits_datatype", 1), Literal("label_datatype", 2),
            Literal("loss_datatype", 3), Literal("reduction", 0),
            Address("logits_address", 1, "p_abs_input"),
            Address("labels_address", 2, "p_abs_data"),
            Address("loss_address", 3, "p_abs_output"),
            Literal("logical_rows", 8), Literal("rank_rows", 4),
            Literal("tp_degree", 2), Literal("vocab_size", 32)}));
    if (opcode == 0x1f)
        return Record(action, opcode, Json::array({
            Literal("logits_datatype", 1), Literal("label_datatype", 2),
            Literal("upstream_datatype", 3), Literal("output_datatype", 1),
            Literal("reduction", 0), Literal("upstream_mode", 1),
            Address("logits_address", 1, "p_abs_input"),
            Address("labels_address", 2, "p_abs_data"),
            Address("upstream_address", 12, "p_abs_aux"),
            Address("logits_grad_address", 3, "p_abs_output"),
            Literal("logical_rows", 8), Literal("rank_rows", 4),
            Literal("tp_degree", 2), Literal("vocab_size", 32),
            Literal("upstream_elements", 4)}));
    if (opcode == 0x20)
        return Record(action, opcode, Json::array({
            Literal("weight_datatype", 1),
            Literal("gradient_datatype", 3),
            Literal("output_datatype", 1), Literal("rounding", 0),
            Address("weight_address", 1, "p_abs_input"),
            Address("gradient_address", 2, "p_abs_data"),
            Address("updated_weight_address", 3, "p_abs_input"),
            Literal("element_count", 32),
            Literal("learning_rate_f64_bits", UINT64_C(0x3fc0000000000000)),
            Literal("momentum_f64_bits", 0)}));
    if (opcode == 0x10)
        return Record(action, opcode, Json::array({
            Literal("datatype", 1),
            Address("input_address", 1, "p_abs_input"),
            Address("data_address", 2, "p_abs_data"),
            Address("output_address", 3, "p_abs_output"),
            Literal("parameters", Json::array({1, 1, 32}))}));
    throw std::runtime_error("unsupported Stage2 synthetic opcode");
}

Json Free(const std::string &action, const std::string &label) {
    return Record(action, 0x86,
                  Json::array({Address("symbol", 7, label)}));
}

Json Relocation(uint64_t record, uint64_t operand, uint64_t kind,
                const std::string &symbol) {
    return {{"record_index", record},
            {"operand_id", operand},
            {"symbol_kind", kind},
            {"symbol_ref", symbol},
            {"addend", 0}};
}

Json Stream(const Json &core, const std::string &action) {
    Json records = Json::array(
        {Alloc(action, "p_label_input", 0x100),
         Alloc(action, "p_label_data", 0x200),
         Alloc(action, "p_label_output", 0x300), Bind(action), Matmul(action),
         Free(action, "p_label_input"), Free(action, "p_label_data"),
         Free(action, "p_label_output")});
    Json relocations = Json::array(
        {Relocation(0, 8, 2, "p_region"),
         Relocation(0, 9, 3, "p_label_input"),
         Relocation(1, 8, 2, "p_region"),
         Relocation(1, 9, 3, "p_label_data"),
         Relocation(2, 8, 2, "p_region"),
         Relocation(2, 9, 3, "p_label_output"),
         Relocation(3, 0x100, 3, "p_label_input"),
         Relocation(3, 0x110, 3, "p_label_output"),
         Relocation(4, 1, 1, "p_abs_input"),
         Relocation(4, 2, 1, "p_abs_data"),
         Relocation(4, 3, 1, "p_abs_output"),
         Relocation(5, 7, 3, "p_label_input"),
         Relocation(6, 7, 3, "p_label_data"),
         Relocation(7, 7, 3, "p_label_output")});
    return {{"logical_core", core},
            {"records", std::move(records)},
            {"runtime_relocations", Json::array()},
            {"address_relocations", std::move(relocations)}};
}

Json ProgramSymbol(const std::string &id, uint64_t kind,
                   const std::string &source) {
    return {{"id", id}, {"kind", kind}, {"source_ref", source}};
}

Json Buffer(const Json &core, const std::string &prefix,
            const std::string &which, uint64_t offset) {
    const std::string id = "b" + prefix + "_" + which;
    return {{"id", id},
            {"schedule_id", "schedule" + prefix},
            {"binding_id", "abs_" + which},
            {"value_id", "value_" + which},
            {"logical_core", core},
            {"tensor_slice",
             {{"value_id", "value_" + which},
              {"offset", Json::array({0, 0})},
              {"shape", Json::array({1, 32})}}},
            {"region_ref", "region"},
            {"region_offset_bytes", offset},
            {"size_bytes", 64},
            {"alignment_bytes", 16},
            {"banks", Json::array({0})},
            {"storage_id", "label_" + which},
            {"alias_of", nullptr},
            {"lifetime_start", 0},
            {"lifetime_end_exclusive", 1},
            {"dtype", "fp16"},
            {"layout", "row_major"},
            {"ownership", "owned"}};
}


Json StateAbi() {
    Json result{{"id", "state_abi"},
                {"state_ref", "state"},
                {"hbm_binding_ref", "hbm_binding"},
                {"kind", "parameter"},
                {"lifetime", "persistent"},
                {"access", "read_only"},
                {"shape", Json::array({1, 32})},
                {"dtype", "fp16"},
                {"layout", "row_major"},
                {"die_id", 0},
                {"address", 0x1000},
                {"size_bytes", 64},
                {"alignment_bytes", 64}};
    result["id"] = StableId(
        "state_abi", "wafer_frontend.state_abi/v1alpha1",
        Without(result, {"id"}));
    return result;
}

Json Definition(const std::string &id, uint64_t kind,
                const std::string &source, const std::string &name,
                uint64_t value, uint64_t size, const Json &cores) {
    return {{"symbol", ProgramSymbol(id, kind, source)},
            {"name", name},
            {"value", value},
            {"size_bytes", size},
            {"logical_cores", cores}};
}

Json AddressBinding(const Json &core, uint64_t record, uint64_t operand,
                    const std::string &buffer) {
    return {{"fragment_id", "fragment"},
            {"logical_core", core},
            {"fragment_record_index", record},
            {"operand_id", operand},
            {"buffer_abi_ids", Json::array({buffer})}};
}

void AppendBindings(Json &bindings, const Json &core,
                    const std::string &prefix) {
    const std::string input = "b" + prefix + "_input";
    const std::string data = "b" + prefix + "_data";
    const std::string output = "b" + prefix + "_output";
    bindings.push_back(AddressBinding(core, 0, 8, input));
    bindings.push_back(AddressBinding(core, 0, 9, input));
    bindings.push_back(AddressBinding(core, 1, 8, data));
    bindings.push_back(AddressBinding(core, 1, 9, data));
    bindings.push_back(AddressBinding(core, 2, 8, output));
    bindings.push_back(AddressBinding(core, 2, 9, output));
    bindings.push_back(AddressBinding(core, 3, 0x100, input));
    bindings.push_back(AddressBinding(core, 3, 0x110, output));
    bindings.push_back(AddressBinding(core, 4, 1, input));
    bindings.push_back(AddressBinding(core, 4, 2, data));
    bindings.push_back(AddressBinding(core, 4, 3, output));
    bindings.push_back(AddressBinding(core, 5, 7, input));
    bindings.push_back(AddressBinding(core, 6, 7, data));
    bindings.push_back(AddressBinding(core, 7, 7, output));
}

void RefreshManifestIds(Json &manifest, bool refresh_buffer_abi_ids = true) {
    std::map<std::string, std::string> buffer_ids;
    std::map<std::string, Json> buffer_slices;
    std::map<std::string, std::string> leaf_ids;
    struct Digest {
        std::string kind;
        std::string id;
        std::string schema;
        std::string digest;
    };
    std::vector<Digest> embedded;
    for (Json &linked : manifest["fragments"]) {
        Json *leaf = &linked;
        const bool region =
            linked["schema_version"] ==
            "wafer_frontend.region_manifest/v1alpha12";
        if (region)
            leaf = &linked["fragment"];
        for (Json &abi : (*leaf)["buffer_abi"]) {
            const std::string old = abi["id"].get<std::string>();
            buffer_slices[old] = abi["tensor_slice"];
            const std::string updated =
                refresh_buffer_abi_ids
                    ? StableId(
                          "buffer_abi",
                          "wafer_frontend.command_fragment/v1alpha13",
                          Without(abi, {"id"}))
                    : old;
            abi["id"] = updated;
            buffer_ids[old] = updated;
        }
        std::sort((*leaf)["buffer_abi"].begin(),
                  (*leaf)["buffer_abi"].end(),
                  [](const Json &left, const Json &right) {
                      return left["id"].get<std::string>() <
                             right["id"].get<std::string>();
                  });
        const std::string old_leaf = (*leaf)["id"].get<std::string>();
        const std::string updated_leaf = StableId(
            "command_fragment", "wafer_frontend.command_fragment/v1alpha13",
            Without(*leaf, {"schema_version", "producer_pass", "id"}));
        (*leaf)["id"] = updated_leaf;
        leaf_ids[old_leaf] = updated_leaf;
        embedded.push_back(
            {"command_fragment", updated_leaf,
             (*leaf)["schema_version"].get<std::string>(),
             CanonicalDigest(*leaf)});
        if (region) {
            linked["id"] = StableId(
                "region_manifest", "wafer_frontend.region_manifest/v1alpha12",
                Without(linked, {"schema_version", "producer_pass", "id"}));
            embedded.push_back(
                {"region_manifest", linked["id"].get<std::string>(),
                 linked["schema_version"].get<std::string>(),
                 CanonicalDigest(linked)});
        }
    }
    for (Json &interface : manifest["fragment_interfaces"]) {
        const std::string old = interface["fragment_id"].get<std::string>();
        if (leaf_ids.count(old) != 0)
            interface["fragment_id"] = leaf_ids.at(old);
    }
    for (Json &stream : manifest["core_streams"]) {
        for (Json &record : stream["records"]) {
            const std::string old = record["fragment_id"].get<std::string>();
            if (leaf_ids.count(old) != 0)
                record["fragment_id"] = leaf_ids.at(old);
        }
    }
    for (Json &binding : manifest["address_operand_bindings"]) {
        const std::string old_fragment =
            binding["fragment_id"].get<std::string>();
        if (leaf_ids.count(old_fragment) != 0)
            binding["fragment_id"] = leaf_ids.at(old_fragment);
        if (!binding.contains("tensor_slices")) {
            Json tensor_slices = Json::array();
            for (const Json &abi : binding["buffer_abi_ids"]) {
                const std::string old = abi.get<std::string>();
                if (buffer_slices.count(old) == 0)
                    throw std::runtime_error(
                        "address witness has no BufferABI tensor slice");
                tensor_slices.push_back(buffer_slices.at(old));
            }
            binding["tensor_slices"] = std::move(tensor_slices);
        }
        for (Json &abi : binding["buffer_abi_ids"]) {
            const std::string old = abi.get<std::string>();
            if (buffer_ids.count(old) != 0)
                abi = buffer_ids.at(old);
        }
    }
    for (Json &binding : manifest["state_operand_bindings"]) {
        const std::string old_fragment =
            binding["fragment_id"].get<std::string>();
        if (leaf_ids.count(old_fragment) != 0)
            binding["fragment_id"] = leaf_ids.at(old_fragment);
    }
    std::sort(manifest["fragments"].begin(), manifest["fragments"].end(),
              [](const Json &left, const Json &right) {
                  return left["id"].get<std::string>() <
                         right["id"].get<std::string>();
              });
    std::sort(manifest["fragment_interfaces"].begin(),
              manifest["fragment_interfaces"].end(),
              [](const Json &left, const Json &right) {
                  return left["fragment_id"].get<std::string>() <
                         right["fragment_id"].get<std::string>();
              });
    auto binding_key = [](const Json &item) {
        const Json &core = item["logical_core"];
        return std::make_tuple(
            core["die_id"].get<uint64_t>(),
            core["local_core_id"].get<uint64_t>(),
            item["fragment_id"].get<std::string>(),
            item["fragment_record_index"].get<uint64_t>(),
            item["operand_id"].get<uint64_t>());
    };
    std::sort(manifest["address_operand_bindings"].begin(),
              manifest["address_operand_bindings"].end(),
              [&](const Json &left, const Json &right) {
                  return binding_key(left) < binding_key(right);
              });
    std::sort(manifest["state_operand_bindings"].begin(),
              manifest["state_operand_bindings"].end(),
              [&](const Json &left, const Json &right) {
                  return binding_key(left) < binding_key(right);
              });
    Json digests = Json::array();
    for (const Json &digest : manifest["input_digests"]) {
        const std::string kind = digest["kind"].get<std::string>();
        if (kind != "command_fragment" && kind != "region_manifest")
            digests.push_back(digest);
    }
    for (const Digest &digest : embedded) {
        digests.push_back({{"kind", digest.kind},
                           {"artifact_id", digest.id},
                           {"schema_version", digest.schema},
                           {"digest", digest.digest}});
    }
    std::sort(digests.begin(), digests.end(),
              [](const Json &left, const Json &right) {
                  return std::make_pair(left["kind"].get<std::string>(),
                                        left["artifact_id"].get<std::string>()) <
                         std::make_pair(right["kind"].get<std::string>(),
                                        right["artifact_id"].get<std::string>());
              });
    manifest["input_digests"] = std::move(digests);
    manifest["id"] = StableId(
        "linked_program_manifest",
        "wafer_frontend.linked_program_manifest/v1alpha14",
        Without(manifest, {"schema_version", "producer_pass", "id"}));
}

Json Manifest() {
    const Json core0 = Core(0, 0);
    const Json core1 = Core(0, 1);
    const Json cores = Json::array({core0, core1});
    Json symbols = Json::array(
        {ProgramSymbol("p_abs_data", 1, "abs_data"),
         ProgramSymbol("p_abs_input", 1, "abs_input"),
         ProgramSymbol("p_abs_output", 1, "abs_output"),
         ProgramSymbol("p_label_data", 3, "label_data"),
         ProgramSymbol("p_label_input", 3, "label_input"),
         ProgramSymbol("p_label_output", 3, "label_output"),
         ProgramSymbol("p_region", 2, "region")});
    Json buffers = Json::array(
        {Buffer(core0, "0", "data", 0x200),
         Buffer(core0, "0", "input", 0x100),
         Buffer(core0, "0", "output", 0x300),
         Buffer(core1, "1", "data", 0x200),
         Buffer(core1, "1", "input", 0x100),
         Buffer(core1, "1", "output", 0x300)});
    Json fragment = {
        {"schema_version", "wafer_frontend.command_fragment/v1alpha13"},
        {"producer_pass", "lowering"},
        {"id", "fragment"},
        {"source_global_dag_id", "global"},
        {"kind", "coarse"},
        {"claimed_action_ids", Json::array({"a0", "a1"})},
        {"core_streams", Json::array({Stream(core0, "a0"),
                                      Stream(core1, "a1")})},
        {"runtime_symbols", Json::array()},
        {"program_symbols", symbols},
        {"buffer_abi", std::move(buffers)},
        {"state_abi", Json::array()}};
    Json definitions = Json::array(
        {Definition("p_abs_data", 1, "abs_data", "abs.data", 0x200, 64,
                    cores),
         Definition("p_abs_input", 1, "abs_input", "abs.input", 0x100, 64,
                    cores),
         Definition("p_abs_output", 1, "abs_output", "abs.output", 0x300,
                    64, cores),
         Definition("p_label_data", 3, "label_data", "label.data", 0, 0,
                    cores),
         Definition("p_label_input", 3, "label_input", "label.input", 0, 0,
                    cores),
         Definition("p_label_output", 3, "label_output", "label.output", 0,
                    0, cores),
         Definition("p_region", 2, "region", "sram_main", 0, 4096, cores)});
    Json linked0 = Json::array();
    Json linked1 = Json::array();
    for (uint64_t index = 0; index < 8; ++index) {
        linked0.push_back({{"fragment_id", "fragment"},
                           {"fragment_record_index", index},
                           {"source_global_action_id", "a0"}});
        linked1.push_back({{"fragment_id", "fragment"},
                           {"fragment_record_index", index},
                           {"source_global_action_id", "a1"}});
    }
    Json bindings = Json::array();
    AppendBindings(bindings, core0, "0");
    AppendBindings(bindings, core1, "1");
    Json manifest = {
        {"schema_version", "wafer_frontend.linked_program_manifest/v1alpha14"},
        {"producer_pass", "linker"},
        {"id", "manifest"},
        {"capabilities", 0},
        {"source_ir1_id", "ir1"},
        {"source_projection_id", "projection"},
        {"source_schedule_set_id", "schedules"},
        {"source_global_dag_id", "global"},
        {"input_digests",
         Json::array(
             {{{"kind", "ir1"},
               {"artifact_id", "ir1"},
               {"schema_version", "wafer_frontend.ir1/v1alpha14"},
               {"digest", std::string(64, '1')}},
              {{"kind", "ir2_projection"},
               {"artifact_id", "projection"},
               {"schema_version",
                "wafer_frontend.ir2_projection_result/v1alpha13"},
               {"digest", std::string(64, '2')}},
              {{"kind", "schedule_set"},
              {"artifact_id", "schedules"},
               {"schema_version",
                "wafer_frontend.intra_die_schedule_set/v1alpha9"},
               {"digest", std::string(64, '3')}},
              {{"kind", "global_action_dag"},
              {"artifact_id", "global"},
               {"schema_version",
                "wafer_frontend.global_action_dag/v1alpha11"},
               {"digest", std::string(64, '4')}}})},
        {"fragments", Json::array({std::move(fragment)})},
        {"fragment_interfaces",
         Json::array({{{"fragment_id", "fragment"},
                       {"runtime_imports", Json::array()},
                       {"runtime_exports", Json::array()},
                       {"program_imports", Json::array()},
                       {"program_exports",
                        Json::array({"p_abs_data", "p_abs_input",
                                     "p_abs_output", "p_label_data",
                                     "p_label_input", "p_label_output",
                                     "p_region"})},
                       {"entry_events", Json::array()},
                       {"exit_events", Json::array()}}})},
        {"core_bindings",
         Json::array({{{"logical_core", core0},
                       {"core_spec_ref", "core0"},
                       {"runtime_core_id", 9},
                       {"sram_profile_ref", "sram"}},
                      {{"logical_core", core1},
                       {"core_spec_ref", "core1"},
                       {"runtime_core_id", 2},
                       {"sram_profile_ref", "sram"}}})},
        {"core_streams",
         Json::array({{{"logical_core", core0},
                       {"runtime_core_id", 9},
                       {"records", std::move(linked0)}},
                      {{"logical_core", core1},
                       {"runtime_core_id", 2},
                       {"records", std::move(linked1)}}})},
        {"runtime_symbol_definitions",
         Json::array({{{"symbol",
                        {{"id", "start0"},
                         {"kind", "start_tag"},
                         {"source_ref", "a0"}}},
                       {"logical_cores", Json::array({core0})},
                       {"source_action_id", nullptr},
                       {"destination_action_id", nullptr}},
                      {{"symbol",
                        {{"id", "start1"},
                         {"kind", "start_tag"},
                         {"source_ref", "a1"}}},
                       {"logical_cores", Json::array({core1})},
                       {"source_action_id", nullptr},
                       {"destination_action_id", nullptr}}})},
        {"program_symbol_definitions", std::move(definitions)},
        {"address_operand_bindings", std::move(bindings)},
        {"state_operand_bindings", Json::array()},
        {"core_groups", Json::array()},
        {"envelope",
         {{"active_cores", cores},
          {"start_events",
           Json::array({{{"target_core", core0},
                         {"tag_symbol_ref", "start0"},
                         {"count", 1}},
                        {{"target_core", core1},
                         {"tag_symbol_ref", "start1"},
                         {"count", 1}}})},
          {"terminal_cores", cores},
          {"expected_ack_cores", cores},
          {"expected_done_cores", cores},
          {"empty_core_ack_policy", "include_empty"},
          {"failure_policy", "abort_all"}}}};
    RefreshManifestIds(manifest);
    return manifest;
}

Json Stage2Manifest(uint64_t opcode) {
    Json manifest = Manifest();
    Json &fragment = manifest["fragments"][0];
    const bool has_data = opcode == 0x10 || opcode == 0x1c ||
                          opcode == 0x1e || opcode == 0x1f ||
                          opcode == 0x20;
    const bool has_aux = opcode == 0x1f;
    const bool binds_data = opcode == 0x1c || opcode == 0x1e ||
                            opcode == 0x1f || opcode == 0x20;
    Json input_shape = Json::array({1, 4, 2});
    Json data_shape = Json::array({1, 32});
    Json output_shape = input_shape;
    uint64_t input_size = 16;
    uint64_t data_size = 64;
    uint64_t output_size = 16;
    std::string input_dtype = "fp16";
    std::string output_dtype = "fp16";
    if (opcode == 0x1b) {
        output_shape = Json::array({1, 2, 2});
        output_size = 8;
    } else if (opcode == 0x1c) {
        input_shape = Json::array({1});
        data_shape = Json::array({8, 8});
        output_shape = Json::array({1, 8});
        input_size = 4;
        data_size = 128;
        output_size = 16;
        input_dtype = "int32";
    } else if (opcode == 0x1d) {
        input_shape = Json::array({1, 8});
        output_shape = Json::array({1});
        input_size = 16;
        output_size = 4;
        output_dtype = "int32";
    } else if (opcode == 0x1e) {
        input_shape = Json::array({4, 32});
        data_shape = Json::array({4});
        output_shape = Json::array({4});
        input_size = 256;
        data_size = 16;
        output_size = 16;
        output_dtype = "fp32";
    } else if (opcode == 0x1f) {
        input_shape = Json::array({4, 32});
        data_shape = Json::array({4});
        output_shape = Json::array({4, 32});
        input_size = 256;
        data_size = 16;
        output_size = 256;
    } else if (opcode == 0x20) {
        input_shape = Json::array({32});
        data_shape = Json::array({32});
        output_shape = input_shape;
        input_size = 64;
        data_size = 128;
        output_size = 64;
        data_size = 128;
    } else if (opcode == 0x10) {
        input_shape = Json::array({1, 1, 32});
        data_shape = Json::array({32});
        output_shape = input_shape;
        input_size = data_size = output_size = 64;
    }
    if (has_aux) {
        const Json cores = Json::array({Core(0, 0), Core(0, 1)});
        for (const std::pair<Json, std::string> &item :
             std::array<std::pair<Json, std::string>, 2>{{
                 {Core(0, 0), "0"}, {Core(0, 1), "1"}}}) {
            Json aux = Buffer(item.first, item.second, "aux", 0x400);
            aux["tensor_slice"]["offset"] = Json::array({0});
            aux["tensor_slice"]["shape"] = Json::array({4});
            aux["size_bytes"] = 16;
            aux["dtype"] = "fp32";
            aux["ownership"] = "borrowed";
            fragment["buffer_abi"].push_back(std::move(aux));
        }
        fragment["program_symbols"].push_back(
            ProgramSymbol("p_abs_aux", 1, "abs_aux"));
        fragment["program_symbols"].push_back(
            ProgramSymbol("p_label_aux", 3, "label_aux"));
        manifest["program_symbol_definitions"].push_back(
            Definition("p_abs_aux", 1, "abs_aux", "abs.aux", 0x400,
                       16, cores));
        manifest["program_symbol_definitions"].push_back(
            Definition("p_label_aux", 3, "label_aux", "label.aux", 0,
                       0, cores));
        Json &exports = manifest["fragment_interfaces"][0]["program_exports"];
        exports.push_back("p_abs_aux");
        exports.push_back("p_label_aux");
        std::sort(fragment["program_symbols"].begin(),
                  fragment["program_symbols"].end(),
                  [](const Json &left, const Json &right) {
                      return left["id"].get<std::string>() <
                             right["id"].get<std::string>();
                  });
        std::sort(manifest["program_symbol_definitions"].begin(),
                  manifest["program_symbol_definitions"].end(),
                  [](const Json &left, const Json &right) {
                      return left["symbol"]["id"].get<std::string>() <
                             right["symbol"]["id"].get<std::string>();
                  });
        std::sort(exports.begin(), exports.end());
    }
    for (Json &abi : fragment["buffer_abi"]) {
        const std::string binding = abi["binding_id"].get<std::string>();
        Json shape;
        uint64_t size = 0;
        std::string dtype = "fp16";
        if (binding == "abs_input") {
            shape = input_shape;
            size = input_size;
            dtype = input_dtype;
        } else if (binding == "abs_data") {
            shape = data_shape;
            size = data_size;
            if (opcode == 0x1e || opcode == 0x1f) dtype = "int32";
            if (opcode == 0x20) dtype = "fp32";
        } else if (binding == "abs_aux") {
            shape = Json::array({4});
            size = 16;
            dtype = "fp32";
        } else {
            shape = output_shape;
            size = output_size;
            dtype = output_dtype;
        }
        abi["tensor_slice"]["offset"] = Json::array();
        for (std::size_t axis = 0; axis < shape.size(); ++axis)
            abi["tensor_slice"]["offset"].push_back(0);
        abi["tensor_slice"]["shape"] = shape;
        abi["size_bytes"] = size;
        abi["dtype"] = dtype;
    }
    const std::string old_fragment_id = fragment["id"].get<std::string>();
    for (Json &stream : fragment["core_streams"]) {
        const std::string action =
            stream["records"][4]["source_global_action_id"].get<std::string>();
        stream["records"][0]["operands"][3]["literal_value"] = input_size;
        stream["records"][1]["operands"][3]["literal_value"] = data_size;
        stream["records"][2]["operands"][3]["literal_value"] = output_size;
        stream["records"][4] = Stage2Compute(action, opcode);
        Json &bind_operands = stream["records"][3]["operands"];
        bind_operands[0]["literal_value"] = has_aux ? 3 : binds_data ? 2 : 1;
        Json &relocations = stream["address_relocations"];
        relocations.erase(
            std::remove_if(relocations.begin(), relocations.end(),
                           [&](const Json &item) {
                               return item["record_index"] == 4 &&
                                      item["operand_id"] == 2;
                           }),
            relocations.end());
        if (binds_data) {
            bind_operands[2] = Address("input_label_1", 0x101,
                                       "p_label_data");
            relocations.push_back(Relocation(3, 0x101, 3, "p_label_data"));
        }
        if (has_aux) {
            bind_operands[3] = Address("input_label_2", 0x102,
                                       "p_label_aux");
            relocations.push_back(Relocation(3, 0x102, 3, "p_label_aux"));
            relocations.push_back(Relocation(4, 12, 1, "p_abs_aux"));
        }
        if (opcode == 0x20) {
            bind_operands[17] = Address("output_label", 0x110,
                                        "p_label_input");
            for (Json &relocation : relocations) {
                const uint64_t record_index =
                    relocation["record_index"].get<uint64_t>();
                const uint64_t operand_id =
                    relocation["operand_id"].get<uint64_t>();
                if (record_index == 3 && operand_id == 0x110)
                    relocation["symbol_ref"] = "p_label_input";
                if (record_index == 4 && operand_id == 3)
                    relocation["symbol_ref"] = "p_abs_input";
            }
            const Json &core = stream["logical_core"];
            const auto input_abi = std::find_if(
                fragment["buffer_abi"].begin(),
                fragment["buffer_abi"].end(), [&](const Json &item) {
                    return item["logical_core"] == core &&
                           item["binding_id"] == "abs_input";
                });
            for (Json &binding : manifest["address_operand_bindings"]) {
                if (binding["logical_core"] != core ||
                    binding["fragment_id"] != old_fragment_id)
                    continue;
                const uint64_t record_index =
                    binding["fragment_record_index"].get<uint64_t>();
                const uint64_t operand_id =
                    binding["operand_id"].get<uint64_t>();
                if ((record_index == 3 && operand_id == 0x110) ||
                    (record_index == 4 && operand_id == 3)) {
                    binding["buffer_abi_ids"] = Json::array(
                        {(*input_abi)["id"].get<std::string>()});
                    binding.erase("tensor_slices");
                }
            }
        }
        if (has_data) {
            relocations.push_back(Relocation(4, 2, 1, "p_abs_data"));
        }
        std::sort(relocations.begin(), relocations.end(),
                  [](const Json &left, const Json &right) {
                      return std::make_pair(
                                 left["record_index"].get<uint64_t>(),
                                 left["operand_id"].get<uint64_t>()) <
                             std::make_pair(
                                 right["record_index"].get<uint64_t>(),
                                 right["operand_id"].get<uint64_t>());
                  });
        if (binds_data) {
            const Json &core = stream["logical_core"];
            const auto abi = std::find_if(
                fragment["buffer_abi"].begin(),
                fragment["buffer_abi"].end(), [&](const Json &item) {
                    return item["logical_core"] == core &&
                           item["binding_id"] == "abs_data";
                });
            Json binding = AddressBinding(core, 3, 0x101,
                                          (*abi)["id"].get<std::string>());
            binding["fragment_id"] = old_fragment_id;
            manifest["address_operand_bindings"].push_back(std::move(binding));
        }
        if (has_aux) {
            const Json &core = stream["logical_core"];
            const auto abi = std::find_if(
                fragment["buffer_abi"].begin(),
                fragment["buffer_abi"].end(), [&](const Json &item) {
                    return item["logical_core"] == core &&
                           item["binding_id"] == "abs_aux";
                });
            Json label_binding = AddressBinding(
                core, 3, 0x102, (*abi)["id"].get<std::string>());
            label_binding["fragment_id"] = old_fragment_id;
            manifest["address_operand_bindings"].push_back(
                std::move(label_binding));
            Json compute_binding = AddressBinding(
                core, 4, 12, (*abi)["id"].get<std::string>());
            compute_binding["fragment_id"] = old_fragment_id;
            manifest["address_operand_bindings"].push_back(
                std::move(compute_binding));
        }
    }
    Json &bindings = manifest["address_operand_bindings"];
    bindings.erase(
        std::remove_if(bindings.begin(), bindings.end(),
                       [&](const Json &item) {
                           return !has_data &&
                                  item["fragment_record_index"] == 4 &&
                                  item["operand_id"] == 2;
                       }),
        bindings.end());
    if (!has_data) {
        Json &symbols = fragment["program_symbols"];
        symbols.erase(
            std::remove_if(symbols.begin(), symbols.end(), [](const Json &item) {
                return item["id"] == "p_abs_data";
            }),
            symbols.end());
        Json &definitions = manifest["program_symbol_definitions"];
        definitions.erase(
            std::remove_if(definitions.begin(), definitions.end(),
                           [](const Json &item) {
                               return item["symbol"]["id"] == "p_abs_data";
                           }),
            definitions.end());
        Json &exports = manifest["fragment_interfaces"][0]["program_exports"];
        exports.erase(std::remove(exports.begin(), exports.end(), "p_abs_data"),
                      exports.end());
    }
    if (opcode == 0x20) {
        Json &symbols = fragment["program_symbols"];
        symbols.erase(
            std::remove_if(symbols.begin(), symbols.end(), [](const Json &item) {
                return item["id"] == "p_abs_output";
            }),
            symbols.end());
        Json &definitions = manifest["program_symbol_definitions"];
        definitions.erase(
            std::remove_if(definitions.begin(), definitions.end(),
                           [](const Json &item) {
                               return item["symbol"]["id"] == "p_abs_output";
                           }),
            definitions.end());
        Json &exports = manifest["fragment_interfaces"][0]["program_exports"];
        exports.erase(
            std::remove(exports.begin(), exports.end(), "p_abs_output"),
            exports.end());
    }
    for (Json &binding : bindings)
        binding.erase("tensor_slices");
    for (Json &definition : manifest["program_symbol_definitions"]) {
        const std::string id = definition["symbol"]["id"].get<std::string>();
        if (id == "p_abs_input") definition["size_bytes"] = input_size;
        if (id == "p_abs_data") definition["size_bytes"] = data_size;
        if (id == "p_abs_output") definition["size_bytes"] = output_size;
    }
    RefreshManifestIds(manifest);
    return manifest;
}
Json TransferManifest() {
    Json manifest = Manifest();
    const Json core0 = Core(0, 0);
    const Json core1 = Core(0, 1);
    Json &base = manifest["fragments"][0];
    const std::string base_id = base["id"].get<std::string>();

    auto stream_for = [](Json &fragment, const Json &core) -> Json & {
        auto found = std::find_if(
            fragment["core_streams"].begin(),
            fragment["core_streams"].end(),
            [&](const Json &item) { return item["logical_core"] == core; });
        if (found == fragment["core_streams"].end())
            throw std::runtime_error("synthetic transfer lacks a base core stream");
        return *found;
    };
    auto remove_record = [](Json &stream, uint64_t removed) {
        stream["records"].erase(
            stream["records"].begin() +
            static_cast<std::ptrdiff_t>(removed));
        for (const char *field :
             {"runtime_relocations", "address_relocations"}) {
            Json shifted = Json::array();
            for (Json relocation : stream[field]) {
                const uint64_t index =
                    relocation["record_index"].get<uint64_t>();
                if (index == removed)
                    continue;
                if (index > removed)
                    relocation["record_index"] = index - 1;
                shifted.push_back(std::move(relocation));
            }
            stream[field] = std::move(shifted);
        }
    };
    auto input_buffer = [](const Json &fragment, const Json &core) {
        const auto found = std::find_if(
            fragment["buffer_abi"].begin(),
            fragment["buffer_abi"].end(),
            [&](const Json &item) {
                return item["logical_core"] == core &&
                       item["binding_id"] == "abs_input";
            });
        if (found == fragment["buffer_abi"].end())
            throw std::runtime_error("synthetic transfer lacks input BufferABI");
        return *found;
    };

    Json source_buffer = input_buffer(base, core0);
    Json destination_buffer = input_buffer(base, core1);
    Json &source_base_stream = stream_for(base, core0);
    Json &destination_base_stream = stream_for(base, core1);
    remove_record(source_base_stream, 5);
    remove_record(destination_base_stream, 0);

    Json shifted_bindings = Json::array();
    for (Json binding : manifest["address_operand_bindings"]) {
        if (binding["fragment_id"] == base_id) {
            const bool source_core = binding["logical_core"] == core0;
            const bool destination_core =
                binding["logical_core"] == core1;
            const uint64_t index =
                binding["fragment_record_index"].get<uint64_t>();
            const uint64_t removed = source_core ? 5 : 0;
            if ((source_core || destination_core) && index == removed)
                continue;
            if ((source_core || destination_core) && index > removed)
                binding["fragment_record_index"] = index - 1;
        }
        shifted_bindings.push_back(std::move(binding));
    }
    manifest["address_operand_bindings"] = std::move(shifted_bindings);

    auto runtime_operand = [](const std::string &name,
                              const std::string &field,
                              const std::string &symbol) {
        return Json{{"name", name},
                    {"kind", "runtime_symbol"},
                    {"literal_value", nullptr},
                    {"runtime_field", field},
                    {"operand_id", nullptr},
                    {"symbol_ref", symbol}};
    };
    auto runtime_symbol = [](const std::string &id,
                             const std::string &kind,
                             const std::string &source) {
        return Json{{"id", id}, {"kind", kind}, {"source_ref", source}};
    };
    auto runtime_relocation = [](uint64_t record,
                                 const std::string &field,
                                 const std::string &symbol) {
        return Json{{"record_index", record},
                    {"field", field},
                    {"symbol_ref", symbol}};
    };
    const std::string fsm = "transfer_fsm";
    const std::string peer_destination = "transfer_peer_destination";
    const std::string peer_source = "transfer_peer_source";
    const std::string token = "transfer_token";
    const std::string send_action = "transfer_send";
    const std::string recv_action = "transfer_recv";
    const std::string wait_action = "transfer_wait";

    Json send = Record(
        send_action, 0x40,
        Json::array({
            Literal("mode", 0),
            Literal("source_space", 0),
            Literal("completion", 1),
            Literal("datatype", 0),
            Literal("reduce_op", 0),
            runtime_operand("fsm_id", "dte_fsm", fsm),
            Literal("token", 0),
            Literal("length_bytes", 64),
            Address("source_address", 4, "p_abs_input"),
            runtime_operand("peer_core", "peer_core", peer_destination),
            Literal("expected_sources", 0),
            Literal("tree_id", 0),
            Literal("group_id", 0),
            Literal("collective_id", 0),
            Literal("epoch", 0)}));
    Json recv = Record(
        recv_action, 0x41,
        Json::array({
            Literal("mode", 0),
            Literal("completion", 0),
            Literal("datatype", 0),
            Literal("reduce_op", 0),
            runtime_operand("fsm_id", "dte_fsm", fsm),
            runtime_operand("token", "dte_token", token),
            Literal("length_bytes", 64),
            Address("destination_address", 5, "p_abs_input"),
            runtime_operand("peer_core", "peer_core", peer_source),
            Literal("expected_sources", 0),
            Literal("tree_id", 0),
            Literal("group_id", 0),
            Literal("collective_id", 0),
            Literal("epoch", 0)}));
    Json wait = Record(
        wait_action, 0xC0,
        Json::array({runtime_operand("token", "dte_token", token)}));

    const std::string source_id = "transfer_source";
    const std::string destination_id = "transfer_destination";
    Json source_fragment{
        {"schema_version", "wafer_frontend.command_fragment/v1alpha13"},
        {"producer_pass", "state_transfer_lowering"},
        {"id", source_id},
        {"source_global_dag_id", "global"},
        {"kind", "state_transfer"},
        {"claimed_action_ids", Json::array({send_action})},
        {"core_streams",
         Json::array(
             {{{"logical_core", core0},
               {"records",
                Json::array({std::move(send),
                             Free(send_action, "p_label_input")})},
               {"runtime_relocations",
                Json::array({
                    runtime_relocation(0, "dte_fsm", fsm),
                    runtime_relocation(
                        0, "peer_core", peer_destination)})},
               {"address_relocations",
                Json::array({
                    Relocation(0, 4, 1, "p_abs_input"),
                    Relocation(1, 7, 3, "p_label_input")})}}})},
        {"runtime_symbols",
         Json::array({
             runtime_symbol(fsm, "dte_fsm", "transfer"),
             runtime_symbol(peer_destination, "runtime_core",
                            "transfer_destination_core")})},
        {"program_symbols",
         Json::array({
             ProgramSymbol("p_abs_input", 1, "abs_input"),
             ProgramSymbol("p_label_input", 3, "label_input")})},
        {"buffer_abi", Json::array({source_buffer})},
        {"state_abi", Json::array()}};
    Json destination_fragment{
        {"schema_version", "wafer_frontend.command_fragment/v1alpha13"},
        {"producer_pass", "state_transfer_lowering"},
        {"id", destination_id},
        {"source_global_dag_id", "global"},
        {"kind", "state_transfer"},
        {"claimed_action_ids",
         Json::array({recv_action, wait_action})},
        {"core_streams",
         Json::array(
             {{{"logical_core", core1},
               {"records",
                Json::array({
                    Alloc(recv_action, "p_label_input", 0x100),
                    std::move(recv), std::move(wait)})},
               {"runtime_relocations",
                Json::array({
                    runtime_relocation(1, "dte_token", token),
                    runtime_relocation(1, "dte_fsm", fsm),
                    runtime_relocation(1, "peer_core", peer_source),
                    runtime_relocation(2, "dte_token", token)})},
               {"address_relocations",
                Json::array({
                    Relocation(0, 8, 2, "p_region"),
                    Relocation(0, 9, 3, "p_label_input"),
                    Relocation(1, 5, 1, "p_abs_input")})}}})},
        {"runtime_symbols",
         Json::array({
             runtime_symbol(fsm, "dte_fsm", "transfer"),
             runtime_symbol(peer_source, "runtime_core",
                            "transfer_source_core"),
             runtime_symbol(token, "dte_token", "transfer_token")})},
        {"program_symbols",
         Json::array({
             ProgramSymbol("p_abs_input", 1, "abs_input"),
             ProgramSymbol("p_label_input", 3, "label_input"),
             ProgramSymbol("p_region", 2, "region")})},
        {"buffer_abi", Json::array({destination_buffer})},
        {"state_abi", Json::array()}};

    auto binding = [](const std::string &fragment_id,
                      const Json &core, uint64_t record,
                      uint64_t operand, const std::string &buffer_id) {
        return Json{{"fragment_id", fragment_id},
                    {"logical_core", core},
                    {"fragment_record_index", record},
                    {"operand_id", operand},
                    {"buffer_abi_ids", Json::array({buffer_id})}};
    };
    const std::string source_buffer_id =
        source_buffer["id"].get<std::string>();
    const std::string destination_buffer_id =
        destination_buffer["id"].get<std::string>();
    for (Json item : {
             binding(source_id, core0, 0, 4, source_buffer_id),
             binding(source_id, core0, 1, 7, source_buffer_id),
             binding(destination_id, core1, 0, 8,
                     destination_buffer_id),
             binding(destination_id, core1, 0, 9,
                     destination_buffer_id),
             binding(destination_id, core1, 1, 5,
                     destination_buffer_id)})
        manifest["address_operand_bindings"].push_back(std::move(item));

    manifest["fragment_interfaces"].push_back(
        {{"fragment_id", source_id},
         {"runtime_imports", Json::array()},
         {"runtime_exports",
          Json::array({fsm, peer_destination})},
         {"program_imports",
          Json::array({"p_abs_input", "p_label_input"})},
         {"program_exports", Json::array()},
         {"entry_events", Json::array()},
         {"exit_events", Json::array()}});
    manifest["fragment_interfaces"].push_back(
        {{"fragment_id", destination_id},
         {"runtime_imports", Json::array({fsm})},
         {"runtime_exports",
          Json::array({peer_source, token})},
         {"program_imports",
          Json::array(
              {"p_abs_input", "p_label_input", "p_region"})},
         {"program_exports", Json::array()},
         {"entry_events", Json::array()},
         {"exit_events", Json::array()}});

    manifest["runtime_symbol_definitions"].push_back(
        {{"symbol", runtime_symbol(fsm, "dte_fsm", "transfer")},
         {"logical_cores", Json::array({core0, core1})},
         {"source_action_id", send_action},
         {"destination_action_id", recv_action}});
    manifest["runtime_symbol_definitions"].push_back(
        {{"symbol",
          runtime_symbol(peer_destination, "runtime_core",
                         "transfer_destination_core")},
         {"logical_cores", Json::array({core1})},
         {"source_action_id", nullptr},
         {"destination_action_id", nullptr}});
    manifest["runtime_symbol_definitions"].push_back(
        {{"symbol",
          runtime_symbol(peer_source, "runtime_core",
                         "transfer_source_core")},
         {"logical_cores", Json::array({core0})},
         {"source_action_id", nullptr},
         {"destination_action_id", nullptr}});
    manifest["runtime_symbol_definitions"].push_back(
        {{"symbol",
          runtime_symbol(token, "dte_token", "transfer_token")},
         {"logical_cores", Json::array({core1})},
         {"source_action_id", recv_action},
         {"destination_action_id", wait_action}});
    std::sort(
        manifest["runtime_symbol_definitions"].begin(),
        manifest["runtime_symbol_definitions"].end(),
        [](const Json &left, const Json &right) {
            return left["symbol"]["id"].get<std::string>() <
                   right["symbol"]["id"].get<std::string>();
        });
    for (Json &definition : manifest["runtime_symbol_definitions"]) {
        if (definition["symbol"]["kind"] == "start_tag" &&
            definition["logical_cores"] == Json::array({core1}))
            definition["symbol"]["source_ref"] = recv_action;
    }

    Json source_links = Json::array();
    for (uint64_t index = 0;
         index < source_base_stream["records"].size(); ++index)
        source_links.push_back(
            {{"fragment_id", base_id},
             {"fragment_record_index", index},
             {"source_global_action_id",
              source_base_stream["records"][index]
                                ["source_global_action_id"]}});
    source_links.push_back(
        {{"fragment_id", source_id},
         {"fragment_record_index", 0},
         {"source_global_action_id", send_action}});
    source_links.push_back(
        {{"fragment_id", source_id},
         {"fragment_record_index", 1},
         {"source_global_action_id", send_action}});

    Json destination_links = Json::array(
        {{{"fragment_id", destination_id},
          {"fragment_record_index", 0},
          {"source_global_action_id", recv_action}},
         {{"fragment_id", destination_id},
          {"fragment_record_index", 1},
          {"source_global_action_id", recv_action}},
         {{"fragment_id", destination_id},
          {"fragment_record_index", 2},
          {"source_global_action_id", wait_action}}});
    for (uint64_t index = 0;
         index < destination_base_stream["records"].size(); ++index)
        destination_links.push_back(
            {{"fragment_id", base_id},
             {"fragment_record_index", index},
             {"source_global_action_id",
              destination_base_stream["records"][index]
                                     ["source_global_action_id"]}});
    for (Json &stream : manifest["core_streams"]) {
        if (stream["logical_core"] == core0)
            stream["records"] = source_links;
        else if (stream["logical_core"] == core1)
            stream["records"] = destination_links;
    }

    manifest["fragments"].push_back(std::move(source_fragment));
    manifest["fragments"].push_back(std::move(destination_fragment));
    RefreshManifestIds(manifest);
    return manifest;
}


Json StateManifest() {
    Json manifest = Manifest();
    const Json core = Core(0, 0);
    Json state_abi = StateAbi();
    const std::string state_abi_id = state_abi["id"].get<std::string>();
    Json local_buffer = Buffer(core, "s", "state", 0x400);
    Json state_fragment{
        {"schema_version", "wafer_frontend.command_fragment/v1alpha13"},
        {"producer_pass", "state_lowering"},
        {"id", "state_fragment"},
        {"source_global_dag_id", "global"},
        {"kind", "state_io"},
        {"claimed_action_ids", Json::array({"state_load"})},
        {"core_streams",
         Json::array({{{"logical_core", core},
                       {"records",
                        Json::array({Record(
                            "state_load", 0x80,
                            Json::array({
                                Address("hbm_address", 6, "p_hbm_state"),
                                Literal("size_bytes", 64),
                                Address("destination_address", 5,
                                        "p_abs_state_local")}))})},
                       {"runtime_relocations", Json::array()},
                       {"address_relocations",
                        Json::array({
                            Relocation(0, 5, 1, "p_abs_state_local"),
                            Relocation(0, 6, 1, "p_hbm_state")})}}})},
        {"runtime_symbols", Json::array()},
        {"program_symbols",
         Json::array({
             ProgramSymbol("p_abs_state_local", 1, "abs_state"),
             ProgramSymbol("p_hbm_state", 1, "hbm_binding")})},
        {"buffer_abi", Json::array({local_buffer})},
        {"state_abi", Json::array({state_abi})}};
    manifest["fragments"].push_back(std::move(state_fragment));
    manifest["fragment_interfaces"].push_back(
        {{"fragment_id", "state_fragment"},
         {"runtime_imports", Json::array()},
         {"runtime_exports", Json::array()},
         {"program_imports", Json::array()},
         {"program_exports",
          Json::array({"p_abs_state_local", "p_hbm_state"})},
         {"entry_events", Json::array()},
         {"exit_events", Json::array()}});
    manifest["program_symbol_definitions"].push_back(Definition(
        "p_abs_state_local", 1, "abs_state", "abs.state.local",
        0x400, 64, Json::array({core})));
    manifest["program_symbol_definitions"].push_back(Definition(
        "p_hbm_state", 1, "hbm_binding", "hbm.state",
        0x1000, 64, Json::array({core})));
    std::sort(manifest["program_symbol_definitions"].begin(),
              manifest["program_symbol_definitions"].end(),
              [](const Json &left, const Json &right) {
                  return left["symbol"]["id"].get<std::string>() <
                         right["symbol"]["id"].get<std::string>();
              });
    auto stream = std::find_if(
        manifest["core_streams"].begin(), manifest["core_streams"].end(),
        [&](const Json &item) { return item["logical_core"] == core; });
    stream->at("records").push_back(
        {{"fragment_id", "state_fragment"},
         {"fragment_record_index", 0},
         {"source_global_action_id", "state_load"}});
    manifest["address_operand_bindings"].push_back(
        AddressBinding(core, 0, 5, "bs_state"));
    manifest["address_operand_bindings"].back()["fragment_id"] =
        "state_fragment";
    manifest["state_operand_bindings"].push_back(
        {{"fragment_id", "state_fragment"},
         {"logical_core", core},
         {"fragment_record_index", 0},
         {"operand_id", 6},
         {"state_abi_id", state_abi_id}});
    RefreshManifestIds(manifest);
    return manifest;
}

Json SharedLifetimeManifest() {
    Json manifest = Manifest();
    const std::string fragment_id =
        manifest["fragments"][0]["id"].get<std::string>();
    Json &stream = manifest["fragments"][0]["core_streams"][0];
    const Json old_records = stream["records"];
    Json records = Json::array();
    for (std::size_t index = 0; index <= 4; ++index)
        records.push_back(old_records[index]);
    records.push_back(Bind("a0_later"));
    records.push_back(Matmul("a0_later"));
    for (std::size_t index = 5; index < old_records.size(); ++index) {
        Json record = old_records[index];
        record["source_global_action_id"] = "a0_later";
        records.push_back(std::move(record));
    }
    stream["records"] = std::move(records);

    Json relocations = Json::array();
    for (Json relocation : stream["address_relocations"]) {
        if (relocation["record_index"].get<uint64_t>() >= 5)
            relocation["record_index"] =
                relocation["record_index"].get<uint64_t>() + 2;
        relocations.push_back(std::move(relocation));
    }
    relocations.push_back(Relocation(5, 0x100, 3, "p_label_input"));
    relocations.push_back(Relocation(5, 0x110, 3, "p_label_output"));
    relocations.push_back(Relocation(6, 1, 1, "p_abs_input"));
    relocations.push_back(Relocation(6, 2, 1, "p_abs_data"));
    relocations.push_back(Relocation(6, 3, 1, "p_abs_output"));
    std::sort(relocations.begin(), relocations.end(),
              [](const Json &left, const Json &right) {
                  return std::make_pair(left["record_index"].get<uint64_t>(),
                                        left["operand_id"].get<uint64_t>()) <
                         std::make_pair(right["record_index"].get<uint64_t>(),
                                        right["operand_id"].get<uint64_t>());
              });
    stream["address_relocations"] = std::move(relocations);
    manifest["fragments"][0]["claimed_action_ids"] =
        Json::array({"a0", "a0_later", "a1"});

    Json linked = Json::array();
    for (uint64_t index = 0; index < 10; ++index) {
        linked.push_back(
            {{"fragment_id", fragment_id},
             {"fragment_record_index", index},
             {"source_global_action_id", index <= 4 ? "a0" : "a0_later"}});
    }
    manifest["core_streams"][0]["records"] = std::move(linked);

    Json bindings = manifest["address_operand_bindings"];
    for (Json &binding : bindings) {
        const Json &core = binding["logical_core"];
        if (core["die_id"].get<uint64_t>() == 0 &&
            core["local_core_id"].get<uint64_t>() == 0 &&
            binding["fragment_record_index"].get<uint64_t>() >= 5) {
            binding["fragment_record_index"] =
                binding["fragment_record_index"].get<uint64_t>() + 2;
        }
    }
    const Json core0 = Core(0, 0);
    const auto abi_id = [&](const std::string &which) {
        const std::string binding_id = "abs_" + which;
        for (const Json &abi : manifest["fragments"][0]["buffer_abi"]) {
            if (abi["binding_id"] == binding_id &&
                abi["logical_core"] == core0)
                return abi["id"].get<std::string>();
        }
        throw std::runtime_error("missing shared-lifetime BufferABI fixture");
    };
    const std::string input_abi = abi_id("input");
    const std::string data_abi = abi_id("data");
    const std::string output_abi = abi_id("output");
    bindings.push_back(AddressBinding(core0, 5, 0x100, input_abi));
    bindings.push_back(AddressBinding(core0, 5, 0x110, output_abi));
    bindings.push_back(AddressBinding(core0, 6, 1, input_abi));
    bindings.push_back(AddressBinding(core0, 6, 2, data_abi));
    bindings.push_back(AddressBinding(core0, 6, 3, output_abi));
    for (Json &binding : bindings) {
        if (binding["fragment_id"] == "fragment")
            binding["fragment_id"] = fragment_id;
    }
    std::sort(bindings.begin(), bindings.end(),
              [](const Json &left, const Json &right) {
                  const Json &left_core = left["logical_core"];
                  const Json &right_core = right["logical_core"];
                  return std::make_tuple(
                             left_core["die_id"].get<uint64_t>(),
                             left_core["local_core_id"].get<uint64_t>(),
                             left["fragment_id"].get<std::string>(),
                             left["fragment_record_index"].get<uint64_t>(),
                             left["operand_id"].get<uint64_t>()) <
                         std::make_tuple(
                             right_core["die_id"].get<uint64_t>(),
                             right_core["local_core_id"].get<uint64_t>(),
                             right["fragment_id"].get<std::string>(),
                             right["fragment_record_index"].get<uint64_t>(),
                             right["operand_id"].get<uint64_t>());
              });
    manifest["address_operand_bindings"] = std::move(bindings);
    RefreshManifestIds(manifest);
    return manifest;
}

Json ViewAddendManifest() {
    Json manifest = Manifest();
    Json &leaf = manifest["fragments"][0];
    const auto root_slice = [] {
        return Json{{"value_id", "value_output"},
                    {"offset", Json::array({0, 0})},
                    {"shape", Json::array({2, 64})}};
    };
    const auto use_slice = [](bool second_row) {
        return Json{{"value_id", "value_output"},
                    {"offset", Json::array({second_row ? 1 : 0, 0})},
                    {"shape", Json::array({1, 16})}};
    };
    for (Json &abi : leaf["buffer_abi"]) {
        if (abi["binding_id"] == "abs_output") {
            abi["tensor_slice"] = root_slice();
            abi["size_bytes"] = 256;
        }
    }
    for (Json &stream : leaf["core_streams"]) {
        const bool second_row =
            stream["logical_core"]["local_core_id"].get<uint64_t>() == 0;
        stream["records"][2]["operands"][3]["literal_value"] = 256;
        for (Json &relocation : stream["address_relocations"]) {
            if (relocation["record_index"] == 4 &&
                relocation["operand_id"] == 3)
                relocation["addend"] = second_row ? 128 : 0;
        }
        for (Json &binding : manifest["address_operand_bindings"]) {
            if (binding["logical_core"] != stream["logical_core"])
                continue;
            const uint64_t record =
                binding["fragment_record_index"].get<uint64_t>();
            const uint64_t operand = binding["operand_id"].get<uint64_t>();
            if ((record == 3 && operand == 0x110) ||
                (record == 4 && operand == 3))
                binding["tensor_slices"] =
                    Json::array({use_slice(second_row)});
            else if ((record == 2 && (operand == 8 || operand == 9)) ||
                     (record == 7 && operand == 7))
                binding["tensor_slices"] = Json::array({root_slice()});
        }
    }
    for (Json &definition : manifest["program_symbol_definitions"]) {
        if (definition["symbol"]["id"] == "p_abs_output")
            definition["size_bytes"] = 256;
    }
    RefreshManifestIds(manifest);
    return manifest;
}

void Require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

void ExpectFailure(const std::function<void()> &operation,
                   const std::string &name) {
    try {
        operation();
    } catch (const ProgramFinalizerError &) {
        return;
    }
    throw std::runtime_error("expected ProgramFinalizerError: " + name);
}

uint32_t RotateRight(uint32_t value, unsigned int shift) {
    return (value >> shift) | (value << (32 - shift));
}

std::string Sha256(const std::vector<uint8_t> &input) {
    static constexpr std::array<uint32_t, 64> constants{{
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2}};
    std::array<uint32_t, 8> state{{
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};
    std::vector<uint8_t> bytes = input;
    const uint64_t bit_length = static_cast<uint64_t>(bytes.size()) * 8;
    bytes.push_back(0x80);
    while (bytes.size() % 64 != 56)
        bytes.push_back(0);
    for (int shift = 56; shift >= 0; shift -= 8)
        bytes.push_back(static_cast<uint8_t>(bit_length >> shift));

    for (std::size_t block = 0; block < bytes.size(); block += 64) {
        std::array<uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            const std::size_t offset = block + index * 4;
            words[index] = (static_cast<uint32_t>(bytes[offset]) << 24) |
                           (static_cast<uint32_t>(bytes[offset + 1]) << 16) |
                           (static_cast<uint32_t>(bytes[offset + 2]) << 8) |
                           static_cast<uint32_t>(bytes[offset + 3]);
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            const uint32_t s0 = RotateRight(words[index - 15], 7) ^
                                RotateRight(words[index - 15], 18) ^
                                (words[index - 15] >> 3);
            const uint32_t s1 = RotateRight(words[index - 2], 17) ^
                                RotateRight(words[index - 2], 19) ^
                                (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const uint32_t sum1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^
                                  RotateRight(e, 25);
            const uint32_t choose = (e & f) ^ ((~e) & g);
            const uint32_t temp1 =
                h + sum1 + choose + constants[index] + words[index];
            const uint32_t sum0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^
                                  RotateRight(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (uint32_t word : state)
        output << std::setw(8) << word;
    return output.str();
}

void Run() {
    Require(Sha256({}) ==
                "e3b0c44298fc1c149afbf4c8996fb924"
                "27ae41e4649b934ca495991b7852b855",
            "self-test SHA-256 implementation is not canonical");
    const ProgramArtifactFinalizer finalizer;
    const Json valid = Manifest();
    const std::string text = valid.dump();
    const auto dto = ProgramArtifactFinalizer::Parse(text);
    const ProgramArtifact artifact = finalizer.Finalize(dto);
    Require(artifact.cores.size() == 2, "expected two cores");
    Require(artifact.cores[0].core_id == 2 && artifact.cores[1].core_id == 9,
            "cores were not sorted by runtime id");
    Require(artifact.cores[0].records.size() == 8 &&
                artifact.cores[1].records.size() == 8,
            "ordinary stream record count mismatch");
    Require(artifact.relocations.size() == 28,
            "relocation count/remap mismatch");
    Require(artifact.relocations.front().core_index == 0 &&
                artifact.relocations.back().core_index == 1,
            "relocations were not remapped to sorted core ordinals");
    Require(artifact.envelope.active_cores == std::vector<uint64_t>({2, 9}),
            "envelope core remap mismatch");
    const std::vector<uint8_t> encoded = finalizer.FinalizeEncoded(text);
    Require(encoded == finalizer.FinalizeEncoded(text),
            "finalization is not deterministic");
    Require(EncodeProgramArtifact(DecodeProgramArtifact(encoded)) == encoded,
            "final artifact did not round-trip canonically");

    for (uint64_t opcode : std::array<uint64_t, 8>{{
             0x10, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20}}) {
        const Json stage2 = Stage2Manifest(opcode);
        const auto stage2_dto =
            ProgramArtifactFinalizer::Parse(stage2.dump());
        const ProgramArtifact stage2_artifact =
            finalizer.Finalize(stage2_dto);
        uint64_t count = 0;
        for (const ProgramCore &core : stage2_artifact.cores)
            for (const ExternalRecord &record : core.records)
                if (static_cast<uint64_t>(record.opcode) == opcode)
                    ++count;
        Require(count == 2,
                "Stage2 synthetic opcode did not finalize on both cores");
        Require(EncodeProgramArtifact(
                    DecodeProgramArtifact(
                        EncodeProgramArtifact(stage2_artifact))) ==
                    EncodeProgramArtifact(stage2_artifact),
                "Stage2 fixed record failed byte round-trip");
    }
    Json bad_attention = Stage2Manifest(0x1b);
    for (Json &stream : bad_attention["fragments"][0]["core_streams"])
        stream["records"][4]["operands"][15]["literal_value"] = 2;
    RefreshManifestIds(bad_attention);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(bad_attention.dump()); },
        "ATTENTION_EXACT stale query-key pair count");
    Json bad_field = Stage2Manifest(0x1a);
    for (Json &stream : bad_field["fragments"][0]["core_streams"])
        stream["records"][4]["operands"][4]["name"] = "tokens";
    RefreshManifestIds(bad_field);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_field.dump()); },
                  "ROPE_QK_EXACT forged field name");
    Json non_absolute = Stage2Manifest(0x1d);
    for (Json &stream : non_absolute["fragments"][0]["core_streams"]) {
        stream["records"][4]["operands"][4]["symbol_ref"] = "p_region";
        for (Json &relocation : stream["address_relocations"])
            if (relocation["record_index"] == 4 &&
                relocation["operand_id"] == 1) {
                relocation["symbol_kind"] = 2;
                relocation["symbol_ref"] = "p_region";
            }
    }
    RefreshManifestIds(non_absolute);
    ExpectFailure([&] { finalizer.FinalizeJson(non_absolute.dump()); },
                  "fixed record non-absolute relocation");
    Json wrong_fixed_dtype = Stage2Manifest(0x1c);
    for (Json &abi : wrong_fixed_dtype["fragments"][0]["buffer_abi"])
        if (abi["binding_id"] == "abs_input")
            abi["dtype"] = "fp16";
    RefreshManifestIds(wrong_fixed_dtype);
    ExpectFailure([&] { finalizer.FinalizeJson(wrong_fixed_dtype.dump()); },
                  "EMBEDDING_LOOKUP indices require INT32 BufferABI");
    Json wide_fixed = Stage2Manifest(0x1d);
    for (Json &stream : wide_fixed["fragments"][0]["core_streams"])
        stream["records"][4]["operands"][7]["literal_value"] =
            uint64_t{1} << 32;
    RefreshManifestIds(wide_fixed);
    ExpectFailure([&] { finalizer.FinalizeJson(wide_fixed.dump()); },
                  "GREEDY_SAMPLE wide token_rows");
    Json bad_ce_dtype = Stage2Manifest(0x1e);
    for (Json &stream : bad_ce_dtype["fragments"][0]["core_streams"])
        stream["records"][4]["operands"][2]["literal_value"] = 1;
    RefreshManifestIds(bad_ce_dtype);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_ce_dtype.dump()); },
                  "CROSS_ENTROPY_FORWARD loss requires FP32");
    Json bad_ce_rows = Stage2Manifest(0x1e);
    for (Json &stream : bad_ce_rows["fragments"][0]["core_streams"])
        stream["records"][4]["operands"][7]["literal_value"] = 7;
    RefreshManifestIds(bad_ce_rows);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_ce_rows.dump()); },
                  "CROSS_ENTROPY_FORWARD TP row quotient");
    Json bad_ce_abi = Stage2Manifest(0x1e);
    for (Json &abi : bad_ce_abi["fragments"][0]["buffer_abi"])
        if (abi["binding_id"] == "abs_output") abi["dtype"] = "fp16";
    RefreshManifestIds(bad_ce_abi);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_ce_abi.dump()); },
                  "CROSS_ENTROPY_FORWARD loss BufferABI requires FP32");
    Json missing_ce_backward_aux = Stage2Manifest(0x1f);
    for (Json &stream : missing_ce_backward_aux["fragments"][0]["core_streams"])
        stream["records"][4]["operands"].erase(
            stream["records"][4]["operands"].begin() + 8);
    RefreshManifestIds(missing_ce_backward_aux);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(missing_ce_backward_aux.dump()); },
        "CROSS_ENTROPY_BACKWARD missing upstream AUX operand");
    Json reordered_ce_backward_aux = Stage2Manifest(0x1f);
    for (Json &stream : reordered_ce_backward_aux["fragments"][0]["core_streams"])
        std::swap(stream["records"][4]["operands"][7],
                  stream["records"][4]["operands"][8]);
    RefreshManifestIds(reordered_ce_backward_aux);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(reordered_ce_backward_aux.dump()); },
        "CROSS_ENTROPY_BACKWARD reordered upstream AUX operand");
    Json bad_ce_backward_dtype = Stage2Manifest(0x1f);
    for (Json &stream : bad_ce_backward_dtype["fragments"][0]["core_streams"])
        stream["records"][4]["operands"][2]["literal_value"] = 1;
    RefreshManifestIds(bad_ce_backward_dtype);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(bad_ce_backward_dtype.dump()); },
        "CROSS_ENTROPY_BACKWARD upstream dtype");
    Json bad_ce_backward_mode = Stage2Manifest(0x1f);
    for (Json &stream : bad_ce_backward_mode["fragments"][0]["core_streams"])
        stream["records"][4]["operands"][5]["literal_value"] = 2;
    RefreshManifestIds(bad_ce_backward_mode);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(bad_ce_backward_mode.dump()); },
        "CROSS_ENTROPY_BACKWARD upstream mode");
    Json bad_sgd_dtype = Stage2Manifest(0x20);
    for (Json &stream : bad_sgd_dtype["fragments"][0]["core_streams"])
        stream["records"][4]["operands"][1]["literal_value"] = 1;
    RefreshManifestIds(bad_sgd_dtype);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_sgd_dtype.dump()); },
                  "SGD_UPDATE gradient dtype");
    Json bad_sgd_lr = Stage2Manifest(0x20);
    for (Json &stream : bad_sgd_lr["fragments"][0]["core_streams"])
        stream["records"][4]["operands"][8]["literal_value"] = 0;
    RefreshManifestIds(bad_sgd_lr);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_sgd_lr.dump()); },
                  "SGD_UPDATE zero learning rate");
    Json bad_sgd_momentum = Stage2Manifest(0x20);
    for (Json &stream : bad_sgd_momentum["fragments"][0]["core_streams"])
        stream["records"][4]["operands"][9]["literal_value"] =
            UINT64_C(0x3fe0000000000000);
    RefreshManifestIds(bad_sgd_momentum);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_sgd_momentum.dump()); },
                  "SGD_UPDATE nonzero momentum");
    Json bad_sgd_alias = Stage2Manifest(0x20);
    for (Json &stream : bad_sgd_alias["fragments"][0]["core_streams"])
        stream["records"][4]["operands"][6]["symbol_ref"] = "p_abs_data";
    RefreshManifestIds(bad_sgd_alias);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_sgd_alias.dump()); },
                  "SGD_UPDATE weight/updated alias");
    Json generic_int32 = Manifest();
    for (Json &abi : generic_int32["fragments"][0]["buffer_abi"])
        if (abi["binding_id"] == "abs_input") {
            abi["dtype"] = "int32";
            abi["tensor_slice"]["shape"] = Json::array({1, 16});
        }
    for (Json &binding : generic_int32["address_operand_bindings"])
        binding.erase("tensor_slices");
    RefreshManifestIds(generic_int32);
    ExpectFailure([&] { finalizer.FinalizeJson(generic_int32.dump()); },
                  "generic compute INT32 BufferABI");
    Json int32_state = StateManifest();
    for (Json &linked : int32_state["fragments"])
        for (Json &abi : linked["state_abi"])
            abi["dtype"] = "int32";
    ExpectFailure([&] { ProgramArtifactFinalizer::Parse(int32_state.dump()); },
                  "StateABI INT32");
    Json trainable_state = StateManifest();
    std::string trainable_state_abi_id;
    for (Json &linked : trainable_state["fragments"])
        for (Json &abi : linked["state_abi"]) {
            abi["kind"] = "trainable_parameter";
            abi["access"] = "read_write";
            abi["id"] = StableId(
                "state_abi", "wafer_frontend.state_abi/v1alpha1",
                Without(abi, {"id"}));
            trainable_state_abi_id = abi["id"].get<std::string>();
        }
    for (Json &binding : trainable_state["state_operand_bindings"])
        binding["state_abi_id"] = trainable_state_abi_id;
    RefreshManifestIds(trainable_state);
    finalizer.FinalizeJson(trainable_state.dump());
    Json bad_trainable_access = trainable_state;
    for (Json &linked : bad_trainable_access["fragments"])
        for (Json &abi : linked["state_abi"])
            abi["access"] = "read_only";
    ExpectFailure(
        [&] { ProgramArtifactFinalizer::Parse(bad_trainable_access.dump()); },
        "TRAINABLE_PARAMETER StateABI requires READ_WRITE");
    Json region7 = Manifest();
    Json region_leaf = region7["fragments"][0];
    region_leaf["kind"] = "isa_region";
    region7["fragments"][0] = {
        {"schema_version", "wafer_frontend.region_manifest/v1alpha12"},
        {"producer_pass", "region_lowering"},
        {"id", "region_manifest"},
        {"region_id", "region0"},
        {"fusion_plan_id", "plan0"},
        {"target_dies", Json::array({0})},
        {"fragment", std::move(region_leaf)}};
    region7["input_digests"].push_back(
        {{"kind", "fusion_plan"},
         {"artifact_id", "plan0"},
         {"schema_version", "wafer_frontend.fusion_plan/v1alpha10"},
         {"digest", std::string(64, '5')}});
    RefreshManifestIds(region7);
    finalizer.FinalizeJson(region7.dump());
    Json old_fusion = region7;
    for (Json &digest : old_fusion["input_digests"])
        if (digest["kind"] == "fusion_plan")
            digest["schema_version"] =
                "wafer_frontend.fusion_plan/v1alpha9";
    RefreshManifestIds(old_fusion);
    ExpectFailure([&] { finalizer.FinalizeJson(old_fusion.dump()); },
                  "old fusion plan trust version");
    Json old_region = region7;
    old_region["fragments"][0]["schema_version"] =
        "wafer_frontend.region_manifest/v1alpha11";
    ExpectFailure([&] { ProgramArtifactFinalizer::Parse(old_region.dump()); },
                  "old RegionManifest version");
    for (const std::pair<std::string, std::string> &stale :
         std::array<std::pair<std::string, std::string>, 4>{{
             {"ir1", "wafer_frontend.ir1/v1alpha12"},
             {"ir2_projection", "wafer_frontend.ir2_projection_result/v1alpha12"},
             {"schedule_set", "wafer_frontend.intra_die_schedule_set/v1alpha8"},
             {"global_action_dag", "wafer_frontend.global_action_dag/v1alpha10"}}}) {
        Json old_trust = Manifest();
        for (Json &digest : old_trust["input_digests"])
            if (digest["kind"] == stale.first)
                digest["schema_version"] = stale.second;
        RefreshManifestIds(old_trust);
        ExpectFailure([&] { finalizer.FinalizeJson(old_trust.dump()); },
                      "old upstream trust version " + stale.first);
    }
    const Json transfer_json = TransferManifest();
    const auto transfer_dto =
        ProgramArtifactFinalizer::Parse(transfer_json.dump());
    const ProgramArtifact transfer_artifact =
        finalizer.Finalize(transfer_dto);
    std::map<Opcode, uint64_t> transfer_opcodes;
    for (const ProgramCore &core : transfer_artifact.cores)
        for (const ExternalRecord &record : core.records)
            ++transfer_opcodes[record.opcode];
    Require(transfer_opcodes[Opcode::DTE_SEND] == 1 &&
                transfer_opcodes[Opcode::DTE_RECV] == 1 &&
                transfer_opcodes[Opcode::DTE_WAIT] == 1,
            "STATE_TRANSFER did not finalize to one SEND/RECV/WAIT");
    const std::vector<uint8_t> transfer_encoded =
        finalizer.FinalizeEncoded(transfer_json.dump());
    Require(
        transfer_encoded ==
            finalizer.FinalizeEncoded(transfer_json.dump()),
        "STATE_TRANSFER finalization is not deterministic");
    auto transfer_leaf = [](
        frontend::LinkedProgramManifestDto &manifest,
        bool source) -> CommandFragmentDto & {
        for (LinkedFragmentDto &linked : manifest.fragments) {
            CommandFragmentDto *leaf =
                std::get_if<CommandFragmentDto>(&linked);
            if (leaf && leaf->kind == FragmentKindDto::STATE_TRANSFER &&
                (leaf->claimed_action_ids.size() == 1) == source)
                return *leaf;
        }
        throw std::runtime_error(
            "synthetic manifest lacks requested STATE_TRANSFER leaf");
    };

    auto old_linked_version = transfer_dto;
    old_linked_version.schema_version =
        "wafer_frontend.linked_program_manifest/v1alpha13";
    ExpectFailure([&] { finalizer.Finalize(old_linked_version); },
                  "old linked manifest version");

    auto old_command_version = transfer_dto;
    transfer_leaf(old_command_version, true).schema_version =
        "wafer_frontend.command_fragment/v1alpha12";
    ExpectFailure([&] { finalizer.Finalize(old_command_version); },
                  "old STATE_TRANSFER command version");
    Json wrapped_transfer = transfer_json;
    for (Json &linked : wrapped_transfer["fragments"])
        if (linked["schema_version"] ==
                "wafer_frontend.command_fragment/v1alpha13" &&
            linked["kind"] == "state_transfer" &&
            linked["claimed_action_ids"].size() == 1) {
            Json leaf = linked;
            linked = {
                {"schema_version",
                 "wafer_frontend.region_manifest/v1alpha12"},
                {"producer_pass", "forged_wrapper"},
                {"id", "forged_transfer_region"},
                {"region_id", "forged_transfer_region"},
                {"fusion_plan_id", "forged_transfer_plan"},
                {"target_dies", Json::array({0})},
                {"fragment", std::move(leaf)}};
            break;
        }
    RefreshManifestIds(wrapped_transfer);
    ExpectFailure(
        [&] {
            const auto dto =
                ProgramArtifactFinalizer::Parse(
                    wrapped_transfer.dump());
            finalizer.Finalize(dto);
        },
        "wrapped STATE_TRANSFER fragment");

    Json unknown_kind = transfer_json;
    for (Json &linked : unknown_kind["fragments"])
        if (linked["schema_version"] ==
                "wafer_frontend.command_fragment/v1alpha13" &&
            linked["kind"] == "state_transfer") {
            linked["kind"] = "future_state_transfer";
            break;
        }
    RefreshManifestIds(unknown_kind);
    ExpectFailure(
        [&] {
            ProgramArtifactFinalizer::Parse(unknown_kind.dump());
        },
        "unknown STATE_TRANSFER fragment kind");

    auto state_abi_forbidden = transfer_dto;
    transfer_leaf(state_abi_forbidden, true).state_abi.push_back(
        frontend::StateAbiDto{});
    ExpectFailure([&] { finalizer.Finalize(state_abi_forbidden); },
                  "STATE_TRANSFER carrying StateABI");

    auto missing_free = transfer_dto;
    transfer_leaf(missing_free, true)
        .core_streams.front().records.pop_back();
    ExpectFailure([&] { finalizer.Finalize(missing_free); },
                  "STATE_TRANSFER source missing FREE");

    auto reordered_destination = transfer_dto;
    {
        CommandFragmentDto &destination =
            transfer_leaf(reordered_destination, false);
        std::swap(destination.core_streams.front().records[1],
                  destination.core_streams.front().records[2]);
    }
    ExpectFailure([&] { finalizer.Finalize(reordered_destination); },
                  "STATE_TRANSFER destination reordered RECV/WAIT");

    auto missing_destination_alloc = transfer_dto;
    {
        CommandFragmentDto &destination =
            transfer_leaf(missing_destination_alloc, false);
        destination.core_streams.front().records.erase(
            destination.core_streams.front().records.begin());
    }
    ExpectFailure([&] { finalizer.Finalize(missing_destination_alloc); },
                  "singleton destination cannot omit ALLOC");

    auto duplicate_destination_alloc = transfer_dto;
    {
        CommandFragmentDto &destination =
            transfer_leaf(duplicate_destination_alloc, false);
        std::vector<frontend::RelocatableRecordDto> &records =
            destination.core_streams.front().records;
        records.insert(records.begin() + 1, records.front());
    }
    ExpectFailure([&] { finalizer.Finalize(duplicate_destination_alloc); },
                  "segmented destination duplicate ALLOC");

    auto reordered_destination_alloc = transfer_dto;
    {
        CommandFragmentDto &destination =
            transfer_leaf(reordered_destination_alloc, false);
        std::swap(destination.core_streams.front().records[0],
                  destination.core_streams.front().records[1]);
    }
    ExpectFailure([&] { finalizer.Finalize(reordered_destination_alloc); },
                  "segmented destination reordered ALLOC");


    auto duplicate_segment_fsm = transfer_dto;
    {
        CommandFragmentDto &source =
            transfer_leaf(duplicate_segment_fsm, true);
        std::vector<frontend::RelocatableRecordDto> &records =
            source.core_streams.front().records;
        frontend::RelocatableRecordDto second = records.front();
        second.source_global_action_id =
            "segmented_send_duplicate_fsm";
        records = {records.front(), std::move(second)};
        source.claimed_action_ids.push_back(
            "segmented_send_duplicate_fsm");
    }
    ExpectFailure([&] { finalizer.Finalize(duplicate_segment_fsm); },
                  "segmented STATE_TRANSFER duplicate DTE_FSM");

    auto duplicate_segment_token = transfer_dto;
    {
        CommandFragmentDto &destination =
            transfer_leaf(duplicate_segment_token, false);
        std::vector<frontend::RelocatableRecordDto> &records =
            destination.core_streams.front().records;
        frontend::RelocatableRecordDto recv = records[1];
        frontend::RelocatableRecordDto wait = records[2];
        recv.source_global_action_id =
            "segmented_recv_duplicate_token";
        wait.source_global_action_id =
            "segmented_wait_duplicate_token";
        recv.operands[4].symbol_ref = "segmented_unique_fsm";
        recv.operands[8].symbol_ref = "segmented_unique_peer";
        records.push_back(std::move(recv));
        records.push_back(std::move(wait));
        destination.claimed_action_ids.push_back(
            "segmented_recv_duplicate_token");
        destination.claimed_action_ids.push_back(
            "segmented_wait_duplicate_token");
    }
    ExpectFailure([&] { finalizer.Finalize(duplicate_segment_token); },
                  "segmented STATE_TRANSFER duplicate DTE_TOKEN");

    auto forged_literal = transfer_dto;
    transfer_leaf(forged_literal, true)
        .core_streams.front().records[0].operands[2].literal_value =
        uint64_t{0};
    ExpectFailure([&] { finalizer.Finalize(forged_literal); },
                  "STATE_TRANSFER SEND completion literal");

    auto mismatched_length = transfer_dto;
    transfer_leaf(mismatched_length, false)
        .core_streams.front().records[1].operands[6].literal_value =
        uint64_t{32};
    ExpectFailure([&] { finalizer.Finalize(mismatched_length); },
                  "STATE_TRANSFER SEND/RECV length mismatch");

    auto forged_token = transfer_dto;
    {
        CommandFragmentDto &destination =
            transfer_leaf(forged_token, false);
        destination.core_streams.front().records[2]
            .operands[0].symbol_ref = "forged_token";
    }
    ExpectFailure([&] { finalizer.Finalize(forged_token); },
                  "STATE_TRANSFER RECV/WAIT token mismatch");

    auto forged_staging = transfer_dto;
    {
        CommandFragmentDto &source =
            transfer_leaf(forged_staging, true);
        const frontend::BufferAbiDto *alternate = nullptr;
        for (LinkedFragmentDto &linked : forged_staging.fragments) {
            CommandFragmentDto *leaf =
                std::get_if<CommandFragmentDto>(&linked);
            if (!leaf)
                leaf =
                    &std::get<frontend::RegionManifestDto>(linked).fragment;
            for (const frontend::BufferAbiDto &abi : leaf->buffer_abi)
                if (abi.logical_core ==
                        source.core_streams.front().logical_core &&
                    abi.id != source.buffer_abi.front().id) {
                    alternate = &abi;
                    break;
                }
            if (alternate)
                break;
        }
        Require(alternate != nullptr,
                "synthetic transfer lacks alternate same-core BufferABI");
        auto binding = std::find_if(
            forged_staging.address_operand_bindings.begin(),
            forged_staging.address_operand_bindings.end(),
            [&](const AddressOperandBindingDto &item) {
                return item.fragment_id == source.id &&
                       item.operand_id ==
                           SemanticOperandId::SOURCE_ADDRESS;
            });
        Require(binding !=
                    forged_staging.address_operand_bindings.end(),
                "synthetic transfer lacks SEND address binding");
        binding->buffer_abi_ids = {alternate->id};
        binding->tensor_slices = {alternate->tensor_slice};
    }
    ExpectFailure([&] { finalizer.Finalize(forged_staging); },
                  "STATE_TRANSFER mixed local staging root");
    const Json state_json = StateManifest();
    const auto state_dto = ProgramArtifactFinalizer::Parse(state_json.dump());
    const ProgramArtifact state_artifact = finalizer.Finalize(state_dto);
    const ExternalRecord *lsu = nullptr;
    for (const ProgramCore &core : state_artifact.cores) {
        for (const ExternalRecord &record : core.records) {
            if (record.opcode == Opcode::LSU_LOAD)
                lsu = &record;
        }
    }
    Require(lsu != nullptr && std::holds_alternative<LsuOperands>(lsu->operands),
            "STATE_IO did not finalize to LSU_LOAD");
    const LsuOperands &lsu_operands = std::get<LsuOperands>(lsu->operands);
    Require(lsu_operands.hbm_address_bytes == 0x1000 &&
                lsu_operands.size_bytes == 64 &&
                lsu_operands.sram.kind == SramAddressKind::ABSOLUTE &&
                lsu_operands.sram.absolute_address_bytes == 0x400,
            "STATE_IO LSU operands do not preserve HBM/local endpoints");
    Require(state_artifact.relocations.size() == 30,
            "STATE_IO must add exact HBM and local relocations");

    auto set_state_load_range = [](Json &manifest, int64_t addend,
                                   uint64_t size_bytes) {
        for (Json &linked : manifest["fragments"]) {
            if (linked["kind"] != "state_io")
                continue;
            Json &stream = linked["core_streams"][0];
            stream["records"][0]["operands"][1]["literal_value"] =
                size_bytes;
            for (Json &relocation : stream["address_relocations"])
                if (relocation["record_index"] == 0 &&
                    relocation["operand_id"] == 6)
                    relocation["addend"] = addend;
            RefreshManifestIds(manifest);
            return;
        }
        throw std::runtime_error("missing STATE_IO range fixture");
    };
    Json ranged_state = state_json;
    set_state_load_range(ranged_state, 48, 16);
    const ProgramArtifact ranged_artifact =
        finalizer.FinalizeJson(ranged_state.dump());
    const ExternalRecord *ranged_lsu = nullptr;
    for (const ProgramCore &core : ranged_artifact.cores)
        for (const ExternalRecord &record : core.records)
            if (record.opcode == Opcode::LSU_LOAD)
                ranged_lsu = &record;
    Require(ranged_lsu != nullptr,
            "ranged STATE_IO did not preserve LSU_LOAD");
    const LsuOperands &ranged_operands =
        std::get<LsuOperands>(ranged_lsu->operands);
    const auto ranged_relocation = std::find_if(
        ranged_artifact.relocations.begin(),
        ranged_artifact.relocations.end(),
        [](const SemanticRelocation &relocation) {
            return relocation.operand_id ==
                       static_cast<uint16_t>(
                           SemanticOperandId::HBM_ADDRESS) &&
                   relocation.addend == 48;
        });
    Require(ranged_operands.hbm_address_bytes == 0x1000 &&
                ranged_operands.size_bytes == 16,
            "ranged STATE_IO did not preserve the exact HBM base and size");
    Require(ranged_relocation != ranged_artifact.relocations.end(),
            "ranged STATE_IO did not preserve the exact HBM addend");

    Json negative_state_range = state_json;
    set_state_load_range(negative_state_range, -1, 16);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(negative_state_range.dump()); },
        "negative StateABI relocation addend");

    Json out_of_bounds_state_range = state_json;
    set_state_load_range(out_of_bounds_state_range, 49, 16);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(out_of_bounds_state_range.dump()); },
        "StateABI relocation range exceeds backing allocation");

    auto missing_state_closure = state_dto;
    missing_state_closure.state_operand_bindings.clear();
    ExpectFailure([&] { finalizer.Finalize(missing_state_closure); },
                  "missing StateOperandBinding closure");

    auto duplicate_state_closure = state_dto;
    duplicate_state_closure.state_operand_bindings.push_back(
        duplicate_state_closure.state_operand_bindings.front());
    ExpectFailure([&] { finalizer.Finalize(duplicate_state_closure); },
                  "duplicate StateOperandBinding closure");

    auto double_closed_hbm = state_dto;
    const StateOperandBindingDto state_binding =
        double_closed_hbm.state_operand_bindings.front();
    const auto local_binding = std::find_if(
        double_closed_hbm.address_operand_bindings.begin(),
        double_closed_hbm.address_operand_bindings.end(),
        [&](const AddressOperandBindingDto &item) {
            return item.fragment_id == state_binding.fragment_id;
        });
    AddressOperandBindingDto forged_hbm = *local_binding;
    forged_hbm.operand_id = SemanticOperandId::HBM_ADDRESS;
    double_closed_hbm.address_operand_bindings.push_back(forged_hbm);
    std::sort(
        double_closed_hbm.address_operand_bindings.begin(),
        double_closed_hbm.address_operand_bindings.end(),
        [](const AddressOperandBindingDto &left,
           const AddressOperandBindingDto &right) {
            return std::make_tuple(
                       left.logical_core, left.fragment_id,
                       left.fragment_record_index,
                       static_cast<uint16_t>(left.operand_id)) <
                   std::make_tuple(
                       right.logical_core, right.fragment_id,
                       right.fragment_record_index,
                       static_cast<uint16_t>(right.operand_id));
        });
    ExpectFailure([&] { finalizer.Finalize(double_closed_hbm); },
                  "HBM relocation double BufferABI/StateABI closure");

    auto forged_hbm_address = state_dto;
    const auto hbm_definition = std::find_if(
        forged_hbm_address.program_symbol_definitions.begin(),
        forged_hbm_address.program_symbol_definitions.end(),
        [](const ProgramSymbolDefinitionDto &item) {
            return item.symbol.source_ref == "hbm_binding";
        });
    hbm_definition->value += 64;
    ExpectFailure([&] { finalizer.Finalize(forged_hbm_address); },
                  "StateABI HBM address mismatch");

    auto forbidden_load = state_dto;
    for (LinkedFragmentDto &linked : forbidden_load.fragments) {
        CommandFragmentDto *leaf = std::get_if<CommandFragmentDto>(&linked);
        if (leaf && leaf->kind == FragmentKindDto::STATE_IO)
            leaf->state_abi.front().access = StateAccessDto::RESERVED;
    }
    ExpectFailure([&] { finalizer.Finalize(forbidden_load); },
                  "LSU_LOAD forbidden by StateABI access");



    const Json view = ViewAddendManifest();
    const ProgramArtifact view_artifact = finalizer.FinalizeJson(view.dump());
    const auto view_relocation = std::find_if(
        view_artifact.relocations.begin(), view_artifact.relocations.end(),
        [](const SemanticRelocation &relocation) {
            return relocation.operand_id ==
                       static_cast<uint16_t>(
                           SemanticOperandId::COMPUTE_OUTPUT_ADDRESS) &&
                   relocation.addend == 128;
        });
    Require(view_relocation != view_artifact.relocations.end(),
            "dense row-major view addend 128 was not preserved");

    auto mutate_view_output = [](Json &manifest,
                                 const std::function<void(Json &, Json &)> &edit) {
        Json &leaf = manifest["fragments"][0];
        for (Json &stream : leaf["core_streams"]) {
            if (stream["logical_core"]["local_core_id"] != 0)
                continue;
            Json *target_relocation = nullptr;
            for (Json &relocation : stream["address_relocations"]) {
                if (relocation["record_index"] == 4 &&
                    relocation["operand_id"] == 3) {
                    target_relocation = &relocation;
                    break;
                }
            }
            Json *target_binding = nullptr;
            for (Json &binding : manifest["address_operand_bindings"]) {
                if (binding["logical_core"] == stream["logical_core"] &&
                    binding["fragment_record_index"] == 4 &&
                    binding["operand_id"] == 3) {
                    target_binding = &binding;
                    break;
                }
            }
            if (!target_relocation || !target_binding)
                throw std::runtime_error("missing view relocation fixture");
            edit(*target_relocation, *target_binding);
            RefreshManifestIds(manifest);
            return;
        }
        throw std::runtime_error("missing view core fixture");
    };

    Json zero_view_addend = view;
    mutate_view_output(
        zero_view_addend, [](Json &relocation, Json &) {
            relocation["addend"] = 0;
        });
    ExpectFailure([&] { finalizer.FinalizeJson(zero_view_addend.dump()); },
                  "forged zero view addend");

    Json negative_view_addend = view;
    mutate_view_output(
        negative_view_addend, [](Json &relocation, Json &) {
            relocation["addend"] = -1;
        });
    ExpectFailure([&] { finalizer.FinalizeJson(negative_view_addend.dump()); },
                  "negative ABS view addend");

    Json out_of_root_view = view;
    mutate_view_output(
        out_of_root_view, [](Json &relocation, Json &binding) {
            relocation["addend"] = 256;
            binding["tensor_slices"][0]["offset"] = Json::array({2, 0});
        });
    ExpectFailure([&] { finalizer.FinalizeJson(out_of_root_view.dump()); },
                  "view outside root backing");

    const Json shared = SharedLifetimeManifest();
    const ProgramArtifact shared_artifact =
        finalizer.FinalizeJson(shared.dump());
    Require(shared_artifact.cores[1].records.size() == 10,
            "cross-action shared-storage stream was not preserved");

    Json duplicate_alloc = valid;
    duplicate_alloc["fragments"][0]["core_streams"][0]["records"][1]
                   ["operands"][1]["symbol_ref"] = "p_label_input";
    duplicate_alloc["fragments"][0]["core_streams"][0]
                   ["address_relocations"][3]["symbol_ref"] = "p_label_input";
    RefreshManifestIds(duplicate_alloc);
    ExpectFailure([&] { finalizer.FinalizeJson(duplicate_alloc.dump()); },
                  "duplicate SRAM allocation");
    Json free_before_alloc = valid;
    Json reordered = Json::array();
    reordered.push_back(free_before_alloc["core_streams"][0]["records"][5]);
    for (std::size_t index = 0;
         index < free_before_alloc["core_streams"][0]["records"].size();
         ++index) {
        if (index != 5)
            reordered.push_back(
                free_before_alloc["core_streams"][0]["records"][index]);
    }
    free_before_alloc["core_streams"][0]["records"] = std::move(reordered);
    RefreshManifestIds(free_before_alloc);
    ExpectFailure([&] { finalizer.FinalizeJson(free_before_alloc.dump()); },
                  "free before allocation");
    Json dangling = valid;
    dangling["fragments"][0]["core_streams"][0]["records"].erase(
        dangling["fragments"][0]["core_streams"][0]["records"].end() - 1);
    dangling["fragments"][0]["core_streams"][0]["address_relocations"].erase(
        dangling["fragments"][0]["core_streams"][0]["address_relocations"].end() - 1);
    dangling["core_streams"][0]["records"].erase(
        dangling["core_streams"][0]["records"].end() - 1);
    Json retained_bindings = Json::array();
    for (const Json &binding : dangling["address_operand_bindings"]) {
        const Json &core = binding["logical_core"];
        if (!(core["die_id"].get<uint64_t>() == 0 &&
              core["local_core_id"].get<uint64_t>() == 0 &&
              binding["fragment_record_index"].get<uint64_t>() == 7 &&
              binding["operand_id"].get<uint64_t>() == 7))
            retained_bindings.push_back(binding);
    }
    dangling["address_operand_bindings"] = std::move(retained_bindings);
    RefreshManifestIds(dangling);
    ExpectFailure([&] { finalizer.FinalizeJson(dangling.dump()); },
                  "dangling TASK allocation");

    Json unknown = valid;
    unknown["unknown"] = 1;
    ExpectFailure([&] { ProgramArtifactFinalizer::Parse(unknown.dump()); },
                  "unknown top-level field");
    const std::array<const char *, 14> moe_input_kinds{{
        "moe_swizzle_scale_spec",
        "moe_swizzle_scale_oracle",
        "moe_swizzle_execution",
        "moe_swizzle_decision",
        "moe_swizzle_workload_selection",
        "moe_swizzle_workload_projection",
        "moe_swizzle_workload_state_abi",
        "moe_swizzle_workload_value_bridge",
        "moe_swizzle_workload_abi",
        "moe_swizzle_hardware_facts",
        "moe_swizzle_overlay",
        "moe_swizzle_projection",
        "moe_swizzle_core_address_abi",
        "moe_swizzle_operand_abi",
    }};
    for (const char *kind : moe_input_kinds) {
        Json parsed_input = valid;
        parsed_input["input_digests"][0]["kind"] = kind;
        RefreshManifestIds(parsed_input);
        static_cast<void>(
            ProgramArtifactFinalizer::Parse(parsed_input.dump()));
    }
    Json unknown_moe_input = valid;
    unknown_moe_input["input_digests"][0]["kind"] =
        "moe_swizzle_unknown";
    RefreshManifestIds(unknown_moe_input);
    ExpectFailure(
        [&] {
            ProgramArtifactFinalizer::Parse(unknown_moe_input.dump());
        },
        "unknown MoE Swizzle input kind");

    Json moe_fragment = valid;
    moe_fragment["fragments"][0]["kind"] = "moe_swizzle";
    moe_fragment["fragments"][0]["producer_pass"] =
        "moe_swizzle_standard_lowering";
    RefreshManifestIds(moe_fragment);
    const auto moe_fragment_dto =
        ProgramArtifactFinalizer::Parse(moe_fragment.dump());
    Require(
        std::get<CommandFragmentDto>(
            moe_fragment_dto.fragments.front()).kind ==
            FragmentKindDto::MOE_SWIZZLE,
        "MoE Swizzle FragmentKind parser changed");
    ExpectFailure(
        [&] { finalizer.FinalizeJson(moe_fragment.dump()); },
        "MoE Swizzle fragment under a foreign top producer");

    Json moe_wrong_kind = valid;
    moe_wrong_kind["fragments"][0]["producer_pass"] =
        "moe_swizzle_standard_lowering";
    RefreshManifestIds(moe_wrong_kind);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(moe_wrong_kind.dump()); },
        "MoE Swizzle producer on a foreign fragment kind");

    Json unadmitted_moe_top = moe_fragment;
    unadmitted_moe_top["producer_pass"] =
        "moe_swizzle_standard_linker";
    RefreshManifestIds(unadmitted_moe_top);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(unadmitted_moe_top.dump()); },
        "MoE Swizzle top before dedicated exact admission");

    Json unknown_moe_fragment = valid;
    unknown_moe_fragment["fragments"][0]["kind"] =
        "moe_swizzle_unknown";
    RefreshManifestIds(unknown_moe_fragment);
    ExpectFailure(
        [&] {
            ProgramArtifactFinalizer::Parse(
                unknown_moe_fragment.dump());
        },
        "unknown MoE Swizzle fragment kind");
    Json missing = valid;
    missing["fragments"][0]["core_streams"][0]["records"][0]["operands"][0].erase("name");
    ExpectFailure([&] { ProgramArtifactFinalizer::Parse(missing.dump()); },
                  "missing nested field");
    Json missing_identity = valid;
    missing_identity["fragments"][0]["buffer_abi"][0].erase("id");
    ExpectFailure(
        [&] { ProgramArtifactFinalizer::Parse(missing_identity.dump()); },
        "missing nested identity must normalize parser exception");
    Json state_abi = valid;
    state_abi["fragments"][0]["state_abi"].push_back(StateAbi());
    RefreshManifestIds(state_abi);
    static_cast<void>(ProgramArtifactFinalizer::Parse(state_abi.dump()));
    Json stale_state_abi = state_abi;
    stale_state_abi["fragments"][0]["state_abi"][0]["id"] = "stale";
    RefreshManifestIds(stale_state_abi);
    ExpectFailure(
        [&] { ProgramArtifactFinalizer::Parse(stale_state_abi.dump()); },
        "unstable StateABI id");

    ExpectFailure(
        [&] {
            ProgramArtifactFinalizer::Parse(
                "{\"schema_version\":1,\"schema_version\":2}");
        },
        "duplicate key");
    Json capability = valid;
    capability["capabilities"] = 1;
    RefreshManifestIds(capability);
    ExpectFailure([&] { ProgramArtifactFinalizer::Parse(capability.dump()); },
                  "unsupported capability");
    Json wrong_type = valid;
    wrong_type["capabilities"] = "0";
    ExpectFailure([&] { ProgramArtifactFinalizer::Parse(wrong_type.dump()); },
                  "integer type coercion");
    Json range = valid;
    range["core_bindings"][0]["runtime_core_id"] = 65536;
    RefreshManifestIds(range);
    ExpectFailure([&] { ProgramArtifactFinalizer::Parse(range.dump()); },
                  "runtime core range");
    Json collective = valid;
    collective["fragments"][0]["core_streams"][0]["records"][4]["opcode"] = 0x40;
    RefreshManifestIds(collective);
    ExpectFailure([&] { finalizer.FinalizeJson(collective.dump()); },
                  "collective opcode fail-closed");
    Json two_bind_inputs = valid;
    two_bind_inputs["fragments"][0]["core_streams"][0]["records"][3]
                   ["operands"][0]["literal_value"] = 2;
    RefreshManifestIds(two_bind_inputs);
    ExpectFailure([&] { finalizer.FinalizeJson(two_bind_inputs.dump()); },
                  "ordinary MATMUL input_count=2");
    Json runtime_relocation = valid;
    runtime_relocation["fragments"][0]["core_streams"][0]
                      ["runtime_relocations"] =
        Json::array({{{"record_index", 0},
                      {"field", "start_tag"},
                      {"symbol_ref", "start0"}}});
    RefreshManifestIds(runtime_relocation);
    ExpectFailure([&] { finalizer.FinalizeJson(runtime_relocation.dump()); },
                  "runtime relocation fail-closed");
    Json region = valid;
    Json leaf = region["fragments"][0];
    region["fragments"][0] =
        {{"schema_version", "wafer_frontend.region_manifest/v1alpha11"},
         {"producer_pass", "region_lowering"},
         {"id", "region_manifest"},
         {"region_id", "region0"},
         {"fusion_plan_id", "plan0"},
         {"target_dies", Json::array({0})},
         {"fragment", std::move(leaf)}};
    RefreshManifestIds(region);
    ExpectFailure([&] { finalizer.FinalizeJson(region.dump()); },
                  "RegionManifest fail-closed");
    Json wide_address = valid;
    wide_address["program_symbol_definitions"][1]["value"] = 0x10000;
    RefreshManifestIds(wide_address);
    ExpectFailure([&] { finalizer.FinalizeJson(wide_address.dump()); },
                  "compute address wire overflow");
    Json stale = valid;
    stale["fragments"][0]["core_streams"][0]["address_relocations"][0]
         ["symbol_ref"] = "p_label_input";
    RefreshManifestIds(stale);
    ExpectFailure([&] { finalizer.FinalizeJson(stale.dump()); },
                  "wrong-kind relocation");

    Json stale_manifest_id = valid;
    stale_manifest_id["id"] = "linked_program_manifest_stale";
    ExpectFailure([&] { finalizer.FinalizeJson(stale_manifest_id.dump()); },
                  "stale manifest stable id");
    Json stale_fragment_id = valid;
    stale_fragment_id["fragments"][0]["id"] = "command_fragment_stale";
    ExpectFailure([&] { finalizer.FinalizeJson(stale_fragment_id.dump()); },
                  "stale fragment stable id");
    Json schema_legal_buffer_id = valid;
    const std::string old_abi_id =
        schema_legal_buffer_id["fragments"][0]["buffer_abi"][0]["id"];
    const std::string arbitrary_abi_id = "abi_schema_legal_fixture";
    schema_legal_buffer_id["fragments"][0]["buffer_abi"][0]["id"] =
        arbitrary_abi_id;
    for (Json &binding : schema_legal_buffer_id["address_operand_bindings"]) {
        for (Json &abi_id : binding["buffer_abi_ids"]) {
            if (abi_id == old_abi_id)
                abi_id = arbitrary_abi_id;
        }
    }
    RefreshManifestIds(schema_legal_buffer_id, false);
    finalizer.FinalizeJson(schema_legal_buffer_id.dump());
    Json stale_fragment_digest = valid;
    for (Json &digest : stale_fragment_digest["input_digests"]) {
        if (digest["kind"] == "command_fragment")
            digest["digest"] = std::string(64, '0');
    }
    stale_fragment_digest["id"] = StableId(
        "linked_program_manifest",
        "wafer_frontend.linked_program_manifest/v1alpha13",
        Without(stale_fragment_digest,
                {"schema_version", "producer_pass", "id"}));
    ExpectFailure([&] { finalizer.FinalizeJson(stale_fragment_digest.dump()); },
                  "stale embedded fragment digest");
    Json wrong_upstream_digest = valid;
    for (Json &digest : wrong_upstream_digest["input_digests"]) {
        if (digest["kind"] == "ir1")
            digest["artifact_id"] = "wrong_ir1";
    }
    RefreshManifestIds(wrong_upstream_digest);
    ExpectFailure([&] { finalizer.FinalizeJson(wrong_upstream_digest.dump()); },
                  "upstream input digest/source mismatch");
}

void RunPythonProducedManifest() {
    const std::string text((std::istreambuf_iterator<char>(std::cin)),
                           std::istreambuf_iterator<char>());
    const frontend::LinkedProgramManifestDto dto =
        ProgramArtifactFinalizer::Parse(text);
    const ProgramArtifactFinalizer finalizer;
    const ProgramArtifact artifact = finalizer.Finalize(dto);
    Require(!artifact.cores.empty(),
            "Python-produced manifest finalized to no core streams");
    const std::vector<uint8_t> bytes = EncodeProgramArtifact(artifact);
    Require(finalizer.FinalizeEncoded(text) == bytes,
            "Python-produced manifest finalization is not deterministic");
    Require(EncodeProgramArtifact(DecodeProgramArtifact(bytes)) == bytes,
            "Python-produced manifest did not round-trip canonically");
    std::size_t records = 0;
    for (const ProgramCore &core : artifact.cores)
        records += core.records.size();
    const bool mixed = std::any_of(
        dto.fragments.begin(), dto.fragments.end(),
        [](const frontend::LinkedFragmentDto &item) {
            return std::holds_alternative<frontend::RegionManifestDto>(item);
        });
    if (mixed) {
        auto forged_ag_addend = dto;
        bool mutated_ag_addend = false;
        for (frontend::LinkedFragmentDto &linked : forged_ag_addend.fragments) {
            frontend::CommandFragmentDto *leaf =
                std::get_if<frontend::CommandFragmentDto>(&linked);
            if (!leaf)
                leaf = &std::get<frontend::RegionManifestDto>(linked).fragment;
            for (frontend::CoreFragmentStreamDto &stream : leaf->core_streams) {
                for (frontend::AddressRelocationDto &relocation :
                     stream.address_relocations) {
                    if (relocation.record_index >= stream.records.size())
                        continue;
                    const Opcode opcode =
                        stream.records[relocation.record_index].opcode;
                    if (relocation.addend == 128 &&
                        (opcode == Opcode::DTE_ISSUE ||
                         opcode == Opcode::DTE_RECV ||
                         opcode == Opcode::DTE_SEND)) {
                        relocation.addend = 0;
                        mutated_ag_addend = true;
                        break;
                    }
                }
                if (mutated_ag_addend) break;
            }
            if (mutated_ag_addend) break;
        }
        Require(mutated_ag_addend,
                "E1 lacks the frozen 128-byte AllGather view addend");
        ExpectFailure([&] { finalizer.Finalize(forged_ag_addend); },
                      "tampered AllGather view addend 128 -> 0");

        auto endpoint = dto;
        auto endpoint_it = std::find_if(
            endpoint.runtime_symbol_definitions.begin(),
            endpoint.runtime_symbol_definitions.end(),
            [](const frontend::RuntimeSymbolDefinitionDto &item) {
                return item.symbol.kind ==
                    frontend::RuntimeSymbolKindDto::DTE_FSM;
            });
        Require(endpoint_it != endpoint.runtime_symbol_definitions.end(),
                "E1 lacks DTE_FSM negative fixture");
        endpoint_it->source_action_id = "unknown_action";
        ExpectFailure([&] { finalizer.Finalize(endpoint); },
                      "tampered runtime endpoint action");

        auto token = dto;
        auto token_it = std::find_if(
            token.runtime_symbol_definitions.begin(),
            token.runtime_symbol_definitions.end(),
            [](const frontend::RuntimeSymbolDefinitionDto &item) {
                return item.symbol.kind ==
                           frontend::RuntimeSymbolKindDto::DTE_TOKEN &&
                       item.destination_action_id.has_value();
            });
        Require(token_it != token.runtime_symbol_definitions.end(),
                "E1 lacks remote DTE_TOKEN negative fixture");
        token_it->destination_action_id.reset();
        ExpectFailure([&] { finalizer.Finalize(token); },
                      "tampered runtime token endpoint");

        auto event = dto;
        auto event_it = std::find_if(
            event.runtime_symbol_definitions.begin(),
            event.runtime_symbol_definitions.end(),
            [](const frontend::RuntimeSymbolDefinitionDto &item) {
                return item.symbol.kind ==
                    frontend::RuntimeSymbolKindDto::EVENT_TAG;
            });
        Require(event_it != event.runtime_symbol_definitions.end(),
                "E1 lacks EVENT_TAG negative fixture");
        event_it->logical_cores.pop_back();
        ExpectFailure([&] { finalizer.Finalize(event); },
                      "tampered event endpoint core");

        auto region = dto;
        auto region_it = std::find_if(
            region.fragments.begin(), region.fragments.end(),
            [](const frontend::LinkedFragmentDto &item) {
                return std::holds_alternative<frontend::RegionManifestDto>(
                    item);
            });
        Require(region_it != region.fragments.end(),
                "E1 lacks RegionManifest negative fixture");
        std::get<frontend::RegionManifestDto>(*region_it).target_dies.clear();
        ExpectFailure([&] { finalizer.Finalize(region); },
                      "tampered region die coverage");

        auto record_ref = dto;
        Require(!record_ref.core_streams.empty() &&
                    !record_ref.core_streams[0].records.empty(),
                "E1 lacks linked-record negative fixture");
        record_ref.core_streams[0].records[0].fragment_record_index =
            std::numeric_limits<uint64_t>::max();
        ExpectFailure([&] { finalizer.Finalize(record_ref); },
                      "tampered linked record reference");

        auto enum_wrap = dto;
        bool mutated_enum = false;
        for (frontend::LinkedFragmentDto &linked : enum_wrap.fragments) {
            frontend::CommandFragmentDto *leaf =
                std::get_if<frontend::CommandFragmentDto>(&linked);
            if (!leaf)
                leaf = &std::get<frontend::RegionManifestDto>(linked).fragment;
            for (frontend::CoreFragmentStreamDto &stream :
                 leaf->core_streams) {
                for (frontend::RelocatableRecordDto &record : stream.records) {
                    if (record.opcode == Opcode::DTE_SEND) {
                        record.operands[0].literal_value = uint64_t{256};
                        mutated_enum = true;
                        break;
                    }
                }
                if (mutated_enum) break;
            }
            if (mutated_enum) break;
        }
        Require(mutated_enum, "E1 lacks DTE_SEND enum negative fixture");
        ExpectFailure([&] { finalizer.Finalize(enum_wrap); },
                      "oversized runtime enum literal truncation");
    }
    Require(artifact.cores.size() == 2 && records == 220 &&
                artifact.relocations.size() == 354,
            "tiny E1 TP2 finalization counts changed");
    Require(dto.state_operand_bindings.size() == 16,
            "tiny E1 TP2 must expose sixteen state operand bindings");

    auto missing_state_binding = dto;
    missing_state_binding.state_operand_bindings.pop_back();
    ExpectFailure([&] { finalizer.Finalize(missing_state_binding); },
                  "real TP2 missing StateABI relocation closure");

    auto forged_state_address = dto;
    const StateOperandBindingDto &state_binding =
        forged_state_address.state_operand_bindings.front();
    const frontend::StateAbiDto *state_abi = nullptr;
    for (frontend::LinkedFragmentDto &linked : forged_state_address.fragments) {
        frontend::CommandFragmentDto *leaf =
            std::get_if<frontend::CommandFragmentDto>(&linked);
        if (!leaf)
            leaf = &std::get<frontend::RegionManifestDto>(linked).fragment;
        if (leaf->id != state_binding.fragment_id)
            continue;
        const auto abi = std::find_if(
            leaf->state_abi.begin(), leaf->state_abi.end(),
            [&](const frontend::StateAbiDto &item) {
                return item.id == state_binding.state_abi_id;
            });
        if (abi != leaf->state_abi.end()) state_abi = &*abi;
        break;
    }
    Require(state_abi != nullptr,
            "real TP2 state binding lacks its leaf StateABI");
    auto state_definition = std::find_if(
        forged_state_address.program_symbol_definitions.begin(),
        forged_state_address.program_symbol_definitions.end(),
        [&](const ProgramSymbolDefinitionDto &item) {
            return item.symbol.source_ref == state_abi->hbm_binding_ref;
        });
    Require(state_definition !=
                forged_state_address.program_symbol_definitions.end(),
            "real TP2 StateABI lacks its HBM symbol definition");
    state_definition->value = state_abi->address + state_abi->alignment_bytes;
    ExpectFailure([&] { finalizer.Finalize(forged_state_address); },
                  "real TP2 tampered StateABI HBM address");

    std::cout << "bytes=" << bytes.size() << " sha256=" << Sha256(bytes)
              << " cores=" << artifact.cores.size()
              << " records=" << records
              << " relocations=" << artifact.relocations.size() << '\n';
}
void RunPd1ProducedManifest() {
    const std::string text((std::istreambuf_iterator<char>(std::cin)),
                           std::istreambuf_iterator<char>());
    const frontend::LinkedProgramManifestDto dto =
        ProgramArtifactFinalizer::Parse(text);
    const ProgramArtifactFinalizer finalizer;
    const ProgramArtifact artifact = finalizer.Finalize(dto);
    const std::vector<uint8_t> bytes = finalizer.FinalizeEncoded(text);
    Require(bytes == finalizer.FinalizeEncoded(text),
            "PD1 finalization is not deterministic");

    std::map<Opcode, uint64_t> raw_opcodes;
    uint64_t raw_records = 0;
    uint64_t transfer_fragments = 0;
    uint64_t transfer_records = 0;
    for (const LinkedFragmentDto &linked : dto.fragments) {
        const CommandFragmentDto &leaf =
            std::holds_alternative<CommandFragmentDto>(linked)
                ? std::get<CommandFragmentDto>(linked)
                : std::get<frontend::RegionManifestDto>(linked).fragment;
        if (leaf.kind == FragmentKindDto::STATE_TRANSFER)
            ++transfer_fragments;
        for (const frontend::CoreFragmentStreamDto &stream :
             leaf.core_streams)
            for (const frontend::RelocatableRecordDto &record :
                 stream.records) {
                ++raw_records;
                ++raw_opcodes[record.opcode];
                if (leaf.kind == FragmentKindDto::STATE_TRANSFER)
                    ++transfer_records;
            }
    }
    Require(dto.fragments.size() == 10 && transfer_fragments == 4 &&
                transfer_records == 10 && raw_records == 30 &&
                dto.address_operand_bindings.size() == 40 &&
                dto.state_operand_bindings.size() == 4,
            "PD1 linked manifest exact counts changed");
    Require(raw_opcodes[Opcode::LSU_LOAD] == 2 &&
                raw_opcodes[Opcode::LSU_STORE] == 2 &&
                raw_opcodes[Opcode::DTE_SEND] == 2 &&
                raw_opcodes[Opcode::DTE_RECV] == 2 &&
                raw_opcodes[Opcode::DTE_WAIT] == 2,
            "PD1 must preserve two K/V LOAD/SEND/RECV/WAIT/STORE paths");

    uint64_t artifact_records = 0;
    std::map<Opcode, uint64_t> artifact_opcodes;
    for (const ProgramCore &core : artifact.cores)
        for (const ExternalRecord &record : core.records) {
            ++artifact_records;
            ++artifact_opcodes[record.opcode];
        }
    Require(artifact.cores.size() == 2 && artifact_records == 30 &&
                artifact_opcodes[Opcode::LSU_LOAD] == 2 &&
                artifact_opcodes[Opcode::LSU_STORE] == 2 &&
                artifact_opcodes[Opcode::DTE_SEND] == 2 &&
                artifact_opcodes[Opcode::DTE_RECV] == 2 &&
                artifact_opcodes[Opcode::DTE_WAIT] == 2,
            "PD1 finalized artifact bridge counts changed");

    auto old_linked = dto;
    old_linked.schema_version =
        "wafer_frontend.linked_program_manifest/v1alpha10";
    ExpectFailure([&] { finalizer.Finalize(old_linked); },
                  "real PD1 old linked manifest version");

    auto old_command = dto;
    bool changed_command = false;
    for (LinkedFragmentDto &linked : old_command.fragments) {
        CommandFragmentDto *leaf =
            std::get_if<CommandFragmentDto>(&linked);
        if (leaf && leaf->kind == FragmentKindDto::STATE_TRANSFER) {
            leaf->schema_version =
                "wafer_frontend.command_fragment/v1alpha9";
            changed_command = true;
            break;
        }
    }
    Require(changed_command, "real PD1 lacks STATE_TRANSFER leaf");
    ExpectFailure([&] { finalizer.Finalize(old_command); },
                  "real PD1 old command version");

    auto mismatched_length = dto;
    bool changed_length = false;
    for (LinkedFragmentDto &linked : mismatched_length.fragments) {
        CommandFragmentDto *leaf =
            std::get_if<CommandFragmentDto>(&linked);
        if (!leaf || leaf->kind != FragmentKindDto::STATE_TRANSFER)
            continue;
        for (frontend::RelocatableRecordDto &record :
             leaf->core_streams.front().records)
            if (record.opcode == Opcode::DTE_RECV) {
                record.operands[6].literal_value = uint64_t{16};
                changed_length = true;
                break;
            }
        if (changed_length)
            break;
    }
    Require(changed_length, "real PD1 lacks DTE_RECV");
    ExpectFailure([&] { finalizer.Finalize(mismatched_length); },
                  "real PD1 mismatched transfer length");

    auto forged_token = dto;
    bool changed_token = false;
    for (LinkedFragmentDto &linked : forged_token.fragments) {
        CommandFragmentDto *leaf =
            std::get_if<CommandFragmentDto>(&linked);
        if (!leaf || leaf->kind != FragmentKindDto::STATE_TRANSFER)
            continue;
        for (frontend::RelocatableRecordDto &record :
             leaf->core_streams.front().records)
            if (record.opcode == Opcode::DTE_WAIT) {
                record.operands[0].symbol_ref = "forged_pd1_token";
                changed_token = true;
                break;
            }
        if (changed_token)
            break;
    }
    Require(changed_token, "real PD1 lacks DTE_WAIT");
    ExpectFailure([&] { finalizer.Finalize(forged_token); },
                  "real PD1 forged transfer token");

    std::cout << "pd1_bytes=" << bytes.size()
              << " sha256=" << Sha256(bytes)
              << " cores=" << artifact.cores.size()
              << " records=" << artifact_records
              << " relocations=" << artifact.relocations.size()
              << " transfer_fragments=" << transfer_fragments << '\n';
}

void RunStage2ProducedManifest() {
    const std::string text((std::istreambuf_iterator<char>(std::cin)),
                           std::istreambuf_iterator<char>());
    const frontend::LinkedProgramManifestDto dto =
        ProgramArtifactFinalizer::Parse(text);
    const ProgramArtifactFinalizer finalizer;
    const ProgramArtifact artifact = finalizer.Finalize(dto);
    const std::vector<uint8_t> bytes = finalizer.FinalizeEncoded(text);
    Require(bytes == finalizer.FinalizeEncoded(text),
            "Stage2 TP1 finalization is not byte deterministic");
    Require(bytes == EncodeProgramArtifact(artifact) &&
                EncodeProgramArtifact(DecodeProgramArtifact(bytes)) == bytes,
            "Stage2 TP1 artifact does not exactly round-trip");

    auto leaf = [](const LinkedFragmentDto &linked)
        -> const CommandFragmentDto & {
        if (const auto *command = std::get_if<CommandFragmentDto>(&linked))
            return *command;
        return std::get<frontend::RegionManifestDto>(linked).fragment;
    };
    auto mutable_leaf = [](LinkedFragmentDto &linked)
        -> CommandFragmentDto & {
        if (auto *command = std::get_if<CommandFragmentDto>(&linked))
            return *command;
        return std::get<frontend::RegionManifestDto>(linked).fragment;
    };

    std::map<Opcode, uint64_t> opcodes;
    uint64_t raw_records = 0;
    uint64_t buffer_relocations = 0;
    uint64_t rms_data_closures = 0;
    uint64_t int32_buffers = 0;
    std::string int32_buffer_id;
    for (const LinkedFragmentDto &linked : dto.fragments) {
        const CommandFragmentDto &fragment = leaf(linked);
        for (const frontend::BufferAbiDto &abi : fragment.buffer_abi) {
            if (abi.dtype != frontend::BufferDTypeDto::INT32)
                continue;
            ++int32_buffers;
            int32_buffer_id = abi.id;
            Require(abi.value_id == "P0.token_ids" &&
                        abi.tensor_slice.shape == std::vector<uint64_t>({8}) &&
                        abi.size_bytes == 32,
                    "Stage2 TP1 INT32 BufferABI is not the exact token input");
        }
        for (const frontend::StateAbiDto &abi : fragment.state_abi)
            Require(abi.dtype != frontend::BufferDTypeDto::INT32,
                    "Stage2 TP1 StateABI unexpectedly carries INT32");
        for (const frontend::CoreFragmentStreamDto &stream :
             fragment.core_streams) {
            for (const frontend::AddressRelocationDto &relocation :
                 stream.address_relocations)
                if (relocation.operand_id != SemanticOperandId::HBM_ADDRESS)
                    ++buffer_relocations;
            for (std::size_t index = 0; index < stream.records.size(); ++index) {
                const frontend::RelocatableRecordDto &record =
                    stream.records[index];
                ++raw_records;
                ++opcodes[record.opcode];
                if (record.opcode != Opcode::RMSNORM)
                    continue;
                Require(record.operands.size() == 5 &&
                            record.operands[2].name == "data_address" &&
                            record.operands[2].kind ==
                                frontend::OperandKindDto::ADDRESS_SYMBOL &&
                            record.operands[2].operand_id ==
                                SemanticOperandId::COMPUTE_DATA_ADDRESS,
                        "Stage2 TP1 RMSNORM lacks its exact DATA operand");
                const uint64_t relocation_count = std::count_if(
                    stream.address_relocations.begin(),
                    stream.address_relocations.end(),
                    [&](const frontend::AddressRelocationDto &item) {
                        return item.record_index == index &&
                               item.operand_id ==
                                   SemanticOperandId::COMPUTE_DATA_ADDRESS &&
                               item.symbol_kind ==
                                   ProgramSymbolKind::ABSOLUTE_ADDRESS;
                    });
                const auto binding = std::find_if(
                    dto.address_operand_bindings.begin(),
                    dto.address_operand_bindings.end(),
                    [&](const AddressOperandBindingDto &item) {
                        return item.fragment_id == fragment.id &&
                               item.logical_core == stream.logical_core &&
                               item.fragment_record_index == index &&
                               item.operand_id ==
                                   SemanticOperandId::COMPUTE_DATA_ADDRESS;
                    });
                Require(relocation_count == 1 &&
                            binding != dto.address_operand_bindings.end() &&
                            binding->buffer_abi_ids.size() == 1 &&
                            binding->tensor_slices.size() == 1 &&
                            binding->tensor_slices[0].shape ==
                                std::vector<uint64_t>({16}),
                        "Stage2 TP1 RMSNORM DATA relocation/binding is not exact");
                const auto abi = std::find_if(
                    fragment.buffer_abi.begin(), fragment.buffer_abi.end(),
                    [&](const frontend::BufferAbiDto &item) {
                        return item.id == binding->buffer_abi_ids.front();
                    });
                Require(abi != fragment.buffer_abi.end() &&
                            abi->dtype == frontend::BufferDTypeDto::FP16 &&
                            abi->size_bytes == 32,
                        "Stage2 TP1 RMS scale must be one FP16[16] buffer");
                ++rms_data_closures;
            }
        }
    }
    Require(dto.fragments.size() == 44 && raw_records == 159 &&
                dto.address_operand_bindings.size() == 278 &&
                buffer_relocations == 278,
            "Stage2 TP1 manifest topology/address closure counts changed");
    Require(opcodes[Opcode::EMBEDDING_LOOKUP] == 1 &&
                opcodes[Opcode::ROPE_QK_EXACT] == 2 &&
                opcodes[Opcode::ATTENTION_EXACT] == 2 &&
                opcodes[Opcode::RMSNORM] == 5 &&
                opcodes[Opcode::MATMUL] == 9 &&
                opcodes[Opcode::GREEDY_SAMPLE] == 0 &&
                rms_data_closures == 5,
            "Stage2 TP1 exact Dense-forward opcode/RMS counts changed");
    Require(int32_buffers == 1 && !int32_buffer_id.empty(),
            "Stage2 TP1 must expose exactly one INT32 token BufferABI");
    const uint64_t int32_binding_refs = std::count_if(
        dto.address_operand_bindings.begin(),
        dto.address_operand_bindings.end(),
        [&](const AddressOperandBindingDto &binding) {
            return std::find(binding.buffer_abi_ids.begin(),
                             binding.buffer_abi_ids.end(),
                             int32_buffer_id) != binding.buffer_abi_ids.end();
        });
    Require(int32_binding_refs == 5,
            "Stage2 TP1 INT32 token buffer must close five lifecycle/compute operands");
    const auto embedding_binding = std::find_if(
        dto.address_operand_bindings.begin(),
        dto.address_operand_bindings.end(),
        [&](const AddressOperandBindingDto &binding) {
            if (binding.operand_id !=
                    SemanticOperandId::COMPUTE_INPUT_ADDRESS ||
                binding.buffer_abi_ids !=
                    std::vector<std::string>{int32_buffer_id})
                return false;
            const auto fragment = std::find_if(
                dto.fragments.begin(), dto.fragments.end(),
                [&](const LinkedFragmentDto &linked) {
                    return leaf(linked).id == binding.fragment_id;
                });
            if (fragment == dto.fragments.end())
                return false;
            const CommandFragmentDto &owner = leaf(*fragment);
            const auto stream = std::find_if(
                owner.core_streams.begin(), owner.core_streams.end(),
                [&](const frontend::CoreFragmentStreamDto &item) {
                    return item.logical_core == binding.logical_core;
                });
            return stream != owner.core_streams.end() &&
                   binding.fragment_record_index < stream->records.size() &&
                   stream->records[binding.fragment_record_index].opcode ==
                       Opcode::EMBEDDING_LOOKUP;
        });
    Require(embedding_binding != dto.address_operand_bindings.end(),
            "Stage2 TP1 embedding input is not closed by the INT32 token buffer");

    auto old_linked = dto;
    old_linked.schema_version =
        "wafer_frontend.linked_program_manifest/v1alpha10";
    ExpectFailure([&] { finalizer.Finalize(old_linked); },
                  "real Stage2 old linked version");
    auto old_command = dto;
    mutable_leaf(old_command.fragments.front()).schema_version =
        "wafer_frontend.command_fragment/v1alpha9";
    ExpectFailure([&] { finalizer.Finalize(old_command); },
                  "real Stage2 old command version");
    auto old_projection = dto;
    for (frontend::ManifestInputDigestDto &digest :
         old_projection.input_digests)
        if (digest.kind == frontend::ManifestInputKindDto::IR2_PROJECTION)
            digest.schema_version =
                "wafer_frontend.ir2_projection_result/v1alpha10";
    ExpectFailure([&] { finalizer.Finalize(old_projection); },
                  "real Stage2 old projection trust version");

    auto bad_attention = dto;
    bool changed_attention = false;
    for (LinkedFragmentDto &linked : bad_attention.fragments) {
        for (frontend::CoreFragmentStreamDto &stream :
             mutable_leaf(linked).core_streams)
            for (frontend::RelocatableRecordDto &record : stream.records)
                if (record.opcode == Opcode::ATTENTION_EXACT) {
                    uint64_t &pairs = std::get<uint64_t>(
                        record.operands[15].literal_value);
                    ++pairs;
                    changed_attention = true;
                    break;
                }
        if (changed_attention) break;
    }
    Require(changed_attention, "real Stage2 manifest lacks ATTENTION_EXACT");
    ExpectFailure([&] { finalizer.Finalize(bad_attention); },
                  "real Stage2 tampered attention formula");

    auto missing_rms_data = dto;
    bool changed_rms = false;
    for (LinkedFragmentDto &linked : missing_rms_data.fragments) {
        for (frontend::CoreFragmentStreamDto &stream :
             mutable_leaf(linked).core_streams)
            for (frontend::RelocatableRecordDto &record : stream.records)
                if (record.opcode == Opcode::RMSNORM) {
                    frontend::RecordOperandDto &data = record.operands[2];
                    data.kind = frontend::OperandKindDto::LITERAL;
                    data.literal_value = uint64_t{0};
                    data.runtime_field.reset();
                    data.operand_id.reset();
                    data.symbol_ref.reset();
                    changed_rms = true;
                    break;
                }
        if (changed_rms) break;
    }
    Require(changed_rms, "real Stage2 manifest lacks RMSNORM");
    ExpectFailure([&] { finalizer.Finalize(missing_rms_data); },
                  "real Stage2 missing RMS DATA operand");

    auto int32_state = dto;
    bool changed_state = false;
    for (LinkedFragmentDto &linked : int32_state.fragments) {
        CommandFragmentDto &fragment = mutable_leaf(linked);
        if (!fragment.state_abi.empty()) {
            fragment.state_abi.front().dtype =
                frontend::BufferDTypeDto::INT32;
            changed_state = true;
            break;
        }
    }
    Require(changed_state, "real Stage2 manifest lacks StateABI");
    ExpectFailure([&] { finalizer.Finalize(int32_state); },
                  "real Stage2 INT32 StateABI");

    uint64_t artifact_records = 0;
    for (const ProgramCore &core : artifact.cores)
        artifact_records += core.records.size();
    Require(artifact_records == 159,
            "Stage2 TP1 finalized record count changed");
    std::cout << "stage2_bytes=" << bytes.size()
              << " sha256=" << Sha256(bytes)
              << " cores=" << artifact.cores.size()
              << " records=" << artifact_records
              << " relocations=" << artifact.relocations.size() << '\n';
}

void RunStage4PdrProducedManifest() {
    const std::string text((std::istreambuf_iterator<char>(std::cin)),
                           std::istreambuf_iterator<char>());
    const frontend::LinkedProgramManifestDto dto =
        ProgramArtifactFinalizer::Parse(text);
    const ProgramArtifactFinalizer finalizer;
    const ProgramArtifact artifact = finalizer.Finalize(dto);
    const std::vector<uint8_t> bytes = finalizer.FinalizeEncoded(text);
    Require(bytes == finalizer.FinalizeEncoded(text),
            "Stage4 PDR finalization is not byte deterministic");
    Require(bytes == EncodeProgramArtifact(artifact) &&
                EncodeProgramArtifact(DecodeProgramArtifact(bytes)) == bytes,
            "Stage4 PDR artifact does not exactly round-trip");

    auto leaf = [](const LinkedFragmentDto &linked)
        -> const CommandFragmentDto & {
        if (const auto *command = std::get_if<CommandFragmentDto>(&linked))
            return *command;
        return std::get<frontend::RegionManifestDto>(linked).fragment;
    };
    auto mutable_leaf = [](LinkedFragmentDto &linked)
        -> CommandFragmentDto & {
        if (auto *command = std::get_if<CommandFragmentDto>(&linked))
            return *command;
        return std::get<frontend::RegionManifestDto>(linked).fragment;
    };

    uint64_t transfer_fragments = 0;
    uint64_t transfer_records = 0;
    std::map<Opcode, uint64_t> opcodes;
    for (const LinkedFragmentDto &linked : dto.fragments) {
        const CommandFragmentDto &fragment = leaf(linked);
        if (fragment.kind != frontend::FragmentKindDto::STATE_TRANSFER)
            continue;
        ++transfer_fragments;
        for (const frontend::CoreFragmentStreamDto &stream :
             fragment.core_streams) {
            transfer_records += stream.records.size();
            for (const frontend::RelocatableRecordDto &record :
                 stream.records)
                ++opcodes[record.opcode];
        }
    }
    Require(transfer_fragments == 16 && transfer_records == 314 &&
                opcodes[Opcode::DTE_SEND] == 64 &&
                opcodes[Opcode::DTE_RECV] == 64 &&
                opcodes[Opcode::DTE_WAIT] == 64 &&
                opcodes[Opcode::EVENT_SET] == 55 &&
                opcodes[Opcode::EVENT_WAIT] == 55,
            "Stage4 PDR state-transfer wave counts changed");

    auto first_record = [&](frontend::LinkedProgramManifestDto &candidate,
                            Opcode opcode)
        -> frontend::RelocatableRecordDto & {
        for (LinkedFragmentDto &linked : candidate.fragments) {
            CommandFragmentDto &fragment = mutable_leaf(linked);
            if (fragment.kind != frontend::FragmentKindDto::STATE_TRANSFER)
                continue;
            for (frontend::CoreFragmentStreamDto &stream :
                 fragment.core_streams)
                for (frontend::RelocatableRecordDto &record : stream.records)
                    if (record.opcode == opcode) return record;
        }
        throw std::runtime_error("missing Stage4 PDR wave record");
    };

    frontend::LinkedProgramManifestDto bad_count = dto;
    first_record(bad_count, Opcode::EVENT_WAIT).operands[3].literal_value =
        uint64_t{2};
    ExpectFailure([&] { finalizer.Finalize(bad_count); },
                  "Stage4 PDR wave count");

    frontend::LinkedProgramManifestDto bad_tag = dto;
    bool changed_tag = false;
    for (frontend::RuntimeSymbolDefinitionDto &definition :
         bad_tag.runtime_symbol_definitions) {
        if (definition.symbol.kind ==
                frontend::RuntimeSymbolKindDto::EVENT_TAG &&
            definition.symbol.source_ref.rfind(
                "state_transfer_wave_binding_", 0) == 0) {
            definition.symbol.source_ref = "forged";
            changed_tag = true;
            break;
        }
    }
    Require(changed_tag, "missing Stage4 PDR wave EVENT_TAG definition");
    ExpectFailure([&] { finalizer.Finalize(bad_tag); },
                  "Stage4 PDR wave tag stable identity");

    frontend::LinkedProgramManifestDto bad_core = dto;
    bool changed_core = false;
    for (frontend::RuntimeSymbolDefinitionDto &definition :
         bad_core.runtime_symbol_definitions) {
        if (definition.symbol.kind ==
                frontend::RuntimeSymbolKindDto::RUNTIME_CORE &&
            definition.symbol.source_ref.rfind(
                "state_transfer_wave_binding_", 0) == 0) {
            definition.symbol.source_ref = "forged";
            changed_core = true;
            break;
        }
    }
    Require(changed_core, "missing Stage4 PDR wave core definition");
    ExpectFailure([&] { finalizer.Finalize(bad_core); },
                  "Stage4 PDR wave core stable identity");

    frontend::LinkedProgramManifestDto bad_owner = dto;
    first_record(bad_owner, Opcode::EVENT_WAIT).source_global_action_id =
        "forged";
    ExpectFailure([&] { finalizer.Finalize(bad_owner); },
                  "Stage4 PDR wave owner");

    frontend::LinkedProgramManifestDto bad_source_order = dto;
    bool swapped_source = false;
    for (LinkedFragmentDto &linked : bad_source_order.fragments) {
        CommandFragmentDto &fragment = mutable_leaf(linked);
        if (fragment.kind != frontend::FragmentKindDto::STATE_TRANSFER ||
            fragment.core_streams.empty())
            continue;
        auto &records = fragment.core_streams.front().records;
        for (std::size_t index = 0; index + 1 < records.size(); ++index) {
            if (records[index].opcode == Opcode::EVENT_WAIT &&
                records[index + 1].opcode == Opcode::DTE_SEND) {
                std::swap(records[index], records[index + 1]);
                swapped_source = true;
                break;
            }
        }
        if (swapped_source) break;
    }
    Require(swapped_source, "missing Stage4 PDR source wave pair");
    ExpectFailure([&] { finalizer.Finalize(bad_source_order); },
                  "Stage4 PDR source wave order");

    frontend::LinkedProgramManifestDto bad_destination_order = dto;
    bool swapped_destination = false;
    for (LinkedFragmentDto &linked : bad_destination_order.fragments) {
        CommandFragmentDto &fragment = mutable_leaf(linked);
        if (fragment.kind != frontend::FragmentKindDto::STATE_TRANSFER ||
            fragment.core_streams.empty())
            continue;
        auto &records = fragment.core_streams.front().records;
        for (std::size_t index = 1; index < records.size(); ++index) {
            if (records[index - 1].opcode == Opcode::DTE_WAIT &&
                records[index].opcode == Opcode::EVENT_SET) {
                std::swap(records[index - 1], records[index]);
                swapped_destination = true;
                break;
            }
        }
        if (swapped_destination) break;
    }
    Require(swapped_destination, "missing Stage4 PDR destination wave pair");
    ExpectFailure([&] { finalizer.Finalize(bad_destination_order); },
                  "Stage4 PDR destination wave order");

    uint64_t artifact_records = 0;
    for (const auto &core : artifact.cores)
        artifact_records += core.records.size();
    std::cout << "stage4_pdr_bytes=" << bytes.size()
              << " sha256=" << Sha256(bytes)
              << " cores=" << artifact.cores.size()
              << " records=" << artifact_records
              << " relocations=" << artifact.relocations.size()
              << " transfer_fragments=" << transfer_fragments
              << " transfer_records=" << transfer_records << '\n';
}

void RunTrainProducedManifest() {
    const std::string text((std::istreambuf_iterator<char>(std::cin)),
                           std::istreambuf_iterator<char>());
    const frontend::LinkedProgramManifestDto dto =
        ProgramArtifactFinalizer::Parse(text);
    const ProgramArtifactFinalizer finalizer;
    const ProgramArtifact artifact = finalizer.Finalize(dto);
    const std::vector<uint8_t> bytes = EncodeProgramArtifact(artifact);
    Require(finalizer.FinalizeEncoded(text) == bytes,
            "Train manifest finalization is not deterministic");

    std::map<frontend::ManifestInputKindDto, std::set<std::string>> lineage;
    std::size_t train_inputs = 0;
    for (const frontend::ManifestInputDigestDto &digest : dto.input_digests) {
        if (digest.kind ==
            frontend::ManifestInputKindDto::TRAIN_LOWERED_PROGRAM) {
            ++train_inputs;
            Require(digest.schema_version ==
                        "wafer_frontend.train_lowered_program/v1alpha2",
                    "Train lowered-program trust version changed");
        } else if (digest.kind == frontend::ManifestInputKindDto::IR1 ||
                   digest.kind ==
                       frontend::ManifestInputKindDto::IR2_PROJECTION ||
                   digest.kind ==
                       frontend::ManifestInputKindDto::SCHEDULE_SET ||
                   digest.kind ==
                       frontend::ManifestInputKindDto::GLOBAL_ACTION_DAG) {
            lineage[digest.kind].insert(digest.artifact_id);
        }
    }
    Require(train_inputs == 1 &&
                lineage[frontend::ManifestInputKindDto::IR1].size() == 2 &&
                lineage[frontend::ManifestInputKindDto::IR2_PROJECTION]
                        .size() == 2 &&
                lineage[frontend::ManifestInputKindDto::SCHEDULE_SET]
                        .size() == 2 &&
                lineage[frontend::ManifestInputKindDto::GLOBAL_ACTION_DAG]
                        .size() == 2,
            "Train manifest must close one anchor and two exact replica lineages");

    auto leaf = [](const LinkedFragmentDto &linked)
        -> const CommandFragmentDto & {
        if (const auto *fragment =
                std::get_if<CommandFragmentDto>(&linked))
            return *fragment;
        return std::get<frontend::RegionManifestDto>(linked).fragment;
    };
    std::set<std::string> leaf_global_dags;
    std::size_t records = 0;
    for (const LinkedFragmentDto &linked : dto.fragments) {
        const CommandFragmentDto &fragment = leaf(linked);
        leaf_global_dags.insert(fragment.source_global_dag_id);
        for (const frontend::CoreFragmentStreamDto &stream :
             fragment.core_streams)
            records += stream.records.size();
    }
    Require(leaf_global_dags ==
                lineage[frontend::ManifestInputKindDto::GLOBAL_ACTION_DAG],
            "Train leaves do not exactly witness both replica GlobalAction DAGs");

    auto old_train = dto;
    for (frontend::ManifestInputDigestDto &digest :
         old_train.input_digests)
        if (digest.kind ==
            frontend::ManifestInputKindDto::TRAIN_LOWERED_PROGRAM)
            digest.schema_version =
                "wafer_frontend.train_lowered_program/v1alpha0";
    ExpectFailure([&] { finalizer.Finalize(old_train); },
                  "Train old lowered-program trust version");

    auto missing_schedule = dto;
    const auto schedule = std::find_if(
        missing_schedule.input_digests.begin(),
        missing_schedule.input_digests.end(),
        [](const frontend::ManifestInputDigestDto &digest) {
            return digest.kind ==
                frontend::ManifestInputKindDto::SCHEDULE_SET;
        });
    Require(schedule != missing_schedule.input_digests.end(),
            "Train manifest lacks schedule lineage fixture");
    missing_schedule.input_digests.erase(schedule);
    ExpectFailure([&] { finalizer.Finalize(missing_schedule); },
                  "Train unequal replica lineage cardinality");

    auto missing_leaf_dag = dto;
    const std::string removed_dag =
        leaf_global_dags.empty() ? std::string{} : *leaf_global_dags.begin();
    const std::string retained_dag =
        leaf_global_dags.size() < 2 ? std::string{} : *std::next(
            leaf_global_dags.begin());
    Require(!removed_dag.empty() && !retained_dag.empty(),
            "Train manifest lacks two GlobalAction DAG leaf fixtures");
    for (LinkedFragmentDto &linked : missing_leaf_dag.fragments) {
        CommandFragmentDto *fragment =
            std::get_if<CommandFragmentDto>(&linked);
        if (!fragment)
            fragment = &std::get<frontend::RegionManifestDto>(linked).fragment;
        if (fragment->source_global_dag_id == removed_dag)
            fragment->source_global_dag_id = retained_dag;
    }
    ExpectFailure([&] { finalizer.Finalize(missing_leaf_dag); },
                  "Train missing replica leaf DAG witness");

    std::cout << "train_bytes=" << bytes.size()
              << " cores=" << artifact.cores.size()
              << " records=" << records
              << " relocations=" << artifact.relocations.size() << '\n';
}

void RunLiteRootedArProducedManifest() {
    const std::string text((std::istreambuf_iterator<char>(std::cin)),
                           std::istreambuf_iterator<char>());
    const frontend::LinkedProgramManifestDto dto =
        ProgramArtifactFinalizer::Parse(text);
    const ProgramArtifactFinalizer finalizer;
    const ProgramArtifact artifact = finalizer.Finalize(dto);
    const std::vector<uint8_t> bytes = finalizer.FinalizeEncoded(text);
    Require(bytes == finalizer.FinalizeEncoded(text),
            "rooted-AR finalization is not byte deterministic");
    Require(bytes == EncodeProgramArtifact(artifact) &&
                EncodeProgramArtifact(DecodeProgramArtifact(bytes)) == bytes,
            "rooted-AR artifact does not exactly round-trip");

    auto leaf = [](const LinkedFragmentDto &linked)
        -> const CommandFragmentDto & {
        if (const auto *command = std::get_if<CommandFragmentDto>(&linked))
            return *command;
        return std::get<frontend::RegionManifestDto>(linked).fragment;
    };
    auto literal = [](const frontend::RecordOperandDto &operand) {
        return std::get<uint64_t>(operand.literal_value);
    };

    std::map<frontend::ManifestInputKindDto, std::set<std::string>> lineage;
    std::size_t top_inputs = 0;
    for (const frontend::ManifestInputDigestDto &digest : dto.input_digests) {
        if (digest.kind ==
            frontend::ManifestInputKindDto::S2_LITE_ROOTED_AR) {
            ++top_inputs;
            Require(digest.schema_version ==
                        "wafer_frontend.s2_lite_rooted_ar_lowered_program/v1alpha1",
                    "rooted-AR lowered-program trust version changed");
        } else if (digest.kind == frontend::ManifestInputKindDto::IR1 ||
                   digest.kind ==
                       frontend::ManifestInputKindDto::IR2_PROJECTION ||
                   digest.kind ==
                       frontend::ManifestInputKindDto::SCHEDULE_SET ||
                   digest.kind ==
                       frontend::ManifestInputKindDto::GLOBAL_ACTION_DAG) {
            lineage[digest.kind].insert(digest.artifact_id);
        }
    }
    Require(top_inputs == 1 &&
                lineage[frontend::ManifestInputKindDto::IR1].size() == 2 &&
                lineage[frontend::ManifestInputKindDto::IR2_PROJECTION]
                        .size() == 2 &&
                lineage[frontend::ManifestInputKindDto::SCHEDULE_SET]
                        .size() == 2 &&
                lineage[frontend::ManifestInputKindDto::GLOBAL_ACTION_DAG]
                        .size() == 2,
            "rooted-AR manifest must close one top anchor and two exact lineages");

    std::size_t overlay_fragments = 0;
    std::size_t overlay_records = 0;
    std::map<Opcode, std::size_t> overlay_opcodes;
    std::set<std::string> local_dags;
    const frontend::RelocatableRecordDto *reduce = nullptr;
    std::string reduce_fragment_id;
    frontend::LogicalCoreDto reduce_core;
    std::size_t reduce_record_index = 0;
    for (const LinkedFragmentDto &linked : dto.fragments) {
        const CommandFragmentDto &fragment = leaf(linked);
        if (fragment.kind != FragmentKindDto::S2_LITE_ROOTED_AR) {
            local_dags.insert(fragment.source_global_dag_id);
            continue;
        }
        ++overlay_fragments;
        Require(fragment.producer_pass == "s2_lite_rooted_ar_lowering" &&
                    fragment.source_global_dag_id == dto.source_global_dag_id &&
                    fragment.core_streams.size() == 1,
                "rooted-AR overlay kind/producer/top carrier changed");
        const auto &stream = fragment.core_streams.front();
        for (std::size_t index = 0; index < stream.records.size(); ++index) {
            const auto &record = stream.records[index];
            ++overlay_records;
            ++overlay_opcodes[record.opcode];
            if (record.opcode == Opcode::LOCAL_REDUCE) {
                Require(reduce == nullptr,
                        "rooted-AR overlay contains multiple LOCAL_REDUCE records");
                reduce = &record;
                reduce_fragment_id = fragment.id;
                reduce_core = stream.logical_core;
                reduce_record_index = index;
            }
        }
    }
    Require(dto.fragments.size() == 98 && dto.core_streams.size() == 2 &&
                dto.address_operand_bindings.size() == 628 &&
                dto.input_digests.size() == 107 &&
                overlay_fragments == 6 && overlay_records == 13 &&
                overlay_opcodes[Opcode::SRAM_ALLOC_AT] == 2 &&
                overlay_opcodes[Opcode::DTE_ISSUE] == 1 &&
                overlay_opcodes[Opcode::DTE_SEND] == 2 &&
                overlay_opcodes[Opcode::DTE_RECV] == 2 &&
                overlay_opcodes[Opcode::DTE_WAIT] == 3 &&
                overlay_opcodes[Opcode::LOCAL_REDUCE] == 1 &&
                overlay_opcodes[Opcode::SRAM_FREE] == 2 &&
                local_dags ==
                    lineage[frontend::ManifestInputKindDto::GLOBAL_ACTION_DAG],
            "rooted-AR production manifest quotient changed");
    std::size_t fragment_records = 0;
    for (const LinkedFragmentDto &linked : dto.fragments)
        for (const auto &stream : leaf(linked).core_streams)
            fragment_records += stream.records.size();
    Require(fragment_records == 351,
            "rooted-AR fragment record count changed");
    Require(reduce != nullptr && reduce->operands.size() == 11 &&
                literal(reduce->operands[0]) == 1 &&
                literal(reduce->operands[1]) == 1 &&
                literal(reduce->operands[2]) == 1 &&
                literal(reduce->operands[3]) == 1 &&
                literal(reduce->operands[4]) == 0 &&
                literal(reduce->operands[5]) == 0 &&
                literal(reduce->operands[6]) == 2 &&
                literal(reduce->operands[7]) == 512 &&
                literal(reduce->operands[8]) == 2048,
            "rooted-AR FP32 LOCAL_REDUCE fixed operands changed");

    const AddressOperandBindingDto *source_binding = nullptr;
    const AddressOperandBindingDto *destination_binding = nullptr;
    for (const AddressOperandBindingDto &binding :
         dto.address_operand_bindings) {
        if (binding.fragment_id != reduce_fragment_id ||
            !(binding.logical_core == reduce_core) ||
            binding.fragment_record_index != reduce_record_index)
            continue;
        if (binding.operand_id == SemanticOperandId::SOURCE_ADDRESS)
            source_binding = &binding;
        if (binding.operand_id == SemanticOperandId::DESTINATION_ADDRESS)
            destination_binding = &binding;
    }
    Require(source_binding != nullptr && destination_binding != nullptr &&
                source_binding->buffer_abi_ids.size() == 2 &&
                destination_binding->buffer_abi_ids.size() == 1 &&
                destination_binding->buffer_abi_ids.front() ==
                    source_binding->buffer_abi_ids.front(),
            "rooted-AR reduce source/destination BufferABI alias changed");
    std::map<std::string, const frontend::BufferAbiDto *> abis;
    for (const LinkedFragmentDto &linked : dto.fragments)
        for (const frontend::BufferAbiDto &abi : leaf(linked).buffer_abi)
            abis.emplace(abi.id, &abi);
    const frontend::BufferAbiDto &rank0 =
        *abis.at(source_binding->buffer_abi_ids[0]);
    const frontend::BufferAbiDto &rank1 =
        *abis.at(source_binding->buffer_abi_ids[1]);
    Require(rank0.dtype == frontend::BufferDTypeDto::FP32 &&
                rank1.dtype == frontend::BufferDTypeDto::FP32 &&
                rank0.size_bytes == 2048 && rank1.size_bytes == 2048 &&
                rank0.region_offset_bytes == 32768 &&
                rank1.region_offset_bytes == 34816,
            "rooted-AR FP32 rank-major scratch span/order changed");

    Json bad_top_kind = Json::parse(text);
    for (Json &digest : bad_top_kind["input_digests"])
        if (digest["kind"] == "s2_lite_rooted_ar") {
            digest["kind"] = "train_lowered_program";
            digest["schema_version"] =
                "wafer_frontend.train_lowered_program/v1alpha3";
            break;
        }
    RefreshManifestIds(bad_top_kind);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_top_kind.dump()); },
                  "restable rooted-AR top kind");

    Json missing_lineage = Json::parse(text);
    auto lineage_it = std::find_if(
        missing_lineage["input_digests"].begin(),
        missing_lineage["input_digests"].end(), [](const Json &digest) {
            return digest["kind"] == "global_action_dag";
        });
    Require(lineage_it != missing_lineage["input_digests"].end(),
            "rooted-AR fixture lacks GlobalAction lineage");
    missing_lineage["input_digests"].erase(lineage_it);
    RefreshManifestIds(missing_lineage);
    ExpectFailure([&] { finalizer.FinalizeJson(missing_lineage.dump()); },
                  "restable rooted-AR missing lineage");

    auto find_rooted_leaf = [](Json &manifest,
                               const std::function<bool(const Json &)> &match)
        -> Json & {
        for (Json &linked : manifest["fragments"]) {
            Json *candidate = &linked;
            if (linked.contains("fragment")) candidate = &linked["fragment"];
            if ((*candidate)["kind"] == "s2_lite_rooted_ar" &&
                match(*candidate))
                return *candidate;
        }
        throw std::runtime_error("missing rooted-AR leaf fixture");
    };
    Json bad_overlay_kind = Json::parse(text);
    find_rooted_leaf(bad_overlay_kind, [](const Json &) { return true; })
        ["kind"] = "moe_transfer";
    RefreshManifestIds(bad_overlay_kind);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_overlay_kind.dump()); },
                  "restable rooted-AR overlay kind");

    Json bad_sequence = Json::parse(text);
    Json &copy_leaf = find_rooted_leaf(
        bad_sequence, [](const Json &candidate) {
            const Json &records = candidate["core_streams"][0]["records"];
            return records.size() == 3 && records[0]["opcode"] == 0x89 &&
                   records[1]["opcode"] == 0x41;
        });
    std::swap(copy_leaf["core_streams"][0]["records"][1],
              copy_leaf["core_streams"][0]["records"][2]);
    for (const char *field : {"runtime_relocations", "address_relocations"})
        for (Json &relocation : copy_leaf["core_streams"][0][field]) {
            const uint64_t index = relocation["record_index"];
            if (index == 1) relocation["record_index"] = 2;
            else if (index == 2) relocation["record_index"] = 1;
        }
    RefreshManifestIds(bad_sequence);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_sequence.dump()); },
                  "restable rooted-AR record sequence");

    Json bad_reduce = Json::parse(text);
    Json &reduce_leaf = find_rooted_leaf(
        bad_reduce, [](const Json &candidate) {
            return std::any_of(
                candidate["core_streams"][0]["records"].begin(),
                candidate["core_streams"][0]["records"].end(),
                [](const Json &record) { return record["opcode"] == 0x43; });
        });
    for (Json &record : reduce_leaf["core_streams"][0]["records"])
        if (record["opcode"] == 0x43)
            record["operands"][6]["literal_value"] = 3;
    RefreshManifestIds(bad_reduce);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_reduce.dump()); },
                  "restable rooted-AR FP32 count");

    Json bad_span_order = Json::parse(text);
    std::string json_reduce_fragment;
    uint64_t json_reduce_index = 0;
    for (const Json &linked : bad_span_order["fragments"]) {
        const Json &candidate = linked.contains("fragment")
                                    ? linked["fragment"] : linked;
        if (candidate["kind"] != "s2_lite_rooted_ar") continue;
        const Json &records = candidate["core_streams"][0]["records"];
        for (std::size_t index = 0; index < records.size(); ++index)
            if (records[index]["opcode"] == 0x43) {
                json_reduce_fragment = candidate["id"];
                json_reduce_index = index;
            }
    }
    bool swapped = false;
    for (Json &binding : bad_span_order["address_operand_bindings"])
        if (binding["fragment_id"] == json_reduce_fragment &&
            binding["fragment_record_index"] == json_reduce_index &&
            binding["operand_id"] == 4) {
            std::swap(binding["buffer_abi_ids"][0],
                      binding["buffer_abi_ids"][1]);
            std::swap(binding["tensor_slices"][0],
                      binding["tensor_slices"][1]);
            swapped = true;
            break;
        }
    Require(swapped, "rooted-AR fixture lacks two-input reduce binding");
    RefreshManifestIds(bad_span_order, false);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_span_order.dump()); },
                  "restable rooted-AR FP32 span order");

    std::cout << "lite_rooted_ar_bytes=" << bytes.size()
              << " cores=" << artifact.cores.size()
              << " records=" << fragment_records
              << " relocations=" << artifact.relocations.size()
              << " fragments=" << dto.fragments.size()
              << " overlay_records=" << overlay_records << '\n';
}

void RunLiteDp4TreeArProducedManifest() {
    const std::string text((std::istreambuf_iterator<char>(std::cin)),
                           std::istreambuf_iterator<char>());
    const frontend::LinkedProgramManifestDto dto =
        ProgramArtifactFinalizer::Parse(text);
    const ProgramArtifactFinalizer finalizer;
    const ProgramArtifact artifact = finalizer.Finalize(dto);
    const std::vector<uint8_t> bytes = finalizer.FinalizeEncoded(text);
    Require(bytes == finalizer.FinalizeEncoded(text) &&
                bytes == EncodeProgramArtifact(artifact) &&
                EncodeProgramArtifact(DecodeProgramArtifact(bytes)) == bytes,
            "DP4 rooted artifact is not byte deterministic");

    auto leaf = [](const LinkedFragmentDto &linked)
        -> const CommandFragmentDto & {
        if (const auto *command = std::get_if<CommandFragmentDto>(&linked))
            return *command;
        return std::get<frontend::RegionManifestDto>(linked).fragment;
    };
    auto literal = [](const frontend::RecordOperandDto &operand) {
        return std::get<uint64_t>(operand.literal_value);
    };
    std::map<frontend::ManifestInputKindDto, std::set<std::string>> lineage;
    std::size_t top_inputs = 0;
    for (const frontend::ManifestInputDigestDto &digest : dto.input_digests) {
        if (digest.kind == frontend::ManifestInputKindDto::S2_LITE_ROOTED_AR) {
            ++top_inputs;
            Require(digest.schema_version ==
                        "wafer_frontend.s2_lite_dp4_tree_ar_lowered_program/v1alpha1",
                    "DP4 rooted top schema changed");
        } else if (digest.kind == frontend::ManifestInputKindDto::IR1 ||
                   digest.kind == frontend::ManifestInputKindDto::IR2_PROJECTION ||
                   digest.kind == frontend::ManifestInputKindDto::SCHEDULE_SET ||
                   digest.kind == frontend::ManifestInputKindDto::GLOBAL_ACTION_DAG) {
            lineage[digest.kind].insert(digest.artifact_id);
        }
    }
    Require(top_inputs == 1 &&
                lineage[frontend::ManifestInputKindDto::IR1].size() == 4 &&
                lineage[frontend::ManifestInputKindDto::IR2_PROJECTION].size() == 4 &&
                lineage[frontend::ManifestInputKindDto::SCHEDULE_SET].size() == 4 &&
                lineage[frontend::ManifestInputKindDto::GLOBAL_ACTION_DAG].size() == 4,
            "DP4 rooted manifest must close four exact lineages");

    std::size_t overlay_fragments = 0;
    std::size_t overlay_records = 0;
    std::size_t overlay_claims = 0;
    std::size_t fragment_records = 0;
    std::map<Opcode, std::size_t> overlay_opcodes;
    std::set<std::string> local_dags;
    std::size_t exact_reduce_bindings = 0;
    for (const LinkedFragmentDto &linked : dto.fragments) {
        const CommandFragmentDto &fragment = leaf(linked);
        for (const auto &stream : fragment.core_streams)
            fragment_records += stream.records.size();
        if (fragment.kind != FragmentKindDto::S2_LITE_ROOTED_AR) {
            local_dags.insert(fragment.source_global_dag_id);
            continue;
        }
        ++overlay_fragments;
        overlay_claims += fragment.claimed_action_ids.size();
        Require(fragment.producer_pass == "s2_lite_rooted_ar_lowering" &&
                    fragment.source_global_dag_id == dto.source_global_dag_id &&
                    fragment.core_streams.size() == 1,
                "DP4 rooted overlay producer/top/core contract changed");
        const auto &stream = fragment.core_streams.front();
        for (std::size_t index = 0; index < stream.records.size(); ++index) {
            const auto &record = stream.records[index];
            ++overlay_records;
            ++overlay_opcodes[record.opcode];
            if (record.opcode == Opcode::DTE_SEND)
                Require(literal(record.operands[7]) == 2048,
                        "DP4 rooted send bytes changed");
            if (record.opcode == Opcode::DTE_RECV)
                Require(literal(record.operands[6]) == 2048,
                        "DP4 rooted recv bytes changed");
            if (record.opcode != Opcode::LOCAL_REDUCE) continue;
            Require(record.operands.size() == 11 &&
                        literal(record.operands[0]) == 1 &&
                        literal(record.operands[1]) == 1 &&
                        literal(record.operands[2]) == 1 &&
                        literal(record.operands[3]) == 1 &&
                        literal(record.operands[4]) == 0 &&
                        literal(record.operands[5]) == 0 &&
                        literal(record.operands[6]) == 2 &&
                        literal(record.operands[7]) == 512 &&
                        literal(record.operands[8]) == 2048,
                    "DP4 rooted FP32 LOCAL_REDUCE operands changed");
            const AddressOperandBindingDto *source_binding = nullptr;
            const AddressOperandBindingDto *destination_binding = nullptr;
            for (const AddressOperandBindingDto &binding : dto.address_operand_bindings) {
                if (binding.fragment_id != fragment.id ||
                    !(binding.logical_core == stream.logical_core) ||
                    binding.fragment_record_index != index)
                    continue;
                if (binding.operand_id == SemanticOperandId::SOURCE_ADDRESS)
                    source_binding = &binding;
                if (binding.operand_id == SemanticOperandId::DESTINATION_ADDRESS)
                    destination_binding = &binding;
            }
            Require(source_binding != nullptr && destination_binding != nullptr &&
                        source_binding->buffer_abi_ids.size() == 2 &&
                        destination_binding->buffer_abi_ids.size() == 1 &&
                        destination_binding->buffer_abi_ids.front() ==
                            source_binding->buffer_abi_ids.front(),
                    "DP4 rooted reduce ABI alias changed");
            ++exact_reduce_bindings;
        }
    }
    Require(dto.fragments.size() == 201 && dto.core_streams.size() == 4 &&
                fragment_records == 709 && dto.input_digests.size() == 218 &&
                overlay_fragments == 17 && overlay_records == 33 &&
                overlay_claims == 23 && exact_reduce_bindings == 3 &&
                overlay_opcodes[Opcode::SRAM_ALLOC_AT] == 4 &&
                overlay_opcodes[Opcode::SRAM_FREE] == 4 &&
                overlay_opcodes[Opcode::DTE_ISSUE] == 2 &&
                overlay_opcodes[Opcode::DTE_SEND] == 6 &&
                overlay_opcodes[Opcode::DTE_RECV] == 6 &&
                overlay_opcodes[Opcode::DTE_WAIT] == 8 &&
                overlay_opcodes[Opcode::LOCAL_REDUCE] == 3 &&
                local_dags == lineage[frontend::ManifestInputKindDto::GLOBAL_ACTION_DAG],
            "DP4 rooted production manifest quotient changed");

    auto find_overlay = [](Json &manifest,
                           const std::function<bool(const Json &)> &match)
        -> Json & {
        for (Json &linked : manifest["fragments"]) {
            Json *candidate = &linked;
            if (linked.contains("fragment")) candidate = &linked["fragment"];
            if ((*candidate)["kind"] == "s2_lite_rooted_ar" && match(*candidate))
                return *candidate;
        }
        throw std::runtime_error("missing DP4 rooted overlay fixture");
    };

    Json wrong_top = Json::parse(text);
    for (Json &digest : wrong_top["input_digests"])
        if (digest["kind"] == "s2_lite_rooted_ar") {
            digest["schema_version"] =
                "wafer_frontend.s2_lite_rooted_ar_lowered_program/v1alpha1";
            break;
        }
    RefreshManifestIds(wrong_top);
    ExpectFailure([&] { finalizer.FinalizeJson(wrong_top.dump()); },
                  "restable DP4 rooted wrong top schema");

    Json missing_lineage = Json::parse(text);
    auto lineage_it = std::find_if(
        missing_lineage["input_digests"].begin(),
        missing_lineage["input_digests"].end(),
        [](const Json &digest) { return digest["kind"] == "global_action_dag"; });
    Require(lineage_it != missing_lineage["input_digests"].end(),
            "DP4 rooted fixture lacks lineage");
    missing_lineage["input_digests"].erase(lineage_it);
    RefreshManifestIds(missing_lineage);
    ExpectFailure([&] { finalizer.FinalizeJson(missing_lineage.dump()); },
                  "restable DP4 rooted missing lineage");

    Json bad_send = Json::parse(text);
    Json &send_leaf = find_overlay(bad_send, [](const Json &candidate) {
        return std::any_of(candidate["core_streams"][0]["records"].begin(),
                           candidate["core_streams"][0]["records"].end(),
                           [](const Json &record) { return record["opcode"] == 0x40; });
    });
    for (Json &record : send_leaf["core_streams"][0]["records"])
        if (record["opcode"] == 0x40) record["operands"][7]["literal_value"] = 1024;
    RefreshManifestIds(bad_send);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_send.dump()); },
                  "restable DP4 rooted send bytes");

    Json bad_reduce = Json::parse(text);
    Json &reduce_leaf = find_overlay(bad_reduce, [](const Json &candidate) {
        return std::any_of(candidate["core_streams"][0]["records"].begin(),
                           candidate["core_streams"][0]["records"].end(),
                           [](const Json &record) { return record["opcode"] == 0x43; });
    });
    for (Json &record : reduce_leaf["core_streams"][0]["records"])
        if (record["opcode"] == 0x43) record["operands"][6]["literal_value"] = 4;
    RefreshManifestIds(bad_reduce);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_reduce.dump()); },
                  "restable DP4 rooted reduce count");

    Json bad_claims = Json::parse(text);
    find_overlay(bad_claims, [](const Json &) { return true; })
        ["claimed_action_ids"] = Json::array();
    RefreshManifestIds(bad_claims);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_claims.dump()); },
                  "restable DP4 rooted claims");

    std::cout << "lite_dp4_tree_ar_bytes=" << bytes.size()
              << " cores=" << artifact.cores.size()
              << " records=" << fragment_records
              << " relocations=" << artifact.relocations.size()
              << " fragments=" << dto.fragments.size()
              << " overlay_records=" << overlay_records << '\n';
}

void RunLiteMoeBackwardProducedManifest() {
    const std::string text((std::istreambuf_iterator<char>(std::cin)),
                           std::istreambuf_iterator<char>());
    const frontend::LinkedProgramManifestDto dto =
        ProgramArtifactFinalizer::Parse(text);
    const ProgramArtifactFinalizer finalizer;
    const ProgramArtifact artifact = finalizer.Finalize(dto);
    const std::vector<uint8_t> bytes = finalizer.FinalizeEncoded(text);
    Require(bytes == finalizer.FinalizeEncoded(text) &&
                bytes == EncodeProgramArtifact(artifact) &&
                EncodeProgramArtifact(DecodeProgramArtifact(bytes)) == bytes,
            "S3-Lite backward artifact is not byte deterministic");

    auto leaf = [](const LinkedFragmentDto &linked)
        -> const CommandFragmentDto & {
        if (const auto *command = std::get_if<CommandFragmentDto>(&linked))
            return *command;
        return std::get<frontend::RegionManifestDto>(linked).fragment;
    };
    std::size_t top_inputs = 0;
    std::size_t overlay_inputs = 0;
    for (const frontend::ManifestInputDigestDto &digest : dto.input_digests) {
        if (digest.kind == frontend::ManifestInputKindDto::S3_LITE_MOE) {
            ++top_inputs;
            Require(
                digest.schema_version ==
                    "wafer_frontend.s3_lite_moe_backward_lowered_program/v1alpha1",
                "S3-Lite backward top trust version changed");
        }
        if (digest.kind ==
                frontend::ManifestInputKindDto::GLOBAL_ACTION_DAG) {
            ++overlay_inputs;
            Require(
                digest.schema_version ==
                    "wafer_frontend.s3_lite_moe_backward_overlay/v1alpha1",
                "S3-Lite backward overlay trust version changed");
        }
    }
    std::map<FragmentKindDto, std::size_t> kinds;
    std::map<Opcode, std::size_t> opcodes;
    std::map<std::string, std::size_t> state_abis;
    std::size_t records = 0;
    std::size_t claims = 0;
    for (const LinkedFragmentDto &linked : dto.fragments) {
        const CommandFragmentDto &fragment = leaf(linked);
        Require(std::holds_alternative<CommandFragmentDto>(linked) &&
                    fragment.producer_pass ==
                        "lite_moe_backward_lowering" &&
                    fragment.source_global_dag_id ==
                        dto.source_global_dag_id &&
                    fragment.core_streams.size() == 1,
                "S3-Lite backward leaf trust changed");
        ++kinds[fragment.kind];
        claims += fragment.claimed_action_ids.size();
        for (const frontend::RelocatableRecordDto &record :
             fragment.core_streams.front().records) {
            ++records;
            ++opcodes[record.opcode];
        }
        for (const frontend::StateAbiDto &state : fragment.state_abi)
            ++state_abis[state.id];
    }
    Require(
        top_inputs == 1 && overlay_inputs == 1 &&
            dto.input_digests.size() == 33 && dto.fragments.size() == 28 &&
            dto.core_streams.size() == 2 &&
            dto.runtime_symbol_definitions.size() == 18 &&
            dto.program_symbol_definitions.size() == 69 &&
            dto.address_operand_bindings.size() == 180 &&
            dto.state_operand_bindings.size() == 8 && records == 104 &&
            claims == 32 &&
            kinds == std::map<FragmentKindDto, std::size_t>{
                         {FragmentKindDto::COARSE, 12},
                         {FragmentKindDto::MOE_TRANSFER, 8},
                         {FragmentKindDto::STATE_IO, 8}} &&
            opcodes == std::map<Opcode, std::size_t>{
                           {Opcode::SRAM_ALLOC_AT, 28},
                           {Opcode::SRAM_FREE, 28},
                           {Opcode::SRAM_BIND, 12},
                           {Opcode::MATMUL, 8},
                           {Opcode::DTE_SEND, 4},
                           {Opcode::DTE_RECV, 4},
                           {Opcode::DTE_WAIT, 4},
                           {Opcode::LOCAL_REDUCE, 4},
                           {Opcode::LSU_LOAD, 4},
                           {Opcode::SGD_UPDATE, 4},
                           {Opcode::LSU_STORE, 4}} &&
            state_abis.size() == 4 &&
            std::all_of(state_abis.begin(), state_abis.end(),
                        [](const auto &entry) { return entry.second == 2; }),
        "S3-Lite backward production manifest quotient changed");

    Json old_top = Json::parse(text);
    for (Json &digest : old_top["input_digests"])
        if (digest["kind"] == "s3_lite_moe") {
            digest["schema_version"] =
                "wafer_frontend.s3_lite_moe_backward_lowered_program/v1alpha0";
            break;
        }
    RefreshManifestIds(old_top);
    ExpectFailure([&] { finalizer.FinalizeJson(old_top.dump()); },
                  "restable S3-Lite backward old top schema");

    Json bad_producer = Json::parse(text);
    bad_producer["fragments"][0]["producer_pass"] = "lowering";
    RefreshManifestIds(bad_producer);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_producer.dump()); },
                  "restable S3-Lite backward producer");

    Json bad_kind = Json::parse(text);
    auto transfer = std::find_if(
        bad_kind["fragments"].begin(), bad_kind["fragments"].end(),
        [](const Json &fragment) {
            return fragment["kind"] == "moe_transfer";
        });
    Require(transfer != bad_kind["fragments"].end(),
            "S3-Lite backward fixture lacks MOE_TRANSFER");
    (*transfer)["kind"] = "coarse";
    RefreshManifestIds(bad_kind);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_kind.dump()); },
                  "restable S3-Lite backward fragment kind");

    auto find_record = [](Json &manifest, uint64_t opcode) -> Json & {
        for (Json &fragment : manifest["fragments"])
            for (Json &record : fragment["core_streams"][0]["records"])
                if (record["opcode"] == opcode) return record;
        throw std::runtime_error(
            "S3-Lite backward fixture lacks requested opcode");
    };
    Json bad_matmul = Json::parse(text);
    find_record(bad_matmul, 0x01)["operands"][4]["literal_value"][1] = 31;
    RefreshManifestIds(bad_matmul);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_matmul.dump()); },
                  "restable S3-Lite backward MATMUL shape");

    Json bad_reduce = Json::parse(text);
    find_record(bad_reduce, 0x43)["operands"][6]["literal_value"] = 3;
    RefreshManifestIds(bad_reduce);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_reduce.dump()); },
                  "restable S3-Lite backward reduce count");

    Json bad_state = Json::parse(text);
    bool changed_state = false;
    for (Json &fragment : bad_state["fragments"])
        if (!fragment["state_abi"].empty()) {
            fragment["state_abi"][0]["access"] = "read_only";
            changed_state = true;
            break;
        }
    Require(changed_state, "S3-Lite backward fixture lacks StateABI");
    RefreshManifestIds(bad_state);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_state.dump()); },
                  "restable S3-Lite backward StateABI permission");

    std::cout << "lite_moe_backward_bytes=" << bytes.size()
              << " cores=" << artifact.cores.size()
              << " records=" << records
              << " relocations=" << artifact.relocations.size()
              << " fragments=" << dto.fragments.size() << '\n';
}

void RunLiteMoeDp4ProducedManifest(const std::string &mode) {
    const std::string text((std::istreambuf_iterator<char>(std::cin)),
                           std::istreambuf_iterator<char>());
    const frontend::LinkedProgramManifestDto dto =
        ProgramArtifactFinalizer::Parse(text);
    const ProgramArtifactFinalizer finalizer;
    const ProgramArtifact artifact = finalizer.Finalize(dto);
    const std::vector<uint8_t> bytes = finalizer.FinalizeEncoded(text);
    Require(bytes == finalizer.FinalizeEncoded(text) &&
                bytes == EncodeProgramArtifact(artifact) &&
                EncodeProgramArtifact(DecodeProgramArtifact(bytes)) == bytes,
            "DP4 MoE artifact is not byte deterministic");

    const bool infer = mode == "infer";
    const bool train_forward = mode == "train_forward";
    Require(infer || train_forward || mode == "backward",
            "unknown DP4 MoE stdin mode");
    const std::string top_schema =
        infer
            ? "wafer_frontend.s3_lite_moe_dp4_infer_lowered_program/v1alpha1"
            : train_forward
            ? "wafer_frontend.s3_lite_moe_dp4_train_forward_lowered_program/v1alpha1"
            : "wafer_frontend.s3_lite_moe_dp4_backward_lowered_program/v1alpha1";
    const std::string global_schema =
        infer
            ? "wafer_frontend.s3_lite_moe_dp4_global/v1alpha1"
            : train_forward
            ? "wafer_frontend.s3_lite_moe_dp4_train_forward/v1alpha1"
            : "wafer_frontend.s3_lite_moe_dp4_backward/v1alpha1";
    const std::size_t expected_fragments = infer ? 80 : train_forward ? 88 : 32;
    const std::size_t expected_records = infer ? 260 : train_forward ? 284 : 114;
    const std::size_t expected_claims = infer ? 92 : train_forward ? 100 : 38;
    std::size_t top_inputs = 0;
    std::size_t global_inputs = 0;
    for (const frontend::ManifestInputDigestDto &digest : dto.input_digests) {
        if (digest.kind == frontend::ManifestInputKindDto::S3_LITE_MOE) {
            ++top_inputs;
            Require(digest.schema_version == top_schema,
                    "DP4 MoE top schema changed");
        }
        if (digest.kind == frontend::ManifestInputKindDto::GLOBAL_ACTION_DAG) {
            ++global_inputs;
            Require(digest.schema_version == global_schema,
                    "DP4 MoE global schema changed");
        }
    }
    std::size_t records = 0;
    std::size_t claims = 0;
    for (const LinkedFragmentDto &linked : dto.fragments) {
        Require(std::holds_alternative<CommandFragmentDto>(linked),
                "DP4 MoE manifest contains a non-command leaf");
        const CommandFragmentDto &fragment =
            std::get<CommandFragmentDto>(linked);
        claims += fragment.claimed_action_ids.size();
        for (const auto &stream : fragment.core_streams)
            records += stream.records.size();
    }
    std::size_t persistent_allocations = 0;
    for (const auto &core : artifact.cores)
        for (const ExternalRecord &record : core.records)
            if (record.opcode == Opcode::SRAM_ALLOC_AT &&
                std::get<SramAllocAtOperands>(record.operands).lifetime ==
                    SramLifetime::PERSISTENT)
                ++persistent_allocations;
    Require(persistent_allocations == (train_forward ? 8 : 0),
            "only DP4 MoE train-forward tape allocations may persist");

    Require(top_inputs == 1 && global_inputs == 1 &&
                dto.fragments.size() == expected_fragments &&
                dto.core_streams.size() == 4 &&
                records == expected_records && claims == expected_claims,
            "DP4 MoE dedicated manifest quotient changed");

    Json old_top = Json::parse(text);
    for (Json &digest : old_top["input_digests"])
        if (digest["kind"] == "s3_lite_moe") {
            digest["schema_version"] = top_schema + ".old";
            break;
        }
    RefreshManifestIds(old_top);
    ExpectFailure([&] { finalizer.FinalizeJson(old_top.dump()); },
                  "restable DP4 MoE old top schema");

    Json wrong_global = Json::parse(text);
    for (Json &digest : wrong_global["input_digests"])
        if (digest["kind"] == "global_action_dag") {
            digest["schema_version"] =
                "wafer_frontend.s3_lite_static_moe_global/v1alpha1";
            break;
        }
    RefreshManifestIds(wrong_global);
    ExpectFailure([&] { finalizer.FinalizeJson(wrong_global.dump()); },
                  "restable DP4 MoE wrong global schema");

    Json bad_producer = Json::parse(text);
    bad_producer["fragments"][0]["producer_pass"] = "lowering";
    RefreshManifestIds(bad_producer);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_producer.dump()); },
                  "restable DP4 MoE producer");

    Json bad_kind = Json::parse(text);
    bad_kind["fragments"][0]["kind"] = "standalone_collective";
    RefreshManifestIds(bad_kind);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_kind.dump()); },
                  "restable DP4 MoE fragment kind");

    Json bad_transport = Json::parse(text);
    bool changed = false;
    for (Json &fragment : bad_transport["fragments"])
        for (Json &record : fragment["core_streams"][0]["records"])
            if (!changed && record["opcode"] == 0x40) {
                record["operands"][7]["literal_value"] = 31;
                changed = true;
            }
    Require(changed, "DP4 MoE fixture lacks DTE_SEND");
    RefreshManifestIds(bad_transport);
    ExpectFailure([&] { finalizer.FinalizeJson(bad_transport.dump()); },
                  "restable DP4 MoE transport bytes");

    std::cout << "lite_moe_dp4_" << mode << "_bytes=" << bytes.size()
              << " cores=" << artifact.cores.size()
              << " records=" << records
              << " relocations=" << artifact.relocations.size()
              << " fragments=" << dto.fragments.size() << '\n';
}

void RunSwizzleProducedManifest(bool require_scale = false) {
    const std::string text((std::istreambuf_iterator<char>(std::cin)),
                           std::istreambuf_iterator<char>());
    const frontend::LinkedProgramManifestDto dto =
        ProgramArtifactFinalizer::Parse(text);
    const ProgramArtifactFinalizer finalizer;
    const ProgramArtifact first = finalizer.Finalize(dto);
    const ProgramArtifact second = finalizer.Finalize(
        ProgramArtifactFinalizer::Parse(text));
    const std::vector<uint8_t> bytes = EncodeProgramArtifact(first);
    Require(EncodeProgramArtifact(second) == bytes &&
                finalizer.FinalizeEncoded(text) == bytes,
            "Swizzle finalization is not deterministic across two parses");
    Require(dto.producer_pass == "swizzle_standard_linker" &&
                dto.input_digests.size() == 9 &&
                dto.fragments.size() == 1 &&
                std::holds_alternative<CommandFragmentDto>(
                    dto.fragments.front()) &&
                std::get<CommandFragmentDto>(dto.fragments.front()).kind ==
                    FragmentKindDto::SWIZZLE,
            "Swizzle stdin is not a dedicated typed-wrapper manifest");
    std::size_t records = 0;
    for (const ProgramCore &core : first.cores)
        records += core.records.size();
    const bool scale = records == 180 || records == 188 ||
        records == 244 || records == 316 || records == 324 ||
        records == 444;
    const std::string pattern =
        records == 38 ? "ag_gemm" :
        records == 44 ? "gemm_rs" :
        records == 50 ? "gemm_ar" :
        records == 104 ? "meshslice_2d_os" :
        records == 180 ? "scale_ag_c8_u1" :
        records == 188 ? "scale_ag_c8_u2" :
        records == 244 ? "scale_rs_c8" :
        records == 316 ? "scale_ag_c16_u1" :
        records == 324 ? "scale_ag_c16_u2" :
        records == 444 ? "scale_rs_c16" : "";
    Require(!pattern.empty(),
            "Swizzle stdin does not match a frozen production quotient");
    Require(scale == require_scale,
            "Swizzle stdin scale class does not match the selected gate");

    Json unknown_input = Json::parse(text);
    unknown_input["input_digests"][0]["kind"] = "swizzle_unknown";
    RefreshManifestIds(unknown_input);
    ExpectFailure(
        [&] { ProgramArtifactFinalizer::Parse(unknown_input.dump()); },
        "restable unknown Swizzle ManifestInputKind");

    Json wrong_schema = Json::parse(text);
    bool changed_schema = false;
    for (Json &digest : wrong_schema["input_digests"])
        if (digest["kind"] == "swizzle_operand_abi") {
            digest["schema_version"] =
                "wafer_frontend.swizzle_operand_abi/v1alpha0";
            changed_schema = true;
        }
    Require(changed_schema, "Swizzle stdin lacks operand ABI digest");
    RefreshManifestIds(wrong_schema);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(wrong_schema.dump()); },
        "restable Swizzle typed input schema");

    Json bad_producer = Json::parse(text);
    bad_producer["fragments"][0]["producer_pass"] = "lowering";
    RefreshManifestIds(bad_producer);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(bad_producer.dump()); },
        "restable Swizzle fragment producer");

    Json bad_ownership = Json::parse(text);
    bool changed_ownership = false;
    for (Json &abi : bad_ownership["fragments"][0]["buffer_abi"])
        if (abi["ownership"] == "borrowed") {
            abi["ownership"] = "owned";
            changed_ownership = true;
            break;
        }
    Require(changed_ownership,
            "Swizzle stdin lacks a borrowed BufferABI");
    RefreshManifestIds(bad_ownership);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(bad_ownership.dump()); },
        "restable Swizzle BufferABI ownership");

    Json bad_fsm = Json::parse(text);
    bool changed_fsm = false;
    for (Json &definition :
         bad_fsm["runtime_symbol_definitions"])
        if (definition["symbol"]["kind"] == "dte_fsm") {
            definition["source_action_id"] =
                "restabled_wrong_swizzle_action";
            changed_fsm = true;
            break;
        }
    Require(changed_fsm, "Swizzle stdin lacks a DTE_FSM");
    RefreshManifestIds(bad_fsm);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(bad_fsm.dump()); },
        "restable Swizzle DTE_FSM endpoint");

    Json bad_terminal = Json::parse(text);
    const std::size_t expected_cores =
        scale || pattern == "meshslice_2d_os" ? 4 : 2;
    Require(bad_terminal["envelope"]["terminal_cores"].size() ==
                expected_cores &&
                bad_terminal["envelope"]["expected_done_cores"].size() ==
                expected_cores,
            "Swizzle stdin lacks its exact terminal cores");
    bad_terminal["envelope"]["terminal_cores"].erase(expected_cores - 1);
    bad_terminal["envelope"]["expected_done_cores"].erase(
        expected_cores - 1);
    RefreshManifestIds(bad_terminal);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(bad_terminal.dump()); },
        "restable Swizzle terminal envelope");

    if (pattern == "meshslice_2d_os" || scale) {
        Json bad_quotient = Json::parse(text);
        bool changed_wait = false;
        for (Json &stream :
             bad_quotient["fragments"][0]["core_streams"])
            for (Json &record : stream["records"])
                if (!changed_wait && record["opcode"] == 0xc0) {
                    record["opcode"] = 0xc2;
                    changed_wait = true;
                }
        Require(changed_wait, "MeshSlice stdin lacks DTE_WAIT");
        RefreshManifestIds(bad_quotient);
        ExpectFailure(
            [&] { finalizer.FinalizeJson(bad_quotient.dump()); },
            "restable MeshSlice opcode quotient");
    }

    if (scale) {
        Json bad_literal = Json::parse(text);
        bool changed_literal = false;
        for (Json &stream :
             bad_literal["fragments"][0]["core_streams"])
            for (Json &record : stream["records"])
                if (!changed_literal && record["opcode"] == 0x01) {
                    Json &parameters =
                        record["operands"].back()["literal_value"];
                    parameters[1] =
                        parameters[1].get<std::uint64_t>() + 1;
                    changed_literal = true;
                }
        Require(changed_literal, "scale Swizzle stdin lacks MATMUL");
        RefreshManifestIds(bad_literal);
        ExpectFailure(
            [&] { finalizer.FinalizeJson(bad_literal.dump()); },
            "restable scale Swizzle literal quotient");

        Json bad_alias = Json::parse(text);
        bool changed_alias = false;
        for (Json &abi : bad_alias["fragments"][0]["buffer_abi"])
            if (!changed_alias &&
                abi["layout"] ==
                    "swizzle_standard_terminal_subview/v1") {
                abi["layout"] = "swizzle_standard_storage_subview/v1";
                changed_alias = true;
            }
        Require(changed_alias,
                "scale Swizzle stdin lacks a terminal subview");
        RefreshManifestIds(bad_alias);
        ExpectFailure(
            [&] { finalizer.FinalizeJson(bad_alias.dump()); },
            "restable scale Swizzle terminal alias");
    }

    if (pattern == "gemm_rs" || pattern == "gemm_ar" ||
        pattern.find("scale_rs_") == 0) {
        Json bad_reduce_order = Json::parse(text);
        bool changed_reduce = false;
        for (Json &binding :
             bad_reduce_order["address_operand_bindings"])
            if (binding["buffer_abi_ids"].size() == 2) {
                std::swap(binding["buffer_abi_ids"][0],
                          binding["buffer_abi_ids"][1]);
                std::swap(binding["tensor_slices"][0],
                          binding["tensor_slices"][1]);
                changed_reduce = true;
                break;
            }
        Require(changed_reduce,
                "Swizzle RS/AR stdin lacks ordered reduce inputs");
        RefreshManifestIds(bad_reduce_order, false);
        ExpectFailure(
            [&] { finalizer.FinalizeJson(bad_reduce_order.dump()); },
            "restable Swizzle ordered reduce span");
    }

    std::cout << "swizzle_" << pattern
              << "_bytes=" << bytes.size()
              << " sha256=" << Sha256(bytes)
              << " cores=" << first.cores.size()
              << " records=" << records
              << " relocations=" << first.relocations.size() << '\n';
}

void RunUnfusedComparisonProducedManifest(bool require_scale = false) {
    const std::string text((std::istreambuf_iterator<char>(std::cin)),
                           std::istreambuf_iterator<char>());
    const frontend::LinkedProgramManifestDto dto =
        ProgramArtifactFinalizer::Parse(text);
    const ProgramArtifactFinalizer finalizer;
    const ProgramArtifact first = finalizer.Finalize(dto);
    const ProgramArtifact second = finalizer.Finalize(
        ProgramArtifactFinalizer::Parse(text));
    const std::vector<uint8_t> bytes = EncodeProgramArtifact(first);
    Require(EncodeProgramArtifact(second) == bytes &&
                finalizer.FinalizeEncoded(text) == bytes,
            "UNFUSED comparison finalization is not deterministic across two parses");
    Require(dto.producer_pass ==
                "unfused_comparison_standard_linker" &&
                dto.input_digests.size() == 8 &&
                dto.fragments.size() == 1 &&
                std::holds_alternative<CommandFragmentDto>(
                    dto.fragments.front()) &&
                std::get<CommandFragmentDto>(dto.fragments.front()).kind ==
                    FragmentKindDto::UNFUSED_COMPARISON &&
                std::get<CommandFragmentDto>(dto.fragments.front()).kind !=
                    FragmentKindDto::SWIZZLE,
            "UNFUSED stdin is not its dedicated typed-wrapper manifest");
    std::size_t records = 0;
    for (const ProgramCore &core : first.cores)
        records += core.records.size();
    const bool scale = records == 68 || records == 104;
    const std::string pattern =
        records == 22 ? "ag_gemm" :
        records == 36 ? "gemm_rs" :
        records == 42 ? "gemm_ar" :
        records == 68 ? "scale_ag" :
        records == 104 ? "scale_rs" : "";
    Require(!pattern.empty(),
            "UNFUSED stdin does not match a frozen production quotient");
    Require(scale == require_scale,
            "UNFUSED stdin scale class does not match the selected gate");

    Json unknown_input = Json::parse(text);
    unknown_input["input_digests"][0]["kind"] =
        "unfused_comparison_unknown";
    RefreshManifestIds(unknown_input);
    ExpectFailure(
        [&] { ProgramArtifactFinalizer::Parse(unknown_input.dump()); },
        "restable unknown UNFUSED ManifestInputKind");

    Json wrong_schema = Json::parse(text);
    bool changed_schema = false;
    for (Json &digest : wrong_schema["input_digests"])
        if (digest["kind"] == "unfused_comparison_operand_abi") {
            digest["schema_version"] =
                "wafer_frontend.unfused_comparison_operand_abi/v1alpha0";
            changed_schema = true;
        }
    Require(changed_schema, "UNFUSED stdin lacks operand ABI digest");
    RefreshManifestIds(wrong_schema);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(wrong_schema.dump()); },
        "restable UNFUSED typed input schema");

    Json bad_producer = Json::parse(text);
    bad_producer["fragments"][0]["producer_pass"] = "lowering";
    RefreshManifestIds(bad_producer);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(bad_producer.dump()); },
        "restable UNFUSED fragment producer");

    Json bad_ownership = Json::parse(text);
    bool changed_ownership = false;
    for (Json &abi : bad_ownership["fragments"][0]["buffer_abi"])
        if (abi["ownership"] == "borrowed") {
            abi["ownership"] = "owned";
            changed_ownership = true;
            break;
        }
    Require(changed_ownership,
            "UNFUSED stdin lacks a borrowed BufferABI");
    RefreshManifestIds(bad_ownership);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(bad_ownership.dump()); },
        "restable UNFUSED BufferABI ownership");

    Json bad_fsm = Json::parse(text);
    bool changed_fsm = false;
    for (Json &definition : bad_fsm["runtime_symbol_definitions"])
        if (definition["symbol"]["kind"] == "dte_fsm") {
            definition["source_action_id"] =
                "restabled_wrong_unfused_action";
            changed_fsm = true;
            break;
        }
    Require(changed_fsm, "UNFUSED stdin lacks a DTE_FSM");
    RefreshManifestIds(bad_fsm);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(bad_fsm.dump()); },
        "restable UNFUSED DTE_FSM endpoint");

    Json bad_terminal = Json::parse(text);
    const std::size_t expected_cores = scale ? 4 : 2;
    Require(bad_terminal["envelope"]["terminal_cores"].size() ==
                expected_cores &&
                bad_terminal["envelope"]["expected_done_cores"].size() ==
                expected_cores,
            "UNFUSED stdin lacks its exact terminal cores");
    bad_terminal["envelope"]["terminal_cores"].erase(expected_cores - 1);
    bad_terminal["envelope"]["expected_done_cores"].erase(
        expected_cores - 1);
    RefreshManifestIds(bad_terminal);
    ExpectFailure(
        [&] { finalizer.FinalizeJson(bad_terminal.dump()); },
        "restable UNFUSED terminal envelope");

    if (pattern == "ag_gemm" || pattern == "gemm_rs") {
        const auto rewrite_address_slices =
            [](Json &manifest,
               const std::map<std::string, Json> &slices) {
                for (Json &binding :
                     manifest["address_operand_bindings"]) {
                    for (std::size_t index = 0;
                         index < binding["buffer_abi_ids"].size();
                         ++index) {
                        const std::string abi_id =
                            binding["buffer_abi_ids"][index]
                                .get<std::string>();
                        const auto found = slices.find(abi_id);
                        if (found != slices.end())
                            binding["tensor_slices"][index] =
                                found->second;
                    }
                }
            };
        const auto terminal_root =
            [](const Json &abi) {
                return abi["ownership"] == "owned" &&
                    abi["alias_of"].is_null() &&
                    abi["layout"] ==
                        "unfused_comparison_storage/v1" &&
                    abi["tensor_slice"]["shape"].size() == 2;
            };

        Json old_replicated_output = Json::parse(text);
        std::set<std::string> terminal_bindings;
        const Json full_shape =
            pattern == "ag_gemm"
                ? Json::array({8, 48})
                : Json::array({8, 16});
        std::map<std::string, Json> replicated_slices;
        std::size_t replicated_roots = 0;
        for (Json &abi :
             old_replicated_output["fragments"][0]["buffer_abi"]) {
            if (!terminal_root(abi))
                continue;
            ++replicated_roots;
            terminal_bindings.insert(
                abi["binding_id"].get<std::string>());
            abi["tensor_slice"]["offset"] =
                Json::array({0, 0});
            abi["tensor_slice"]["shape"] = full_shape;
            replicated_slices.emplace(
                abi["id"].get<std::string>(),
                abi["tensor_slice"]);
        }
        for (Json &abi :
             old_replicated_output["fragments"][0]["buffer_abi"]) {
            if (abi["alias_of"].is_null() ||
                terminal_bindings.count(
                    abi["alias_of"].get<std::string>()) == 0)
                continue;
            abi["tensor_slice"]["offset"] =
                Json::array({0, 0});
            abi["tensor_slice"]["shape"] = full_shape;
            replicated_slices.emplace(
                abi["id"].get<std::string>(),
                abi["tensor_slice"]);
        }
        Require(replicated_roots == 2,
                "UNFUSED S0 stdin lacks two rank-local terminal roots");
        rewrite_address_slices(
            old_replicated_output, replicated_slices);
        RefreshManifestIds(old_replicated_output);
        ExpectFailure(
            [&] {
                finalizer.FinalizeJson(
                    old_replicated_output.dump());
            },
            "restable UNFUSED old full-output-per-rank terminal");

        Json wrong_rank_slice = Json::parse(text);
        std::map<std::string, Json> wrong_slices;
        std::size_t changed_rank_slices = 0;
        for (Json &abi :
             wrong_rank_slice["fragments"][0]["buffer_abi"]) {
            if (!terminal_root(abi))
                continue;
            abi["tensor_slice"]["offset"] =
                Json::array({0, 0});
            wrong_slices.emplace(
                abi["id"].get<std::string>(),
                abi["tensor_slice"]);
            ++changed_rank_slices;
        }
        Require(changed_rank_slices == 2,
                "UNFUSED S0 stdin lacks exact rank terminal slices");
        rewrite_address_slices(wrong_rank_slice, wrong_slices);
        RefreshManifestIds(wrong_rank_slice);
        ExpectFailure(
            [&] {
                finalizer.FinalizeJson(wrong_rank_slice.dump());
            },
            "restable UNFUSED duplicate rank terminal slice");
    }

    if (scale) {
        Json bad_quotient = Json::parse(text);
        bool changed_wait = false;
        for (Json &stream :
             bad_quotient["fragments"][0]["core_streams"])
            for (Json &record : stream["records"])
                if (!changed_wait && record["opcode"] == 0xc0) {
                    record["opcode"] = 0xc2;
                    changed_wait = true;
                }
        Require(changed_wait, "scale UNFUSED stdin lacks DTE_WAIT");
        RefreshManifestIds(bad_quotient);
        ExpectFailure(
            [&] { finalizer.FinalizeJson(bad_quotient.dump()); },
            "restable scale UNFUSED opcode quotient");

        Json bad_literal = Json::parse(text);
        bool changed_literal = false;
        for (Json &stream :
             bad_literal["fragments"][0]["core_streams"])
            for (Json &record : stream["records"])
                if (!changed_literal && record["opcode"] == 0x01) {
                    Json &parameters =
                        record["operands"].back()["literal_value"];
                    parameters[1] =
                        parameters[1].get<std::uint64_t>() + 1;
                    changed_literal = true;
                }
        Require(changed_literal, "scale UNFUSED stdin lacks MATMUL");
        RefreshManifestIds(bad_literal);
        ExpectFailure(
            [&] { finalizer.FinalizeJson(bad_literal.dump()); },
            "restable scale UNFUSED literal quotient");
    }

    if (pattern != "ag_gemm" && pattern != "scale_ag") {
        Json bad_reduce_order = Json::parse(text);
        bool changed_reduce = false;
        for (Json &binding :
             bad_reduce_order["address_operand_bindings"])
            if (binding["buffer_abi_ids"].size() == 2) {
                std::swap(binding["buffer_abi_ids"][0],
                          binding["buffer_abi_ids"][1]);
                std::swap(binding["tensor_slices"][0],
                          binding["tensor_slices"][1]);
                changed_reduce = true;
                break;
            }
        Require(changed_reduce,
                "UNFUSED RS/AR stdin lacks ordered reduce inputs");
        RefreshManifestIds(bad_reduce_order, false);
        ExpectFailure(
            [&] { finalizer.FinalizeJson(bad_reduce_order.dump()); },
            "restable UNFUSED ordered reduce span");
    }

    std::cout << "unfused_comparison_" << pattern
              << "_bytes=" << bytes.size()
              << " sha256=" << Sha256(bytes)
              << " cores=" << first.cores.size()
              << " records=" << records
              << " relocations=" << first.relocations.size() << '\n';
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 1) {
            Run();
        } else if (argc == 2 && std::string(argv[1]) == "--stdin") {
            RunPythonProducedManifest();
        } else if (argc == 2 &&
                   std::string(argv[1]) == "--pd1-stdin") {
            RunPd1ProducedManifest();
        } else if (argc == 2 &&
                   std::string(argv[1]) == "--stage2-stdin") {
            RunStage2ProducedManifest();
        } else if (argc == 2 &&
                   std::string(argv[1]) == "--stage4-pdr-stdin") {
            RunStage4PdrProducedManifest();
        } else if (argc == 2 &&
                   std::string(argv[1]) == "--train-stdin") {
            RunTrainProducedManifest();
        } else if (argc == 2 &&
                   std::string(argv[1]) == "--lite-rooted-ar-stdin") {
            RunLiteRootedArProducedManifest();
        } else if (argc == 2 &&
                   std::string(argv[1]) == "--lite-dp4-tree-ar-stdin") {
            RunLiteDp4TreeArProducedManifest();
        } else if (argc == 2 &&
                   std::string(argv[1]) ==
                       "--lite-moe-backward-stdin") {
            RunLiteMoeBackwardProducedManifest();
        } else if (argc == 2 &&
                   std::string(argv[1]) ==
                       "--lite-moe-dp4-infer-stdin") {
            RunLiteMoeDp4ProducedManifest("infer");
        } else if (argc == 2 &&
                   std::string(argv[1]) ==
                       "--lite-moe-dp4-train-forward-stdin") {
            RunLiteMoeDp4ProducedManifest("train_forward");
        } else if (argc == 2 &&
                   std::string(argv[1]) ==
                       "--lite-moe-dp4-backward-stdin") {
            RunLiteMoeDp4ProducedManifest("backward");
        } else if (argc == 2 &&
                   std::string(argv[1]) == "--swizzle-stdin") {
            RunSwizzleProducedManifest(false);
        } else if (argc == 2 &&
                   std::string(argv[1]) == "--swizzle-scale-stdin") {
            RunSwizzleProducedManifest(true);
        } else if (argc == 2 &&
                   std::string(argv[1]) ==
                       "--unfused-comparison-stdin") {
            RunUnfusedComparisonProducedManifest(false);
        } else if (argc == 2 &&
                   std::string(argv[1]) == "--unfused-scale-stdin") {
            RunUnfusedComparisonProducedManifest(true);
        } else {
            throw std::runtime_error(
                "usage: program_finalizer_selftest "
                "[--stdin|--pd1-stdin|--stage2-stdin|--stage4-pdr-stdin|--train-stdin|--lite-rooted-ar-stdin|--lite-dp4-tree-ar-stdin|--lite-moe-backward-stdin|--lite-moe-dp4-infer-stdin|--lite-moe-dp4-train-forward-stdin|--lite-moe-dp4-backward-stdin|--swizzle-stdin|--swizzle-scale-stdin|--unfused-comparison-stdin|--unfused-scale-stdin]");
        }
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "program finalizer self-test failed: " << error.what()
                  << '\n';
        return EXIT_FAILURE;
    }
}
