#include "memory/moe_inference_mid_program_pager.h"

#include "frontend/program_finalizer.h"
#include "frontend/program_io.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace external_memory {
namespace {
using Json = nlohmann::json;
constexpr uint64_t kHome1 = uint64_t{1} << 30;

[[noreturn]] void Fail(const std::string &detail) {
    throw std::invalid_argument("MoE inference paged runtime: " + detail);
}

void Exact(const Json &value, std::initializer_list<const char *> fields) {
    if (!value.is_object() || value.size() != fields.size())
        Fail("JSON object shape drifted");
    for (const char *field : fields)
        if (!value.contains(field)) Fail(std::string("missing field ") + field);
}

std::string String(const Json &object, const char *field) {
    if (!object.at(field).is_string()) Fail(std::string(field) + " must be string");
    const auto value = object.at(field).get<std::string>();
    if (value.empty()) Fail(std::string(field) + " must not be empty");
    return value;
}

uint64_t Number(const Json &object, const char *field) {
    const auto &value = object.at(field);
    if (!value.is_number_unsigned() &&
        !(value.is_number_integer() && value.get<int64_t>() >= 0))
        Fail(std::string(field) + " must be nonnegative uint64");
    return value.get<uint64_t>();
}

Json ReadJson(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) Fail("cannot open " + path.string());
    Json result;
    input >> result;
    if (!input.eof() && input.peek() != EOF)
        Fail("trailing JSON after " + path.string());
    return result;
}

std::vector<uint8_t> Hex(const std::string &value) {
    if (value.empty() || value.size() % 2) Fail("seed hex length invalid");
    const auto digit = [](char ch) -> uint8_t {
        if (ch >= '0' && ch <= '9') return static_cast<uint8_t>(ch - '0');
        if (ch >= 'a' && ch <= 'f') return static_cast<uint8_t>(ch - 'a' + 10);
        Fail("seed must be lowercase hexadecimal");
    };
    std::vector<uint8_t> result;
    result.reserve(value.size() / 2);
    for (size_t index = 0; index < value.size(); index += 2)
        result.push_back(static_cast<uint8_t>(
            digit(value[index]) * 16 + digit(value[index + 1])));
    return result;
}

std::string SemanticId(Json value) {
    value.erase("id");
    return "moe_inference_paged_runtime_" +
        frontend::program_io::Sha256Hex(value.dump()).substr(0, 20);
}

FabricConfig ParseTwoDieFabric(const Json &value,
                               std::string &external_ref,
                               std::map<uint64_t, std::string> &connections,
                               std::map<uint64_t, std::string> &hbm_refs) {
    Exact(value, {"schema_version", "producer_pass", "id",
                  "external_capacities", "hbm_capacities", "links",
                  "connections"});
    if (String(value, "schema_version") !=
            "wafer_frontend.external_memory_fabric/v1alpha2" ||
        String(value, "producer_pass") != "external_memory_fabric_builder" ||
        !value.at("external_capacities").is_array() ||
        value.at("external_capacities").size() != 1 ||
        !value.at("hbm_capacities").is_array() ||
        value.at("hbm_capacities").size() != 2 ||
        !value.at("links").is_array() || value.at("links").size() != 1 ||
        !value.at("connections").is_array() ||
        value.at("connections").size() != 2)
        Fail("requires official shared one-link/two-die fabric");
    FabricConfig fabric;
    const auto &external = value.at("external_capacities")[0];
    Exact(external, {"id", "tier", "location_ref", "base_address",
                     "capacity_bytes", "alignment_bytes"});
    if (String(external, "tier") != "external" ||
        String(external, "location_ref") != "host:0" ||
        Number(external, "base_address") != 0 ||
        Number(external, "capacity_bytes") != 2304 ||
        Number(external, "alignment_bytes") != 16)
        Fail("host effective capacity/base changed");
    external_ref = String(external, "id");
    fabric.external_capacities.push_back({external_ref, "host:0", 0, 2304});
    for (const auto &raw : value.at("hbm_capacities")) {
        Exact(raw, {"id", "tier", "location_ref", "base_address",
                    "capacity_bytes", "alignment_bytes"});
        const auto location = String(raw, "location_ref");
        const uint64_t die = location == "die:0" ? 0 : location == "die:1" ? 1 : 2;
        if (die > 1 || String(raw, "tier") != "hbm" ||
            Number(raw, "base_address") != (die ? kHome1 : 0) ||
            Number(raw, "capacity_bytes") != 1024 ||
            Number(raw, "alignment_bytes") != 16 ||
            !hbm_refs.emplace(die, String(raw, "id")).second)
            Fail("physical per-die 1024B HBM home changed");
        fabric.hbm_capacities.push_back(
            {String(raw, "id"), die, die ? kHome1 : 0, 1024});
    }
    if (hbm_refs.size() != 2) Fail("both physical HBM homes required");
    const auto &link = value.at("links")[0];
    Exact(link, {"id", "external_capacity_ref", "ingress_die_id",
                 "bytes_per_cycle", "latency_cycles", "queue_depth",
                 "max_outstanding", "duplex"});
    if (String(link, "external_capacity_ref") != external_ref ||
        Number(link, "ingress_die_id") != 0 ||
        Number(link, "bytes_per_cycle") != 256 ||
        Number(link, "latency_cycles") != 2 ||
        Number(link, "queue_depth") != 2 ||
        Number(link, "max_outstanding") != 2 ||
        String(link, "duplex") != "half_duplex_shared")
        Fail("one shared external link service shape changed");
    const auto link_ref = String(link, "id");
    fabric.links.push_back({link_ref, external_ref, 0, 256, 2, 2, 2});
    for (const auto &raw : value.at("connections")) {
        Exact(raw, {"id", "link_ref", "hbm_capacity_ref",
                    "target_die_id", "route_die_ids",
                    "route_latency_cycles", "route_bytes_per_cycle"});
        const uint64_t die = Number(raw, "target_die_id");
        if (die > 1 || String(raw, "link_ref") != link_ref ||
            String(raw, "hbm_capacity_ref") != hbm_refs.at(die) ||
            !connections.emplace(die, String(raw, "id")).second)
            Fail("two die HBM connections do not share source link");
        if (!raw.at("route_die_ids").is_array() ||
            raw.at("route_die_ids") !=
                (die ? Json::array({0, 1}) : Json::array({0})) ||
            Number(raw, "route_latency_cycles") != die ||
            (die ? !raw.at("route_bytes_per_cycle").is_number_integer() ||
                   Number(raw, "route_bytes_per_cycle") != 256
                 : !raw.at("route_bytes_per_cycle").is_null()))
            Fail("official die0/die1 route service shape changed");
        fabric.connections.push_back({String(raw, "id"), link_ref,
                                      hbm_refs.at(die), die,
                                      die ? std::vector<uint64_t>{0, 1}
                                          : std::vector<uint64_t>{0},
                                      die, die ? std::optional<uint64_t>{256}
                                               : std::nullopt});
    }
    ValidateFabricConfig(fabric);
    return fabric;
}

uint64_t RecordBytes(const frontend::RelocatableRecordDto &record) {
    uint64_t found = 0;
    for (const auto &operand : record.operands)
        if (operand.name == "size_bytes") {
            const auto *literal = std::get_if<uint64_t>(&operand.literal_value);
            if (!literal || found) Fail("real LSU size literal missing/duplicated");
            found = *literal;
        }
    if (!found) Fail("real LSU has no positive size literal");
    return found;
}

int64_t HbmAddend(const frontend::CoreFragmentStreamDto &stream,
                  uint64_t record_index) {
    bool found = false;
    int64_t result = 0;
    for (const auto &relocation : stream.address_relocations)
        if (relocation.record_index == record_index &&
            relocation.operand_id == SemanticOperandId::HBM_ADDRESS) {
            if (found) Fail("real LSU has duplicate HBM relocation");
            found = true;
            result = relocation.addend;
        }
    if (!found) Fail("real LSU has no HBM relocation");
    return result;
}
} // namespace

MoeInferenceMidProgramPager::MoeInferenceMidProgramPager(
    const sc_core::sc_module_name &name, const std::filesystem::path &sidecar,
    std::vector<std::string> manifest_texts,
    std::map<std::pair<uint64_t, uint64_t>, HBMBackend *> backends,
    sc_core::sc_time cycle_time)
    : cycle_time_(cycle_time) {
    const Json contract = ReadJson(sidecar);
    Exact(contract, {"schema_version", "producer_pass", "id",
                     "request_digest", "model_digest", "logical_graph_digest",
                     "offload_memory_plan_digest",
                     "source_external_parameter_allocation_refs",
                     "source_rank0_kv_inventory_digest", "linked_manifest_ids",
                     "linked_manifest_digests", "hbm_capacity_bytes_per_die",
                     "workspace_end_bytes_per_die", "physical_parameter_bytes",
                     "physical_kv_bytes", "highest_relative_state_end_bytes",
                     "fabric", "parameter_spans", "kv_spans", "events", "seeds"});
    if (String(contract, "schema_version") !=
            "wafer_frontend.moe_inference_paged_runtime/v1alpha1" ||
        String(contract, "producer_pass") != "moe_inference_paged_runtime" ||
        String(contract, "id") != SemanticId(contract) ||
        Number(contract, "hbm_capacity_bytes_per_die") != 1024 ||
        Number(contract, "workspace_end_bytes_per_die") != 464 ||
        Number(contract, "physical_parameter_bytes") != 1384 ||
        Number(contract, "physical_kv_bytes") != 256 ||
        Number(contract, "highest_relative_state_end_bytes") != 960 ||
        manifest_texts.size() != 3 || backends.size() != 2 ||
        !backends.count({0, 0}) || !backends.count({1, 0}) ||
        !backends.at({0, 0}) || !backends.at({1, 0}))
        Fail("source-signed fixed full MoE two-die identity changed");
    source_ref_ = String(contract, "id");
    std::map<uint64_t, std::string> hbm_refs;
    const FabricConfig fabric = ParseTwoDieFabric(
        contract.at("fabric"), external_capacity_ref_, connection_by_die_,
        hbm_refs);
    const auto &raw_parameters = contract.at("parameter_spans");
    const auto &raw_kv = contract.at("kv_spans");
    const auto &raw_events = contract.at("events");
    const auto &raw_seeds = contract.at("seeds");
    if (!raw_parameters.is_array() || raw_parameters.size() != 19 ||
        !raw_kv.is_array() || raw_kv.size() != 4 ||
        !raw_events.is_array() || raw_events.size() != 89 ||
        !raw_seeds.is_array() || raw_seeds.size() != 23)
        Fail("19 physical weights/four KV pages/89 LSU gates required");
    std::set<std::pair<uint64_t, uint64_t>> sources;
    std::set<std::pair<uint64_t, uint64_t>> external_ranges;
    uint64_t effective_parameter_bytes = 0;
    for (const auto &raw : raw_parameters) {
        Exact(raw, {"die_id", "kind", "source_state_ref",
                    "source_hbm_address", "external_address",
                    "hbm_address", "size_bytes"});
        MoeInferencePagerSpan span{
            Number(raw, "die_id"), String(raw, "kind"),
            String(raw, "source_state_ref"), Number(raw, "source_hbm_address"),
            Number(raw, "external_address"), Number(raw, "hbm_address"),
            Number(raw, "size_bytes")};
        if (span.die_id > 1 || span.size_bytes == 0 ||
            span.size_bytes > 192 || span.hbm_address !=
                (span.die_id ? kHome1 : 0) + 512 ||
            (span.kind != "shared_or_router" &&
             span.kind != "expert_retention") ||
            !sources.emplace(span.die_id, span.source_hbm_address).second ||
            !external_ranges.emplace(span.external_address,
                                     span.external_address + span.size_bytes).second)
            Fail("true physical parameter page inventory changed");
        if (span.kind == "expert_retention" && span.size_bytes != 192)
            Fail("expert real weight page must be 192B");
        effective_parameter_bytes += span.size_bytes;
        parameters_.push_back(std::move(span));
    }
    if (effective_parameter_bytes != 1384)
        Fail("physical parameter payload is not 1384B exact");
    std::sort(parameters_.begin(), parameters_.end(),
              [](const auto &a, const auto &b) {
                  return a.external_address < b.external_address;
              });
    const std::array<std::pair<uint64_t, uint64_t>, 4> groups{{
        {0, 384}, {384, 968}, {976, 1360}, {1360, 1392}}};
    for (const auto &[begin, end] : groups) {
        uint64_t cursor = begin;
        for (const auto &span : parameters_)
            if (span.external_address >= begin && span.external_address < end) {
                if (span.external_address != cursor)
                    Fail("physical parameter pages do not tightly cover P3 subrange");
                cursor += span.size_bytes;
            }
        if (cursor != end)
            Fail("physical parameter payload P3 subrange is incomplete");
    }
    for (const auto &raw : raw_kv) {
        Exact(raw, {"die_id", "kind", "source_state_ref",
                    "source_hbm_address", "external_address",
                    "hbm_address", "size_bytes"});
        MoeInferencePagerSpan span{
            Number(raw, "die_id"), String(raw, "kind"),
            String(raw, "source_state_ref"), Number(raw, "source_hbm_address"),
            Number(raw, "external_address"), Number(raw, "hbm_address"),
            Number(raw, "size_bytes")};
        if (span.die_id != 0 || span.size_bytes != 64 ||
            (span.kind != "kv_key" && span.kind != "kv_value") ||
            span.source_hbm_address < 1344 || span.external_address < 1952 ||
            span.hbm_address < 704 ||
            !sources.emplace(span.die_id, span.source_hbm_address).second)
            Fail("four real source rank0 KV pages changed");
        kv_pinned_.emplace(span.hbm_address, false);
        kv_pages_.push_back(std::move(span));
    }
    std::sort(kv_pages_.begin(), kv_pages_.end(),
              [](const auto &a, const auto &b) {
                  return a.source_hbm_address < b.source_hbm_address;
              });
    for (size_t index = 0; index < 4; ++index)
        if (kv_pages_[index].source_hbm_address != 1344 + 64 * index ||
            kv_pages_[index].external_address != 1952 + 64 * index ||
            kv_pages_[index].hbm_address != 704 + 64 * index)
            Fail("KV source/external/physical page continuity changed");
    if (std::count_if(kv_pages_.begin(), kv_pages_.end(),
                      [](const auto &span) { return span.kind == "kv_key"; }) != 2)
        Fail("two key/two value KV role pages required");
    const auto &ids = contract.at("linked_manifest_ids");
    const auto &digests = contract.at("linked_manifest_digests");
    if (!ids.is_array() || ids.size() != 3 || !digests.is_array() ||
        digests.size() != 3)
        Fail("three source-signed linked MoE segment ids/digests required");
    std::vector<frontend::LinkedProgramManifestDto> manifests;
    for (size_t segment = 0; segment < 3; ++segment) {
        if (!ids[segment].is_string() || !digests[segment].is_string())
            Fail("linked manifest id/digest must be strings");
        const auto parsed = frontend::ProgramArtifactFinalizer::Parse(
            manifest_texts[segment]);
        if (parsed.id != ids[segment].get<std::string>() ||
            frontend::ProgramArtifactFinalizer::CanonicalManifestDigest(
                manifest_texts[segment]) != digests[segment].get<std::string>() ||
            parsed.producer_pass != "moe_full_model_region_linker" ||
            parsed.fragments.size() != (segment ? 43 : 39) ||
            parsed.core_streams.size() != 2 ||
            parsed.core_streams[0].runtime_core_id != 0 ||
            parsed.core_streams[1].runtime_core_id != 16 ||
            parsed.core_streams[0].records.size() != (segment ? 199 : 195) ||
            parsed.core_streams[1].records.size() != 52 ||
            parsed.state_operand_bindings.size() != (segment ? 31 : 27))
            Fail("production full MoE linked segment identity/shape drifted");
        manifests.push_back(std::move(parsed));
    }
    for (size_t index = 0; index < raw_events.size(); ++index) {
        const auto &raw = raw_events[index];
        Exact(raw, {"segment_index", "runtime_core_id", "linked_record_index",
                    "fragment_id", "fragment_record_index", "kind",
                    "state_ref", "state_abi_id", "source_hbm_address",
                    "external_address", "hbm_address", "lsu_address",
                    "lsu_size_bytes", "dma_size_bytes"});
        MoeInferencePagerEvent event{
            index, Number(raw, "segment_index"),
            Number(raw, "runtime_core_id"), Number(raw, "linked_record_index"),
            String(raw, "fragment_id"), Number(raw, "fragment_record_index"),
            String(raw, "kind"), String(raw, "state_ref"),
            String(raw, "state_abi_id"), Number(raw, "source_hbm_address"),
            Number(raw, "external_address"), Number(raw, "hbm_address"),
            Number(raw, "lsu_address"), Number(raw, "lsu_size_bytes"),
            Number(raw, "dma_size_bytes")};
        if (event.segment_index > 2 ||
            (event.runtime_core_id != 0 && event.runtime_core_id != 16))
            Fail("signed DMA gate leaves fixed two-core sequence");
        event_indices_by_core_[event.runtime_core_id].push_back(index);
        events_.push_back(std::move(event));
    }
    if (event_indices_by_core_.at(0).size() != 71 ||
        event_indices_by_core_.at(16).size() != 18)
        Fail("89 source LSU gates must be Core0=71/Core16=18");

    size_t event_cursor = 0;
    uint64_t restore_bytes = 0, writeback_bytes = 0;
    for (size_t segment = 0; segment < 3; ++segment) {
        const auto &manifest = manifests[segment];
        std::map<std::string, const frontend::CommandFragmentDto *> fragments;
        std::map<std::string, frontend::StateAbiDto> abis;
        for (const auto &linked : manifest.fragments) {
            const auto *fragment = std::get_if<frontend::CommandFragmentDto>(&linked);
            if (!fragment) Fail("full MoE source requires true command fragments");
            if (!fragments.emplace(fragment->id, fragment).second)
                Fail("duplicate source command fragment id");
            for (const auto &abi : fragment->state_abi) {
                const auto [it, inserted] = abis.emplace(abi.id, abi);
                if (!inserted &&
                    (it->second.state_ref != abi.state_ref ||
                     it->second.hbm_binding_ref != abi.hbm_binding_ref ||
                     it->second.kind != abi.kind ||
                     it->second.die_id != abi.die_id ||
                     it->second.address != abi.address ||
                     it->second.size_bytes != abi.size_bytes))
                    Fail("shared KV StateABI has conflicting physical definitions");
            }
        }
        if (abis.size() != 23)
            Fail("23 true source shared/router/expert/KV StateABI required");
        std::map<std::tuple<std::string, uint64_t, uint64_t>,
                 frontend::StateOperandBindingDto> bindings;
        for (const auto &binding : manifest.state_operand_bindings)
            if (!bindings.emplace(
                    std::make_tuple(binding.fragment_id,
                                    binding.fragment_record_index,
                                    binding.logical_core.die_id), binding).second)
                Fail("duplicate StateOperandBinding at physical MoE core");
        const std::array<uint64_t, 4> expected_counts{{
            19, 4, segment ? 4u : 0u, 4}};
        std::array<uint64_t, 4> counts{{0, 0, 0, 0}};
        for (const auto &core : manifest.core_streams) {
            const uint64_t core_id = core.runtime_core_id;
            const uint64_t die = core.logical_core.die_id;
            if ((core_id == 0 && die != 0) ||
                (core_id == 16 && die != 1))
                Fail("source runtime core physical die mapping changed");
            for (size_t linked_index = 0;
                 linked_index < core.records.size(); ++linked_index) {
                const auto &ref = core.records[linked_index];
                const auto fragment = fragments.find(ref.fragment_id);
                if (fragment == fragments.end())
                    Fail("linked record lost source command fragment");
                const frontend::CoreFragmentStreamDto *stream = nullptr;
                for (const auto &candidate : fragment->second->core_streams)
                    if (candidate.logical_core.die_id == die &&
                        candidate.logical_core.local_core_id ==
                            core.logical_core.local_core_id)
                        stream = &candidate;
                if (!stream || ref.fragment_record_index >= stream->records.size())
                    Fail("linked source fragment local record closure drifted");
                const auto &record = stream->records[ref.fragment_record_index];
                const auto binding = bindings.find(
                    std::make_tuple(ref.fragment_id,
                                    ref.fragment_record_index, die));
                if (binding == bindings.end()) {
                    if (record.opcode == Opcode::LSU_LOAD ||
                        record.opcode == Opcode::LSU_STORE)
                        Fail("an actual MoE LSU lacks StateOperandBinding");
                    continue;
                }
                if (event_cursor >= events_.size())
                    Fail("more real LSU records than 89 signed DMA gates");
                const auto &event = events_[event_cursor++];
                const auto abi = abis.find(binding->second.state_abi_id);
                if (abi == abis.end() ||
                    event.segment_index != segment ||
                    event.runtime_core_id != core_id ||
                    event.linked_record_index != linked_index ||
                    event.fragment_id != ref.fragment_id ||
                    event.fragment_record_index != ref.fragment_record_index ||
                    event.state_abi_id != abi->second.id ||
                    event.state_ref != abi->second.state_ref ||
                    event.hbm_address != abi->second.address ||
                    event.dma_size_bytes != abi->second.size_bytes ||
                    abi->second.die_id != die ||
                    abi->second.alignment_bytes != 64 ||
                    event.lsu_size_bytes != RecordBytes(record))
                    Fail("89 real linked MoE LSU gates no longer bind physical StateABI");
                const int64_t addend = HbmAddend(*stream,
                                                  ref.fragment_record_index);
                if (addend < 0 || event.lsu_address !=
                        abi->second.address + static_cast<uint64_t>(addend))
                    Fail("actual LSU address relocation differs from signed gate");
                const auto parameter = std::find_if(
                    parameters_.begin(), parameters_.end(),
                    [&](const auto &span) {
                        return span.die_id == die &&
                               span.source_hbm_address ==
                                   event.source_hbm_address;
                    });
                const auto kv = std::find_if(
                    kv_pages_.begin(), kv_pages_.end(),
                    [&](const auto &span) {
                        return span.die_id == die &&
                               span.source_hbm_address ==
                                   event.source_hbm_address;
                    });
                if (abi->second.kind == frontend::StateKindDto::PARAMETER ||
                    abi->second.kind ==
                        frontend::StateKindDto::TRAINABLE_PARAMETER) {
                    if (parameter == parameters_.end() ||
                        event.external_address != parameter->external_address ||
                        event.hbm_address != parameter->hbm_address ||
                        abi->second.size_bytes != parameter->size_bytes ||
                        event.lsu_address != abi->second.address ||
                        event.lsu_size_bytes != abi->second.size_bytes)
                        Fail("weight LSU is not a whole true physical page");
                    if (record.opcode == Opcode::LSU_LOAD &&
                        event.kind == "weight_restore_before_load") {
                        ++counts[0];
                        restore_bytes += event.dma_size_bytes;
                    } else if (record.opcode == Opcode::LSU_STORE &&
                               abi->second.kind ==
                                   frontend::StateKindDto::TRAINABLE_PARAMETER &&
                               parameter->kind == "expert_retention" &&
                               event.kind == "expert_writeback_after_store") {
                        ++counts[1];
                        writeback_bytes += event.dma_size_bytes;
                    } else {
                        Fail("only exact expert retention may Store immutable inference weight");
                    }
                } else if (abi->second.kind == frontend::StateKindDto::KV_KEY ||
                           abi->second.kind == frontend::StateKindDto::KV_VALUE) {
                    const uint64_t page_bytes = 32 + 16 * segment;
                    if (kv == kv_pages_.end() || die != 0 ||
                        event.external_address != kv->external_address ||
                        event.hbm_address != kv->hbm_address ||
                        abi->second.size_bytes != page_bytes ||
                        (abi->second.kind == frontend::StateKindDto::KV_KEY
                             ? "kv_key" : "kv_value") != kv->kind)
                        Fail("versioned KV page does not match source physical role");
                    if (record.opcode == Opcode::LSU_LOAD &&
                        event.kind == "kv_restore_before_load" &&
                        event.lsu_address == abi->second.address &&
                        event.lsu_size_bytes == page_bytes) {
                        ++counts[2];
                        restore_bytes += page_bytes;
                    } else if (record.opcode == Opcode::LSU_STORE &&
                               event.kind == "kv_writeback_after_store" &&
                               event.lsu_address == abi->second.address +
                                   (segment ? 32 + 16 * (segment - 1) : 0) &&
                               event.lsu_size_bytes ==
                                   (segment ? 16 : 32)) {
                        ++counts[3];
                        writeback_bytes += page_bytes;
                    } else {
                        Fail("KV suffix Store must write back complete external page");
                    }
                } else {
                    Fail("unexpected persistent MoE inference StateABI role");
                }
            }
        }
        if (counts != expected_counts ||
            event_cursor != (segment == 0 ? 27 : segment == 1 ? 58 : 89))
            Fail("segment lacks 19 weights/4 expert stores/4 real KV lifecycle");
    }
    if (event_cursor != events_.size() ||
        restore_bytes != 4600 || writeback_bytes != 2880)
        Fail("89 real StateABI gates disagree with exact traffic oracle");

    std::map<std::string, HBMBackend *> hbm{
        {hbm_refs.at(0), backends.at({0, 0})},
        {hbm_refs.at(1), backends.at({1, 0})}};
    runtime_ = std::make_unique<ExternalMemoryRuntimeBridge>(
        name, fabric, std::move(hbm), cycle_time_);
    std::set<std::pair<uint64_t, uint64_t>> seeded;
    for (const auto &raw : raw_seeds) {
        Exact(raw, {"external_address", "payload_hex"});
        const uint64_t address = Number(raw, "external_address");
        const auto payload = Hex(String(raw, "payload_hex"));
        const auto present = std::find_if(
            parameters_.begin(), parameters_.end(),
            [&](const auto &span) {
                return span.external_address == address &&
                       span.size_bytes == payload.size();
            });
        const auto kv_present = std::find_if(
            kv_pages_.begin(), kv_pages_.end(),
            [&](const auto &span) {
                return span.external_address == address &&
                       span.size_bytes == payload.size();
            });
        if ((present == parameters_.end() && kv_present == kv_pages_.end()) ||
            !seeded.emplace(address, payload.size()).second)
            Fail("seed fabricated or duplicated a physical StateABI range");
        if (kv_present != kv_pages_.end() &&
            std::any_of(payload.begin(), payload.end(),
                        [](uint8_t byte) { return byte != 0; }))
            Fail("initial KV future page bytes must be empty");
        runtime_->SeedExternal(external_capacity_ref_, address, payload);
    }
    if (seeded.size() != 23)
        Fail("external authority seed omitted a true physical page");
    initial_parameter_digest_ = frontend::program_io::Sha256Hex(
        ProbeParameters());
    parameter_digest_ = initial_parameter_digest_;
}

const MoeInferencePagerEvent &MoeInferenceMidProgramPager::NextEvent(
    uint64_t core_id, uint64_t address, uint64_t size) const {
    const auto events = event_indices_by_core_.find(core_id);
    const auto cursor = next_by_core_.find(core_id);
    if (events == event_indices_by_core_.end() ||
        cursor == next_by_core_.end() ||
        cursor->second >= events->second.size())
        Fail("Core0/Core16 issued LSU beyond source-signed gates");
    const auto &event = events_[events->second[cursor->second]];
    if (event.runtime_core_id != core_id ||
        event.lsu_address != address || event.lsu_size_bytes != size)
        Fail("actual MoE LSU core/address/size differs from signed gate " +
             std::to_string(event.index));
    return event;
}

void MoeInferenceMidProgramPager::Transfer(
    const MoeInferencePagerEvent &event, TransferDirection direction) {
    const std::string id = "moe_inference_paged_" +
        std::to_string(event.index) + "_" + event.state_abi_id;
    const auto quantum = cycle_time_.value();
    const auto remainder = sc_core::sc_time_stamp().value() % quantum;
    if (remainder != 0)
        sc_core::wait(sc_core::sc_time::from_value(quantum - remainder));
    const uint64_t cycle = sc_core::sc_time_stamp().value() / quantum;
    const uint64_t die = event.runtime_core_id == 16 ? 1 : 0;
    runtime_->Submit({id, connection_by_die_.at(die), direction,
                      event.external_address, event.hbm_address,
                      event.dma_size_bytes, cycle,
                      kExternalDmaRequestSchemaVersion});
    const auto completion = runtime_->Wait(id);
    if (completion.status != 0 ||
        completion.payload_bytes != event.dma_size_bytes)
        Fail("real shared-link DMA completion failed for " +
             event.state_ref + ": " + completion.error);
    ++completed_events_;
    std::cout << "[MOE_INFERENCE_PAGED_DMA_EVENT] index=" << event.index
              << " segment=" << event.segment_index
              << " core=" << event.runtime_core_id
              << " linked_record=" << event.linked_record_index
              << " kind=" << event.kind
              << " state_ref=" << event.state_ref
              << " lsu_bytes=" << event.lsu_size_bytes
              << " dma_bytes=" << completion.payload_bytes
              << " issue_cycle=" << cycle
              << " completed_at_ticks=" << completion.completed_at.value()
              << " lsu_dependency_complete=1 pass=1" << std::endl;
}

void MoeInferenceMidProgramPager::BeforeLoad(
    uint64_t core_id, uint64_t address, uint64_t size) {
    if (awaiting_load_.count(core_id))
        Fail("previous real MoE LSU Load has no completion hook");
    const auto &event = NextEvent(core_id, address, size);
    const uint64_t die = core_id == 16 ? 1 : 0;
    if (event.kind == "weight_restore_before_load") {
        if (weight_pinned_.at(die))
            Fail("parameter page slot reused before prior LSU completion");
        weight_pinned_.at(die) = true;
    } else if (event.kind == "kv_restore_before_load") {
        if (die != 0 || kv_pinned_.at(event.hbm_address))
            Fail("KV physical page restored twice before suffix writeback");
        kv_pinned_.at(event.hbm_address) = true;
    } else {
        Fail("actual MoE LSU Load has non-restore source role");
    }
    Transfer(event, TransferDirection::kExternalToHbm);
    awaiting_load_.emplace(core_id, event.index);
    ++next_by_core_.at(core_id);
}

void MoeInferenceMidProgramPager::AfterLoad(
    uint64_t core_id, uint64_t address, uint64_t size) {
    const auto awaiting = awaiting_load_.find(core_id);
    if (awaiting == awaiting_load_.end())
        Fail("real MoE LSU Load completion lacks prior external restore");
    const auto &event = events_[awaiting->second];
    if (event.lsu_address != address || event.lsu_size_bytes != size)
        Fail("completed MoE LSU Load differs from signed restore gate");
    if (event.kind == "weight_restore_before_load") {
        const uint64_t die = core_id == 16 ? 1 : 0;
        if (!weight_pinned_.at(die)) Fail("parameter slot pin vanished early");
        weight_pinned_.at(die) = false;
        const auto span = std::find_if(
            parameters_.begin(), parameters_.end(),
            [&](const auto &item) {
                return item.die_id == die &&
                       item.source_hbm_address == event.source_hbm_address;
            });
        if (span == parameters_.end()) Fail("restored weight has no real page");
        if (span->kind == "expert_retention" && span->size_bytes == 192 &&
            !expert_awaiting_store_.emplace(
                die, span->source_hbm_address).second)
            Fail("expert inference retention Load repeated before its Store");
    } else if (event.kind != "kv_restore_before_load" ||
               !kv_pinned_.at(event.hbm_address)) {
        Fail("KV page pin vanished before suffix Store");
    }
    awaiting_load_.erase(awaiting);
}

void MoeInferenceMidProgramPager::AfterStore(
    uint64_t core_id, uint64_t address, uint64_t size) {
    if (awaiting_load_.count(core_id))
        Fail("MoE LSU Store raced with incomplete same-core Load");
    const auto &event = NextEvent(core_id, address, size);
    const uint64_t die = core_id == 16 ? 1 : 0;
    if (weight_pinned_.at(die))
        Fail("expert/KV Store raced with an incomplete weight slot restore");
    if (event.kind == "expert_writeback_after_store") {
        if (!expert_awaiting_store_.erase(
                {die, event.source_hbm_address}))
            Fail("expert retention Store lacks exact prior weight Load");
    } else if (event.kind == "kv_writeback_after_store") {
        if (die != 0 ||
            (event.segment_index != 0 &&
             !kv_pinned_.at(event.hbm_address)))
            Fail("Decode KV suffix Store lacks completed full-prefix restore");
    } else {
        Fail("actual MoE LSU Store lacks signed dirty writeback role");
    }
    ++dirty_;
    Transfer(event, TransferDirection::kHbmToExternal);
    --dirty_;
    if (event.kind == "kv_writeback_after_store")
        kv_pinned_.at(event.hbm_address) = false;
    ++next_by_core_.at(core_id);
}

std::vector<uint8_t> MoeInferenceMidProgramPager::ProbeKvPages(
    uint64_t page_bytes) const {
    auto ordered = kv_pages_;
    std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
        return std::tie(a.kind, a.hbm_address) <
               std::tie(b.kind, b.hbm_address);
    });
    std::vector<uint8_t> authority;
    authority.reserve(4 * page_bytes);
    for (const auto &span : ordered) {
        const auto payload = runtime_->ProbeExternal(
            external_capacity_ref_, span.external_address, page_bytes);
        if (payload.size() != page_bytes)
            Fail("external KV authority probe missed exact page extent");
        authority.insert(authority.end(), payload.begin(), payload.end());
    }
    return authority;
}

std::vector<uint8_t> MoeInferenceMidProgramPager::ProbeParameters() const {
    std::vector<uint8_t> authority;
    authority.reserve(1384);
    for (const auto &span : parameters_) {
        const auto payload = runtime_->ProbeExternal(
            external_capacity_ref_, span.external_address, span.size_bytes);
        if (payload.size() != span.size_bytes)
            Fail("external parameter authority missed true physical bytes");
        authority.insert(authority.end(), payload.begin(), payload.end());
    }
    if (authority.size() != 1384)
        Fail("physical parameter authority must cover 1384B without padding");
    return authority;
}

std::string MoeInferenceMidProgramPager::ProbeInitialKvAuthority() const {
    if (completed_events_ != 0 || Pending() != 0 || Pinned() != 0)
        Fail("KV authority initial probe must precede all real DMA/compute");
    const auto capacity = ProbeKvPages(64);
    if (capacity.size() != 256 ||
        std::any_of(capacity.begin(), capacity.end(),
                    [](uint8_t byte) { return byte != 0; }))
        Fail("initial physical rank0 KV future-page payload is not empty");
    return frontend::program_io::Sha256Hex(std::vector<uint8_t>{});
}

void MoeInferenceMidProgramPager::CompleteSegment(uint64_t segment) {
    const uint64_t expected = segment == 0 ? 27 : segment == 1 ? 58 : 89;
    if (segment > 2 || completed_events_ != expected || Pending() != 0 ||
        Pinned() != 0 || dirty_ != 0 || !awaiting_load_.empty() ||
        !expert_awaiting_store_.empty())
        Fail("full MoE segment ended with pending/dirty/pinned state or missed expert Store");
    for (const auto &[core, indices] : event_indices_by_core_)
        for (size_t index = 0; index < next_by_core_.at(core); ++index)
            if (events_[indices[index]].segment_index > segment)
                Fail("Core0/Core16 consumed a future-segment source gate early");
    const uint64_t page_bytes = 32 + 16 * segment;
    const auto kv = ProbeKvPages(page_bytes);
    external_kv_probes_ += 4;
    kv_bytes_ = kv.size();
    if (kv_bytes_ != 4 * page_bytes ||
        external_kv_probes_ != 4 * (segment + 1))
        Fail("four true external KV pages did not advance extent");
    kv_digest_ = frontend::program_io::Sha256Hex(kv);
    parameter_digest_ = frontend::program_io::Sha256Hex(ProbeParameters());
    if (parameter_digest_ != initial_parameter_digest_)
        Fail("expert retention Store changed immutable MoE inference weights");
    std::cout << "[MOE_INFERENCE_PAGED_PARAMETER_AUTHORITY] version="
              << segment + 1 << " physical_bytes=1384 digest="
              << parameter_digest_ << " immutable=1 functional=0 pass=1"
              << std::endl;
}

uint64_t MoeInferenceMidProgramPager::Pending() const {
    return runtime_->Outstanding();
}

uint64_t MoeInferenceMidProgramPager::Pinned() const {
    uint64_t count = 0;
    for (const auto &item : weight_pinned_)
        if (item.second) ++count;
    for (const auto &item : kv_pinned_)
        if (item.second) ++count;
    return count;
}

const RuntimeStats &MoeInferenceMidProgramPager::Stats() const {
    return runtime_->Stats();
}
} // namespace external_memory
