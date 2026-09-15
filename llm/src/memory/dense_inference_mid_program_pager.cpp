#include "memory/dense_inference_mid_program_pager.h"

#include "frontend/program_finalizer.h"
#include "frontend/program_io.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace external_memory {
namespace {
using Json = nlohmann::json;

[[noreturn]] void Fail(const std::string &detail) {
    throw std::invalid_argument("Dense inference paged runtime: " + detail);
}

void Exact(const Json &object, std::initializer_list<const char *> names) {
    if (!object.is_object() || object.size() != names.size())
        Fail("JSON object shape drifted");
    for (const char *name : names)
        if (!object.contains(name)) Fail(std::string("missing field ") + name);
}

std::string String(const Json &object, const char *key) {
    if (!object.at(key).is_string()) Fail(std::string(key) + " must be string");
    const auto value = object.at(key).get<std::string>();
    if (value.empty()) Fail(std::string(key) + " must not be empty");
    return value;
}

uint64_t Number(const Json &object, const char *key) {
    const auto &value = object.at(key);
    if (!value.is_number_unsigned() &&
        !(value.is_number_integer() && value.get<int64_t>() >= 0))
        Fail(std::string(key) + " must be nonnegative uint64");
    return value.get<uint64_t>();
}

Json ReadJson(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) Fail("cannot open " + path.string());
    Json value;
    input >> value;
    if (!input.eof() && input.peek() != EOF)
        Fail("trailing content in " + path.string());
    return value;
}

std::vector<uint8_t> Hex(const std::string &value) {
    if (value.empty() || value.size() % 2) Fail("seed is not even-length hex");
    const auto digit = [](char character) -> uint8_t {
        if (character >= '0' && character <= '9')
            return static_cast<uint8_t>(character - '0');
        if (character >= 'a' && character <= 'f')
            return static_cast<uint8_t>(character - 'a' + 10);
        Fail("seed must use lowercase hexadecimal");
    };
    std::vector<uint8_t> result;
    result.reserve(value.size() / 2);
    for (size_t i = 0; i < value.size(); i += 2)
        result.push_back(static_cast<uint8_t>(
            digit(value[i]) * 16 + digit(value[i + 1])));
    return result;
}

FabricConfig ParseFixedFabric(const Json &value) {
    Exact(value, {"schema_version", "producer_pass", "id",
                  "external_capacities", "hbm_capacities", "links",
                  "connections"});
    if (String(value, "schema_version") !=
            "wafer_frontend.external_memory_fabric/v1alpha2" ||
        String(value, "producer_pass") != "external_memory_fabric_builder" ||
        !value.at("external_capacities").is_array() ||
        !value.at("hbm_capacities").is_array() ||
        !value.at("links").is_array() ||
        !value.at("connections").is_array() ||
        value.at("external_capacities").size() != 1 ||
        value.at("hbm_capacities").size() != 1 ||
        value.at("links").size() != 1 ||
        value.at("connections").size() != 1)
        Fail("requires official one-link die0 ExternalMemoryFabric");
    const auto &external = value.at("external_capacities")[0];
    Exact(external, {"id", "tier", "location_ref", "base_address",
                     "capacity_bytes", "alignment_bytes"});
    const auto &hbm = value.at("hbm_capacities")[0];
    Exact(hbm, {"id", "tier", "location_ref", "base_address",
                "capacity_bytes", "alignment_bytes"});
    const auto &link = value.at("links")[0];
    Exact(link, {"id", "external_capacity_ref", "ingress_die_id",
                 "bytes_per_cycle", "latency_cycles", "queue_depth",
                 "max_outstanding", "duplex"});
    const auto &connection = value.at("connections")[0];
    Exact(connection, {"id", "link_ref", "hbm_capacity_ref",
                       "target_die_id", "route_die_ids",
                       "route_latency_cycles", "route_bytes_per_cycle"});
    if (String(external, "tier") != "external" ||
        String(external, "location_ref") != "host:0" ||
        Number(external, "base_address") != 0 ||
        Number(external, "capacity_bytes") != 54400 ||
        Number(external, "alignment_bytes") != 64 ||
        String(hbm, "tier") != "hbm" ||
        String(hbm, "location_ref") != "die:0" ||
        Number(hbm, "base_address") != 0 ||
        Number(hbm, "capacity_bytes") != 12288 ||
        Number(hbm, "alignment_bytes") != 64 ||
        String(link, "external_capacity_ref") != String(external, "id") ||
        Number(link, "ingress_die_id") != 0 ||
        Number(link, "bytes_per_cycle") != 256 ||
        Number(link, "latency_cycles") != 2 ||
        Number(link, "queue_depth") != 2 ||
        Number(link, "max_outstanding") != 2 ||
        String(link, "duplex") != "half_duplex_shared" ||
        String(connection, "link_ref") != String(link, "id") ||
        String(connection, "hbm_capacity_ref") != String(hbm, "id") ||
        Number(connection, "target_die_id") != 0 ||
        !connection.at("route_die_ids").is_array() ||
        connection.at("route_die_ids") != Json::array({0}) ||
        Number(connection, "route_latency_cycles") != 0 ||
        !connection.at("route_bytes_per_cycle").is_null())
        Fail("fixed low-HBM/shared bridge fabric shape changed");
    FabricConfig result;
    result.external_capacities.push_back(
        {String(external, "id"), "host:0", 0, 54400});
    result.hbm_capacities.push_back(
        {String(hbm, "id"), 0, 0, 12288});
    result.links.push_back({String(link, "id"), String(external, "id"),
                            0, 256, 2, 2, 2});
    result.connections.push_back({String(connection, "id"),
                                  String(link, "id"), String(hbm, "id"),
                                  0, {0}, 0, std::nullopt});
    ValidateFabricConfig(result);
    return result;
}

uint64_t ExpectedKvPageBytes(uint64_t segment) {
    if (segment > 2) Fail("KV segment exceeds Prefill+2Decode");
    return 128 + 32 * segment;
}

std::string DigestWithoutId(Json value) {
    value.erase("id");
    return frontend::program_io::Sha256Hex(value.dump());
}
} // namespace

DenseInferenceMidProgramPager::DenseInferenceMidProgramPager(
    const sc_core::sc_module_name &name,
    const std::filesystem::path &sidecar,
    std::vector<std::string> manifest_texts,
    std::map<std::pair<uint64_t, uint64_t>, HBMBackend *> backends,
    sc_core::sc_time cycle_time)
    : cycle_time_(cycle_time) {
    const Json contract = ReadJson(sidecar);
    Exact(contract, {"schema_version", "producer_pass", "id",
                     "request_digest", "model_digest",
                     "logical_graph_digest", "offload_memory_plan_digest",
                     "source_parameter_allocation_ref",
                     "source_kv_inventory_digest", "linked_manifest_ids",
                     "linked_manifest_digests", "hbm_capacity_bytes",
                     "workspace_end_bytes", "parameter_state_bytes",
                     "kv_capacity_bytes", "highest_paged_end_bytes",
                     "fabric", "weight_spans", "kv_spans", "events",
                     "parameter_seed_hex", "kv_seed_hex"});
    if (String(contract, "schema_version") !=
            "wafer_frontend.dense_inference_paged_runtime/v1alpha1" ||
        String(contract, "producer_pass") !=
            "dense_inference_paged_runtime" ||
        String(contract, "id") != "dense_inference_paged_runtime_" +
            DigestWithoutId(contract).substr(0, 20) ||
        Number(contract, "hbm_capacity_bytes") != 12288 ||
        Number(contract, "workspace_end_bytes") != 1600 ||
        Number(contract, "parameter_state_bytes") != 53568 ||
        Number(contract, "kv_capacity_bytes") != 768 ||
        Number(contract, "highest_paged_end_bytes") != 10560 ||
        manifest_texts.size() != 3)
        Fail("source-signed fixed two-layer bounded identity changed");
    source_ref_ = String(contract, "id");
    const FabricConfig fabric = ParseFixedFabric(contract.at("fabric"));
    external_capacity_ref_ = fabric.external_capacities.front().id;
    connection_ref_ = fabric.connections.front().id;
    if (backends.size() != 1 || !backends.count({0, 0}) ||
        !backends.at({0, 0}))
        Fail("physical die0 stack0/channel0 HBM backend unavailable");

    const auto &raw_weights = contract.at("weight_spans");
    const auto &raw_kv = contract.at("kv_spans");
    const auto &raw_events = contract.at("events");
    if (!raw_weights.is_array() || raw_weights.size() != 15 ||
        !raw_kv.is_array() || raw_kv.size() != 4 ||
        !raw_events.is_array() || raw_events.size() != 65)
        Fail("requires 15 physical weights, four KV pages and 65 LSU gates");
    std::set<std::string> weight_refs;
    for (const auto &raw : raw_weights) {
        Exact(raw, {"kind", "state_ref", "source_hbm_address",
                    "external_address", "hbm_address", "size_bytes"});
        DenseInferencePagerSpan span{
            String(raw, "kind"), String(raw, "state_ref"),
            Number(raw, "source_hbm_address"),
            Number(raw, "external_address"), Number(raw, "hbm_address"),
            Number(raw, "size_bytes")};
        if (span.kind != "parameter" ||
            !weight_refs.insert(span.state_ref).second ||
            span.external_address != span.source_hbm_address ||
            span.hbm_address != 1600 || span.size_bytes == 0 ||
            span.size_bytes > 8192)
            Fail("parameter ABI is not a true single-slot page");
        weights_.push_back(std::move(span));
    }
    std::sort(weights_.begin(), weights_.end(), [](const auto &a, const auto &b) {
        return a.external_address < b.external_address;
    });
    uint64_t cursor = 0;
    for (const auto &span : weights_) {
        if (span.external_address != cursor)
            Fail("15 physical parameter slices do not cover tight P3 allocation");
        cursor += span.size_bytes;
    }
    if (cursor != 53568)
        Fail("P3 aggregate parameter allocation is not 53,568 exact bytes");

    std::set<uint64_t> kv_source_addresses, kv_hbm_addresses;
    std::map<uint64_t, std::string> kv_kinds;
    for (const auto &raw : raw_kv) {
        Exact(raw, {"kind", "state_ref", "source_hbm_address",
                    "external_address", "hbm_address", "size_bytes"});
        DenseInferencePagerSpan span{
            String(raw, "kind"), String(raw, "state_ref"),
            Number(raw, "source_hbm_address"),
            Number(raw, "external_address"), Number(raw, "hbm_address"),
            Number(raw, "size_bytes")};
        if ((span.kind != "kv_key" && span.kind != "kv_value") ||
            !kv_source_addresses.insert(span.source_hbm_address).second ||
            !kv_hbm_addresses.insert(span.hbm_address).second ||
            span.size_bytes != 192 || span.hbm_address < 9792 ||
            span.hbm_address + 192 > 10560)
            Fail("KV page is not a stable disjoint 192B slot");
        kv_kinds.emplace(span.hbm_address, span.kind);
        kv_pinned_.emplace(span.hbm_address, false);
        kv_pages_.push_back(std::move(span));
    }
    std::sort(kv_pages_.begin(), kv_pages_.end(),
              [](const auto &a, const auto &b) {
                  return a.source_hbm_address < b.source_hbm_address;
              });
    for (size_t i = 0; i < 4; ++i)
        if (kv_pages_[i].source_hbm_address != 53568 + 192 * i ||
            kv_pages_[i].external_address != 53568 + 192 * i ||
            kv_pages_[i].hbm_address != 9792 + 192 * i)
            Fail("KV external/home/physical page slot continuity drifted");
    if (std::count_if(kv_pages_.begin(), kv_pages_.end(),
                      [](const auto &span) { return span.kind == "kv_key"; }) != 2)
        Fail("KV key/value physical page roles are incomplete");

    const auto parameter_seed = Hex(String(contract, "parameter_seed_hex"));
    const auto kv_seed = Hex(String(contract, "kv_seed_hex"));
    if (parameter_seed.size() != 53568 || kv_seed.size() != 768 ||
        std::all_of(parameter_seed.begin(), parameter_seed.end(),
                    [](uint8_t value) { return value == 0; }) ||
        std::any_of(kv_seed.begin(), kv_seed.end(),
                    [](uint8_t value) { return value != 0; }))
        Fail("external source parameter/KV initial payload is not exact");

    const auto &ids = contract.at("linked_manifest_ids");
    const auto &digests = contract.at("linked_manifest_digests");
    if (!ids.is_array() || ids.size() != 3 ||
        !digests.is_array() || digests.size() != 3)
        Fail("three source linked manifest identities required");
    std::vector<frontend::LinkedProgramManifestDto> manifests;
    for (size_t segment = 0; segment < 3; ++segment) {
        auto parsed = frontend::ProgramArtifactFinalizer::Parse(
            manifest_texts[segment]);
        if (parsed.id != ids[segment].get<std::string>() ||
            frontend::ProgramArtifactFinalizer::CanonicalManifestDigest(
                manifest_texts[segment]) !=
                digests[segment].get<std::string>() ||
            parsed.producer_pass != "manifest_linker" ||
            parsed.core_streams.size() != 1 ||
            parsed.core_streams.front().runtime_core_id != 0 ||
            parsed.fragments.size() != (segment ? 48 : 44) ||
            parsed.core_streams.front().records.size() !=
                (segment ? 163 : 159) ||
            parsed.state_operand_bindings.size() !=
                (segment ? 23 : 19))
            Fail("production Dense linked segment identity/shape drifted");
        manifests.push_back(std::move(parsed));
    }

    for (const auto &raw : raw_events) {
        Exact(raw, {"segment_index", "linked_record_index", "fragment_id",
                    "fragment_record_index", "kind", "state_ref",
                    "state_abi_id", "source_hbm_address",
                    "external_address", "hbm_address", "lsu_address",
                    "lsu_size_bytes", "dma_size_bytes"});
        events_.push_back({
            Number(raw, "segment_index"), Number(raw, "linked_record_index"),
            String(raw, "fragment_id"), Number(raw, "fragment_record_index"),
            String(raw, "kind"), String(raw, "state_ref"),
            String(raw, "state_abi_id"), Number(raw, "source_hbm_address"),
            Number(raw, "external_address"), Number(raw, "hbm_address"),
            Number(raw, "lsu_address"), Number(raw, "lsu_size_bytes"),
            Number(raw, "dma_size_bytes")});
    }
    size_t event_cursor = 0;
    for (size_t segment = 0; segment < 3; ++segment) {
        const auto &manifest = manifests[segment];
        std::map<std::string, const frontend::CommandFragmentDto *> fragments;
        std::map<std::string, frontend::StateAbiDto> abis;
        for (const auto &linked : manifest.fragments) {
            const auto *fragment =
                std::get_if<frontend::CommandFragmentDto>(&linked);
            if (!fragment) Fail("fixed source requires 44/48 command fragments");
            fragments.emplace(fragment->id, fragment);
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
        std::map<std::pair<std::string, uint64_t>,
                 frontend::StateOperandBindingDto> bindings;
        for (const auto &binding : manifest.state_operand_bindings)
            if (!bindings.emplace(
                    std::make_pair(binding.fragment_id,
                                   binding.fragment_record_index), binding).second)
                Fail("duplicate linked StateOperandBinding");
        size_t weight_loads = 0, kv_loads = 0, kv_stores = 0;
        for (size_t linked_index = 0;
             linked_index < manifest.core_streams.front().records.size();
             ++linked_index) {
            const auto &ref = manifest.core_streams.front().records[linked_index];
            const auto fragment = fragments.find(ref.fragment_id);
            if (fragment == fragments.end() ||
                fragment->second->core_streams.size() != 1 ||
                ref.fragment_record_index >=
                    fragment->second->core_streams.front().records.size())
                Fail("linked fragment record source closure changed");
            const auto &record = fragment->second->core_streams.front()
                                     .records[ref.fragment_record_index];
            const auto binding = bindings.find(
                {ref.fragment_id, ref.fragment_record_index});
            if (binding == bindings.end()) {
                if (record.opcode == Opcode::LSU_LOAD ||
                    record.opcode == Opcode::LSU_STORE)
                    Fail("an actual LSU lacks signed StateOperandBinding");
                continue;
            }
            if (event_cursor >= events_.size())
                Fail("more signed state LSU records than 65 DMA gates");
            const auto &event = events_[event_cursor++];
            const auto abi = abis.find(binding->second.state_abi_id);
            if (abi == abis.end() || event.segment_index != segment ||
                event.linked_record_index != linked_index ||
                event.fragment_id != ref.fragment_id ||
                event.fragment_record_index != ref.fragment_record_index ||
                event.state_abi_id != abi->second.id ||
                event.state_ref != abi->second.state_ref ||
                event.hbm_address != abi->second.address ||
                event.dma_size_bytes != abi->second.size_bytes)
                Fail("65 real linked LSU gates no longer bind paged StateABI");
            if (abi->second.kind == frontend::StateKindDto::PARAMETER) {
                const auto span = std::find_if(
                    weights_.begin(), weights_.end(), [&](const auto &item) {
                        return item.state_ref == event.state_ref;
                    });
                if (span == weights_.end() ||
                    event.kind != "weight_restore_before_load" ||
                    record.opcode != Opcode::LSU_LOAD ||
                    event.source_hbm_address != span->source_hbm_address ||
                    event.external_address != span->external_address ||
                    event.lsu_address != span->hbm_address ||
                    event.lsu_size_bytes != span->size_bytes ||
                    abi->second.size_bytes != span->size_bytes)
                    Fail("parameter LSU must restore the entire true ABI page");
                ++weight_loads;
            } else if (abi->second.kind == frontend::StateKindDto::KV_KEY ||
                       abi->second.kind == frontend::StateKindDto::KV_VALUE) {
                const auto span = std::find_if(
                    kv_pages_.begin(), kv_pages_.end(), [&](const auto &item) {
                        return item.hbm_address == event.hbm_address;
                    });
                if (span == kv_pages_.end() ||
                    event.source_hbm_address != span->source_hbm_address ||
                    event.external_address != span->external_address ||
                    abi->second.size_bytes != ExpectedKvPageBytes(segment) ||
                    (abi->second.kind == frontend::StateKindDto::KV_KEY
                         ? "kv_key" : "kv_value") != span->kind)
                    Fail("versioned KV source/physical page role drifted");
                if (record.opcode == Opcode::LSU_LOAD &&
                    event.kind == "kv_restore_before_load" &&
                    event.lsu_address == span->hbm_address &&
                    event.lsu_size_bytes == abi->second.size_bytes) {
                    ++kv_loads;
                } else if (record.opcode == Opcode::LSU_STORE &&
                           event.kind == "kv_writeback_after_store" &&
                           event.lsu_address == span->hbm_address +
                               (segment ? ExpectedKvPageBytes(segment - 1) : 0) &&
                           event.lsu_size_bytes == (segment ? 32 : 128)) {
                    ++kv_stores;
                } else {
                    Fail("KV suffix Store must write back the full authority page");
                }
            } else {
                Fail("fixed paged source has an unexpected persistent state");
            }
        }
        if (weight_loads != 15 || kv_loads != (segment ? 4 : 0) ||
            kv_stores != 4 ||
            event_cursor != (segment == 0 ? 19 : segment == 1 ? 42 : 65))
            Fail("segment LSU coverage is not exact 15 weight + 4 KV page lifecycle");
    }
    if (event_cursor != events_.size()) Fail("unbound signed DMA gates remain");

    std::map<std::string, HBMBackend *> hbm{
        {fabric.hbm_capacities.front().id, backends.at({0, 0})}};
    runtime_ = std::make_unique<ExternalMemoryRuntimeBridge>(
        name, fabric, std::move(hbm), cycle_time_);
    runtime_->SeedExternal(external_capacity_ref_, 0, parameter_seed);
    runtime_->SeedExternal(external_capacity_ref_, 53568, kv_seed);
}

const DenseInferencePagerEvent &DenseInferenceMidProgramPager::NextEvent(
    const char *kind, uint64_t address, uint64_t size) const {
    if (next_event_ >= events_.size())
        Fail("runtime issued an LSU beyond 65 signed gates");
    const auto &event = events_[next_event_];
    if (event.kind != kind || event.lsu_address != address ||
        event.lsu_size_bytes != size)
        Fail("actual LSU direction/address/size changed at gate " +
             std::to_string(next_event_));
    return event;
}

void DenseInferenceMidProgramPager::Transfer(
    const DenseInferencePagerEvent &event, TransferDirection direction) {
    const std::string id = "dense_inference_paged_" +
        std::to_string(next_event_) + "_" + event.state_abi_id;
    const auto quantum = cycle_time_.value();
    const auto remainder = sc_core::sc_time_stamp().value() % quantum;
    if (remainder != 0)
        sc_core::wait(sc_core::sc_time::from_value(quantum - remainder));
    const uint64_t cycle = sc_core::sc_time_stamp().value() / quantum;
    runtime_->Submit({id, connection_ref_, direction, event.external_address,
                      event.hbm_address, event.dma_size_bytes, cycle,
                      kExternalDmaRequestSchemaVersion});
    const auto completion = runtime_->Wait(id);
    if (completion.status != 0 ||
        completion.payload_bytes != event.dma_size_bytes)
        Fail("actual shared DMA completion failed for " + event.state_ref +
             ": " + completion.error);
    std::cout << "[DENSE_INFERENCE_PAGED_DMA_EVENT] index=" << next_event_
              << " segment=" << event.segment_index
              << " linked_record=" << event.linked_record_index
              << " kind=" << event.kind
              << " state_ref=" << event.state_ref
              << " lsu_bytes=" << event.lsu_size_bytes
              << " dma_bytes=" << completion.payload_bytes
              << " issue_cycle=" << cycle
              << " completed_at_ticks=" << completion.completed_at.value()
              << " lsu_dependency_complete=1 pass=1" << std::endl;
}

void DenseInferenceMidProgramPager::BeforeLoad(uint64_t address,
                                               uint64_t size) {
    if (awaiting_after_load_) Fail("previous LSU Load has no completion hook");
    if (next_event_ >= events_.size()) Fail("weight/KV load beyond signed sequence");
    const char *kind = events_[next_event_].kind ==
                               "weight_restore_before_load"
                           ? "weight_restore_before_load"
                           : "kv_restore_before_load";
    const auto &event = NextEvent(kind, address, size);
    if (event.kind == "weight_restore_before_load") {
        if (weight_pinned_) Fail("weight slot reused before old LSU completion");
        weight_pinned_ = true;
    } else {
        if (kv_pinned_.at(event.hbm_address))
            Fail("KV page restored twice without dirty writeback");
        kv_pinned_.at(event.hbm_address) = true;
    }
    Transfer(event, TransferDirection::kExternalToHbm);
    awaiting_after_load_ = true;
    ++next_event_;
}

void DenseInferenceMidProgramPager::AfterLoad(uint64_t address,
                                              uint64_t size) {
    if (!awaiting_after_load_ || next_event_ == 0)
        Fail("LSU Load completion lacks its page restore");
    const auto &event = events_[next_event_ - 1];
    if (event.lsu_address != address || event.lsu_size_bytes != size)
        Fail("completed LSU Load differs from its restore gate");
    if (event.kind == "weight_restore_before_load") {
        if (!weight_pinned_) Fail("weight slot pin vanished early");
        weight_pinned_ = false;
    } else if (event.kind != "kv_restore_before_load" ||
               !kv_pinned_.at(event.hbm_address)) {
        Fail("KV Load page pin vanished before suffix Store");
    }
    awaiting_after_load_ = false;
}

void DenseInferenceMidProgramPager::AfterStore(uint64_t address,
                                               uint64_t size) {
    if (awaiting_after_load_ || weight_pinned_)
        Fail("KV Store raced with incomplete weight/KV Load");
    const auto &event = NextEvent("kv_writeback_after_store", address, size);
    if (event.segment_index != 0 && !kv_pinned_.at(event.hbm_address))
        Fail("Decode KV suffix Store lacks full-prefix restore");
    ++dirty_;
    Transfer(event, TransferDirection::kHbmToExternal);
    --dirty_;
    kv_pinned_.at(event.hbm_address) = false;
    ++next_event_;
}

std::vector<uint8_t> DenseInferenceMidProgramPager::ProbeKvPages(
    uint64_t page_bytes) const {
    auto ordered = kv_pages_;
    std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
        return std::tie(a.kind, a.hbm_address) <
               std::tie(b.kind, b.hbm_address);
    });
    std::vector<uint8_t> authority;
    authority.reserve(ordered.size() * page_bytes);
    for (const auto &span : ordered) {
        const auto payload = runtime_->ProbeExternal(
            external_capacity_ref_, span.external_address, page_bytes);
        if (payload.size() != page_bytes)
            Fail("external KV authority probe did not cover exact current page");
        authority.insert(authority.end(), payload.begin(), payload.end());
    }
    return authority;
}

std::string DenseInferenceMidProgramPager::ProbeInitialKvAuthority() const {
    if (next_event_ != 0 || Pending() != 0 || Pinned() != 0)
        Fail("initial KV authority must precede all compute/DMA");
    const auto capacity = ProbeKvPages(192);
    if (capacity.size() != 768 ||
        std::any_of(capacity.begin(), capacity.end(),
                    [](uint8_t value) { return value != 0; }))
        Fail("initial external KV future page capacity is not empty");
    return frontend::program_io::Sha256Hex(std::vector<uint8_t>{});
}

void DenseInferenceMidProgramPager::CompleteSegment(uint64_t segment) {
    const uint64_t expected = segment == 0 ? 19 : segment == 1 ? 42 : 65;
    if (segment > 2 || next_event_ != expected || Pending() != 0 ||
        Pinned() != 0 || dirty_ != 0 || awaiting_after_load_)
        Fail("segment ended with incomplete DMA, dirty state or active pin");
    const uint64_t page_bytes = ExpectedKvPageBytes(segment);
    const auto authority = ProbeKvPages(page_bytes);
    external_kv_probes_ += 4;
    kv_bytes_ = authority.size();
    if (kv_bytes_ != 4 * page_bytes ||
        external_kv_probes_ != 4 * (segment + 1))
        Fail("four external KV authority pages did not advance extent");
    kv_digest_ = frontend::program_io::Sha256Hex(authority);
}

uint64_t DenseInferenceMidProgramPager::Pending() const {
    return runtime_->Outstanding();
}

uint64_t DenseInferenceMidProgramPager::Pinned() const {
    return (weight_pinned_ ? 1 : 0) +
           std::count_if(kv_pinned_.begin(), kv_pinned_.end(),
                         [](const auto &item) { return item.second; });
}

const RuntimeStats &DenseInferenceMidProgramPager::Stats() const {
    return runtime_->Stats();
}
} // namespace external_memory
