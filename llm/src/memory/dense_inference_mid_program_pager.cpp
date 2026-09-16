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

FabricConfig ParseRectFabric(const Json &value, uint64_t rows,
                             uint64_t columns) {
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
        value.at("hbm_capacities").size() != 4 ||
        value.at("links").size() != 1 ||
        value.at("connections").size() != 4)
        Fail("requires one external authority and four physical HBM homes");
    const auto &external = value.at("external_capacities")[0];
    Exact(external, {"id", "tier", "location_ref", "base_address",
                     "capacity_bytes", "alignment_bytes"});
    if (String(external, "tier") != "external" ||
        String(external, "location_ref") != "host:0" ||
        Number(external, "base_address") != 0 ||
        Number(external, "capacity_bytes") != 131072 ||
        Number(external, "alignment_bytes") != 64)
        Fail("rectangular external capacity identity changed");
    FabricConfig result;
    result.external_capacities.push_back(
        {String(external, "id"), "host:0", 0, 131072});
    std::set<uint64_t> hbm_dies, connection_dies;
    for (const auto &hbm : value.at("hbm_capacities")) {
        Exact(hbm, {"id", "tier", "location_ref", "base_address",
                    "capacity_bytes", "alignment_bytes"});
        const std::string location = String(hbm, "location_ref");
        if (String(hbm, "tier") != "hbm" ||
            location.rfind("die:", 0) != 0 ||
            Number(hbm, "base_address") != 0 ||
            Number(hbm, "capacity_bytes") != 12288 ||
            Number(hbm, "alignment_bytes") != 64)
            Fail("rectangular HBM capacity identity changed");
        const uint64_t die = std::stoull(location.substr(4));
        if (die > 3 || !hbm_dies.insert(die).second)
            Fail("rectangular HBM owners are not exact TP4 Dies");
        result.hbm_capacities.push_back(
            {String(hbm, "id"), die, 0, 12288});
    }
    for (const auto &link : value.at("links")) {
        Exact(link, {"id", "external_capacity_ref", "ingress_die_id",
                     "bytes_per_cycle", "latency_cycles", "queue_depth",
                     "max_outstanding", "duplex"});
        const uint64_t die = Number(link, "ingress_die_id");
        if (die != 0 ||
            String(link, "external_capacity_ref") != String(external, "id") ||
            Number(link, "bytes_per_cycle") != 256 ||
            Number(link, "latency_cycles") != 2 ||
            Number(link, "queue_depth") != 4 ||
            Number(link, "max_outstanding") != 4 ||
            String(link, "duplex") != "half_duplex_shared")
            Fail("rectangular shared external ingress changed");
        result.links.push_back({String(link, "id"), String(external, "id"),
                                die, 256, 2, 4, 4});
    }
    for (const auto &connection : value.at("connections")) {
        Exact(connection, {"id", "link_ref", "hbm_capacity_ref",
                           "target_die_id", "route_die_ids",
                           "route_latency_cycles", "route_bytes_per_cycle"});
        const uint64_t die = Number(connection, "target_die_id");
        const std::array<std::vector<uint64_t>, 4> expected_routes =
            rows == 2 && columns == 2
                ? std::array<std::vector<uint64_t>, 4>{
                      std::vector<uint64_t>{0}, std::vector<uint64_t>{0, 1},
                      std::vector<uint64_t>{0, 2},
                      std::vector<uint64_t>{0, 1, 3}}
                : std::array<std::vector<uint64_t>, 4>{
                      std::vector<uint64_t>{0}, std::vector<uint64_t>{0, 1},
                      std::vector<uint64_t>{0, 1, 2},
                      std::vector<uint64_t>{0, 1, 2, 3}};
        std::vector<uint64_t> route;
        if (!connection.at("route_die_ids").is_array())
            Fail("rectangular external route is not an array");
        for (const auto &hop : connection.at("route_die_ids"))
            route.push_back(hop.get<uint64_t>());
        const bool direct = die == 0;
        if (die > 3 || !connection_dies.insert(die).second ||
            route != expected_routes.at(die) ||
            Number(connection, "route_latency_cycles") != route.size() - 1 ||
            (direct ? !connection.at("route_bytes_per_cycle").is_null()
                    : Number(connection, "route_bytes_per_cycle") != 256))
            Fail("rectangular routed external connection changed");
        result.connections.push_back(
            {String(connection, "id"), String(connection, "link_ref"),
             String(connection, "hbm_capacity_ref"), die, route,
             route.size() - 1, direct ? std::nullopt
                                      : std::optional<uint64_t>(256)});
    }
    if (hbm_dies != std::set<uint64_t>({0, 1, 2, 3}) ||
        connection_dies != hbm_dies)
        Fail("rectangular fabric does not cover four Dies exactly");
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
    if (String(contract, "schema_version") ==
            "wafer_frontend.dense_inference_paged_runtime/v1alpha2") {
        Exact(contract, {"schema_version", "producer_pass", "id",
                         "physical_source_digest",
                         "external_materialization_digest",
                         "linked_manifest_ids", "linked_manifest_digests",
                         "mesh_rows", "mesh_columns",
                         "hbm_capacity_bytes_per_die",
                         "weight_slot_base_bytes_per_die",
                         "parameter_state_bytes", "kv_capacity_bytes",
                         "highest_paged_end_bytes_per_die", "fabric",
                         "spans", "events", "external_seed_hex"});
        const uint64_t rows = Number(contract, "mesh_rows");
        const uint64_t columns = Number(contract, "mesh_columns");
        if (String(contract, "producer_pass") !=
                "dense_inference_rect_paged_runtime" ||
            String(contract, "id") !=
                "dense_inference_rect_paged_runtime_" +
                    DigestWithoutId(contract).substr(0, 20) ||
            !((rows == 1 && columns == 4) ||
              (rows == 2 && columns == 2) ||
              (rows == 4 && columns == 1)) ||
            Number(contract, "hbm_capacity_bytes_per_die") != 12288 ||
            Number(contract, "weight_slot_base_bytes_per_die") != 1600 ||
            Number(contract, "parameter_state_bytes") != 107776 ||
            Number(contract, "kv_capacity_bytes") != 6144 ||
            Number(contract, "highest_paged_end_bytes_per_die") != 11328 ||
            manifest_texts.size() != 3)
            Fail("source-signed TP4 rectangular bounded identity changed");
        source_ref_ = String(contract, "id");
        const FabricConfig fabric = ParseRectFabric(contract.at("fabric"),
                                                    rows, columns);
        external_capacity_ref_ = fabric.external_capacities.front().id;
        if (backends.size() != 4)
            Fail("four physical TP4 HBM backends required");
        for (uint64_t die = 0; die < 4; ++die)
            if (!backends.count({die, 0}) || !backends.at({die, 0}))
                Fail("one physical stack0/channel0 backend per Die required");

        const auto &raw_spans = contract.at("spans");
        const auto &raw_events = contract.at("events");
        if (!raw_spans.is_array() || raw_spans.size() != 76 ||
            !raw_events.is_array() || raw_events.size() != 260)
            Fail("requires 60 physical weights, 16 KV pages and 260 gates");
        std::map<uint64_t, uint64_t> weight_count, kv_count;
        std::map<uint64_t, uint64_t> weight_bytes, kv_bytes;
        std::set<std::pair<uint64_t, uint64_t>> source_homes;
        std::set<uint64_t> external_homes;
        for (const auto &raw : raw_spans) {
            Exact(raw, {"kind", "state_ref", "die_id",
                        "runtime_core_id", "connection_ref",
                        "source_hbm_address", "external_address",
                        "hbm_address", "size_bytes"});
            DenseInferencePagerSpan span;
            span.kind = String(raw, "kind");
            span.state_ref = String(raw, "state_ref");
            span.die_id = Number(raw, "die_id");
            span.runtime_core_id = Number(raw, "runtime_core_id");
            span.connection_ref = String(raw, "connection_ref");
            span.source_hbm_address = Number(raw, "source_hbm_address");
            span.external_address = Number(raw, "external_address");
            span.hbm_address = Number(raw, "hbm_address");
            span.size_bytes = Number(raw, "size_bytes");
            if (span.die_id > 3 ||
                !source_homes.emplace(span.die_id,
                                      span.source_hbm_address).second ||
                !external_homes.insert(span.external_address).second ||
                span.size_bytes == 0)
                Fail("rectangular physical span identity repeats");
            if (span.kind == "parameter") {
                if (span.hbm_address != 1600 || span.size_bytes > 8192)
                    Fail("rectangular weight is not a one-slot page");
                ++weight_count[span.die_id];
                weight_bytes[span.die_id] += span.size_bytes;
                weights_.push_back(span);
            } else if (span.kind == "kv_key" || span.kind == "kv_value") {
                if (span.size_bytes != 384 || span.hbm_address < 9792 ||
                    span.hbm_address + span.size_bytes > 11328)
                    Fail("rectangular KV page is outside its stable slot");
                ++kv_count[span.die_id];
                kv_bytes[span.die_id] += span.size_bytes;
                kv_pinned_.emplace(
                    std::make_pair(span.runtime_core_id, span.hbm_address),
                    false);
                kv_pages_.push_back(span);
            } else {
                Fail("rectangular span has an unexpected persistent role");
            }
            weight_pinned_.emplace(span.runtime_core_id, false);
            awaiting_after_load_.emplace(span.runtime_core_id, false);
            next_event_by_core_.emplace(span.runtime_core_id, 0);
        }
        for (uint64_t die = 0; die < 4; ++die)
            if (weight_count[die] != 15 || kv_count[die] != 4 ||
                weight_bytes[die] != 26944 || kv_bytes[die] != 1536)
                Fail("rectangular per-Die 15+4 StateABI coverage changed");
        if (weight_pinned_.size() != 4)
            Fail("rectangular spans do not resolve four runtime cores");

        const auto &ids = contract.at("linked_manifest_ids");
        const auto &digests = contract.at("linked_manifest_digests");
        if (!ids.is_array() || ids.size() != 3 ||
            !digests.is_array() || digests.size() != 3)
            Fail("three rectangular linked identities required");
        for (size_t segment = 0; segment < 3; ++segment) {
            const auto parsed = frontend::ProgramArtifactFinalizer::Parse(
                manifest_texts[segment]);
            const std::vector<size_t> expected_records =
                segment == 0 ? std::vector<size_t>{375, 359, 359, 359}
                             : std::vector<size_t>{379, 363, 363, 363};
            std::vector<size_t> records;
            std::set<uint64_t> dies, cores;
            for (const auto &stream : parsed.core_streams) {
                dies.insert(stream.logical_core.die_id);
                cores.insert(stream.runtime_core_id);
                records.push_back(stream.records.size());
            }
            std::sort(records.begin(), records.end(), std::greater<size_t>());
            auto sorted_expected = expected_records;
            std::sort(sorted_expected.begin(), sorted_expected.end(),
                      std::greater<size_t>());
            if (parsed.id != ids[segment].get<std::string>() ||
                frontend::ProgramArtifactFinalizer::CanonicalManifestDigest(
                    manifest_texts[segment]) !=
                    digests[segment].get<std::string>() ||
                parsed.producer_pass != "manifest_linker" ||
                parsed.fragments.size() != (segment ? 196 : 180) ||
                parsed.core_streams.size() != 4 ||
                parsed.state_operand_bindings.size() !=
                    (segment ? 92 : 76) ||
                dies != std::set<uint64_t>({0, 1, 2, 3}) ||
                cores.size() != 4 || records != sorted_expected)
                Fail("production rectangular linked segment shape drifted");
        }

        std::map<std::tuple<uint64_t, uint64_t, uint64_t>, uint64_t> counts;
        uint64_t read_bytes = 0, write_bytes = 0;
        for (size_t index = 0; index < raw_events.size(); ++index) {
            const auto &raw = raw_events[index];
            Exact(raw, {"event_index", "segment_index",
                        "linked_record_index", "fragment_id",
                        "fragment_record_index", "kind", "state_ref",
                        "state_abi_id", "die_id", "runtime_core_id",
                        "connection_ref", "source_hbm_address",
                        "external_address", "hbm_address", "lsu_address",
                        "lsu_size_bytes", "dma_size_bytes"});
            DenseInferencePagerEvent event;
            event.event_index = Number(raw, "event_index");
            event.segment_index = Number(raw, "segment_index");
            event.linked_record_index = Number(raw, "linked_record_index");
            event.fragment_id = String(raw, "fragment_id");
            event.fragment_record_index = Number(raw, "fragment_record_index");
            event.kind = String(raw, "kind");
            event.state_ref = String(raw, "state_ref");
            event.state_abi_id = String(raw, "state_abi_id");
            event.die_id = Number(raw, "die_id");
            event.runtime_core_id = Number(raw, "runtime_core_id");
            event.connection_ref = String(raw, "connection_ref");
            event.source_hbm_address = Number(raw, "source_hbm_address");
            event.external_address = Number(raw, "external_address");
            event.hbm_address = Number(raw, "hbm_address");
            event.lsu_address = Number(raw, "lsu_address");
            event.lsu_size_bytes = Number(raw, "lsu_size_bytes");
            event.dma_size_bytes = Number(raw, "dma_size_bytes");
            if (event.event_index != index || event.segment_index > 2 ||
                event.die_id > 3 ||
                !next_event_by_core_.count(event.runtime_core_id))
                Fail("rectangular event identity is outside TP4 timeline");
            const auto span = std::find_if(
                raw_spans.begin(), raw_spans.end(), [&](const auto &item) {
                    return Number(item, "die_id") == event.die_id &&
                           Number(item, "source_hbm_address") ==
                               event.source_hbm_address;
                });
            if (span == raw_spans.end() ||
                Number(*span, "runtime_core_id") != event.runtime_core_id ||
                String(*span, "connection_ref") != event.connection_ref ||
                Number(*span, "external_address") != event.external_address ||
                Number(*span, "hbm_address") != event.hbm_address ||
                event.dma_size_bytes > Number(*span, "size_bytes"))
                Fail("rectangular event does not bind its signed physical span");
            uint64_t role = 0;
            if (event.kind == "weight_restore_before_load") {
                role = 0;
                if (event.dma_size_bytes != Number(*span, "size_bytes"))
                    Fail("rectangular weight event changed its physical extent");
                read_bytes += event.dma_size_bytes;
            } else if (event.kind == "kv_restore_before_load") {
                role = 1;
                if (event.dma_size_bytes != 256 + 64 * event.segment_index)
                    Fail("rectangular KV restore changed its version extent");
                read_bytes += event.dma_size_bytes;
            } else if (event.kind == "kv_writeback_after_store") {
                role = 2;
                if (event.dma_size_bytes != 256 + 64 * event.segment_index)
                    Fail("rectangular KV writeback changed its version extent");
                write_bytes += event.dma_size_bytes;
            } else {
                Fail("rectangular event has an unexpected page transition");
            }
            ++counts[{event.segment_index, event.die_id, role}];
            events_.push_back(event);
            event_indices_by_core_[event.runtime_core_id].push_back(index);
        }
        for (uint64_t segment = 0; segment < 3; ++segment)
            for (uint64_t die = 0; die < 4; ++die)
                if (counts[{segment, die, 0}] != 15 ||
                    counts[{segment, die, 1}] != (segment ? 4 : 0) ||
                    counts[{segment, die, 2}] != 4)
                    Fail("rectangular per-core segment gate coverage changed");
        if (read_bytes != 334592 || write_bytes != 15360)
            Fail("rectangular StateABI traffic byte oracle changed");

        const auto seed = Hex(String(contract, "external_seed_hex"));
        if (seed.size() != 131072)
            Fail("rectangular external seed capacity changed");
        for (const auto &span : weights_)
            if (std::all_of(seed.begin() + span.external_address,
                            seed.begin() + span.external_address + span.size_bytes,
                            [](uint8_t value) { return value == 0; }))
                Fail("a rectangular parameter source page is all zero");
        for (const auto &span : kv_pages_)
            if (std::any_of(seed.begin() + span.external_address,
                            seed.begin() + span.external_address + span.size_bytes,
                            [](uint8_t value) { return value != 0; }))
                Fail("rectangular initial KV authority is not empty");

        std::map<std::string, HBMBackend *> hbm;
        for (const auto &capacity : fabric.hbm_capacities)
            hbm.emplace(capacity.id,
                        backends.at({capacity.owner_die_id, 0}));
        runtime_ = std::make_unique<ExternalMemoryRuntimeBridge>(
            name, fabric, std::move(hbm), cycle_time_);
        runtime_->SeedExternal(external_capacity_ref_, 0, seed);
        kv_page_initial_bytes_ = 256;
        kv_page_step_bytes_ = 64;
        return;
    }
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
        DenseInferencePagerSpan span;
        span.kind = String(raw, "kind");
        span.state_ref = String(raw, "state_ref");
        span.source_hbm_address = Number(raw, "source_hbm_address");
        span.external_address = Number(raw, "external_address");
        span.hbm_address = Number(raw, "hbm_address");
        span.size_bytes = Number(raw, "size_bytes");
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
        DenseInferencePagerSpan span;
        span.kind = String(raw, "kind");
        span.state_ref = String(raw, "state_ref");
        span.source_hbm_address = Number(raw, "source_hbm_address");
        span.external_address = Number(raw, "external_address");
        span.hbm_address = Number(raw, "hbm_address");
        span.size_bytes = Number(raw, "size_bytes");
        if ((span.kind != "kv_key" && span.kind != "kv_value") ||
            !kv_source_addresses.insert(span.source_hbm_address).second ||
            !kv_hbm_addresses.insert(span.hbm_address).second ||
            span.size_bytes != 192 || span.hbm_address < 9792 ||
            span.hbm_address + 192 > 10560)
            Fail("KV page is not a stable disjoint 192B slot");
        kv_kinds.emplace(span.hbm_address, span.kind);
        kv_pinned_.emplace(std::make_pair(0, span.hbm_address), false);
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
        DenseInferencePagerEvent event;
        event.event_index = events_.size();
        event.segment_index = Number(raw, "segment_index");
        event.linked_record_index = Number(raw, "linked_record_index");
        event.fragment_id = String(raw, "fragment_id");
        event.fragment_record_index = Number(raw, "fragment_record_index");
        event.kind = String(raw, "kind");
        event.state_ref = String(raw, "state_ref");
        event.state_abi_id = String(raw, "state_abi_id");
        event.source_hbm_address = Number(raw, "source_hbm_address");
        event.external_address = Number(raw, "external_address");
        event.hbm_address = Number(raw, "hbm_address");
        event.lsu_address = Number(raw, "lsu_address");
        event.lsu_size_bytes = Number(raw, "lsu_size_bytes");
        event.dma_size_bytes = Number(raw, "dma_size_bytes");
        event.connection_ref = connection_ref_;
        events_.push_back(std::move(event));
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
    weight_pinned_.emplace(0, false);
    awaiting_after_load_.emplace(0, false);
    next_event_by_core_.emplace(0, 0);
    for (uint64_t index = 0; index < events_.size(); ++index)
        event_indices_by_core_[0].push_back(index);
}

const DenseInferencePagerEvent &DenseInferenceMidProgramPager::NextEvent(
    uint64_t runtime_core_id, const char *kind, uint64_t address,
    uint64_t size) const {
    const auto queue = event_indices_by_core_.find(runtime_core_id);
    const auto cursor = next_event_by_core_.find(runtime_core_id);
    if (queue == event_indices_by_core_.end() ||
        cursor == next_event_by_core_.end() ||
        cursor->second >= queue->second.size())
        Fail("runtime core issued an LSU beyond its signed gates");
    const auto &event = events_.at(queue->second.at(cursor->second));
    if (event.runtime_core_id != runtime_core_id || event.kind != kind ||
        event.lsu_address != address || event.lsu_size_bytes != size)
        Fail("actual LSU core/direction/address/size changed at gate " +
             std::to_string(event.event_index));
    return event;
}

void DenseInferenceMidProgramPager::Transfer(
    const DenseInferencePagerEvent &event, TransferDirection direction) {
    const std::string id = "dense_inference_paged_" +
        std::to_string(event.event_index) + "_" + event.state_abi_id;
    const auto quantum = cycle_time_.value();
    const auto remainder = sc_core::sc_time_stamp().value() % quantum;
    if (remainder != 0)
        sc_core::wait(sc_core::sc_time::from_value(quantum - remainder));
    const uint64_t cycle = sc_core::sc_time_stamp().value() / quantum;
    const std::string &connection = event.connection_ref.empty()
        ? connection_ref_ : event.connection_ref;
    runtime_->Submit({id, connection, direction, event.external_address,
                      event.hbm_address, event.dma_size_bytes, cycle,
                      kExternalDmaRequestSchemaVersion});
    const auto completion = runtime_->Wait(id);
    if (completion.status != 0 ||
        completion.payload_bytes != event.dma_size_bytes)
        Fail("actual shared DMA completion failed for " + event.state_ref +
             ": " + completion.error);
    std::cout << "[DENSE_INFERENCE_PAGED_DMA_EVENT] index="
              << event.event_index
              << " segment=" << event.segment_index
              << " linked_record=" << event.linked_record_index
              << " kind=" << event.kind
              << " state_ref=" << event.state_ref
              << " lsu_bytes=" << event.lsu_size_bytes
              << " dma_bytes=" << completion.payload_bytes
              << " issue_cycle=" << cycle
              << " completed_at_ticks=" << completion.completed_at.value()
              << " lsu_dependency_complete=1 pass=1"
              << " die=" << event.die_id
              << " runtime_core=" << event.runtime_core_id << std::endl;
}

void DenseInferenceMidProgramPager::BeforeLoad(
    uint64_t runtime_core_id, uint64_t address, uint64_t size) {
    if (awaiting_after_load_.at(runtime_core_id))
        Fail("previous LSU Load has no completion hook");
    const auto queue = event_indices_by_core_.find(runtime_core_id);
    const auto cursor = next_event_by_core_.find(runtime_core_id);
    if (queue == event_indices_by_core_.end() ||
        cursor == next_event_by_core_.end() ||
        cursor->second >= queue->second.size())
        Fail("weight/KV load beyond signed core sequence");
    const auto &pending = events_.at(queue->second.at(cursor->second));
    const char *kind = pending.kind == "weight_restore_before_load"
                           ? "weight_restore_before_load"
                           : "kv_restore_before_load";
    const auto &event = NextEvent(runtime_core_id, kind, address, size);
    if (event.kind == "weight_restore_before_load") {
        if (weight_pinned_.at(runtime_core_id))
            Fail("weight slot reused before old LSU completion");
        weight_pinned_.at(runtime_core_id) = true;
    } else {
        const auto key = std::make_pair(runtime_core_id, event.hbm_address);
        if (kv_pinned_.at(key))
            Fail("KV page restored twice without dirty writeback");
        kv_pinned_.at(key) = true;
    }
    Transfer(event, TransferDirection::kExternalToHbm);
    awaiting_after_load_.at(runtime_core_id) = true;
    ++next_event_by_core_.at(runtime_core_id);
    ++completed_events_;
}

void DenseInferenceMidProgramPager::AfterLoad(
    uint64_t runtime_core_id, uint64_t address, uint64_t size) {
    const auto position = next_event_by_core_.at(runtime_core_id);
    if (!awaiting_after_load_.at(runtime_core_id) || position == 0)
        Fail("LSU Load completion lacks its page restore");
    const auto &event = events_.at(
        event_indices_by_core_.at(runtime_core_id).at(position - 1));
    if (event.lsu_address != address || event.lsu_size_bytes != size)
        Fail("completed LSU Load differs from its restore gate");
    if (event.kind == "weight_restore_before_load") {
        if (!weight_pinned_.at(runtime_core_id))
            Fail("weight slot pin vanished early");
        weight_pinned_.at(runtime_core_id) = false;
    } else {
        const auto key = std::make_pair(runtime_core_id, event.hbm_address);
        if (event.kind != "kv_restore_before_load" ||
            !kv_pinned_.at(key))
            Fail("KV Load page pin vanished before suffix Store");
    }
    awaiting_after_load_.at(runtime_core_id) = false;
}

void DenseInferenceMidProgramPager::AfterStore(
    uint64_t runtime_core_id, uint64_t address, uint64_t size) {
    if (awaiting_after_load_.at(runtime_core_id) ||
        weight_pinned_.at(runtime_core_id))
        Fail("KV Store raced with incomplete weight/KV Load");
    const auto &event = NextEvent(runtime_core_id,
                                  "kv_writeback_after_store",
                                  address, size);
    const auto key = std::make_pair(runtime_core_id, event.hbm_address);
    if (event.segment_index != 0 && !kv_pinned_.at(key))
        Fail("Decode KV suffix Store lacks full-prefix restore");
    ++dirty_;
    Transfer(event, TransferDirection::kHbmToExternal);
    --dirty_;
    kv_pinned_.at(key) = false;
    ++next_event_by_core_.at(runtime_core_id);
    ++completed_events_;
}

std::vector<uint8_t> DenseInferenceMidProgramPager::ProbeKvPages(
    uint64_t page_bytes) const {
    auto ordered = kv_pages_;
    std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
        return std::tie(a.die_id, a.kind, a.hbm_address) <
               std::tie(b.die_id, b.kind, b.hbm_address);
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
    if (completed_events_ != 0 || Pending() != 0 || Pinned() != 0)
        Fail("initial KV authority must precede all compute/DMA");
    const uint64_t capacity_bytes =
        kv_page_initial_bytes_ + 2 * kv_page_step_bytes_;
    const auto capacity = ProbeKvPages(capacity_bytes);
    if (capacity.size() != kv_pages_.size() * capacity_bytes ||
        std::any_of(capacity.begin(), capacity.end(),
                    [](uint8_t value) { return value != 0; }))
        Fail("initial external KV future page capacity is not empty");
    return frontend::program_io::Sha256Hex(std::vector<uint8_t>{});
}

void DenseInferenceMidProgramPager::CompleteSegment(uint64_t segment) {
    if (segment > 2 || Pending() != 0 || Pinned() != 0 || dirty_ != 0)
        Fail("segment ended with incomplete DMA, dirty state or active pin");
    for (const auto &[core, indices] : event_indices_by_core_) {
        const uint64_t expected = std::count_if(
            indices.begin(), indices.end(), [&](uint64_t index) {
                return events_.at(index).segment_index <= segment;
            });
        if (next_event_by_core_.at(core) != expected ||
            awaiting_after_load_.at(core))
            Fail("a physical core ended outside its signed segment boundary");
    }
    const uint64_t page_bytes =
        kv_page_initial_bytes_ + kv_page_step_bytes_ * segment;
    const auto authority = ProbeKvPages(page_bytes);
    external_kv_probes_ += kv_pages_.size();
    kv_bytes_ = authority.size();
    if (kv_bytes_ != kv_pages_.size() * page_bytes ||
        external_kv_probes_ != kv_pages_.size() * (segment + 1))
        Fail("external KV authority pages did not advance exact extent");
    kv_digest_ = frontend::program_io::Sha256Hex(authority);
}

uint64_t DenseInferenceMidProgramPager::Pending() const {
    return runtime_->Outstanding();
}

uint64_t DenseInferenceMidProgramPager::Pinned() const {
    return std::count_if(weight_pinned_.begin(), weight_pinned_.end(),
                         [](const auto &item) { return item.second; }) +
           std::count_if(kv_pinned_.begin(), kv_pinned_.end(),
                         [](const auto &item) { return item.second; });
}

const RuntimeStats &DenseInferenceMidProgramPager::Stats() const {
    return runtime_->Stats();
}
} // namespace external_memory
