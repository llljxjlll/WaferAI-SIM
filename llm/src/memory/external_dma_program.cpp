#include "memory/external_dma_program.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace external_memory {
namespace {

using Json = nlohmann::json;

[[noreturn]] void Fail(const std::string &path,
                       const std::string &message) {
    throw std::invalid_argument(path + ": " + message);
}

void ExactObject(
    const Json &value, const std::string &path,
    std::initializer_list<const char *> fields) {
    if (!value.is_object()) Fail(path, "must be an object");
    std::set<std::string> expected;
    for (const char *field : fields) expected.emplace(field);
    for (const auto &item : value.items())
        if (expected.count(item.key()) == 0)
            Fail(path + "." + item.key(), "unknown field");
    for (const auto &field : expected)
        if (!value.contains(field))
            Fail(path + "." + field, "missing required field");
}

const Json &Field(const Json &value, const char *name,
                  const std::string &path) {
    if (!value.contains(name))
        Fail(path + "." + name, "missing required field");
    return value.at(name);
}

std::string String(const Json &value, const std::string &path) {
    if (!value.is_string()) Fail(path, "must be a string");
    const std::string result = value.get<std::string>();
    if (result.empty()) Fail(path, "must not be empty");
    return result;
}

uint64_t U64(const Json &value, const std::string &path) {
    if (!value.is_number_unsigned() &&
        !(value.is_number_integer() && value.get<int64_t>() >= 0))
        Fail(path, "must be an unsigned 64-bit integer");
    try {
        return value.get<uint64_t>();
    } catch (const std::exception &) {
        Fail(path, "must be an unsigned 64-bit integer");
    }
}

const Json &Array(const Json &value, const std::string &path) {
    if (!value.is_array()) Fail(path, "must be an array");
    return value;
}

void RequireDigest(const std::string &value,
                   const std::string &path) {
    if (value.size() != 64 ||
        !std::all_of(
            value.begin(), value.end(), [](char character) {
                return (character >= '0' && character <= '9') ||
                       (character >= 'a' && character <= 'f');
            }))
        Fail(path, "must be a lowercase SHA-256 digest");
}

std::vector<uint8_t> HexPayload(
    const Json &value, const std::string &path) {
    const std::string encoded = String(value, path);
    if (encoded.size() % 2 != 0)
        Fail(path, "must have even length");
    auto digit = [&](char character) -> uint8_t {
        if (character >= '0' && character <= '9')
            return static_cast<uint8_t>(character - '0');
        if (character >= 'a' && character <= 'f')
            return static_cast<uint8_t>(character - 'a' + 10);
        Fail(path, "must use lowercase hexadecimal");
    };
    std::vector<uint8_t> result;
    result.reserve(encoded.size() / 2);
    for (size_t index = 0; index < encoded.size(); index += 2)
        result.push_back(static_cast<uint8_t>(
            digit(encoded[index]) * 16 + digit(encoded[index + 1])));
    if (result.empty()) Fail(path, "must not be empty");
    return result;
}

std::vector<std::string> StringArray(
    const Json &value, const std::string &path) {
    Array(value, path);
    std::vector<std::string> result;
    std::set<std::string> seen;
    for (size_t index = 0; index < value.size(); ++index) {
        const std::string item = String(
            value[index], path + "[" + std::to_string(index) + "]");
        if (!seen.insert(item).second)
            Fail(path, "contains duplicate entries");
        result.push_back(item);
    }
    return result;
}

uint64_t OwnerDie(const std::string &location,
                  const std::string &path) {
    constexpr const char *prefix = "die:";
    if (location.rfind(prefix, 0) != 0)
        Fail(path, "HBM owner must use die:<id>");
    const std::string raw = location.substr(4);
    if (raw.empty() ||
        !std::all_of(raw.begin(), raw.end(), [](char character) {
            return character >= '0' && character <= '9';
        }) ||
        (raw.size() > 1 && raw.front() == '0'))
        Fail(path, "HBM owner must use canonical die:<id>");
    try {
        size_t consumed = 0;
        const uint64_t value = std::stoull(raw, &consumed);
        if (consumed != raw.size())
            Fail(path, "HBM owner must use canonical die:<id>");
        return value;
    } catch (const std::exception &) {
        Fail(path, "HBM owner die exceeds uint64");
    }
}

void ValidateRange(uint64_t address, uint64_t size_bytes,
                   uint64_t base, uint64_t capacity,
                   const std::string &path) {
    if (size_bytes == 0 ||
        address > std::numeric_limits<uint64_t>::max() - size_bytes ||
        base > std::numeric_limits<uint64_t>::max() - capacity ||
        address < base || address + size_bytes > base + capacity)
        Fail(path, "byte range exceeds capacity");
}

FabricConfig ParseFabric(const Json &value,
                         const std::string &path) {
    ExactObject(
        value, path,
        {"schema_version", "producer_pass", "id",
         "external_capacities", "hbm_capacities",
         "links", "connections"});
    if (String(Field(value, "schema_version", path),
               path + ".schema_version") !=
        "wafer_frontend.external_memory_fabric/v1alpha2")
        Fail(path + ".schema_version", "unsupported schema version");
    if (String(Field(value, "producer_pass", path),
               path + ".producer_pass") !=
        "external_memory_fabric_builder")
        Fail(path + ".producer_pass", "unexpected producer");
    String(Field(value, "id", path), path + ".id");

    FabricConfig result;
    const Json &external = Array(
        Field(value, "external_capacities", path),
        path + ".external_capacities");
    for (size_t index = 0; index < external.size(); ++index) {
        const std::string item_path =
            path + ".external_capacities[" +
            std::to_string(index) + "]";
        const Json &item = external[index];
        ExactObject(
            item, item_path,
            {"id", "tier", "location_ref", "base_address",
             "capacity_bytes", "alignment_bytes"});
        if (String(Field(item, "tier", item_path),
                   item_path + ".tier") != "external")
            Fail(item_path + ".tier", "must be external");
        ExternalCapacityConfig capacity;
        capacity.id = String(
            Field(item, "id", item_path), item_path + ".id");
        capacity.owner_ref = String(
            Field(item, "location_ref", item_path),
            item_path + ".location_ref");
        capacity.base_address = U64(
            Field(item, "base_address", item_path),
            item_path + ".base_address");
        capacity.capacity_bytes = U64(
            Field(item, "capacity_bytes", item_path),
            item_path + ".capacity_bytes");
        U64(Field(item, "alignment_bytes", item_path),
            item_path + ".alignment_bytes");
        result.external_capacities.push_back(std::move(capacity));
    }

    const Json &hbm = Array(
        Field(value, "hbm_capacities", path),
        path + ".hbm_capacities");
    for (size_t index = 0; index < hbm.size(); ++index) {
        const std::string item_path =
            path + ".hbm_capacities[" +
            std::to_string(index) + "]";
        const Json &item = hbm[index];
        ExactObject(
            item, item_path,
            {"id", "tier", "location_ref", "base_address",
             "capacity_bytes", "alignment_bytes"});
        if (String(Field(item, "tier", item_path),
                   item_path + ".tier") != "hbm")
            Fail(item_path + ".tier", "must be hbm");
        HbmCapacityConfig capacity;
        capacity.id = String(
            Field(item, "id", item_path), item_path + ".id");
        const std::string owner = String(
            Field(item, "location_ref", item_path),
            item_path + ".location_ref");
        capacity.owner_die_id = OwnerDie(
            owner, item_path + ".location_ref");
        capacity.base_address = U64(
            Field(item, "base_address", item_path),
            item_path + ".base_address");
        capacity.capacity_bytes = U64(
            Field(item, "capacity_bytes", item_path),
            item_path + ".capacity_bytes");
        U64(Field(item, "alignment_bytes", item_path),
            item_path + ".alignment_bytes");
        result.hbm_capacities.push_back(std::move(capacity));
    }

    const Json &links = Array(
        Field(value, "links", path), path + ".links");
    for (size_t index = 0; index < links.size(); ++index) {
        const std::string item_path =
            path + ".links[" + std::to_string(index) + "]";
        const Json &item = links[index];
        ExactObject(
            item, item_path,
            {"id", "external_capacity_ref", "ingress_die_id",
             "bytes_per_cycle", "latency_cycles", "queue_depth",
             "max_outstanding", "duplex"});
        if (String(Field(item, "duplex", item_path),
                   item_path + ".duplex") != "half_duplex_shared")
            Fail(item_path + ".duplex", "unsupported duplex mode");
        result.links.push_back(
            {String(Field(item, "id", item_path),
                    item_path + ".id"),
             String(Field(item, "external_capacity_ref", item_path),
                    item_path + ".external_capacity_ref"),
             U64(Field(item, "ingress_die_id", item_path),
                 item_path + ".ingress_die_id"),
             U64(Field(item, "bytes_per_cycle", item_path),
                 item_path + ".bytes_per_cycle"),
             U64(Field(item, "latency_cycles", item_path),
                 item_path + ".latency_cycles"),
             U64(Field(item, "queue_depth", item_path),
                 item_path + ".queue_depth"),
             U64(Field(item, "max_outstanding", item_path),
                 item_path + ".max_outstanding")});
    }

    const Json &connections = Array(
        Field(value, "connections", path),
        path + ".connections");
    for (size_t index = 0; index < connections.size(); ++index) {
        const std::string item_path =
            path + ".connections[" + std::to_string(index) + "]";
        const Json &item = connections[index];
        ExactObject(
            item, item_path,
            {"id", "link_ref", "hbm_capacity_ref",
             "target_die_id", "route_die_ids",
             "route_latency_cycles", "route_bytes_per_cycle"});
        ConnectionConfig connection;
        connection.id = String(
            Field(item, "id", item_path), item_path + ".id");
        connection.link_ref = String(
            Field(item, "link_ref", item_path),
            item_path + ".link_ref");
        connection.hbm_capacity_ref = String(
            Field(item, "hbm_capacity_ref", item_path),
            item_path + ".hbm_capacity_ref");
        connection.target_die_id = U64(
            Field(item, "target_die_id", item_path),
            item_path + ".target_die_id");
        const Json &route = Array(
            Field(item, "route_die_ids", item_path),
            item_path + ".route_die_ids");
        for (size_t route_index = 0;
             route_index < route.size(); ++route_index)
            connection.route_die_ids.push_back(U64(
                route[route_index],
                item_path + ".route_die_ids[" +
                    std::to_string(route_index) + "]"));
        connection.route_latency_cycles = U64(
            Field(item, "route_latency_cycles", item_path),
            item_path + ".route_latency_cycles");
        const Json &route_bandwidth =
            Field(item, "route_bytes_per_cycle", item_path);
        if (!route_bandwidth.is_null())
            connection.route_bytes_per_cycle = U64(
                route_bandwidth,
                item_path + ".route_bytes_per_cycle");
        result.connections.push_back(std::move(connection));
    }
    ValidateFabricConfig(result);
    return result;
}

DmaBackendBinding ParseBinding(
    const Json &value, const std::string &path) {
    ExactObject(
        value, path,
        {"id", "hbm_capacity_ref", "owner_die_id",
         "stack_id", "channel_id"});
    return {
        String(Field(value, "id", path), path + ".id"),
        String(Field(value, "hbm_capacity_ref", path),
               path + ".hbm_capacity_ref"),
        U64(Field(value, "owner_die_id", path),
            path + ".owner_die_id"),
        U64(Field(value, "stack_id", path), path + ".stack_id"),
        U64(Field(value, "channel_id", path), path + ".channel_id")};
}

TransferDirection ParseDirection(
    const Json &value, const std::string &path) {
    const std::string direction = String(value, path);
    if (direction == "external_to_hbm")
        return TransferDirection::kExternalToHbm;
    if (direction == "hbm_to_external")
        return TransferDirection::kHbmToExternal;
    Fail(path, "unsupported transfer direction");
}

DmaDescriptor ParseDescriptor(
    const Json &value, const std::string &path) {
    ExactObject(
        value, path,
        {"id", "sequence", "operation_ref",
         "source_transfer_request_ref", "source_operation_deps",
         "depends_on", "request_schema_version", "connection_ref",
         "direction", "external_address", "hbm_address",
         "size_bytes", "planned_issue_cycle", "planned_ready_cycle"});
    DmaDescriptor result;
    result.id = String(Field(value, "id", path), path + ".id");
    result.sequence =
        U64(Field(value, "sequence", path), path + ".sequence");
    result.operation_ref = String(
        Field(value, "operation_ref", path),
        path + ".operation_ref");
    result.source_transfer_request_ref = String(
        Field(value, "source_transfer_request_ref", path),
        path + ".source_transfer_request_ref");
    result.source_operation_deps = StringArray(
        Field(value, "source_operation_deps", path),
        path + ".source_operation_deps");
    result.depends_on = StringArray(
        Field(value, "depends_on", path), path + ".depends_on");
    result.request_schema_version = String(
        Field(value, "request_schema_version", path),
        path + ".request_schema_version");
    if (result.request_schema_version !=
        kExternalDmaRequestSchemaVersion)
        Fail(path + ".request_schema_version",
             "unsupported request schema version");
    result.connection_ref = String(
        Field(value, "connection_ref", path),
        path + ".connection_ref");
    result.direction = ParseDirection(
        Field(value, "direction", path), path + ".direction");
    result.external_address = U64(
        Field(value, "external_address", path),
        path + ".external_address");
    result.hbm_address = U64(
        Field(value, "hbm_address", path),
        path + ".hbm_address");
    result.size_bytes = U64(
        Field(value, "size_bytes", path), path + ".size_bytes");
    result.planned_issue_cycle = U64(
        Field(value, "planned_issue_cycle", path),
        path + ".planned_issue_cycle");
    result.planned_ready_cycle = U64(
        Field(value, "planned_ready_cycle", path),
        path + ".planned_ready_cycle");
    if (result.size_bytes == 0 ||
        result.planned_ready_cycle <= result.planned_issue_cycle)
        Fail(path, "descriptor size/timing is invalid");
    return result;
}

DmaExternalSeed ParseSeed(
    const Json &value, const std::string &path) {
    ExactObject(
        value, path,
        {"id", "external_capacity_ref", "address", "payload_hex"});
    return {
        String(Field(value, "id", path), path + ".id"),
        String(Field(value, "external_capacity_ref", path),
               path + ".external_capacity_ref"),
        U64(Field(value, "address", path), path + ".address"),
        HexPayload(Field(value, "payload_hex", path),
                   path + ".payload_hex")};
}

DmaExternalProbe ParseProbe(
    const Json &value, const std::string &path) {
    ExactObject(
        value, path,
        {"id", "external_capacity_ref", "address",
         "expected_payload_hex"});
    return {
        String(Field(value, "id", path), path + ".id"),
        String(Field(value, "external_capacity_ref", path),
               path + ".external_capacity_ref"),
        U64(Field(value, "address", path), path + ".address"),
        HexPayload(Field(value, "expected_payload_hex", path),
                   path + ".expected_payload_hex")};
}

template <typename Item>
void RequireCanonicalIo(
    const std::vector<Item> &items, const std::string &path) {
    if (items.empty()) Fail(path, "must not be empty");
    for (size_t index = 1; index < items.size(); ++index) {
        const auto prior = std::tie(
            items[index - 1].external_capacity_ref,
            items[index - 1].address,
            items[index - 1].id);
        const auto current = std::tie(
            items[index].external_capacity_ref,
            items[index].address,
            items[index].id);
        if (!(prior < current))
            Fail(path, "must use unique canonical address order");
    }
}

void ValidateProgram(ExternalDmaProgram &program,
                     const ExternalDmaExpectedSource &expected) {
    for (const auto &entry : {
             std::pair{program.case_digest, "case_digest"},
             std::pair{program.request_digest, "request_digest"},
             std::pair{program.logical_graph_digest,
                       "logical_graph_digest"},
             std::pair{program.source_memory_plan_digest,
                       "source_memory_plan_digest"},
             std::pair{program.blocking_offload_plan_digest,
                       "blocking_offload_plan_digest"}})
        RequireDigest(entry.first,
                      "external_dma_program." +
                          std::string(entry.second));
    const auto require_match = [&](const std::string &actual,
                                   const std::string &wanted,
                                   const char *field) {
        RequireDigest(wanted, std::string("expected_source.") + field);
        if (actual != wanted)
            Fail(std::string("external_dma_program.") + field,
                 "source binding mismatch");
    };
    require_match(
        program.case_digest, expected.case_digest, "case_digest");
    require_match(
        program.request_digest, expected.request_digest,
        "request_digest");
    require_match(
        program.logical_graph_digest, expected.logical_graph_digest,
        "logical_graph_digest");
    require_match(
        program.source_memory_plan_digest,
        expected.source_memory_plan_digest,
        "source_memory_plan_digest");
    require_match(
        program.blocking_offload_plan_digest,
        expected.blocking_offload_plan_digest,
        "blocking_offload_plan_digest");

    std::map<std::string, uint64_t> hbm_owner;
    for (const auto &capacity : program.fabric.hbm_capacities)
        hbm_owner.emplace(capacity.id, capacity.owner_die_id);
    std::set<std::string> bound_capacities;
    std::set<HbmEndpoint> bound_endpoints;
    std::set<std::string> binding_ids;
    for (const auto &binding : program.backend_bindings) {
        if (binding.stack_id >
                static_cast<uint64_t>(
                    std::numeric_limits<int>::max()) ||
            binding.channel_id >
                static_cast<uint64_t>(
                    std::numeric_limits<int>::max()))
            Fail("external_dma_program.backend_bindings",
                 "stack/channel exceeds HBMRuntime integer range");
        const auto capacity = hbm_owner.find(
            binding.hbm_capacity_ref);
        if (capacity == hbm_owner.end())
            Fail("external_dma_program.backend_bindings",
                 "references unknown HBM capacity");
        if (capacity->second != binding.owner_die_id)
            Fail("external_dma_program.backend_bindings",
                 "owner differs from HBM capacity");
        if (!binding_ids.insert(binding.id).second ||
            !bound_capacities.insert(
                binding.hbm_capacity_ref).second ||
            !bound_endpoints.insert(
                {binding.stack_id, binding.channel_id}).second)
            Fail("external_dma_program.backend_bindings",
                 "capacity or endpoint is bound more than once");
    }
    if (bound_capacities.size() != hbm_owner.size())
        Fail("external_dma_program.backend_bindings",
             "must exactly cover HBM capacities");
    for (size_t index = 1;
         index < program.backend_bindings.size(); ++index)
        if (program.backend_bindings[index - 1].hbm_capacity_ref >=
            program.backend_bindings[index].hbm_capacity_ref)
            Fail("external_dma_program.backend_bindings",
                 "must use canonical capacity order");

    if (program.descriptors.empty())
        Fail("external_dma_program.descriptors",
             "must contain at least one transfer");
    std::set<std::string> descriptor_ids;
    std::set<std::string> operation_refs;
    std::set<std::string> transfer_refs;
    std::map<std::string, const ConnectionConfig *> connections;
    std::map<std::string, const LinkConfig *> links;
    std::map<std::string, const ExternalCapacityConfig *> external;
    std::map<std::string, const HbmCapacityConfig *> hbm;
    for (const auto &item : program.fabric.connections)
        connections.emplace(item.id, &item);
    for (const auto &item : program.fabric.links)
        links.emplace(item.id, &item);
    for (const auto &item : program.fabric.external_capacities)
        external.emplace(item.id, &item);
    for (const auto &item : program.fabric.hbm_capacities)
        hbm.emplace(item.id, &item);
    for (size_t index = 0;
         index < program.descriptors.size(); ++index) {
        const auto &descriptor = program.descriptors[index];
        if (descriptor.sequence != index)
            Fail("external_dma_program.descriptors",
                 "sequence must be contiguous from zero");
        const std::vector<std::string> expected_deps =
            index == 0
                ? std::vector<std::string>{}
                : std::vector<std::string>{
                      program.descriptors[index - 1].id};
        if (descriptor.depends_on != expected_deps)
            Fail("external_dma_program.descriptors",
                 "blocking dependency must name prior descriptor");
        if (index > 0 &&
            descriptor.planned_issue_cycle <
                program.descriptors[index - 1].planned_ready_cycle)
            Fail("external_dma_program.descriptors",
                 "planned transfer starts before its dependency");
        if (!descriptor_ids.insert(descriptor.id).second ||
            !operation_refs.insert(
                descriptor.operation_ref).second ||
            !transfer_refs.insert(
                descriptor.source_transfer_request_ref).second)
            Fail("external_dma_program.descriptors",
                 "descriptor provenance must be unique");
        const auto connection = connections.find(
            descriptor.connection_ref);
        if (connection == connections.end())
            Fail("external_dma_program.descriptors",
                 "references unknown connection");
        const auto link = links.at(connection->second->link_ref);
        const auto external_capacity =
            external.at(link->external_capacity_ref);
        const auto hbm_capacity =
            hbm.at(connection->second->hbm_capacity_ref);
        ValidateRange(
            descriptor.external_address, descriptor.size_bytes,
            external_capacity->base_address,
            external_capacity->capacity_bytes,
            "external_dma_program.descriptors.external_address");
        ValidateRange(
            descriptor.hbm_address, descriptor.size_bytes,
            hbm_capacity->base_address,
            hbm_capacity->capacity_bytes,
            "external_dma_program.descriptors.hbm_address");
    }

    RequireCanonicalIo(
        program.external_seeds,
        "external_dma_program.external_seeds");
    RequireCanonicalIo(
        program.external_probes,
        "external_dma_program.external_probes");
    const auto require_nonoverlap = [](
        const auto &items, const std::string &path,
        const auto &payload) {
        std::map<std::string,
                 std::vector<std::pair<uint64_t, uint64_t>>> ranges;
        std::set<std::string> ids;
        for (const auto &item : items) {
            if (!ids.insert(item.id).second)
                Fail(path, "record IDs must be unique");
            const uint64_t size = payload(item).size();
            if (item.address >
                std::numeric_limits<uint64_t>::max() - size)
                Fail(path, "byte range overflows uint64");
            const uint64_t end = item.address + size;
            auto &capacity_ranges =
                ranges[item.external_capacity_ref];
            for (const auto &prior : capacity_ranges)
                if (item.address < prior.second &&
                    prior.first < end)
                    Fail(path, "records overlap");
            capacity_ranges.emplace_back(item.address, end);
        }
    };
    require_nonoverlap(
        program.external_seeds,
        "external_dma_program.external_seeds",
        [](const DmaExternalSeed &item)
            -> const std::vector<uint8_t> & {
            return item.payload;
        });
    require_nonoverlap(
        program.external_probes,
        "external_dma_program.external_probes",
        [](const DmaExternalProbe &item)
            -> const std::vector<uint8_t> & {
            return item.expected_payload;
        });
    for (const auto &seed : program.external_seeds) {
        const auto capacity = external.find(
            seed.external_capacity_ref);
        if (capacity == external.end())
            Fail("external_dma_program.external_seeds",
                 "references unknown external capacity");
        ValidateRange(
            seed.address, seed.payload.size(),
            capacity->second->base_address,
            capacity->second->capacity_bytes,
            "external_dma_program.external_seeds");
    }
    for (const auto &probe : program.external_probes) {
        const auto capacity = external.find(
            probe.external_capacity_ref);
        if (capacity == external.end())
            Fail("external_dma_program.external_probes",
                 "references unknown external capacity");
        ValidateRange(
            probe.address, probe.expected_payload.size(),
            capacity->second->base_address,
            capacity->second->capacity_bytes,
            "external_dma_program.external_probes");
    }
}

std::map<std::string, HBMBackend *> ResolveBackends(
    const ExternalDmaProgram &program,
    const std::map<HbmEndpoint, HBMBackend *> &endpoints) {
    std::map<std::string, HBMBackend *> result;
    for (const auto &binding : program.backend_bindings) {
        const auto found = endpoints.find(
            {binding.stack_id, binding.channel_id});
        if (found == endpoints.end() || found->second == nullptr)
            throw std::invalid_argument(
                "external DMA backend binding cannot be resolved");
        result.emplace(binding.hbm_capacity_ref, found->second);
    }
    return result;
}

uint64_t CurrentCycle(sc_core::sc_time cycle_time) {
    return sc_core::sc_time_stamp().value() / cycle_time.value();
}

void AlignToCycle(sc_core::sc_time cycle_time) {
    const auto now = sc_core::sc_time_stamp().value();
    const auto quantum = cycle_time.value();
    const auto remainder = now % quantum;
    if (remainder != 0)
        sc_core::wait(sc_core::sc_time::from_value(
            quantum - remainder));
}

} // namespace

ExternalDmaProgram ParseExternalDmaProgram(
    std::string_view json,
    const ExternalDmaExpectedSource &expected_source) {
    Json value;
    try {
        std::vector<std::set<std::string>> object_keys;
        auto callback = [&object_keys](
                            int, Json::parse_event_t event,
                            Json &parsed) {
            if (event == Json::parse_event_t::object_start) {
                object_keys.emplace_back();
            } else if (event == Json::parse_event_t::key) {
                if (object_keys.empty())
                    throw std::invalid_argument(
                        "invalid duplicate-key parser state");
                const std::string key =
                    parsed.get<std::string>();
                if (!object_keys.back().insert(key).second)
                    throw std::invalid_argument(
                        "duplicate object key " + key);
            } else if (event == Json::parse_event_t::object_end) {
                if (object_keys.empty())
                    throw std::invalid_argument(
                        "invalid duplicate-key parser state");
                object_keys.pop_back();
            }
            return true;
        };
        value = Json::parse(
            json.begin(), json.end(), callback, true, false);
        if (value.is_discarded())
            throw std::invalid_argument("invalid JSON");
    } catch (const std::exception &error) {
        throw std::invalid_argument(
            std::string("external DMA program JSON: ") +
            error.what());
    }
    const std::string path = "external_dma_program";
    ExactObject(
        value, path,
        {"schema_version", "producer_pass", "id", "case_digest",
         "request_digest", "logical_graph_digest",
         "source_memory_plan_digest", "blocking_offload_plan_id",
         "blocking_offload_plan_digest", "fabric",
         "backend_bindings", "descriptors",
         "external_seeds", "external_probes"});
    ExternalDmaProgram result;
    result.schema_version = String(
        Field(value, "schema_version", path),
        path + ".schema_version");
    if (result.schema_version != kExternalDmaProgramSchemaVersion)
        Fail(path + ".schema_version", "unsupported schema version");
    result.producer_pass = String(
        Field(value, "producer_pass", path),
        path + ".producer_pass");
    if (result.producer_pass != "external_dma_program_finalizer")
        Fail(path + ".producer_pass", "unexpected producer");
    result.id = String(Field(value, "id", path), path + ".id");
    result.case_digest = String(
        Field(value, "case_digest", path),
        path + ".case_digest");
    result.request_digest = String(
        Field(value, "request_digest", path),
        path + ".request_digest");
    result.logical_graph_digest = String(
        Field(value, "logical_graph_digest", path),
        path + ".logical_graph_digest");
    result.source_memory_plan_digest = String(
        Field(value, "source_memory_plan_digest", path),
        path + ".source_memory_plan_digest");
    result.blocking_offload_plan_id = String(
        Field(value, "blocking_offload_plan_id", path),
        path + ".blocking_offload_plan_id");
    result.blocking_offload_plan_digest = String(
        Field(value, "blocking_offload_plan_digest", path),
        path + ".blocking_offload_plan_digest");
    result.fabric = ParseFabric(
        Field(value, "fabric", path), path + ".fabric");

    const Json &bindings = Array(
        Field(value, "backend_bindings", path),
        path + ".backend_bindings");
    for (size_t index = 0; index < bindings.size(); ++index)
        result.backend_bindings.push_back(ParseBinding(
            bindings[index],
            path + ".backend_bindings[" +
                std::to_string(index) + "]"));
    const Json &descriptors = Array(
        Field(value, "descriptors", path),
        path + ".descriptors");
    for (size_t index = 0; index < descriptors.size(); ++index)
        result.descriptors.push_back(ParseDescriptor(
            descriptors[index],
            path + ".descriptors[" +
                std::to_string(index) + "]"));
    const Json &seeds = Array(
        Field(value, "external_seeds", path),
        path + ".external_seeds");
    for (size_t index = 0; index < seeds.size(); ++index)
        result.external_seeds.push_back(ParseSeed(
            seeds[index],
            path + ".external_seeds[" +
                std::to_string(index) + "]"));
    const Json &probes = Array(
        Field(value, "external_probes", path),
        path + ".external_probes");
    for (size_t index = 0; index < probes.size(); ++index)
        result.external_probes.push_back(ParseProbe(
            probes[index],
            path + ".external_probes[" +
                std::to_string(index) + "]"));
    ValidateProgram(result, expected_source);
    return result;
}

ExternalDmaProgram LoadExternalDmaProgram(
    const std::filesystem::path &path,
    const ExternalDmaExpectedSource &expected_source) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::invalid_argument(
            "cannot open external DMA program: " + path.string());
    std::ostringstream content;
    content << input.rdbuf();
    if (input.bad())
        throw std::invalid_argument(
            "failed reading external DMA program: " +
            path.string());
    return ParseExternalDmaProgram(content.str(), expected_source);
}

ExternalDmaProgramExecutor::ExternalDmaProgramExecutor(
    const sc_core::sc_module_name &name,
    ExternalDmaProgram program,
    std::map<HbmEndpoint, HBMBackend *> hbm_backends,
    sc_core::sc_time cycle_time)
    : sc_core::sc_module(name),
      program_(std::move(program)),
      runtime_(std::make_unique<ExternalMemoryRuntimeBridge>(
          sc_core::sc_gen_unique_name("external_dma_runtime"),
          program_.fabric,
          ResolveBackends(program_, hbm_backends),
          cycle_time)),
      cycle_time_(cycle_time) {
    if (cycle_time_ <= sc_core::SC_ZERO_TIME)
        throw std::invalid_argument(
            "external DMA executor cycle time must be > 0");
    SC_THREAD(Run);
}

ExternalDmaProgramExecutor::~ExternalDmaProgramExecutor() = default;

void ExternalDmaProgramExecutor::Run() {
    ExternalDmaProgramExecution result;
    result.program_ref = program_.id;
    try {
        for (const auto &seed : program_.external_seeds)
            runtime_->SeedExternal(
                seed.external_capacity_ref,
                seed.address, seed.payload);
        std::set<std::string> completed;
        for (const auto &descriptor : program_.descriptors) {
            for (const auto &dependency : descriptor.depends_on)
                if (completed.count(dependency) == 0)
                    throw std::logic_error(
                        "external DMA dependency is incomplete");
            AlignToCycle(cycle_time_);
            const uint64_t current = CurrentCycle(cycle_time_);
            if (current < descriptor.planned_issue_cycle) {
                const uint64_t remaining =
                    descriptor.planned_issue_cycle - current;
                if (remaining >
                    std::numeric_limits<uint64_t>::max() /
                        cycle_time_.value())
                    throw std::overflow_error(
                        "external DMA planned issue time overflows");
                sc_core::wait(sc_core::sc_time::from_value(
                    remaining * cycle_time_.value()));
            }
            const uint64_t issue_cycle = CurrentCycle(cycle_time_);
            TransferRequest request{
                descriptor.id,
                descriptor.connection_ref,
                descriptor.direction,
                descriptor.external_address,
                descriptor.hbm_address,
                descriptor.size_bytes,
                issue_cycle,
                descriptor.request_schema_version};
            runtime_->Submit(request);
            RuntimeCompletion completion =
                runtime_->Wait(descriptor.id);
            result.completions.push_back(completion);
            if (completion.status != 0)
                throw std::runtime_error(
                    "external DMA request failed: " +
                    completion.error);
            completed.insert(descriptor.id);
        }
        for (const auto &probe : program_.external_probes) {
            DmaProbeResult observed;
            observed.probe_ref = probe.id;
            observed.payload = runtime_->ProbeExternal(
                probe.external_capacity_ref, probe.address,
                probe.expected_payload.size());
            observed.matched =
                observed.payload == probe.expected_payload;
            result.probes.push_back(observed);
            if (!observed.matched)
                throw std::runtime_error(
                    "external DMA output probe mismatch");
        }
        result.completed = true;
    } catch (const std::exception &error) {
        result.error = error.what();
    }
    result.stats = runtime_->Stats();
    result.pending_requests = runtime_->Outstanding();
    execution_ = std::move(result);
    done_.notify(sc_core::SC_ZERO_TIME);
}

std::optional<ExternalDmaProgramExecution>
ExternalDmaProgramExecutor::Poll() const {
    return execution_;
}

ExternalDmaProgramExecution ExternalDmaProgramExecutor::Wait() {
    while (!execution_.has_value()) sc_core::wait(done_);
    return *execution_;
}

const sc_core::sc_event &
ExternalDmaProgramExecutor::CompletionEvent() const {
    return done_;
}

} // namespace external_memory
