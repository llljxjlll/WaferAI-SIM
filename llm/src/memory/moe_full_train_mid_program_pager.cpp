#include "memory/moe_full_train_mid_program_pager.h"

#include "frontend/program_finalizer.h"
#include "frontend/program_io.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace external_memory {
namespace {
using Json = nlohmann::json;

[[noreturn]] void Fail(const std::string &detail) {
    throw std::invalid_argument("MoE train paged runtime: " + detail);
}

void Exact(const Json &value, std::initializer_list<const char *> fields) {
    if (!value.is_object() || value.size() != fields.size())
        Fail("JSON object shape drifted");
    for (const char *field : fields)
        if (!value.contains(field)) Fail(std::string("missing field ") + field);
}

std::string String(const Json &value, const char *field) {
    if (!value.at(field).is_string()) Fail(std::string(field) + " must be string");
    const auto result = value.at(field).get<std::string>();
    if (result.empty()) Fail(std::string(field) + " must be nonempty");
    return result;
}

uint64_t Number(const Json &value, const char *field) {
    const auto &item = value.at(field);
    if (!item.is_number_unsigned() &&
        !(item.is_number_integer() && item.get<int64_t>() >= 0))
        Fail(std::string(field) + " must be uint64");
    return item.get<uint64_t>();
}

Json ReadJson(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) Fail("cannot open " + path.string());
    Json value;
    input >> value;
    input >> std::ws;
    if (input.peek() != EOF)
        Fail("trailing JSON after " + path.string());
    return value;
}

std::vector<uint8_t> Hex(const std::string &value) {
    if (value.empty() || value.size() % 2) Fail("seed hex length changed");
    const auto digit = [](char c) -> uint8_t {
        if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
        Fail("seed must be lowercase hex");
    };
    std::vector<uint8_t> result;
    result.reserve(value.size() / 2);
    for (size_t i = 0; i < value.size(); i += 2)
        result.push_back(static_cast<uint8_t>(digit(value[i]) * 16 +
                                              digit(value[i + 1])));
    return result;
}

uint64_t RecordBytes(const frontend::RelocatableRecordDto &record) {
    uint64_t found = 0;
    for (const auto &operand : record.operands)
        if (operand.name == "size_bytes") {
            const auto *literal = std::get_if<uint64_t>(&operand.literal_value);
            if (!literal || found) Fail("LSU size literal missing/duplicated");
            found = *literal;
        }
    if (!found) Fail("LSU size literal absent");
    return found;
}

int64_t HbmAddend(const frontend::CoreFragmentStreamDto &stream,
                  uint64_t record_index) {
    bool found = false;
    int64_t result = 0;
    for (const auto &relocation : stream.address_relocations)
        if (relocation.record_index == record_index &&
            relocation.operand_id == SemanticOperandId::HBM_ADDRESS) {
            if (found) Fail("duplicate LSU HBM relocation");
            found = true;
            result = relocation.addend;
        }
    if (!found) Fail("LSU HBM relocation absent");
    return result;
}

FabricConfig ParseFabric(const Json &raw, std::string &external_ref,
                         std::string &hbm_ref,
                         std::string &connection_ref) {
    Exact(raw, {"schema_version", "producer_pass", "id",
                "external_capacities", "hbm_capacities", "links",
                "connections"});
    if (String(raw, "schema_version") !=
            "wafer_frontend.external_memory_fabric/v1alpha2" ||
        String(raw, "producer_pass") != "external_memory_fabric_builder" ||
        !raw.at("external_capacities").is_array() ||
        raw.at("external_capacities").size() != 1 ||
        !raw.at("hbm_capacities").is_array() ||
        raw.at("hbm_capacities").size() != 1 ||
        !raw.at("links").is_array() || raw.at("links").size() != 1 ||
        !raw.at("connections").is_array() ||
        raw.at("connections").size() != 1)
        Fail("one-link one-die fabric changed");
    const auto &external = raw.at("external_capacities")[0];
    const auto &hbm = raw.at("hbm_capacities")[0];
    const auto &link = raw.at("links")[0];
    const auto &connection = raw.at("connections")[0];
    Exact(external, {"id", "tier", "location_ref", "base_address",
                     "capacity_bytes", "alignment_bytes"});
    Exact(hbm, {"id", "tier", "location_ref", "base_address",
                "capacity_bytes", "alignment_bytes"});
    Exact(link, {"id", "external_capacity_ref", "ingress_die_id",
                 "bytes_per_cycle", "latency_cycles", "queue_depth",
                 "max_outstanding", "duplex"});
    Exact(connection, {"id", "link_ref", "hbm_capacity_ref",
                       "target_die_id", "route_die_ids",
                       "route_latency_cycles", "route_bytes_per_cycle"});
    external_ref = String(external, "id");
    hbm_ref = String(hbm, "id");
    connection_ref = String(connection, "id");
    if (String(external, "tier") != "external" ||
        String(external, "location_ref") != "host:0" ||
        Number(external, "base_address") != 0 ||
        Number(external, "capacity_bytes") != 2048 ||
        Number(external, "alignment_bytes") != 16 ||
        String(hbm, "tier") != "hbm" ||
        String(hbm, "location_ref") != "die:0" ||
        Number(hbm, "base_address") != 0 ||
        Number(hbm, "capacity_bytes") != 2560 ||
        Number(hbm, "alignment_bytes") != 16 ||
        String(link, "external_capacity_ref") != external_ref ||
        Number(link, "ingress_die_id") != 0 ||
        Number(link, "bytes_per_cycle") != 256 ||
        Number(link, "latency_cycles") != 2 ||
        Number(link, "queue_depth") != 2 ||
        Number(link, "max_outstanding") != 2 ||
        String(link, "duplex") != "half_duplex_shared" ||
        String(connection, "link_ref") != String(link, "id") ||
        String(connection, "hbm_capacity_ref") != hbm_ref ||
        Number(connection, "target_die_id") != 0 ||
        connection.at("route_die_ids") != Json::array({0}) ||
        Number(connection, "route_latency_cycles") != 0 ||
        !connection.at("route_bytes_per_cycle").is_null())
        Fail("source external capacity/link/route changed");
    FabricConfig fabric;
    fabric.external_capacities.push_back({external_ref, "host:0", 0, 2048});
    fabric.hbm_capacities.push_back({hbm_ref, 0, 0, 2560});
    fabric.links.push_back({String(link, "id"), external_ref, 0, 256, 2, 2, 2});
    fabric.connections.push_back({connection_ref, String(link, "id"),
                                  hbm_ref, 0, {0}, 0, std::nullopt});
    ValidateFabricConfig(fabric);
    return fabric;
}
} // namespace

MoeFullTrainMidProgramPager::MoeFullTrainMidProgramPager(
    const sc_core::sc_module_name &name,
    const std::filesystem::path &sidecar,
    std::vector<std::string> manifest_texts,
    std::map<HbmEndpoint, HBMBackend *> backends,
    sc_core::sc_time cycle_time)
    : cycle_time_(cycle_time) {
    const Json contract = ReadJson(sidecar);
    Exact(contract, {"schema_version", "producer_pass", "id",
                     "request_digest", "logical_graph_digest",
                     "offload_memory_plan_digest",
                     "source_external_allocation_refs",
                     "linked_manifest_ids", "linked_manifest_digests",
                     "hbm_capacity_bytes", "workspace_end_bytes",
                     "weight_slot_address", "weight_slot_bytes",
                     "pinned_route_addresses", "external_capacity_bytes",
                     "expected_events", "expected_external_read_bytes",
                     "expected_external_write_bytes", "fabric",
                     "parameter_spans", "events", "seeds"});
    if (String(contract, "schema_version") !=
            "wafer_frontend.moe_full_train_paged_runtime/v1alpha1" ||
        String(contract, "producer_pass") != "moe_full_train_paged_runtime" ||
        Number(contract, "hbm_capacity_bytes") != 2560 ||
        Number(contract, "workspace_end_bytes") != 2016 ||
        Number(contract, "weight_slot_address") != 2048 ||
        Number(contract, "weight_slot_bytes") != 128 ||
        contract.at("pinned_route_addresses") != Json::array({2304, 2432}) ||
        Number(contract, "external_capacity_bytes") != 2048 ||
        Number(contract, "expected_events") != 130 ||
        Number(contract, "expected_external_read_bytes") != 4864 ||
        Number(contract, "expected_external_write_bytes") != 1904 ||
        manifest_texts.size() != 2 || backends.size() != 1 ||
        backends.count({0, 0}) != 1)
        Fail("unsupported full two-step bounded MoE training contract");
    Json semantic = contract;
    semantic.erase("id");
    source_ref_ = "moe_full_train_paged_runtime_" +
        frontend::program_io::Sha256Hex(semantic.dump()).substr(0, 20);
    if (source_ref_ != String(contract, "id"))
        Fail("source sidecar identity changed");
    for (const char *field : {"request_digest", "logical_graph_digest",
                              "offload_memory_plan_digest"})
        if (String(contract, field).size() != 64)
            Fail(std::string(field) + " must be a SHA-256 digest");
    if (contract.at("source_external_allocation_refs").size() != 2)
        Fail("two actual P3 external parameter allocations required");
    const std::string shared_allocation =
        contract.at("source_external_allocation_refs")[0].get<std::string>();
    const std::string expert_allocation =
        contract.at("source_external_allocation_refs")[1].get<std::string>();
    if (shared_allocation.empty() || expert_allocation.empty() ||
        shared_allocation == expert_allocation)
        Fail("P3 external parameter allocation identity changed");
    std::string hbm_ref;
    const FabricConfig fabric = ParseFabric(contract.at("fabric"),
                                            external_capacity_ref_,
                                            hbm_ref, connection_ref_);
    if (!contract.at("linked_manifest_ids").is_array() ||
        contract.at("linked_manifest_ids").size() != 2 ||
        !contract.at("linked_manifest_digests").is_array() ||
        contract.at("linked_manifest_digests").size() != 2)
        Fail("two linked manifests require source IDs and digests");
    std::vector<frontend::LinkedProgramManifestDto> manifests;
    for (size_t step = 0; step < 2; ++step) {
        const auto manifest = frontend::ProgramArtifactFinalizer::Parse(
            manifest_texts[step]);
        if (manifest.id != contract.at("linked_manifest_ids")[step].get<std::string>() ||
            frontend::ProgramArtifactFinalizer::CanonicalManifestDigest(
                manifest_texts[step]) !=
                contract.at("linked_manifest_digests")[step].get<std::string>() ||
            manifest.producer_pass != "manifest_linker" ||
            manifest.fragments.size() != 160 ||
            manifest.core_streams.size() != 1 ||
            manifest.core_streams[0].runtime_core_id != 0 ||
            manifest.core_streams[0].records.size() != 621 ||
            manifest.state_operand_bindings.size() != 67)
            Fail("two-step physical linked manifest identity/shape drifted");
        manifests.push_back(std::move(manifest));
    }
    if (!contract.at("parameter_spans").is_array() ||
        contract.at("parameter_spans").size() != 19 ||
        !contract.at("events").is_array() ||
        contract.at("events").size() != 130 ||
        !contract.at("seeds").is_array() ||
        contract.at("seeds").size() != 19)
        Fail("19 true trainable pages and 130 LSU gates required");
    std::set<std::string> state_refs, abi_ids;
    uint64_t cursor = 0;
    uint64_t total_bytes = 0;
    for (size_t index = 0; index < 19; ++index) {
        const auto &raw = contract.at("parameter_spans")[index];
        Exact(raw, {"state_ref", "state_abi_id", "source_allocation_ref",
                    "group", "source_hbm_address", "external_address",
                    "hbm_address", "size_bytes"});
        MoeFullTrainPagerSpan span{
            String(raw, "state_ref"), String(raw, "state_abi_id"),
            String(raw, "source_allocation_ref"), String(raw, "group"),
            Number(raw, "source_hbm_address"),
            Number(raw, "external_address"), Number(raw, "hbm_address"),
            Number(raw, "size_bytes")};
        const bool expert = span.group == "parameter.expert.0.tp0.rank0";
        if ((span.group != "parameter.shared.tp0.rank0" && !expert) ||
            span.source_allocation_ref !=
                (expert ? expert_allocation : shared_allocation) ||
            span.size_bytes == 0 || span.size_bytes > 128 ||
            span.hbm_address != 2048 ||
            !state_refs.insert(span.state_ref).second ||
            !abi_ids.insert(span.state_abi_id).second)
            Fail("MoE paged StateABI identity/role/slot invalid");
        if (index == 13 && cursor == 568) cursor = 576;
        if (span.external_address != cursor ||
            span.external_address + span.size_bytes >
                (expert ? 960u : 568u) ||
            (expert != (index >= 13)))
            Fail("actual P3 shared/expert allocations have a page gap");
        cursor += span.size_bytes;
        total_bytes += span.size_bytes;
        spans_.push_back(std::move(span));
    }
    if (cursor != 960 || total_bytes != 952)
        Fail("P3 568B shared + 384B expert external extent changed");
    std::map<std::string, MoeFullTrainPagerSpan> by_ref;
    for (const auto &span : spans_) by_ref.emplace(span.state_ref, span);
    uint64_t restores = 0, stores = 0, restore_bytes = 0, store_bytes = 0;
    for (size_t index = 0; index < 130; ++index) {
        const auto &raw = contract.at("events")[index];
        Exact(raw, {"step_index", "runtime_core_id", "linked_record_index",
                    "fragment_id", "fragment_record_index", "kind",
                    "state_ref", "state_abi_id", "source_hbm_address",
                    "external_address", "hbm_address", "size_bytes"});
        MoeFullTrainPagerEvent event{
            index, Number(raw, "step_index"),
            Number(raw, "linked_record_index"), String(raw, "fragment_id"),
            Number(raw, "fragment_record_index"), String(raw, "kind"),
            String(raw, "state_ref"), String(raw, "state_abi_id"),
            Number(raw, "source_hbm_address"),
            Number(raw, "external_address"), Number(raw, "hbm_address"),
            Number(raw, "size_bytes")};
        const auto found = by_ref.find(event.state_ref);
        if (event.step_index != index / 65 ||
            Number(raw, "runtime_core_id") != 0 ||
            found == by_ref.end() ||
            event.state_abi_id != found->second.state_abi_id ||
            event.source_hbm_address != found->second.source_hbm_address ||
            event.external_address != found->second.external_address ||
            event.hbm_address != 2048 ||
            event.size_bytes != found->second.size_bytes)
            Fail("source-signed physical MoE LSU event differs from StateABI page");
        if (event.kind == "restore_before_load") {
            ++restores; restore_bytes += event.size_bytes;
        } else if (event.kind == "writeback_after_store") {
            ++stores; store_bytes += event.size_bytes;
        } else Fail("unknown MoE parameter DMA event kind");
        events_.push_back(std::move(event));
    }
    if (restores != 92 || stores != 38 ||
        restore_bytes != 4864 || store_bytes != 1904)
        Fail("exact physical MoE two-step external traffic changed");
    size_t signed_event = 0;
    for (size_t step = 0; step < 2; ++step) {
        const auto &manifest = manifests[step];
        std::map<std::string, const frontend::CommandFragmentDto *> fragments;
        std::map<std::string, frontend::StateAbiDto> states;
        for (const auto &linked : manifest.fragments) {
            const auto *fragment = std::get_if<frontend::CommandFragmentDto>(&linked);
            if (!fragment || !fragments.emplace(fragment->id, fragment).second)
                Fail("MoE training requires 160 distinct real command fragments");
            for (const auto &state : fragment->state_abi) {
                const auto [it, inserted] = states.emplace(state.id, state);
                if (!inserted &&
                    (it->second.state_ref != state.state_ref ||
                     it->second.kind != state.kind ||
                     it->second.address != state.address ||
                     it->second.size_bytes != state.size_bytes))
                    Fail("shared physical StateABI definitions conflict");
            }
        }
        if (states.size() != 21)
            Fail("19 trainables and two route StateABIs required");
        std::map<std::pair<std::string, uint64_t>,
                 frontend::StateOperandBindingDto> bindings;
        for (const auto &binding : manifest.state_operand_bindings)
            if (!bindings.emplace(std::make_pair(binding.fragment_id,
                                                 binding.fragment_record_index),
                                  binding).second)
                Fail("duplicate MoE physical StateOperandBinding");
        uint64_t route_loads = 0, weight_loads = 0, weight_stores = 0;
        const auto &core = manifest.core_streams[0];
        for (size_t linked_index = 0; linked_index < core.records.size();
             ++linked_index) {
            const auto &ref = core.records[linked_index];
            const auto fragment = fragments.find(ref.fragment_id);
            if (fragment == fragments.end())
                Fail("linked record lost physical source fragment");
            const frontend::CoreFragmentStreamDto *stream = nullptr;
            for (const auto &candidate : fragment->second->core_streams)
                if (candidate.logical_core.die_id == 0 &&
                    candidate.logical_core.local_core_id == 0)
                    stream = &candidate;
            if (!stream || ref.fragment_record_index >= stream->records.size())
                Fail("fragment-local MoE record unavailable");
            const auto &record = stream->records[ref.fragment_record_index];
            const auto binding = bindings.find({ref.fragment_id,
                                                 ref.fragment_record_index});
            if (binding == bindings.end()) {
                if (record.opcode == Opcode::LSU_LOAD ||
                    record.opcode == Opcode::LSU_STORE)
                    Fail("actual LSU lacks StateOperandBinding");
                continue;
            }
            const auto state = states.find(binding->second.state_abi_id);
            if (state == states.end() ||
                RecordBytes(record) != state->second.size_bytes ||
                HbmAddend(*stream, ref.fragment_record_index) != 0)
                Fail("real LSU size/HBM relocation changed");
            if (state->second.kind == frontend::StateKindDto::MOE_STATIC_ROUTE) {
                if (record.opcode != Opcode::LSU_LOAD ||
                    state->second.size_bytes != 80 ||
                    (state->second.address != 2304 &&
                     state->second.address != 2432))
                    Fail("pinned route HBM LSU changed");
                ++route_loads;
                continue;
            }
            if (state->second.kind !=
                    frontend::StateKindDto::TRAINABLE_PARAMETER ||
                signed_event >= events_.size())
                Fail("trainable parameter LSU lacks a signed DMA event");
            const auto &event = events_[signed_event++];
            if (event.step_index != step ||
                event.linked_record_index != linked_index ||
                event.fragment_id != ref.fragment_id ||
                event.fragment_record_index != ref.fragment_record_index ||
                event.state_ref != state->second.state_ref ||
                event.state_abi_id != state->second.id ||
                event.hbm_address != state->second.address ||
                event.size_bytes != state->second.size_bytes ||
                ((record.opcode == Opcode::LSU_LOAD &&
                  event.kind != "restore_before_load") ||
                 (record.opcode == Opcode::LSU_STORE &&
                  event.kind != "writeback_after_store")))
                Fail("actual linked LSU order differs from external DMA gate");
            if (record.opcode == Opcode::LSU_LOAD) ++weight_loads;
            else if (record.opcode == Opcode::LSU_STORE) ++weight_stores;
            else Fail("trainable state uses unsupported native opcode");
        }
        if (route_loads != 2 || weight_loads != 46 ||
            weight_stores != 19 || signed_event != 65 * (step + 1))
            Fail("MoE step lacks 46 restore/19 writeback/two pinned routes");
    }
    if (signed_event != events_.size())
        Fail("signed MoE DMA events exceed actual LSU records");
    runtime_ = std::make_unique<ExternalMemoryRuntimeBridge>(
        name, fabric, std::map<std::string, HBMBackend *>{{hbm_ref,
                                                            backends.at({0, 0})}},
        cycle_time_);
    std::set<uint64_t> seeded;
    for (const auto &raw : contract.at("seeds")) {
        Exact(raw, {"external_address", "payload_hex"});
        const uint64_t address = Number(raw, "external_address");
        const auto payload = Hex(String(raw, "payload_hex"));
        const auto found = std::find_if(spans_.begin(), spans_.end(),
            [&](const auto &span) {
                return span.external_address == address &&
                       span.size_bytes == payload.size();
            });
        if (found == spans_.end() || !seeded.insert(address).second ||
            std::none_of(payload.begin(), payload.end(),
                         [](uint8_t value) { return value != 0; }))
            Fail("external seed omitted/duplicated an actual nonzero weight");
        runtime_->SeedExternal(external_capacity_ref_, address, payload);
    }
    if (seeded.size() != 19)
        Fail("all 19 true trainable state seeds required");
    authority_digest_ = frontend::program_io::Sha256Hex(ProbeParameters());
    std::cout << "[MOE_TRAIN_PAGED_STATE] version=0 bytes=952 digest="
              << authority_digest_ << " content_changed=0 functional=0 "
              << "pass=1" << std::endl;
}

std::vector<uint8_t> MoeFullTrainMidProgramPager::ProbeParameters() const {
    std::vector<uint8_t> payload;
    for (const auto &span : spans_) {
        const auto bytes = runtime_->ProbeExternal(
            external_capacity_ref_, span.external_address, span.size_bytes);
        payload.insert(payload.end(), bytes.begin(), bytes.end());
    }
    if (payload.size() != 952)
        Fail("external trainable authority lost exact 952B extent");
    return payload;
}

const MoeFullTrainPagerEvent &MoeFullTrainMidProgramPager::NextEvent(
    uint64_t core_id, uint64_t address, uint64_t size) const {
    if (core_id != 0 || next_event_ >= events_.size())
        Fail("unbound MoE physical core or LSU after signed event tail");
    const auto &event = events_[next_event_];
    if (event.hbm_address != address || event.size_bytes != size)
        Fail("actual MoE LSU address/extent differs from next signed event");
    return event;
}

void MoeFullTrainMidProgramPager::Transfer(
    const MoeFullTrainPagerEvent &event, TransferDirection direction) {
    const auto quantum = cycle_time_.value();
    if (!quantum) Fail("nonpositive NpuSim cycle time");
    const auto remainder = sc_core::sc_time_stamp().value() % quantum;
    if (remainder)
        sc_core::wait(sc_core::sc_time::from_value(quantum - remainder));
    const auto cycle = sc_core::sc_time_stamp().value() / quantum;
    const std::string request = "moe_train_paged_" +
        std::to_string(event.index) + "_" + event.state_abi_id;
    runtime_->Submit({request, connection_ref_, direction,
                      event.external_address, event.hbm_address,
                      event.size_bytes, cycle,
                      kExternalDmaRequestSchemaVersion});
    const auto completion = runtime_->Wait(request);
    if (completion.status != 0 ||
        completion.payload_bytes != event.size_bytes)
        Fail("real MoE train DMA completion failed: " + completion.error);
    ++completed_events_;
    std::cout << "[MOE_TRAIN_PAGED_DMA_EVENT] index=" << event.index
              << " step=" << event.step_index
              << " linked_record=" << event.linked_record_index
              << " kind=" << event.kind
              << " state_ref=" << event.state_ref
              << " bytes=" << event.size_bytes
              << " issue_cycle=" << cycle
              << " completed_at_ticks=" << completion.completed_at.value()
              << " lsu_dependency_complete=1 pass=1" << std::endl;
}

void MoeFullTrainMidProgramPager::BeforeLoad(
    uint64_t core_id, uint64_t address, uint64_t size) {
    if (core_id != 0) Fail("MoE training pager only accepts physical core0");
    if ((address == 2304 || address == 2432) && size == 80) {
        if (awaiting_load_) Fail("resident route Load raced weight restore");
        ++route_loads_;
        return;
    }
    if (awaiting_load_) Fail("prior weight LSU Load has not completed");
    const auto &event = NextEvent(core_id, address, size);
    if (event.kind != "restore_before_load")
        Fail("actual MoE LSU Load is not a signed restore");
    Transfer(event, TransferDirection::kExternalToHbm);
    awaiting_load_ = event.index;
    ++next_event_;
}

void MoeFullTrainMidProgramPager::AfterLoad(
    uint64_t core_id, uint64_t address, uint64_t size) {
    if (core_id != 0) Fail("MoE training Load completed on wrong core");
    if ((address == 2304 || address == 2432) && size == 80) {
        if (awaiting_load_) Fail("resident route Load interleaved weight completion");
        return;
    }
    if (!awaiting_load_) Fail("weight Load has no completed external restore");
    const auto &event = events_[*awaiting_load_];
    if (event.hbm_address != address || event.size_bytes != size)
        Fail("completed weight Load differs from signed restored page");
    loaded_states_.insert(event.state_ref);
    awaiting_load_.reset();
}

void MoeFullTrainMidProgramPager::AfterStore(
    uint64_t core_id, uint64_t address, uint64_t size) {
    if (awaiting_load_) Fail("weight Store raced incomplete Load");
    const auto &event = NextEvent(core_id, address, size);
    if (event.kind != "writeback_after_store" ||
        loaded_states_.erase(event.state_ref) != 1)
        Fail("trainable Store lacks earlier same-version parameter Load");
    Transfer(event, TransferDirection::kHbmToExternal);
    ++next_event_;
}

void MoeFullTrainMidProgramPager::CompleteStep(uint64_t step) {
    if (step > 1 || next_event_ != 65 * (step + 1) ||
        completed_events_ != next_event_ ||
        route_loads_ != 2 * (step + 1) || awaiting_load_ ||
        !loaded_states_.empty() || Pending())
        Fail("MoE SGD step boundary has missing DMA, route or pending LSU");
    const auto &stats = runtime_->Stats();
    if (stats.external_read_bytes != 2432 * (step + 1) ||
        stats.external_write_bytes != 952 * (step + 1))
        Fail("MoE external transfer bytes differ from actual StateABI gates");
    const auto prior = authority_digest_;
    authority_digest_ = frontend::program_io::Sha256Hex(ProbeParameters());
    std::cout << "[MOE_TRAIN_PAGED_STATE] version=" << step + 1
              << " bytes=952 digest=" << authority_digest_
              << " content_changed=" << (authority_digest_ != prior ? 1 : 0)
              << " functional=0 pass=1" << std::endl;
    std::cout << "[MOE_TRAIN_PAGED_STEP] index=" << step
              << " restore=46 writeback=19 route_loads=2"
              << " external_read_bytes=" << stats.external_read_bytes
              << " external_write_bytes=" << stats.external_write_bytes
              << " pending=" << Pending() << " pass=1" << std::endl;
}

uint64_t MoeFullTrainMidProgramPager::Pending() const {
    return runtime_->Outstanding();
}

const RuntimeStats &MoeFullTrainMidProgramPager::Stats() const {
    return runtime_->Stats();
}

} // namespace external_memory
