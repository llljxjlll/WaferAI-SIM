#include "memory/dense_adamw_mid_program_pager.h"

#include "frontend/program_finalizer.h"
#include "frontend/program_io.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace external_memory {
namespace {
using Json = nlohmann::json;

[[noreturn]] void Fail(const std::string &detail) {
    throw std::invalid_argument("Dense AdamW paged runtime: " + detail);
}

void Exact(const Json &object, std::initializer_list<const char *> names) {
    if (!object.is_object() || object.size() != names.size())
        Fail("JSON object shape drifted");
    for (const char *name : names)
        if (!object.contains(name)) Fail(std::string("missing field ") + name);
}

std::string String(const Json &object, const char *key) {
    if (!object.at(key).is_string()) Fail(std::string(key) + " must be string");
    const std::string result = object.at(key).get<std::string>();
    if (result.empty()) Fail(std::string(key) + " must not be empty");
    return result;
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

std::vector<std::string> TwoStrings(const Json &value, const char *key) {
    if (!value.at(key).is_array() || value.at(key).size() != 2)
        Fail(std::string(key) + " requires two source artifacts");
    const std::string first = value.at(key)[0].get<std::string>();
    const std::string second = value.at(key)[1].get<std::string>();
    if (first.empty() || second.empty() || first == second)
        Fail(std::string(key) + " changed identities");
    return {first, second};
}

bool CoveredBySeed(const DenseAdamwPagerSpan &span,
                   const ExternalDmaProgram &program) {
    return std::any_of(program.external_seeds.begin(),
                       program.external_seeds.end(), [&](const auto &seed) {
        return seed.address <= span.external_address &&
               span.external_address + span.size_bytes <=
                   seed.address + seed.payload.size();
    });
}

void EmitRoleValues(const std::map<std::string, std::vector<uint8_t>> &roles,
                    uint64_t version, uint64_t pending) {
    const std::map<std::string, std::pair<size_t, size_t>> expected{
        {"trainable_parameter", {15, 4576}},
        {"optimizer_master", {17, 9152}},
        {"optimizer_moment1", {17, 9152}},
        {"optimizer_moment2", {17, 9152}},
        {"optimizer_step", {17, 68}},
    };
    if (roles.size() != expected.size() || pending != 0)
        Fail("external role probe coverage/pending drifted");
    for (const auto &[name, count_bytes] : expected) {
        const auto found = roles.find(name);
        if (found == roles.end() || found->second.size() != count_bytes.second)
            Fail("real external role probe bytes changed for " + name);
        std::cout << "[DENSE_ADAMW_EXTERNAL_ROLE_VALUE] role=" << name
                  << " version=" << version << " bytes=" << found->second.size()
                  << " digest=" << frontend::program_io::Sha256Hex(found->second)
                  << " state_count=" << count_bytes.first
                  << " pending=" << pending << " functional=0 pass=1"
                  << std::endl;
    }
}
} // namespace

DenseAdamwMidProgramPager::DenseAdamwMidProgramPager(
    const sc_core::sc_module_name &name,
    const std::filesystem::path &sidecar,
    std::vector<std::string> manifest_texts,
    std::map<HbmEndpoint, HBMBackend *> backends,
    sc_core::sc_time cycle_time)
    : cycle_time_(cycle_time) {
    const Json contract = ReadJson(sidecar);
    Exact(contract, {"schema_version", "producer_pass", "id",
                     "source_dma_program_relative_path", "source_dma_program_id",
                     "source_dma_program_digest", "request_digest",
                     "logical_graph_digest", "source_memory_plan_digest",
                     "blocking_offload_plan_digest", "linked_manifest_ids",
                     "linked_manifest_digests", "hbm_capacity_bytes",
                     "workspace_end_bytes", "state_bytes", "state_spans", "events"});
    if (String(contract, "schema_version") !=
            "wafer_frontend.dense_adamw_paged_runtime/v1alpha1" ||
        String(contract, "producer_pass") != "dense_adamw_paged_runtime" ||
        String(contract, "source_dma_program_relative_path") !=
            "artifacts/external_dma_program.json" ||
        manifest_texts.size() != 2 || Number(contract, "state_bytes") != 32100 ||
        Number(contract, "hbm_capacity_bytes") != 36864 ||
        Number(contract, "workspace_end_bytes") != 9248)
        Fail("unsupported two-layer bounded production identity");
    const auto ids = TwoStrings(contract, "linked_manifest_ids");
    const auto digests = TwoStrings(contract, "linked_manifest_digests");
    std::vector<frontend::LinkedProgramManifestDto> manifests;
    for (std::size_t step = 0; step < 2; ++step) {
        auto parsed = frontend::ProgramArtifactFinalizer::Parse(
            manifest_texts[step]);
        if (parsed.id != ids[step] ||
            frontend::ProgramArtifactFinalizer::CanonicalManifestDigest(
                manifest_texts[step]) != digests[step])
            Fail("source-signed linked manifest identity changed");
        manifests.push_back(std::move(parsed));
    }
    const std::filesystem::path source_path =
        sidecar.parent_path() / "artifacts/external_dma_program.json";
    const Json source_json = ReadJson(source_path);
    if (frontend::program_io::Sha256Hex(source_json.dump()) !=
        String(contract, "source_dma_program_digest"))
        Fail("canonical source DMA program digest changed");
    ExternalDmaExpectedSource expected;
    expected.case_digest = String(source_json, "case_digest");
    expected.request_digest = String(contract, "request_digest");
    expected.logical_graph_digest = String(contract, "logical_graph_digest");
    expected.source_memory_plan_digest =
        String(contract, "source_memory_plan_digest");
    expected.blocking_offload_plan_digest =
        String(contract, "blocking_offload_plan_digest");
    const ExternalDmaProgram program =
        LoadExternalDmaProgram(source_path, expected);
    source_ref_ = program.id;
    if (source_ref_ != String(contract, "source_dma_program_id") ||
        program.backend_bindings.size() != 1 ||
        program.fabric.hbm_capacities.size() != 1 ||
        program.fabric.connections.size() != 1 ||
        program.external_seeds.size() != 5 ||
        program.external_probes.size() != 5 ||
        program.fabric.hbm_capacities.front().capacity_bytes !=
            Number(contract, "hbm_capacity_bytes"))
        Fail("shared source DMA fabric/seed/physical HBM changed");
    connection_ref_ = program.fabric.connections.front().id;
    external_capacity_ref_ = program.fabric.external_capacities.front().id;
    if (!contract.at("state_spans").is_array() ||
        contract.at("state_spans").size() != 83 ||
        !contract.at("events").is_array() ||
        contract.at("events").size() != 332)
        Fail("83 physical spans/332 real LSU gates required");
    std::set<std::string> state_refs, abi_ids;
    uint64_t exact_bytes = 0;
    for (const auto &raw : contract.at("state_spans")) {
        Exact(raw, {"state_ref", "state_abi_id", "source_allocation_ref",
                    "external_address", "hbm_address", "size_bytes", "kind"});
        DenseAdamwPagerSpan span{
            String(raw, "state_ref"), String(raw, "state_abi_id"),
            String(raw, "source_allocation_ref"), String(raw, "kind"),
            Number(raw, "external_address"), Number(raw, "hbm_address"),
            Number(raw, "size_bytes")};
        if (!state_refs.insert(span.state_ref).second ||
            !abi_ids.insert(span.state_abi_id).second ||
            span.size_bytes == 0 ||
            span.hbm_address < Number(contract, "workspace_end_bytes") ||
            span.hbm_address + span.size_bytes >
                Number(contract, "hbm_capacity_bytes") ||
            !CoveredBySeed(span, program))
            Fail("physical StateABI span, source seed or HBM workspace invalid");
        exact_bytes += span.size_bytes;
        resident_.emplace(span.state_ref, false);
        spans_.push_back(std::move(span));
    }
    if (exact_bytes != 32100 ||
        !std::is_sorted(spans_.begin(), spans_.end(),
                        [](const auto &a, const auto &b) {
                            return a.state_ref < b.state_ref;
                        }))
        Fail("83 source StateABI bytes are not exact/canonical");
    const std::map<std::string, std::pair<frontend::StateKindDto, size_t>> roles{
        {"trainable_parameter", {frontend::StateKindDto::TRAINABLE_PARAMETER, 15}},
        {"optimizer_master", {frontend::StateKindDto::OPTIMIZER_MASTER, 17}},
        {"optimizer_moment1", {frontend::StateKindDto::OPTIMIZER_MOMENT1, 17}},
        {"optimizer_moment2", {frontend::StateKindDto::OPTIMIZER_MOMENT2, 17}},
        {"optimizer_step", {frontend::StateKindDto::OPTIMIZER_STEP, 17}},
    };
    std::map<std::string, std::vector<DenseAdamwPagerSpan>> groups;
    std::set<std::string> allocation_refs;
    for (const auto &span : spans_) {
        if (roles.count(span.kind) == 0)
            Fail("unknown AdamW physical StateABI role");
        groups[span.kind].push_back(span);
    }
    if (groups.size() != 5) Fail("source external roles are incomplete");
    for (const auto &[role, group] : groups) {
        if (group.size() != roles.at(role).second)
            Fail("15 physical carriers and 17 optimizer states per role required");
        auto sorted = group;
        std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
            return a.external_address < b.external_address;
        });
        const std::string allocation_ref = sorted.front().source_allocation_ref;
        if (!allocation_refs.insert(allocation_ref).second)
            Fail("two physical roles alias a P3 source allocation");
        uint64_t cursor = sorted.front().external_address;
        for (const auto &span : sorted) {
            if (span.source_allocation_ref != allocation_ref ||
                span.external_address != cursor)
                Fail("P3 external source allocation has an ABI gap/duplicate");
            cursor += span.size_bytes;
        }
        const bool exact_seed = std::any_of(
            program.external_seeds.begin(), program.external_seeds.end(),
            [&](const auto &seed) {
                return seed.external_capacity_ref == external_capacity_ref_ &&
                    seed.address == sorted.front().external_address &&
                    seed.payload.size() == cursor - sorted.front().external_address;
            });
        if (!exact_seed)
            Fail("source five allocation seed groups are not real ABI-tight");
    }
    std::map<std::string, DenseAdamwPagerSpan> by_ref;
    for (const auto &span : spans_) by_ref.emplace(span.state_ref, span);
    for (const auto &raw : contract.at("events")) {
        Exact(raw, {"step_index", "linked_record_index", "kind",
                    "state_ref", "external_address", "hbm_address",
                    "size_bytes"});
        DenseAdamwPagerEvent event{
            Number(raw, "step_index"), Number(raw, "linked_record_index"),
            String(raw, "kind"), String(raw, "state_ref"),
            Number(raw, "external_address"), Number(raw, "hbm_address"),
            Number(raw, "size_bytes")};
        const auto found = by_ref.find(event.state_ref);
        if (found == by_ref.end() || event.step_index > 1 ||
            (event.kind != "restore_before_lsu_load" &&
             event.kind != "writeback_after_lsu_store") ||
            std::tie(event.external_address, event.hbm_address,
                     event.size_bytes) !=
                std::tie(found->second.external_address,
                         found->second.hbm_address, found->second.size_bytes))
            Fail("paged event no longer binds physical signed state");
        events_.push_back(std::move(event));
    }
    for (std::size_t step = 0; step < 2; ++step) {
        const auto &manifest = manifests[step];
        if (manifest.fragments.size() != 1 ||
            !std::holds_alternative<frontend::CommandFragmentDto>(
                manifest.fragments.front()) ||
            manifest.state_operand_bindings.size() != 166)
            Fail("paged manifest is not a true production AdamW source");
        const auto &fragment = std::get<frontend::CommandFragmentDto>(
            manifest.fragments.front());
        if (fragment.state_abi.size() != 83 ||
            fragment.core_streams.size() != 1)
            Fail("paged manifest StateABI/core identity drifted");
        std::map<std::string, frontend::StateAbiDto> abis;
        for (const auto &abi : fragment.state_abi)
            abis.emplace(abi.state_ref, abi);
        for (const auto &span : spans_) {
            const auto it = abis.find(span.state_ref);
            if (it == abis.end() || it->second.id != span.state_abi_id ||
                it->second.kind != roles.at(span.kind).first ||
                it->second.die_id != 0 ||
                it->second.address != span.hbm_address ||
                it->second.size_bytes != span.size_bytes)
                Fail("linked StateABI source range differs from pager");
        }
        auto bindings = manifest.state_operand_bindings;
        std::sort(bindings.begin(), bindings.end(),
                  [](const auto &a, const auto &b) {
                      return a.fragment_record_index < b.fragment_record_index;
                  });
        std::set<std::string> read, write;
        uint64_t previous_index = 0;
        for (std::size_t i = 0; i < 166; ++i) {
            const auto &event = events_[step * 166 + i];
            const auto &binding = bindings[i];
            const auto &records = fragment.core_streams.front().records;
            if (event.step_index != step ||
                binding.fragment_record_index >= records.size() ||
                (i && event.linked_record_index <= previous_index) ||
                binding.fragment_record_index != event.linked_record_index ||
                binding.state_abi_id != by_ref.at(event.state_ref).state_abi_id ||
                (records[binding.fragment_record_index].opcode !=
                    (event.kind == "restore_before_lsu_load"
                         ? Opcode::LSU_LOAD : Opcode::LSU_STORE)))
                Fail("linked record/StateABI/LSU order changed after pager signing");
            previous_index = event.linked_record_index;
            (event.kind == "restore_before_lsu_load" ? read : write)
                .insert(event.state_ref);
            if (step &&
                (event.linked_record_index != events_[i].linked_record_index ||
                 event.kind != events_[i].kind ||
                 event.state_ref != events_[i].state_ref))
                Fail("step0/step1 physical LSU order changed");
        }
        if (read != state_refs || write != state_refs)
            Fail("each step must restore and write back exactly 83 states");
    }
    std::map<std::string, HBMBackend *> hbm;
    for (const auto &binding : program.backend_bindings) {
        const auto found = backends.find(
            {binding.stack_id, binding.channel_id});
        if (found == backends.end() || !found->second)
            Fail("source-bound physical HBM backend is unavailable");
        hbm.emplace(binding.hbm_capacity_ref, found->second);
    }
    runtime_ = std::make_unique<ExternalMemoryRuntimeBridge>(
        name, program.fabric, std::move(hbm), cycle_time_);
    for (const auto &seed : program.external_seeds)
        runtime_->SeedExternal(seed.external_capacity_ref, seed.address,
                               seed.payload);
    probes_ = program.external_probes;
}

const DenseAdamwPagerEvent &DenseAdamwMidProgramPager::NextEvent(
    const char *kind, uint64_t address, uint64_t size_bytes) const {
    if (next_event_ >= events_.size())
        Fail("runtime issued an LSU beyond 332 signed gates");
    const auto &event = events_[next_event_];
    if (event.kind != kind || event.hbm_address != address ||
        event.size_bytes != size_bytes)
        Fail("runtime LSU direction/address/size changed from signed order at " +
             std::to_string(next_event_));
    return event;
}

void DenseAdamwMidProgramPager::Transfer(
    const DenseAdamwPagerEvent &event, TransferDirection direction) {
    const std::string id = "dense_adamw_paged_" +
        std::to_string(next_event_) + "_" + event.state_ref;
    const auto quantum = cycle_time_.value();
    const auto remainder = sc_core::sc_time_stamp().value() % quantum;
    if (remainder != 0)
        sc_core::wait(sc_core::sc_time::from_value(quantum - remainder));
    const uint64_t cycle = sc_core::sc_time_stamp().value() / quantum;
    runtime_->Submit({id, connection_ref_, direction, event.external_address,
                      event.hbm_address, event.size_bytes, cycle,
                      kExternalDmaRequestSchemaVersion});
    const auto completion = runtime_->Wait(id);
    if (completion.status != 0 || completion.payload_bytes != event.size_bytes)
        Fail("actual DMA completion failed for " + event.state_ref + ": " +
             completion.error);
    std::cout << "[DENSE_ADAMW_PAGED_DMA_EVENT] index=" << next_event_
              << " step=" << event.step_index
              << " linked_record=" << event.linked_record_index
              << " direction="
              << (direction == TransferDirection::kExternalToHbm
                      ? "restore_before_lsu_load"
                      : "writeback_after_lsu_store")
              << " state_ref=" << event.state_ref
              << " bytes=" << completion.payload_bytes
              << " issue_cycle=" << cycle
              << " completed_at_ticks=" << completion.completed_at.value()
              << " lsu_dependency_complete=1 pass=1" << std::endl;
}

void DenseAdamwMidProgramPager::BeforeLoad(uint64_t address, uint64_t size) {
    const auto &event = NextEvent("restore_before_lsu_load", address, size);
    if (resident_.at(event.state_ref))
        Fail("duplicate restore before dirty writeback");
    for (const auto &span : spans_)
        if (resident_.at(span.state_ref) &&
            address < span.hbm_address + span.size_bytes &&
            span.hbm_address < address + size)
            Fail("paged HBM slot conflicts with still pinned state");
    Transfer(event, TransferDirection::kExternalToHbm);
    resident_.at(event.state_ref) = true;
    ++next_event_;
}

void DenseAdamwMidProgramPager::AfterStore(uint64_t address, uint64_t size) {
    const auto &event = NextEvent("writeback_after_lsu_store", address, size);
    if (!resident_.at(event.state_ref))
        Fail("dirty state was not restored before actual LSU store");
    Transfer(event, TransferDirection::kHbmToExternal);
    resident_.at(event.state_ref) = false;
    ++next_event_;
}

void DenseAdamwMidProgramPager::CompleteStep(uint64_t index) {
    if (index > 1 || next_event_ != (index + 1) * 166 ||
        runtime_->Outstanding() != 0 ||
        std::any_of(resident_.begin(), resident_.end(), [](const auto &item) {
            return item.second;
        }))
        Fail("segment ended with incomplete DMA, dirty state or active pin");
    std::vector<uint8_t> authority;
    authority.reserve(32100);
    std::map<std::string, std::vector<uint8_t>> role_values;
    for (const auto &span : spans_) {
        auto payload = runtime_->ProbeExternal(
            external_capacity_ref_, span.external_address, span.size_bytes);
        if (payload.size() != span.size_bytes)
            Fail("physical external StateABI readback was incomplete");
        authority.insert(authority.end(), payload.begin(), payload.end());
        role_values[span.kind].insert(role_values[span.kind].end(),
                                     payload.begin(), payload.end());
        ++external_probes_;
    }
    if (authority.size() != 32100 || external_probes_ != (index + 1) * 83)
        Fail("physical external authority does not cover all 83 states");
    authority_digest_ = frontend::program_io::Sha256Hex(authority);
    if (index == 1) {
        for (const auto &probe : probes_)
            if (runtime_->ProbeExternal(probe.external_capacity_ref,
                                        probe.address,
                                        probe.expected_payload.size()) !=
                probe.expected_payload)
                Fail("external final true source probe disagrees with seed");
    }
    EmitRoleValues(role_values, index + 1, runtime_->Outstanding());
}

std::string DenseAdamwMidProgramPager::ProbeInitialAuthority() const {
    if (next_event_ != 0 || runtime_->Outstanding() != 0)
        Fail("initial external authority must be probed before compute");
    std::vector<uint8_t> authority;
    authority.reserve(32100);
    std::map<std::string, std::vector<uint8_t>> role_values;
    for (const auto &span : spans_) {
        const auto payload = runtime_->ProbeExternal(
            external_capacity_ref_, span.external_address, span.size_bytes);
        if (payload.size() != span.size_bytes)
            Fail("initial external StateABI probe read was incomplete");
        authority.insert(authority.end(), payload.begin(), payload.end());
        role_values[span.kind].insert(role_values[span.kind].end(),
                                     payload.begin(), payload.end());
    }
    if (authority.size() != 32100)
        Fail("initial external authority does not cover 83 physical spans");
    EmitRoleValues(role_values, 0, runtime_->Outstanding());
    return frontend::program_io::Sha256Hex(authority);
}

uint64_t DenseAdamwMidProgramPager::Pending() const {
    return runtime_->Outstanding();
}

const RuntimeStats &DenseAdamwMidProgramPager::Stats() const {
    return runtime_->Stats();
}
} // namespace external_memory
