#include "memory/full_dense_adamw_pager.h"

#include "frontend/program_finalizer.h"
#include "frontend/program_io.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace external_memory {
namespace {
using Json = nlohmann::json;

[[noreturn]] void Fail(const std::string &message) {
    throw std::invalid_argument("Full Dense AdamW pager: " + message);
}

void Exact(const Json &value, std::initializer_list<const char *> fields) {
    if (!value.is_object() || value.size() != fields.size())
        Fail("JSON object has an unknown or missing field");
    for (const char *name : fields)
        if (!value.contains(name)) Fail(std::string("missing ") + name);
}

std::string String(const Json &value, const char *name) {
    const auto &item = value.at(name);
    if (!item.is_string() || item.get<std::string>().empty())
        Fail(std::string(name) + " must be nonempty string");
    return item.get<std::string>();
}

uint64_t Number(const Json &value, const char *name) {
    const auto &item = value.at(name);
    if (!item.is_number_unsigned() &&
        !(item.is_number_integer() && item.get<int64_t>() >= 0))
        Fail(std::string(name) + " must be nonnegative integer");
    return item.get<uint64_t>();
}

std::string ReadText(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) Fail("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

std::vector<uint8_t> Hex(const std::string &hex) {
    if (hex.size() % 2) Fail("seed hex length is odd");
    std::vector<uint8_t> bytes;
    bytes.reserve(hex.size() / 2);
    const auto digit = [](char value) -> int {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        return -1;
    };
    for (size_t i = 0; i < hex.size(); i += 2) {
        const int hi = digit(hex[i]), lo = digit(hex[i + 1]);
        if (hi < 0 || lo < 0) Fail("seed must use lowercase hex");
        bytes.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return bytes;
}

using RecordKey = std::pair<std::string, uint64_t>;

std::map<std::string, Json> Fragments(const Json &manifest) {
    std::map<std::string, Json> result;
    for (const auto &fragment : manifest.at("fragments"))
        if (!result.emplace(String(fragment, "id"), fragment).second)
            Fail("duplicate linked fragment");
    return result;
}

std::map<RecordKey, std::string> StateBindings(const Json &manifest) {
    std::map<RecordKey, std::string> result;
    for (const auto &item : manifest.at("state_operand_bindings")) {
        const RecordKey key{String(item, "fragment_id"),
                            Number(item, "fragment_record_index")};
        if (!result.emplace(key, String(item, "state_abi_id")).second)
            Fail("duplicate state operand binding");
    }
    return result;
}

std::map<std::string, Json> StateAbis(const Json &manifest) {
    std::map<std::string, Json> result;
    for (const auto &fragment : manifest.at("fragments"))
        for (const auto &abi : fragment.at("state_abi")) {
            const std::string id = String(abi, "id");
            const auto [it, inserted] = result.emplace(id, abi);
            if (!inserted && it->second != abi)
                Fail("one StateABI id has inconsistent fragments");
        }
    return result;
}

Json CoreRecord(const std::map<std::string, Json> &fragments,
                const Json &program_record) {
    const auto &fragment = fragments.at(String(program_record, "fragment_id"));
    const uint64_t index = Number(program_record, "fragment_record_index");
    if (fragment.at("core_streams").size() != 1 ||
        index >= fragment.at("core_streams")[0].at("records").size())
        Fail("global program record does not resolve to one local core record");
    return fragment.at("core_streams")[0].at("records")[index];
}

void VerifyManifestPair(const Json &source, const Json &paged,
                        const std::vector<FullDenseAdamwStateSpan> &states,
                        const std::vector<FullDenseAdamwDmaEvent> &events) {
    if (source.at("fragments").size() != 520 ||
        paged.at("fragments").size() != 520 ||
        source.at("core_streams").size() != 1 ||
        paged.at("core_streams").size() != 1 ||
        source.at("state_operand_bindings").size() != 350 ||
        paged.at("state_operand_bindings").size() != 350 ||
        events.size() != 350 || states.size() != 75 ||
        source.at("source_ir1_id") != paged.at("source_ir1_id"))
        Fail("source/paged full two-step physical inventory differs");
    const auto source_fragments = Fragments(source);
    const auto paged_fragments = Fragments(paged);
    const auto source_bindings = StateBindings(source);
    const auto paged_bindings = StateBindings(paged);
    const auto source_abis = StateAbis(source);
    const auto paged_abis = StateAbis(paged);
    std::map<std::string, FullDenseAdamwStateSpan> by_ref;
    for (const auto &span : states) {
        if (!by_ref.emplace(span.state_ref, span).second)
            Fail("duplicate signed state reference");
        const auto old = source_abis.find(span.source_abi_id);
        const auto now = paged_abis.find(span.paged_abi_id);
        if (old == source_abis.end() || now == paged_abis.end() ||
            String(old->second, "state_ref") != span.state_ref ||
            String(now->second, "state_ref") != span.state_ref ||
            String(old->second, "kind") != span.kind ||
            String(now->second, "kind") != span.kind ||
            Number(old->second, "address") != span.external_address ||
            Number(now->second, "address") != span.hbm_address ||
            Number(old->second, "size_bytes") != span.size_bytes ||
            Number(now->second, "size_bytes") != span.size_bytes)
            Fail("source/paged StateABI span identity or range changed");
    }
    const auto &old_records = source.at("core_streams")[0].at("records");
    const auto &new_records = paged.at("core_streams")[0].at("records");
    if (old_records.size() != new_records.size() ||
        old_records.size() != 1406)
        Fail("full training global record sequence changed");
    size_t event_index = 0;
    for (size_t i = 0; i < old_records.size(); ++i) {
        const auto &old_ref = old_records[i];
        const auto &new_ref = new_records[i];
        if (old_ref.at("source_global_action_id") !=
                new_ref.at("source_global_action_id") ||
            Number(old_ref, "fragment_record_index") !=
                Number(new_ref, "fragment_record_index"))
            Fail("full training action or local record order changed");
        const auto old_native = CoreRecord(source_fragments, old_ref);
        const auto new_native = CoreRecord(paged_fragments, new_ref);
        if (old_native != new_native)
            Fail("paged source modified native compute/LSU record");
        const uint64_t opcode = Number(old_native, "opcode");
        if (opcode != 128 && opcode != 129) continue; // LSU_LOAD/LSU_STORE
        if (event_index >= events.size()) Fail("unplanned HBM LSU record");
        const auto &event = events[event_index];
        const RecordKey old_key{String(old_ref, "fragment_id"),
                                Number(old_ref, "fragment_record_index")};
        const RecordKey new_key{String(new_ref, "fragment_id"),
                                Number(new_ref, "fragment_record_index")};
        const auto old_binding = source_bindings.find(old_key);
        const auto new_binding = paged_bindings.find(new_key);
        const auto state = by_ref.find(event.state_ref);
        if (old_binding == source_bindings.end() ||
            new_binding == paged_bindings.end() || state == by_ref.end() ||
            old_binding->second != state->second.source_abi_id ||
            new_binding->second != state->second.paged_abi_id ||
            event.index != event_index || event.program_record_index != i ||
            event.kind != (opcode == 128 ? "restore_before_lsu_load"
                                        : "writeback_after_lsu_store") ||
            event.external_address != state->second.external_address ||
            event.hbm_address != state->second.hbm_address ||
            event.size_bytes != state->second.size_bytes ||
            event.step != (event_index < 175 ? 0 : 1))
            Fail("actual source/paged StateABI LSU order differs from signed DMA");
        ++event_index;
    }
    if (event_index != events.size()) Fail("unused signed DMA event");
    for (size_t step = 0; step < 2; ++step) {
        std::set<std::string> loads, stores;
        size_t read_count = 0, write_count = 0;
        for (size_t i = step * 175; i < (step + 1) * 175; ++i) {
            const auto &event = events[i];
            (event.kind == "restore_before_lsu_load" ? loads : stores)
                .insert(event.state_ref);
            (event.kind == "restore_before_lsu_load" ? read_count : write_count)++;
        }
        if (loads.size() != 75 || stores.size() != 75 ||
            read_count != 100 || write_count != 75)
            Fail("each full step must restore all states and write back all dirty states");
    }
}
} // namespace

FullDenseAdamwPager::FullDenseAdamwPager(
    const sc_core::sc_module_name &name,
    const std::filesystem::path &sidecar,
    const std::vector<uint8_t> &program_bytes,
    HBMBackend *hbm, sc_core::sc_time cycle_time)
    : cycle_time_(cycle_time) {
    if (!hbm || cycle_time_ <= sc_core::SC_ZERO_TIME)
        Fail("requires physical HBM backend and positive cycle time");
    const Json contract = Json::parse(ReadText(sidecar));
    Exact(contract, {"schema_version", "producer_pass",
        "source_manifest_relative_path", "paged_manifest_relative_path",
        "source_manifest_id", "source_manifest_digest",
        "paged_manifest_id", "paged_manifest_digest",
        "physical_dag_digest", "source_ir1_id", "hbm_capacity_bytes",
        "external_capacity_bytes", "slot_address", "slot_bytes",
        "state_payload_bytes", "link_bytes_per_cycle",
        "link_latency_cycles", "queue_depth", "max_outstanding",
        "states", "events"});
    if (String(contract, "schema_version") !=
            "wafer_frontend.full_dense_adamw_paged_runtime/v1alpha1" ||
        String(contract, "producer_pass") != "full_dense_adamw_paged")
        Fail("unsupported complete AdamW pager schema");
    source_digest_ = String(contract, "source_manifest_digest");
    paged_digest_ = String(contract, "paged_manifest_digest");
    const auto source_path = sidecar.parent_path() /
        String(contract, "source_manifest_relative_path");
    const auto paged_path = sidecar.parent_path() /
        String(contract, "paged_manifest_relative_path");
    const std::string source_text = ReadText(source_path);
    const std::string paged_text = ReadText(paged_path);
    if (frontend::ProgramArtifactFinalizer::CanonicalManifestDigest(
            source_text) != source_digest_ ||
        frontend::ProgramArtifactFinalizer::CanonicalManifestDigest(
            paged_text) != paged_digest_)
        Fail("source or paged manifest digest changed");
    const auto source_manifest = frontend::ProgramArtifactFinalizer::Parse(
        source_text);
    const auto paged_manifest = frontend::ProgramArtifactFinalizer::Parse(
        paged_text);
    if (source_manifest.id != String(contract, "source_manifest_id") ||
        paged_manifest.id != String(contract, "paged_manifest_id") ||
        EncodeProgramArtifact(
            frontend::ProgramArtifactFinalizer{}.Finalize(paged_manifest)) !=
            program_bytes)
        Fail("paged native NPUP is not finalized from signed manifest");
    const Json source = Json::parse(source_text);
    const Json paged = Json::parse(paged_text);
    if (String(source, "source_ir1_id") !=
            String(contract, "source_ir1_id"))
        Fail("signed IR1 source identity changed");
    hbm_capacity_ = Number(contract, "hbm_capacity_bytes");
    external_capacity_ = Number(contract, "external_capacity_bytes");
    slot_bytes_ = Number(contract, "slot_bytes");
    const uint64_t slot_address = Number(contract, "slot_address");
    if (hbm_capacity_ != 4096 || external_capacity_ != 8192 ||
        slot_address != 0 || slot_bytes_ == 0 ||
        slot_bytes_ > hbm_capacity_ ||
        Number(contract, "link_bytes_per_cycle") == 0 ||
        Number(contract, "queue_depth") == 0 ||
        Number(contract, "max_outstanding") == 0)
        Fail("finite external/HBM blocking window is malformed");
    if (!contract.at("states").is_array() ||
        !contract.at("events").is_array())
        Fail("states and events must be arrays");
    uint64_t payload_bytes = 0;
    for (const auto &item : contract.at("states")) {
        Exact(item, {"state_ref", "source_abi_id", "paged_abi_id",
                     "kind", "external_address", "hbm_address",
                     "size_bytes", "seed_hex"});
        FullDenseAdamwStateSpan span{
            String(item, "state_ref"), String(item, "source_abi_id"),
            String(item, "paged_abi_id"), String(item, "kind"),
            Number(item, "external_address"), Number(item, "hbm_address"),
            Number(item, "size_bytes"), Hex(String(item, "seed_hex"))};
        if (span.seed.size() != span.size_bytes ||
            span.external_address + span.size_bytes > external_capacity_ ||
            span.hbm_address != slot_address ||
            span.size_bytes > slot_bytes_)
            Fail("physical external seed or HBM slot is out of bounds");
        payload_bytes += span.size_bytes;
        spans_.push_back(std::move(span));
    }
    if (payload_bytes != Number(contract, "state_payload_bytes") ||
        payload_bytes != 5716 || spans_.size() != 75 ||
        !std::is_sorted(spans_.begin(), spans_.end(),
            [](const auto &a, const auto &b) {
                return a.state_ref < b.state_ref;
            }))
        Fail("75 external state payloads are not canonical/exact");
    for (const auto &item : contract.at("events")) {
        Exact(item, {"index", "program_record_index", "step", "kind",
                     "state_ref", "external_address", "hbm_address",
                     "size_bytes"});
        events_.push_back({Number(item, "index"),
                           Number(item, "program_record_index"),
                           Number(item, "step"), String(item, "kind"),
                           String(item, "state_ref"),
                           Number(item, "external_address"),
                           Number(item, "hbm_address"),
                           Number(item, "size_bytes")});
    }
    VerifyManifestPair(source, paged, spans_, events_);
    external_ref_ = "full_dense_adamw_host";
    const std::string hbm_ref = "full_dense_adamw_hbm0";
    const std::string link_ref = "full_dense_adamw_shared_link";
    connection_ref_ = "full_dense_adamw_connection";
    FabricConfig fabric;
    fabric.external_capacities.push_back(
        {external_ref_, "host0", 0, external_capacity_});
    fabric.hbm_capacities.push_back({hbm_ref, 0, 0, hbm_capacity_});
    fabric.links.push_back({link_ref, external_ref_, 0,
        Number(contract, "link_bytes_per_cycle"),
        Number(contract, "link_latency_cycles"),
        Number(contract, "queue_depth"),
        Number(contract, "max_outstanding")});
    fabric.connections.push_back({connection_ref_, link_ref, hbm_ref,
                                  0, {0}, 0, std::nullopt});
    runtime_ = std::make_unique<ExternalMemoryRuntimeBridge>(
        name, std::move(fabric),
        std::map<std::string, HBMBackend *>{{hbm_ref, hbm}}, cycle_time_);
    for (const auto &span : spans_)
        runtime_->SeedExternal(external_ref_, span.external_address, span.seed);
    initial_digest_ = ProbeAuthority(0);
}

const FullDenseAdamwDmaEvent &FullDenseAdamwPager::Next(
    const char *kind, uint64_t address, uint64_t size) const {
    if (next_event_ >= events_.size())
        Fail("runtime issued an LSU beyond 350 signed state accesses");
    const auto &event = events_[next_event_];
    if (event.kind != kind || event.hbm_address != address ||
        event.size_bytes != size)
        Fail("actual LSU direction/slot/size differs from signed source at " +
             std::to_string(next_event_));
    return event;
}

void FullDenseAdamwPager::Transfer(
    const FullDenseAdamwDmaEvent &event, TransferDirection direction) {
    const uint64_t quantum = cycle_time_.value();
    const uint64_t remainder = sc_core::sc_time_stamp().value() % quantum;
    if (remainder)
        sc_core::wait(sc_core::sc_time::from_value(quantum - remainder));
    const uint64_t cycle = sc_core::sc_time_stamp().value() / quantum;
    const std::string id = "full_dense_adamw_" + std::to_string(next_event_);
    runtime_->Submit({id, connection_ref_, direction,
        event.external_address, event.hbm_address, event.size_bytes,
        cycle, kExternalDmaRequestSchemaVersion});
    const auto completion = runtime_->Wait(id);
    if (completion.status != 0 ||
        completion.payload_bytes != event.size_bytes)
        Fail("physical external DMA request did not complete: " +
             completion.error);
    std::cout << "[FULL_DENSE_ADAMW_DMA_EVENT] index=" << event.index
              << " step=" << event.step << " program_record="
              << event.program_record_index << " direction=" << event.kind
              << " state_ref=" << event.state_ref
              << " bytes=" << completion.payload_bytes
              << " issue_cycle=" << cycle
              << " completed_at_ticks=" << completion.completed_at.value()
              << " lsu_dependency_complete=1 pass=1" << std::endl;
}

void FullDenseAdamwPager::BeforeLoad(uint64_t address, uint64_t size) {
    const auto &event = Next("restore_before_lsu_load", address, size);
    Transfer(event, TransferDirection::kExternalToHbm);
    ++next_event_;
}

void FullDenseAdamwPager::AfterStore(uint64_t address, uint64_t size) {
    const auto &event = Next("writeback_after_lsu_store", address, size);
    Transfer(event, TransferDirection::kHbmToExternal);
    ++next_event_;
    if (next_event_ == 175 || next_event_ == 350) {
        const auto digest = ProbeAuthority(next_event_ / 175);
        if (next_event_ == 350) final_digest_ = digest;
    }
}

std::string FullDenseAdamwPager::ProbeAuthority(uint64_t version) {
    if (runtime_->Outstanding())
        Fail("external authority probed before DMA drain");
    std::vector<uint8_t> bytes;
    for (const auto &span : spans_) {
        const auto value = runtime_->ProbeExternal(
            external_ref_, span.external_address, span.size_bytes);
        if (value.size() != span.size_bytes)
            Fail("external authority readback was incomplete");
        if (version == 0 && value != span.seed)
            Fail("initial external authority differs from source seed");
        bytes.insert(bytes.end(), value.begin(), value.end());
    }
    if (bytes.size() != 5716)
        Fail("external authority omitted one physical state");
    const auto digest = frontend::program_io::Sha256Hex(bytes);
    std::cout << "[FULL_DENSE_ADAMW_EXTERNAL_STATE] version=" << version
              << " states=" << spans_.size() << " bytes=" << bytes.size()
              << " digest=" << digest
              << " pending=" << runtime_->Outstanding()
              << " functional=0 pass=1" << std::endl;
    return digest;
}

void FullDenseAdamwPager::RequireComplete() const {
    const auto &stats = runtime_->Stats();
    if (next_event_ != events_.size() || runtime_->Outstanding() ||
        final_digest_.empty() || stats.submitted_requests != 350 ||
        stats.completed_requests != 350 || stats.failed_requests != 0 ||
        stats.external_read_bytes != 14584 ||
        stats.external_write_bytes != 11432 ||
        stats.hbm_read_bytes != 11432 ||
        stats.hbm_write_bytes != 14584)
        Fail("full two-step DMA event/byte/authority drain incomplete");
}

uint64_t FullDenseAdamwPager::Outstanding() const {
    return runtime_->Outstanding();
}

const RuntimeStats &FullDenseAdamwPager::Stats() const {
    return runtime_->Stats();
}

} // namespace external_memory
