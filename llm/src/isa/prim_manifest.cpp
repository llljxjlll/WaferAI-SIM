#include "isa/prim_manifest.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

namespace {

using E = PrimManifestEntry;
using C = PrimCategory;
using V = PrimVisibility;
using L = PrimLifecycle;
using S = PrimSupport;

constexpr std::array<E, kPrimManifestSize> kManifest{{
    {PrimId::ATTENTION_F, "Attention_f", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::AVAILABLE},
    {PrimId::BATCHNORM_F, "Batchnorm_f", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::UNSUPPORTED},
    {PrimId::CONV_F, "Conv_f", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::AVAILABLE},
    {PrimId::DUMMY_P, "Dummy_p", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::AVAILABLE},
    {PrimId::GATE_FORWARD, "gate_forward", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::AVAILABLE},
    {PrimId::GELU_F, "Gelu_f", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::AVAILABLE},
    {PrimId::GEMM_RS_SWIZZLE, "Gemm_rs_swizzle", C::COMPUTE, V::PUBLIC,
     L::STABLE, S::EXPERIMENTAL},
    {PrimId::LAYERNORM_F, "Layernorm_f", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::AVAILABLE},
    {PrimId::MATMUL_F, "Matmul_f", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::AVAILABLE},
    {PrimId::MATMUL_F_MLA, "Matmul_f_mla", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::EXPERIMENTAL},
    {PrimId::MAX_POOL, "Max_pool", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::AVAILABLE},
    {PrimId::MERGE_CONV, "Merge_conv", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::UNSUPPORTED},
    {PrimId::MERGE_MATMUL, "Merge_matmul", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::AVAILABLE},
    {PrimId::PARSE_INPUT, "parse_input", C::MEMORY, V::INTERNAL, L::STABLE,
     S::AVAILABLE},
    {PrimId::PARSE_OUTPUT, "parse_output", C::MEMORY, V::INTERNAL, L::STABLE,
     S::AVAILABLE},
    {PrimId::RECV_GLOBAL_MEMORY, "Recv_global_memory", C::MEMORY, V::INTERNAL,
     L::STABLE, S::UNSUPPORTED},
    {PrimId::RELU_F, "Relu_f", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::AVAILABLE},
    {PrimId::RESIDUAL_F, "Residual_f", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::AVAILABLE},
    {PrimId::RMSNORM_FORWARD, "rmsnorm_forward", C::COMPUTE, V::PUBLIC,
     L::STABLE, S::AVAILABLE},
    {PrimId::ROPE_FORWARD, "rope_forward", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::AVAILABLE},
    {PrimId::SEND_GLOBAL_MEMORY, "Send_global_memory", C::MEMORY, V::INTERNAL,
     L::STABLE, S::UNSUPPORTED},
    {PrimId::SILU_FORWARD, "silu_forward", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::AVAILABLE},
    {PrimId::SPLIT_CONV, "Split_conv", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::UNSUPPORTED},
    {PrimId::SPLIT_MATMUL, "Split_matmul", C::COMPUTE, V::PUBLIC, L::STABLE,
     S::AVAILABLE},
    {PrimId::SWIGLU_FORWARD, "swiglu_forward", C::COMPUTE, V::PUBLIC,
     L::STABLE, S::AVAILABLE},
    {PrimId::SWITCH_DATA, "switch_data", C::MEMORY, V::INTERNAL, L::STABLE,
     S::AVAILABLE},
    {PrimId::ATTENTION_F_GPU, "Attention_f_gpu", C::COMPUTE, V::INTERNAL,
     L::STABLE, S::AVAILABLE},
    {PrimId::ATTENTION_FORWARD_GPU_PD, "attention_forward_gpu_pd", C::COMPUTE,
     V::INTERNAL, L::STABLE, S::AVAILABLE},
    {PrimId::GELU_F_GPU, "Gelu_f_gpu", C::COMPUTE, V::INTERNAL, L::STABLE,
     S::AVAILABLE},
    {PrimId::LAYERNORM_F_GPU, "Layernorm_f_gpu", C::COMPUTE, V::INTERNAL,
     L::STABLE, S::AVAILABLE},
    {PrimId::MATMUL_F_GPU, "Matmul_f_gpu", C::COMPUTE, V::INTERNAL, L::STABLE,
     S::AVAILABLE},
    {PrimId::MATMUL_FORWARD_GPU_PD, "matmul_forward_gpu_pd", C::COMPUTE,
     V::INTERNAL, L::STABLE, S::AVAILABLE},
    {PrimId::RESIDUAL_F_GPU, "Residual_f_gpu", C::COMPUTE, V::INTERNAL,
     L::STABLE, S::AVAILABLE},
    {PrimId::LOAD_EXPERT, "load_expert", C::MEMORY, V::INTERNAL, L::STABLE,
     S::AVAILABLE},
    {PrimId::MATMUL_FORWARD_MOE, "matmul_forward_moe", C::COMPUTE, V::PUBLIC,
     L::STABLE, S::AVAILABLE},
    {PrimId::CLEAR_SRAM, "Clear_sram", C::MEMORY, V::INTERNAL, L::STABLE,
     S::AVAILABLE},
    {PrimId::COLLECTIVE_DATA, "Collective_data_prim", C::COMMUNICATION,
     V::INTERNAL, L::STABLE, S::AVAILABLE},
    {PrimId::COLLECTIVE, "Collective_prim", C::SYNCHRONIZATION, V::INTERNAL,
     L::STABLE, S::AVAILABLE},
    {PrimId::DTE_ASYNC, "Dte_async", C::DYNAMIC, V::INTERNAL, L::STABLE,
     S::AVAILABLE},
    {PrimId::LEGACY_LOAD, "Load_prim", C::MEMORY, V::INTERNAL, L::DEPRECATED,
     S::AVAILABLE},
    {PrimId::LSU_MEM, "Lsu_mem", C::DYNAMIC, V::INTERNAL, L::STABLE,
     S::AVAILABLE},
    {PrimId::RECV, "Recv_prim", C::COMMUNICATION, V::INTERNAL, L::STABLE,
     S::AVAILABLE},
    {PrimId::REDUCE_COMPUTE, "Reduce_compute_prim", C::COMMUNICATION,
     V::INTERNAL, L::STABLE, S::AVAILABLE},
    {PrimId::SEND, "Send_prim", C::COMMUNICATION, V::INTERNAL, L::STABLE,
     S::AVAILABLE},
    {PrimId::SET_ADDR, "Set_addr", C::MEMORY, V::INTERNAL, L::STABLE,
     S::AVAILABLE},
    {PrimId::SET_BATCH, "Set_batch", C::MEMORY, V::INTERNAL, L::STABLE,
     S::AVAILABLE},
    {PrimId::SRAM_PIPELINE, "Sram_pipeline", C::MEMORY, V::INTERNAL, L::STABLE,
     S::EXPERIMENTAL},
    {PrimId::LEGACY_STORE, "Store_prim", C::MEMORY, V::INTERNAL,
     L::DEPRECATED, S::AVAILABLE},
    {PrimId::ATTENTION_FORWARD_PD, "Attention_f_pd", C::COMPUTE, V::PUBLIC,
     L::STABLE, S::EXPERIMENTAL},
    {PrimId::MATMUL_FORWARD_PD, "matmul_forward_pd", C::COMPUTE, V::PUBLIC,
     L::STABLE, S::EXPERIMENTAL},
    {PrimId::ROPE_FORWARD_PD, "rope_forward_pd", C::COMPUTE, V::PUBLIC,
     L::STABLE, S::EXPERIMENTAL},
    {PrimId::SRAM_BIND_ONESHOT, "Sram_bind_oneshot", C::MEMORY, V::INTERNAL,
     L::STABLE, S::AVAILABLE},
    {PrimId::SRAM_LIFECYCLE, "Sram_lifecycle", C::MEMORY, V::INTERNAL,
     L::STABLE, S::AVAILABLE},
    {PrimId::GROUP_SYNC, "Group_sync_prim", C::SYNCHRONIZATION, V::INTERNAL,
     L::STABLE, S::AVAILABLE},
    {PrimId::EVENT_CONTROL, "Event_control_prim", C::SYNCHRONIZATION,
     V::INTERNAL, L::STABLE, S::AVAILABLE},
    {PrimId::DTE_SEND_ENDPOINT, "Dte_send_endpoint_prim", C::COMMUNICATION,
     V::INTERNAL, L::STABLE, S::AVAILABLE},
    {PrimId::DTE_RECV_ENDPOINT, "Dte_recv_endpoint_prim", C::COMMUNICATION,
     V::INTERNAL, L::STABLE, S::AVAILABLE},
    {PrimId::COLLECTIVE_DATA_V1, "Collective_data_v1_prim",
     C::COMMUNICATION, V::INTERNAL, L::STABLE, S::AVAILABLE},
    {PrimId::COLLECTIVE_PHASE_BARRIER_V1,
     "Collective_phase_barrier_v1_prim", C::SYNCHRONIZATION,
     V::INTERNAL, L::STABLE, S::AVAILABLE},
    {PrimId::COLLECTIVE_LAUNCH_V1, "Collective_launch_v1_prim",
     C::COMMUNICATION, V::INTERNAL, L::STABLE, S::AVAILABLE},
    {PrimId::ROPE_QK_EXACT, "Rope_qk_exact_prim", C::COMPUTE, V::PUBLIC,
     L::STABLE, S::AVAILABLE},
    {PrimId::ATTENTION_EXACT, "Attention_exact_prim", C::COMPUTE, V::PUBLIC,
     L::STABLE, S::AVAILABLE},
    {PrimId::EMBEDDING_LOOKUP, "Embedding_lookup_prim", C::COMPUTE,
     V::PUBLIC, L::STABLE, S::AVAILABLE},
    {PrimId::GREEDY_SAMPLE, "Greedy_sample_prim", C::COMPUTE, V::PUBLIC,
     L::STABLE, S::AVAILABLE},
    {PrimId::CROSS_ENTROPY_FORWARD, "Cross_entropy_forward_prim",
     C::COMPUTE, V::PUBLIC, L::STABLE, S::AVAILABLE},
    {PrimId::CROSS_ENTROPY_BACKWARD, "Cross_entropy_backward_prim",
     C::COMPUTE, V::PUBLIC, L::STABLE, S::AVAILABLE},
    {PrimId::SGD_UPDATE, "Sgd_update_prim", C::COMPUTE, V::PUBLIC,
     L::STABLE, S::AVAILABLE},
}};

bool SetError(std::string *error, std::string message) {
    if (error != nullptr)
        *error = std::move(message);
    return false;
}

bool ValidCategory(PrimCategory category) noexcept {
    return category == C::COMPUTE || category == C::COMMUNICATION ||
           category == C::MEMORY || category == C::SYNCHRONIZATION ||
           category == C::DYNAMIC;
}


bool ValidVisibility(PrimVisibility visibility) noexcept {
    return visibility == V::PUBLIC || visibility == V::INTERNAL;
}

bool ValidLifecycle(PrimLifecycle lifecycle) noexcept {
    return lifecycle == L::STABLE || lifecycle == L::DEPRECATED ||
           lifecycle == L::RESERVED || lifecycle == L::TOMBSTONE;
}

bool ValidSupport(PrimSupport support) noexcept {
    return support == S::AVAILABLE || support == S::UNSUPPORTED ||
           support == S::EXPERIMENTAL;
}

} // namespace

const std::array<PrimManifestEntry, kPrimManifestSize> &
PrimManifest() noexcept {
    return kManifest;
}

const PrimManifestEntry *LookupPrim(uint16_t raw_id) noexcept {
    if (raw_id == 0 || raw_id > UINT8_MAX)
        return nullptr;
    const auto it = std::lower_bound(
        kManifest.begin(), kManifest.end(), raw_id,
        [](const PrimManifestEntry &entry, uint16_t candidate) {
            return static_cast<uint16_t>(PrimIdValue(entry.id)) < candidate;
        });
    if (it == kManifest.end() || PrimIdValue(it->id) != raw_id)
        return nullptr;
    return &*it;
}

const PrimManifestEntry *LookupPrim(std::string_view factory_name) noexcept {
    const auto it = std::find_if(
        kManifest.begin(), kManifest.end(),
        [factory_name](const PrimManifestEntry &entry) {
            return entry.factory_name == factory_name;
        });
    return it == kManifest.end() ? nullptr : &*it;
}

bool ValidatePrimManifestEntries(
    const std::vector<PrimManifestEntry> &entries, std::string *error) {
    if (error != nullptr)
        error->clear();
    if (entries.size() != kPrimManifestSize)
        return SetError(error, "Prim manifest must contain exactly " +
                                   std::to_string(kPrimManifestSize) +
                                   " entries");

    std::array<bool, kPrimManifestSize + 1> seen_ids{};
    std::vector<std::string_view> seen_names;
    seen_names.reserve(entries.size());
    for (const PrimManifestEntry &entry : entries) {
        const uint16_t raw_id = PrimIdValue(entry.id);
        if (raw_id == 0)
            return SetError(error, "PrimId::INVALID cannot appear in manifest");
        if (raw_id > kPrimManifestSize)
            return SetError(error, "PrimId is outside frozen contiguous range");
        if (seen_ids[raw_id])
            return SetError(error, "duplicate PrimId " +
                                       std::to_string(raw_id));
        seen_ids[raw_id] = true;
        if (entry.factory_name.empty())
            return SetError(error, "factory name must not be empty");
        if (std::find(seen_names.begin(), seen_names.end(),
                      entry.factory_name) != seen_names.end()) {
            return SetError(error, "duplicate factory name: " +
                                       std::string(entry.factory_name));
        }
        seen_names.push_back(entry.factory_name);
        if (!ValidCategory(entry.primary_category))
            return SetError(error, "invalid primary category: " +
                                       std::string(entry.factory_name));
        if (!ValidVisibility(entry.visibility))
            return SetError(error, "invalid visibility: " +
                                       std::string(entry.factory_name));
        if (!ValidLifecycle(entry.lifecycle))
            return SetError(error, "invalid lifecycle: " +
                                       std::string(entry.factory_name));
        if (!ValidSupport(entry.support))
            return SetError(error, "invalid support: " +
                                       std::string(entry.factory_name));
        const bool must_be_dynamic = entry.id == PrimId::DTE_ASYNC ||
                                     entry.id == PrimId::LSU_MEM;
        if (must_be_dynamic !=
            (entry.primary_category == PrimCategory::DYNAMIC)) {
            return SetError(error, "dynamic category mismatch: " +
                                       std::string(entry.factory_name));
        }
        if ((entry.lifecycle == L::RESERVED ||
             entry.lifecycle == L::TOMBSTONE) &&
            entry.support == S::AVAILABLE) {
            return SetError(error, "reserved/tombstone Prim cannot be available: " +
                                       std::string(entry.factory_name));
        }
    }
    for (std::size_t raw_id = 1; raw_id <= kPrimManifestSize; ++raw_id) {
        if (!seen_ids[raw_id])
            return SetError(error, "missing PrimId " +
                                       std::to_string(raw_id));
    }
    return true;
}

bool ValidatePrimManifest(std::string *error) {
    const std::vector<PrimManifestEntry> entries(kManifest.begin(),
                                                 kManifest.end());
    if (!ValidatePrimManifestEntries(entries, error))
        return false;
    for (std::size_t i = 0; i < kManifest.size(); ++i) {
        if (PrimIdValue(kManifest[i].id) != i + 1)
            return SetError(error, "Prim manifest is not sorted and contiguous");
    }
    if (kMaxAssignedPrimId > UINT8_MAX)
        return SetError(error, "PrimId exceeds the 8-bit internal wire");
    return true;
}
