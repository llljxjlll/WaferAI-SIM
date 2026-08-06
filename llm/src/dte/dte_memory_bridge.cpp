#include "dte/dte_memory_bridge.h"

#include "trace/Event_engine.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

DteMemoryBridge::DteMemoryBridge(sc_module_name name,
                                 sram::RegionTable &regions,
                                 sram::AccessUnit &sram_access,
                                 sram::HbmByteTransport &hbm,
                                 uint32_t queue_depth, uint32_t workers,
                                 Event_engine *event_engine, int core_id)
    : sc_module(name), regions_(regions), sram_access_(sram_access), hbm_(hbm),
      queue_depth_(queue_depth), worker_count_(workers),
      event_engine_(event_engine), core_id_(core_id) {
    if (queue_depth_ == 0 || worker_count_ == 0)
        throw std::invalid_argument(
            "DTE memory queue depth and worker count must be non-zero");
    for (uint32_t worker = 0; worker < worker_count_; ++worker)
        sc_spawn(sc_bind(&DteMemoryBridge::Worker, this),
                 sc_gen_unique_name("dte-memory-worker"));
}

void DteMemoryBridge::Validate(uint32_t token, DteDir direction,
                               uint64_t hbm_addr, uint64_t sram_addr,
                               uint64_t size_bytes) const {
    if (records_.count(token) != 0)
        throw std::invalid_argument(
            "DTE memory logical token is already outstanding");
    if (direction != DteDir::DRAM_TO_SPM &&
        direction != DteDir::SPM_TO_DRAM &&
        direction != DteDir::SPM_TO_SPM)
        throw std::invalid_argument(
            "DTE memory bridge supports DRAM_TO_SPM, SPM_TO_DRAM, "
            "and SPM_TO_SPM");
    if (size_bytes == 0)
        throw std::invalid_argument(
            "DTE memory descriptor size must be non-zero");
    if (hbm_addr > std::numeric_limits<uint64_t>::max() - size_bytes)
        throw std::overflow_error(
            direction == DteDir::SPM_TO_SPM
                ? "DTE memory destination SRAM range overflows uint64_t"
                : "DTE memory HBM range overflows uint64_t");
    const auto source_command = direction == DteDir::DRAM_TO_SPM
                                    ? sram::Command::kWrite
                                    : sram::Command::kRead;
    regions_.ResolveAbsolute(sram_addr, size_bytes, sram::Initiator::kDte,
                             source_command);
    if (direction == DteDir::SPM_TO_SPM)
        regions_.ResolveAbsolute(hbm_addr, size_bytes,
                                 sram::Initiator::kDte,
                                 sram::Command::kWrite);
}

void DteMemoryBridge::Reserve(uint32_t token, DteDir direction,
                              uint64_t hbm_addr, uint64_t sram_addr,
                              uint64_t size_bytes) {
    Validate(token, direction, hbm_addr, sram_addr, size_bytes);
    if (pending_.size() + reserved_ >= queue_depth_)
        throw std::runtime_error("DTE memory issue queue is full");
    auto record = std::make_shared<Record>();
    record->token = token;
    record->direction = direction;
    record->hbm_addr = hbm_addr;
    record->sram_addr = sram_addr;
    record->size_bytes = size_bytes;
    record->issue_time = sc_time_stamp();
    const uint64_t lease_group = static_cast<uint64_t>(token) + 1;
    try {
        if (direction == DteDir::DRAM_TO_SPM) {
            record->hazard_leases.push_back(
                sram_access_.DeclareRangeLease(
                    sram::Initiator::kDte, sram::Command::kWrite,
                    sram_addr, size_bytes, lease_group));
        } else {
            record->hazard_leases.push_back(
                sram_access_.DeclareRangeLease(
                    sram::Initiator::kDte, sram::Command::kRead,
                    sram_addr, size_bytes, lease_group));
            if (direction == DteDir::SPM_TO_SPM)
                record->hazard_leases.push_back(
                    sram_access_.DeclareRangeLease(
                        sram::Initiator::kDte, sram::Command::kWrite,
                        hbm_addr, size_bytes, lease_group));
        }
    } catch (...) {
        for (uint64_t lease : record->hazard_leases)
            sram_access_.ReleaseRangeLease(lease);
        throw;
    }
    records_.emplace(token, record);
    ++reserved_;
}

void DteMemoryBridge::Commit(uint32_t token) {
    const auto record = Find(token);
    if (record->status != Status::kReserved)
        throw std::logic_error("DTE memory token is not reserved");
    if (reserved_ == 0)
        throw std::logic_error("DTE memory reservation accounting underflow");
    --reserved_;
    record->status = Status::kQueued;
    pending_.push_back(record);
    ++stats_.issued;
    stats_.peak_outstanding =
        std::max<uint64_t>(stats_.peak_outstanding, records_.size());
    TraceStage("DTE_mem_commit", "B", token);
    pending_changed_.notify(SC_ZERO_TIME);
}

void DteMemoryBridge::Abort(uint32_t token) {
    const auto record = Find(token);
    if (record->status != Status::kReserved)
        throw std::runtime_error("can abort only a reserved DTE memory token");
    for (uint64_t lease : record->hazard_leases)
        sram_access_.ReleaseRangeLease(lease);
    if (reserved_ == 0)
        throw std::logic_error("DTE memory reservation accounting underflow");
    --reserved_;
    records_.erase(token);
}

void DteMemoryBridge::Issue(uint32_t token, DteDir direction,
                            uint64_t hbm_addr, uint64_t sram_addr,
                            uint64_t size_bytes) {
    Reserve(token, direction, hbm_addr, sram_addr, size_bytes);
    try {
        Commit(token);
    } catch (...) {
        Abort(token);
        throw;
    }
}

std::shared_ptr<DteMemoryBridge::Record>
DteMemoryBridge::Find(uint32_t token) const {
    const auto it = records_.find(token);
    if (it == records_.end())
        throw std::out_of_range("unknown DTE memory token");
    return it->second;
}

void DteMemoryBridge::Wait(uint32_t token) {
    const auto record = Find(token);
    while (record->status == Status::kQueued ||
           record->status == Status::kRunning)
        wait(record->done);
    if (record->status == Status::kCancelled)
        throw std::runtime_error("DTE memory token was cancelled");
    if (record->status == Status::kFailed) {
        if (record->error) std::rethrow_exception(record->error);
        throw std::runtime_error("DTE memory token failed");
    }
}

bool DteMemoryBridge::Poll(uint32_t token) const {
    const auto status = Find(token)->status;
    return status == Status::kComplete || status == Status::kCancelled ||
           status == Status::kFailed;
}

bool DteMemoryBridge::CanCancel(uint32_t token) const {
    return Find(token)->status == Status::kQueued;
}

void DteMemoryBridge::Cancel(uint32_t token) {
    const auto record = Find(token);
    if (record->status != Status::kQueued)
        throw std::runtime_error(
            "DTE memory can cancel only a queued transfer");
    const auto it = std::find(pending_.begin(), pending_.end(), record);
    if (it == pending_.end())
        throw std::logic_error("queued DTE memory token is absent");
    pending_.erase(it);
    for (uint64_t lease : record->hazard_leases)
        sram_access_.ReleaseRangeLease(lease);
    record->hazard_leases.clear();
    record->status = Status::kCancelled;
    ++stats_.cancelled;
    TraceStage("DTE_mem_commit", "E", token, "cancelled=1");
    record->done.notify(SC_ZERO_TIME);
}

void DteMemoryBridge::Release(uint32_t token) {
    const auto record = Find(token);
    if (record->status == Status::kQueued ||
        record->status == Status::kRunning)
        throw std::runtime_error(
            "cannot release an incomplete DTE memory token");
    records_.erase(token);
}

void DteMemoryBridge::TraceStage(const char *stage, const char *phase,
                                 uint32_t token,
                                 const std::string &extra) const {
    if (event_engine_ == nullptr) return;
    std::string detail = "token=" + std::to_string(token);
    if (!extra.empty()) detail += " " + extra;
    event_engine_->add_event(
        "DTE_mem_" + std::to_string(core_id_), stage, phase,
        Trace_event_util(detail), SC_ZERO_TIME, token);
}

void DteMemoryBridge::Worker() {
    while (true) {
        if (pending_.empty()) {
            wait(pending_changed_);
            continue;
        }
        const auto record = pending_.front();
        pending_.pop_front();
        if (record->status == Status::kCancelled) continue;
        record->status = Status::kRunning;
        record->run_time = sc_time_stamp();
        ++running_;
        stats_.peak_running =
            std::max<uint64_t>(stats_.peak_running, running_);
        bool axi_stage_open = false;
        bool hbm_stage_open = false;
        bool spm_stage_open = false;
        try {
            for (uint64_t lease : record->hazard_leases)
                sram_access_.WaitRangeLease(lease);
            if (record->direction == DteDir::DRAM_TO_SPM) {
                TraceStage("DTE_mem_axi", "B", record->token, "read=1");
                axi_stage_open = true;
                TraceStage("DTE_mem_hbm", "B", record->token, "read=1");
                hbm_stage_open = true;
                auto payload =
                    hbm_.Read(record->hbm_addr, record->size_bytes);
                TraceStage("DTE_mem_hbm", "E", record->token, "read=1");
                hbm_stage_open = false;
                TraceStage("DTE_mem_axi", "E", record->token, "read=1");
                axi_stage_open = false;
                stats_.hbm_read_bytes += record->size_bytes;
                sram::Request request;
                request.initiator = sram::Initiator::kDte;
                request.command = sram::Command::kWrite;
                request.address = record->sram_addr;
                request.size_bytes = record->size_bytes;
                request.payload = std::move(payload);
                request.hazard_lease = record->hazard_leases.front();
                TraceStage("DTE_mem_spm", "B", record->token, "write=1");
                spm_stage_open = true;
                sram_access_.Access(request);
                TraceStage("DTE_mem_spm", "E", record->token, "write=1");
                spm_stage_open = false;
                stats_.sram_write_bytes += record->size_bytes;
            } else if (record->direction == DteDir::SPM_TO_DRAM) {
                sram::Request request;
                request.initiator = sram::Initiator::kDte;
                request.command = sram::Command::kRead;
                request.address = record->sram_addr;
                request.size_bytes = record->size_bytes;
                request.hazard_lease = record->hazard_leases.front();
                TraceStage("DTE_mem_spm", "B", record->token, "read=1");
                spm_stage_open = true;
                auto payload = sram_access_.Access(request).payload;
                TraceStage("DTE_mem_spm", "E", record->token, "read=1");
                spm_stage_open = false;
                stats_.sram_read_bytes += record->size_bytes;
                TraceStage("DTE_mem_axi", "B", record->token, "write=1");
                axi_stage_open = true;
                TraceStage("DTE_mem_hbm", "B", record->token, "write=1");
                hbm_stage_open = true;
                hbm_.Write(record->hbm_addr, payload);
                TraceStage("DTE_mem_hbm", "E", record->token, "write=1");
                hbm_stage_open = false;
                TraceStage("DTE_mem_axi", "E", record->token, "write=1");
                axi_stage_open = false;
                stats_.hbm_write_bytes += record->size_bytes;
            } else {
                sram::Request read;
                read.initiator = sram::Initiator::kDte;
                read.command = sram::Command::kRead;
                read.address = record->sram_addr;
                read.size_bytes = record->size_bytes;
                read.hazard_lease = record->hazard_leases.at(0);
                TraceStage("DTE_mem_spm", "B", record->token, "copy_read=1");
                spm_stage_open = true;
                auto payload = sram_access_.Access(read).payload;
                TraceStage("DTE_mem_spm", "E", record->token, "copy_read=1");
                spm_stage_open = false;
                stats_.sram_read_bytes += record->size_bytes;

                sram::Request write;
                write.initiator = sram::Initiator::kDte;
                write.command = sram::Command::kWrite;
                write.address = record->hbm_addr;
                write.size_bytes = record->size_bytes;
                write.payload = std::move(payload);
                write.hazard_lease = record->hazard_leases.at(1);
                TraceStage("DTE_mem_spm", "B", record->token, "copy_write=1");
                spm_stage_open = true;
                sram_access_.Access(write);
                TraceStage("DTE_mem_spm", "E", record->token, "copy_write=1");
                spm_stage_open = false;
                stats_.sram_write_bytes += record->size_bytes;
                stats_.sram_copy_bytes += record->size_bytes;
            }
            for (uint64_t lease : record->hazard_leases)
                sram_access_.ReleaseRangeLease(lease);
            record->hazard_leases.clear();
            record->status = Status::kComplete;
            ++stats_.completed;
        } catch (...) {
            if (spm_stage_open)
                TraceStage("DTE_mem_spm", "E", record->token, "failed=1");
            if (hbm_stage_open)
                TraceStage("DTE_mem_hbm", "E", record->token, "failed=1");
            if (axi_stage_open)
                TraceStage("DTE_mem_axi", "E", record->token, "failed=1");
            for (uint64_t lease : record->hazard_leases) {
                try {
                    sram_access_.ReleaseRangeLease(lease);
                } catch (...) {
                }
            }
            record->hazard_leases.clear();
            record->error = std::current_exception();
            record->status = Status::kFailed;
            ++stats_.failed;
        }
        --running_;
        DteMemoryTraceRecord trace;
        trace.token = record->token;
        trace.direction = record->direction;
        trace.source_addr = record->direction == DteDir::DRAM_TO_SPM
                                ? record->hbm_addr
                                : record->sram_addr;
        trace.destination_addr = record->direction == DteDir::DRAM_TO_SPM
                                     ? record->sram_addr
                                     : record->hbm_addr;
        trace.size_bytes = record->size_bytes;
        trace.issue_time = record->issue_time;
        trace.run_time = record->run_time;
        trace.completion_time = sc_time_stamp();
        trace_.push_back(trace);
        TraceStage("DTE_mem_commit", "E", record->token);
        record->done.notify(SC_ZERO_TIME);
    }
}
