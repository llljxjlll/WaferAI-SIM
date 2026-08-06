#include "memory/core_lsu_unit.h"

#include "trace/Event_engine.h"

#include <limits>
#include <stdexcept>

namespace sram {

CoreLsuUnit::CoreLsuUnit(sc_module_name name, RegionTable &regions,
                         AccessUnit &sram_access, HbmByteTransport &hbm,
                         uint32_t queue_depth, uint32_t max_outstanding,
                         uint64_t issue_latency_ns,
                         Event_engine *event_engine, int core_id)
    : sc_module(name), regions_(regions), sram_access_(sram_access), hbm_(hbm),
      queue_("queue", queue_depth), queue_depth_(queue_depth),
      max_outstanding_(max_outstanding),
      issue_latency_ns_(issue_latency_ns), event_engine_(event_engine),
      core_id_(core_id) {
    if (queue_depth == 0 || max_outstanding == 0 ||
        max_outstanding > queue_depth)
        throw std::invalid_argument(
            "LSU requires 0 < max_outstanding <= queue_depth");
    for (uint32_t worker = 0; worker < max_outstanding_; ++worker)
        sc_spawn(sc_bind(&CoreLsuUnit::Worker, this),
                 sc_gen_unique_name("lsu-worker"));
}

void CoreLsuUnit::Validate(const LsuDescriptor &descriptor) const {
    if (descriptor.size_bytes == 0)
        throw std::invalid_argument("LSU descriptor size must be non-zero");
    if (descriptor.hbm_addr >
        std::numeric_limits<uint64_t>::max() - descriptor.size_bytes)
        throw std::overflow_error("LSU HBM range overflows uint64_t");
    if (!descriptor.byte_enable.empty() &&
        descriptor.byte_enable.size() != descriptor.size_bytes)
        throw std::invalid_argument(
            "LSU byte-enable length must equal descriptor size");
    const Command command =
        descriptor.direction == LsuDirection::kHbmToSram
            ? Command::kWrite
            : Command::kRead;
    regions_.ResolveAbsolute(descriptor.sram_addr, descriptor.size_bytes,
                             Initiator::kLsu, command);
}

std::shared_ptr<CoreLsuUnit::TokenRecord>
CoreLsuUnit::Find(LsuToken token) const {
    const auto it = records_.find(token);
    if (it == records_.end())
        throw std::out_of_range("unknown or already consumed LSU token");
    return it->second;
}

LsuToken CoreLsuUnit::Issue(const LsuDescriptor &descriptor,
                            LsuToken requested_token) {
    Validate(descriptor);
    if (records_.size() >= queue_depth_)
        throw std::runtime_error("LSU issue queue capacity exhausted");
    auto record = std::make_shared<TokenRecord>();
    if (requested_token != 0) {
        if (records_.count(requested_token) != 0)
            throw std::invalid_argument("duplicate outstanding LSU token");
        record->token = requested_token;
    } else {
        while (next_token_ != 0 && records_.count(next_token_) != 0)
            ++next_token_;
        if (next_token_ == 0)
            throw std::overflow_error("LSU token space exhausted");
        record->token = next_token_++;
    }
    const sc_time issue_begin = sc_time_stamp();
    record->descriptor = descriptor;
    record->issue_time = issue_begin;
    TraceStage("LSU_issue", "B", record->token);
    try {
    if (issue_latency_ns_ != 0)
        wait(sc_time(static_cast<double>(issue_latency_ns_), SC_NS));
    stats_.issue_latency_ns +=
        (sc_time_stamp() - issue_begin).to_seconds() * 1e9;
    const Command lease_command =
        descriptor.direction == LsuDirection::kHbmToSram
            ? Command::kWrite
            : Command::kRead;
    record->hazard_lease = sram_access_.DeclareRangeLease(
        Initiator::kLsu, lease_command, descriptor.sram_addr,
        descriptor.size_bytes);
    if (!queue_.nb_write(record)) {
        sram_access_.ReleaseRangeLease(record->hazard_lease);
        throw std::runtime_error("LSU issue queue is full");
    }
    records_.emplace(record->token, record);
    ++stats_.issued;
    stats_.peak_outstanding =
        std::max<uint64_t>(stats_.peak_outstanding, records_.size());
    TraceStage("LSU_issue", "E", record->token);
    return record->token;
    } catch (...) {
        TraceStage("LSU_issue", "E", record->token, "failed=1");
        throw;
    }
}

LsuToken CoreLsuUnit::IssueLoad(uint64_t hbm_addr, uint64_t sram_addr,
                                uint64_t size_bytes) {
    return Issue({LsuDirection::kHbmToSram, hbm_addr, sram_addr, size_bytes,
                  {}});
}

LsuToken CoreLsuUnit::IssueStore(uint64_t sram_addr, uint64_t hbm_addr,
                                 uint64_t size_bytes,
                                 std::vector<uint8_t> byte_enable) {
    return Issue({LsuDirection::kSramToHbm, hbm_addr, sram_addr, size_bytes,
                  std::move(byte_enable)});
}

LsuToken CoreLsuUnit::IssueLoadRegion(uint64_t hbm_addr,
                                      std::string_view region,
                                      uint64_t offset,
                                      uint64_t size_bytes) {
    const auto range = regions_.Resolve(region, offset, size_bytes,
                                        Initiator::kLsu, Command::kWrite);
    return IssueLoad(hbm_addr, range.address, range.size_bytes);
}

LsuToken CoreLsuUnit::IssueStoreRegion(
    std::string_view region, uint64_t offset, uint64_t hbm_addr,
    uint64_t size_bytes, std::vector<uint8_t> byte_enable) {
    const auto range = regions_.Resolve(region, offset, size_bytes,
                                        Initiator::kLsu, Command::kRead);
    return IssueStore(range.address, hbm_addr, range.size_bytes,
                      std::move(byte_enable));
}

void CoreLsuUnit::Wait(LsuToken token) {
    const auto record = Find(token);
    TraceStage("LSU_wait", "B", token);
    while (record->status == LsuTokenStatus::kQueued ||
           record->status == LsuTokenStatus::kRunning)
        wait(record->done);
    const auto error = record->error;
    const auto status = record->status;
    records_.erase(token);
    TraceStage("LSU_wait", "E", token,
               "status=" + std::to_string(static_cast<int>(status)));
    if (status == LsuTokenStatus::kCancelled)
        throw std::runtime_error("LSU token was cancelled");
    if (status == LsuTokenStatus::kFailed) {
        if (error) std::rethrow_exception(error);
        throw std::runtime_error("LSU token failed");
    }
}

bool CoreLsuUnit::Poll(LsuToken token) const {
    const auto status = Find(token)->status;
    return status == LsuTokenStatus::kComplete ||
           status == LsuTokenStatus::kCancelled ||
           status == LsuTokenStatus::kFailed;
}

bool CoreLsuUnit::Cancel(LsuToken token) {
    const auto record = Find(token);
    if (record->status != LsuTokenStatus::kQueued) return false;
    record->cancel_requested = true;
    return true;
}

void CoreLsuUnit::Fence() {
    std::vector<LsuToken> tokens;
    tokens.reserve(records_.size());
    for (const auto &entry : records_) tokens.push_back(entry.first);
    for (LsuToken token : tokens) Wait(token);
}

void CoreLsuUnit::Load(uint64_t hbm_addr, uint64_t sram_addr,
                       uint64_t size_bytes) {
    Wait(IssueLoad(hbm_addr, sram_addr, size_bytes));
}

void CoreLsuUnit::Store(uint64_t sram_addr, uint64_t hbm_addr,
                        uint64_t size_bytes,
                        std::vector<uint8_t> byte_enable) {
    Wait(IssueStore(sram_addr, hbm_addr, size_bytes,
                    std::move(byte_enable)));
}

void CoreLsuUnit::Complete(const std::shared_ptr<TokenRecord> &record,
                           LsuTokenStatus status) {
    record->status = status;
    LsuTraceRecord trace;
    trace.token = record->token;
    trace.direction = record->descriptor.direction;
    trace.hbm_addr = record->descriptor.hbm_addr;
    trace.sram_addr = record->descriptor.sram_addr;
    trace.size_bytes = record->descriptor.size_bytes;
    trace.issue_time = record->issue_time;
    trace.run_time = record->run_time;
    trace.completion_time = sc_time_stamp();
    trace_.push_back(trace);
    record->done.notify(SC_ZERO_TIME);
}

void CoreLsuUnit::TraceStage(const char *stage, const char *phase,
                             LsuToken token,
                             const std::string &extra) const {
    if (event_engine_ == nullptr) return;
    std::string detail = "token=" + std::to_string(token);
    if (!extra.empty()) detail += " " + extra;
    event_engine_->add_event(
        "LSU_" + std::to_string(core_id_), stage, phase,
        Trace_event_util(detail), SC_ZERO_TIME,
        static_cast<unsigned>(token));
}

void CoreLsuUnit::Worker() {
    while (true) {
        const auto record = queue_.read();
        if (record->cancel_requested) {
            sram_access_.ReleaseRangeLease(record->hazard_lease);
            ++stats_.cancelled;
            Complete(record, LsuTokenStatus::kCancelled);
            continue;
        }
        record->status = LsuTokenStatus::kRunning;
        record->run_time = sc_time_stamp();
        ++running_;
        stats_.peak_running = std::max<uint64_t>(stats_.peak_running, running_);
        bool hbm_stage_open = false;
        bool sram_stage_open = false;
        try {
            sram_access_.WaitRangeLease(record->hazard_lease);
            const auto &d = record->descriptor;
            if (d.direction == LsuDirection::kHbmToSram) {
                TraceStage("LSU_hbm", "B", record->token, "read=1");
                hbm_stage_open = true;
                auto payload = hbm_.Read(d.hbm_addr, d.size_bytes);
                TraceStage("LSU_hbm", "E", record->token, "read=1");
                hbm_stage_open = false;
                stats_.hbm_read_bytes += d.size_bytes;
                Request request;
                request.initiator = Initiator::kLsu;
                request.command = Command::kWrite;
                request.address = d.sram_addr;
                request.size_bytes = d.size_bytes;
                request.payload = std::move(payload);
                request.byte_enable = d.byte_enable;
                request.hazard_lease = record->hazard_lease;
                TraceStage("LSU_sram", "B", record->token, "write=1");
                sram_stage_open = true;
                sram_access_.Access(request);
                TraceStage("LSU_sram", "E", record->token, "write=1");
                sram_stage_open = false;
                stats_.sram_write_bytes += d.size_bytes;
            } else {
                Request request;
                request.initiator = Initiator::kLsu;
                request.command = Command::kRead;
                request.address = d.sram_addr;
                request.size_bytes = d.size_bytes;
                request.hazard_lease = record->hazard_lease;
                TraceStage("LSU_sram", "B", record->token, "read=1");
                sram_stage_open = true;
                auto payload = sram_access_.Access(request).payload;
                TraceStage("LSU_sram", "E", record->token, "read=1");
                sram_stage_open = false;
                stats_.sram_read_bytes += d.size_bytes;
                TraceStage("LSU_hbm", "B", record->token, "write=1");
                hbm_stage_open = true;
                hbm_.Write(d.hbm_addr, payload, d.byte_enable);
                TraceStage("LSU_hbm", "E", record->token, "write=1");
                hbm_stage_open = false;
                stats_.hbm_write_bytes += d.size_bytes;
            }
            sram_access_.ReleaseRangeLease(record->hazard_lease);
            ++stats_.completed;
            --running_;
            Complete(record, LsuTokenStatus::kComplete);
        } catch (...) {
            if (sram_stage_open)
                TraceStage("LSU_sram", "E", record->token, "failed=1");
            if (hbm_stage_open)
                TraceStage("LSU_hbm", "E", record->token, "failed=1");
            try {
                sram_access_.ReleaseRangeLease(record->hazard_lease);
            } catch (...) {
            }
            record->error = std::current_exception();
            ++stats_.failed;
            --running_;
            Complete(record, LsuTokenStatus::kFailed);
        }
    }
}

} // namespace sram
