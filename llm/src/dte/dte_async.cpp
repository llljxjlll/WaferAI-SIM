#include "dte/dte_async.h"

#include "trace/Event_engine.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

uint64_t DteAggregationMetrics::BandwidthUtilizationPpm() const {
    if (bus_capacity_bits == 0)
        return 0;
    const long double ratio =
        static_cast<long double>(useful_payload_bits) * 1000000.0L /
        static_cast<long double>(bus_capacity_bits);
    return static_cast<uint64_t>(ratio + 0.5L);
}

void ValidateDteAggregationConfig(const DteAggregationConfig &config) {
    if (!config.enabled)
        return;
    if (config.max_descriptors < 2)
        throw std::invalid_argument(
            "DTE V3b aggregation_max_descriptors must be >= 2");
    if (config.max_payload_bytes == 0)
        throw std::invalid_argument(
            "DTE V3b aggregation_max_bytes must be > 0");
    if (config.timeout_cycles == 0)
        throw std::invalid_argument(
            "DTE V3b aggregation_timeout must be > 0 cycles");
    if (config.address_block_bytes == 0)
        throw std::invalid_argument(
            "DTE V3b aggregation_address_block_bytes must be > 0");
}

DteAsyncTracker::DteAsyncTracker(
    const sc_module_name &name, DTEUnit &unit,
    const DteAggregationConfig &aggregation, int core_id,
    Event_engine *event_engine)
    : sc_module(name), unit_(unit), aggregation_(aggregation),
      core_id_(core_id), event_engine_(event_engine) {
    ValidateDteAggregationConfig(aggregation_);
    SC_THREAD(timeoutWorker);
}

DteAsyncAccess DteAsyncTracker::AccessForDirection(DteDir dir) {
    switch (dir) {
    case DteDir::SPM_TO_REMOTE:
    case DteDir::SPM_TO_DRAM:
        return DteAsyncAccess::READ;
    case DteDir::REMOTE_TO_SPM:
    case DteDir::DRAM_TO_SPM:
        return DteAsyncAccess::WRITE;
    case DteDir::SPM_TO_SPM:
        return DteAsyncAccess::READ_WRITE;
    case DteDir::DRAM_TO_REMOTE:
        return DteAsyncAccess::NONE;
    }
    throw std::invalid_argument("DTE async direction is invalid");
}

bool DteAsyncTracker::RangeOverlaps(uint64_t lhs_addr, uint64_t lhs_size,
                                    uint64_t rhs_addr, uint64_t rhs_size) {
    if (lhs_size == 0 || rhs_size == 0)
        return false;
    return lhs_addr < rhs_addr + rhs_size &&
           rhs_addr < lhs_addr + lhs_size;
}

bool DteAsyncTracker::HasHazard(const DteAsyncRecord &existing,
                                const DteAsyncRecord &incoming) {
    return RangeOverlaps(existing.spm_write_addr,
                         existing.spm_write_size,
                         incoming.spm_read_addr,
                         incoming.spm_read_size) ||
           RangeOverlaps(existing.spm_write_addr,
                         existing.spm_write_size,
                         incoming.spm_write_addr,
                         incoming.spm_write_size) ||
           RangeOverlaps(existing.spm_read_addr,
                         existing.spm_read_size,
                         incoming.spm_write_addr,
                         incoming.spm_write_size);
}

void DteAsyncTracker::PopulateSpmRanges(DteAsyncRecord &record) {
    switch (record.direction) {
    case DteDir::SPM_TO_REMOTE:
    case DteDir::SPM_TO_DRAM:
        record.spm_read_addr = record.spm_addr;
        record.spm_read_size = record.spm_size;
        break;
    case DteDir::REMOTE_TO_SPM:
    case DteDir::DRAM_TO_SPM:
        record.spm_write_addr = record.spm_addr;
        record.spm_write_size = record.spm_size;
        break;
    case DteDir::SPM_TO_SPM:
        record.spm_read_addr = record.spm_addr;
        record.spm_read_size = record.spm_size;
        record.spm_write_addr = record.remote_addr;
        record.spm_write_size = record.spm_size;
        break;
    case DteDir::DRAM_TO_REMOTE:
        break;
    }
}

void DteAsyncTracker::ValidateIssue(
    uint32_t token, uint64_t payload_bits, DteDir dir,
    uint64_t spm_addr, uint64_t spm_size, uint32_t remote_peer,
    uint64_t remote_addr, uint32_t address_block) const {
    if (records_.count(token))
        throw std::invalid_argument(
            "DTE async logical token is already outstanding");
    if (payload_bits == 0)
        throw std::invalid_argument("DTE async payload_bits must be > 0");
    (void)AccessForDirection(dir);
    const bool v4_direction =
        dir == DteDir::SPM_TO_SPM || dir == DteDir::SPM_TO_DRAM ||
        dir == DteDir::DRAM_TO_SPM || dir == DteDir::DRAM_TO_REMOTE;
    if (v4_direction && !unit_.config().fine_grained_resources)
        throw std::invalid_argument(
            "DTE V4 direction requires fine_grained_resources=true");

    const uint64_t payload_bytes = CeilDivU64(payload_bits, uint64_t(8));
    if (dir == DteDir::DRAM_TO_REMOTE) {
        if (spm_addr != 0 || spm_size != 0)
            throw std::invalid_argument(
                "DRAM_TO_REMOTE must not declare a local SPM range");
    } else {
        if (spm_size == 0)
            throw std::invalid_argument("DTE async spm_size must be > 0");
        if (spm_addr > std::numeric_limits<uint64_t>::max() - spm_size)
            throw std::overflow_error("DTE async SPM address range overflows");
        if (payload_bytes > spm_size)
            throw std::invalid_argument(
                "DTE async payload does not fit in the declared SPM range");
    }
    const uint64_t remote_range_bytes =
        dir == DteDir::SPM_TO_SPM ? spm_size : payload_bytes;
    if ((dir == DteDir::SPM_TO_SPM || dir == DteDir::SPM_TO_DRAM ||
         dir == DteDir::DRAM_TO_SPM || dir == DteDir::DRAM_TO_REMOTE) &&
        remote_addr > std::numeric_limits<uint64_t>::max() -
                          remote_range_bytes)
        throw std::overflow_error(
            "DTE V4 secondary address range overflows");
    if (dir == DteDir::DRAM_TO_REMOTE &&
        remote_peer == DTE_ASYNC_INVALID_REMOTE_PEER)
        throw std::invalid_argument(
            "DRAM_TO_REMOTE requires remote_peer");

    if (!aggregation_.enabled)
        return;
    if (dir != DteDir::SPM_TO_REMOTE && dir != DteDir::REMOTE_TO_SPM)
        throw std::invalid_argument(
            "DTE V3b aggregation supports only remote SPM endpoint directions");
    if ((payload_bits % 8) != 0 || payload_bytes != spm_size)
        throw std::invalid_argument(
            "DTE V3b aggregation requires a byte-aligned payload whose "
            "size equals spm_size");
    if (remote_peer == DTE_ASYNC_INVALID_REMOTE_PEER)
        throw std::invalid_argument(
            "DTE V3b aggregation requires remote_peer");
    if (remote_addr >
        std::numeric_limits<uint64_t>::max() - payload_bytes)
        throw std::overflow_error(
            "DTE V3b remote address range overflows");
    const uint64_t first_block =
        remote_addr / aggregation_.address_block_bytes;
    const uint64_t last_block =
        (remote_addr + payload_bytes - 1) /
        aggregation_.address_block_bytes;
    if (first_block != last_block || first_block != address_block)
        throw std::invalid_argument(
            "DTE V3b descriptor must fit in its declared address_block");
}

bool DteAsyncTracker::CanAppend(const OpenGroup &group,
                                uint64_t payload_bits,
                                uint64_t spm_addr, uint64_t spm_size,
                                uint64_t remote_addr) const {
    const uint64_t payload_bytes = payload_bits / 8;
    if (group.tokens.size() >= aggregation_.max_descriptors)
        return false;
    if (group.total_payload_bytes >
        aggregation_.max_payload_bytes - payload_bytes)
        return false;
    return group.next_spm_addr == spm_addr &&
           group.next_remote_addr == remote_addr &&
           spm_size == payload_bytes;
}

uint64_t DteAsyncTracker::IssuePhysical(
    const std::vector<uint32_t> &tokens, uint64_t payload_bits, DteDir dir) {
    if (tokens.empty())
        throw std::logic_error("DTE async physical batch has no tokens");
    unit_.WaitForCredit();
    DteTransferContext &context = unit_.Issue(payload_bits, dir);
    PhysicalBatch batch;
    batch.xfer_id = context.xfer_id;
    batch.context = &context;
    batch.remaining_tokens = tokens.size();
    batch.member_count = tokens.size();
    batch.payload_bits = payload_bits;
    physical_batches_.emplace(context.xfer_id, batch);

    for (uint32_t token : tokens) {
        DteAsyncRecord &record = records_.at(token);
        record.xfer_id = context.xfer_id;
        record.context = &context;
        record.staged = false;
    }

    ++aggregation_metrics_.physical_transfers;
    aggregation_metrics_.useful_payload_bits += payload_bits;
    const uint64_t accounting_width =
        unit_.config().fine_grained_resources
            ? (dir == DteDir::SPM_TO_REMOTE
                   ? unit_.config().spm_read_width_bits
                   : unit_.config().spm_write_width_bits)
            : unit_.config().bit_width_bits;
    const uint64_t cycles = CeilDivU64(payload_bits, accounting_width);
    if (cycles > std::numeric_limits<uint64_t>::max() /
                     accounting_width)
        throw std::overflow_error(
            "DTE aggregation bus capacity accounting overflows");
    aggregation_metrics_.bus_capacity_bits +=
        cycles * accounting_width;
    if (tokens.size() > 1) {
        aggregation_metrics_.coalesced_descriptors += tokens.size();
        aggregation_metrics_.launch_savings += tokens.size() - 1;
    }
    return context.xfer_id;
}

uint64_t DteAsyncTracker::StartAggregationGroup(uint32_t token) {
    DteAsyncRecord &record = records_.at(token);
    OpenGroup group;
    group.group_id = next_group_id_++;
    group.key = GroupKey(static_cast<uint8_t>(record.direction),
                         record.remote_peer, record.address_block);
    group.direction = record.direction;
    group.tokens.push_back(token);
    group.total_payload_bits = record.payload_bits;
    group.total_payload_bytes = record.payload_bits / 8;
    group.next_spm_addr = record.spm_addr + record.spm_size;
    group.next_remote_addr = record.remote_addr + record.payload_bits / 8;
    group.first_issue_sequence = record.issue_sequence;
    group.deadline = sc_time_stamp() +
                     sc_time(aggregation_.timeout_cycles * CYCLE, SC_NS);
    record.group_id = group.group_id;
    record.staged = true;
    group_by_key_[group.key] = group.group_id;
    open_groups_.emplace(group.group_id, group);
    aggregation_changed_.notify(SC_ZERO_TIME);
    return group.group_id;
}

uint64_t DteAsyncTracker::AppendToAggregationGroup(uint64_t group_id,
                                                   uint32_t token) {
    OpenGroup &group = open_groups_.at(group_id);
    DteAsyncRecord &record = records_.at(token);
    group.tokens.push_back(token);
    group.total_payload_bits += record.payload_bits;
    group.total_payload_bytes += record.payload_bits / 8;
    group.next_spm_addr = record.spm_addr + record.spm_size;
    group.next_remote_addr = record.remote_addr + record.payload_bits / 8;
    record.group_id = group_id;
    record.staged = true;
    Trace("DTE_coalesce_collect", "B", token,
          DTE_ASYNC_INVALID_XFER_ID,
          "group=" + std::to_string(group_id) +
              " members=" + std::to_string(group.tokens.size()));
    Trace("DTE_coalesce_collect", "E", token,
          DTE_ASYNC_INVALID_XFER_ID,
          "group=" + std::to_string(group_id) +
              " members=" + std::to_string(group.tokens.size()));
    return group_id;
}

uint64_t DteAsyncTracker::FlushGroup(uint64_t group_id,
                                     const char *reason) {
    auto it = open_groups_.find(group_id);
    if (it == open_groups_.end())
        return DTE_ASYNC_INVALID_XFER_ID;
    OpenGroup group = it->second;
    const uint64_t xfer =
        IssuePhysical(group.tokens, group.total_payload_bits, group.direction);
    group_by_key_.erase(group.key);
    open_groups_.erase(it);
    aggregation_changed_.notify(SC_ZERO_TIME);

    for (uint32_t token : group.tokens) {
        const std::string bind_extra =
            "group=" + std::to_string(group_id) +
            " head=" + std::to_string(group.tokens.front()) +
            " members=" + std::to_string(group.tokens.size());
        Trace("DTE_coalesce_bind", "B", token, xfer, bind_extra);
        Trace("DTE_coalesce_bind", "E", token, xfer, bind_extra);
    }

    const uint32_t head = group.tokens.front();
    const std::string extra =
        "group=" + std::to_string(group_id) +
        " members=" + std::to_string(group.tokens.size()) +
        " bits=" + std::to_string(group.total_payload_bits) +
        " reason=" + reason +
        " saved=" + std::to_string(group.tokens.size() - 1) +
        " cumulative_utilization_ppm=" +
        std::to_string(aggregation_metrics_.BandwidthUtilizationPpm());
    Trace("DTE_coalesce_flush", "B", head, xfer, extra);
    Trace("DTE_coalesce_flush", "E", head, xfer, extra);
    return xfer;
}

void DteAsyncTracker::FlushAllOpenGroups(const char *reason) {
    std::vector<std::pair<uint64_t, uint64_t>> ordered;
    ordered.reserve(open_groups_.size());
    for (const auto &[group_id, group] : open_groups_)
        ordered.emplace_back(group.first_issue_sequence, group_id);
    std::sort(ordered.begin(), ordered.end());
    for (const auto &[sequence, group_id] : ordered) {
        (void)sequence;
        FlushGroup(group_id, reason);
    }
}

uint64_t DteAsyncTracker::IssueToken(
    uint32_t token, uint64_t payload_bits, DteDir dir,
    uint64_t spm_addr, uint64_t spm_size, uint32_t remote_peer,
    uint64_t remote_addr, uint32_t address_block) {
    ValidateIssue(token, payload_bits, dir, spm_addr, spm_size,
                  remote_peer, remote_addr, address_block);

    const DteAsyncAccess access = AccessForDirection(dir);
    DteAsyncRecord incoming;
    incoming.direction = dir;
    incoming.spm_addr = spm_addr;
    incoming.spm_size = spm_size;
    incoming.remote_addr = remote_addr;
    incoming.access = access;
    PopulateSpmRanges(incoming);
    std::vector<std::pair<uint64_t, uint32_t>> conflicts;
    for (const auto &[other_token, record] : records_) {
        const bool complete =
            record.context != nullptr &&
            record.context->state == DteTransferState::COMPLETED;
        if (!complete && HasHazard(record, incoming))
            conflicts.emplace_back(record.issue_sequence, other_token);
    }
    std::sort(conflicts.begin(), conflicts.end());
    for (const auto &[sequence, other_token] : conflicts) {
        (void)sequence;
        const uint64_t other_xfer = records_.at(other_token).xfer_id;
        Trace("DTE_async_hazard", "B", token, other_xfer,
              "depends_on=" + std::to_string(other_token));
        WaitForCompletion(other_token, false);
        Trace("DTE_async_hazard", "E", token,
              records_.at(other_token).xfer_id,
              "depends_on=" + std::to_string(other_token));
    }

    DteAsyncRecord record;
    record.token = token;
    record.payload_bits = payload_bits;
    record.direction = dir;
    record.spm_addr = spm_addr;
    record.spm_size = spm_size;
    record.access = access;
    record.issue_sequence = next_issue_sequence_++;
    record.remote_peer = remote_peer;
    record.remote_addr = remote_addr;
    record.address_block = address_block;
    PopulateSpmRanges(record);
    records_.emplace(token, record);
    ++aggregation_metrics_.logical_descriptors;

    uint64_t xfer = DTE_ASYNC_INVALID_XFER_ID;
    std::string issue_extra;
    if (!aggregation_.enabled) {
        xfer = IssuePhysical({token}, payload_bits, dir);
    } else {
        const uint64_t payload_bytes = payload_bits / 8;
        const GroupKey key(static_cast<uint8_t>(dir), remote_peer,
                           address_block);
        if (payload_bytes > aggregation_.max_payload_bytes) {
            xfer = IssuePhysical({token}, payload_bits, dir);
            const std::string extra =
                "group=standalone members=1 bits=" +
                std::to_string(payload_bits) +
                " reason=oversize saved=0 cumulative_utilization_ppm=" +
                std::to_string(
                    aggregation_metrics_.BandwidthUtilizationPpm());
            Trace("DTE_coalesce_flush", "B", token, xfer, extra);
            Trace("DTE_coalesce_flush", "E", token, xfer, extra);
        } else {
            auto key_it = group_by_key_.find(key);
            uint64_t group_id;
            if (key_it == group_by_key_.end()) {
                group_id = StartAggregationGroup(token);
            } else {
                const uint64_t current_id = key_it->second;
                const OpenGroup &current = open_groups_.at(current_id);
                if (CanAppend(current, payload_bits, spm_addr, spm_size,
                              remote_addr)) {
                    group_id = AppendToAggregationGroup(current_id, token);
                } else {
                    const bool contiguous =
                        current.next_spm_addr == spm_addr &&
                        current.next_remote_addr == remote_addr;
                    FlushGroup(current_id,
                               contiguous ? "limit" : "incompatible");
                    group_id = StartAggregationGroup(token);
                }
            }
            const OpenGroup &group = open_groups_.at(group_id);
            if (group.tokens.size() >= aggregation_.max_descriptors ||
                group.total_payload_bytes >=
                    aggregation_.max_payload_bytes)
                xfer = FlushGroup(group_id, "limit");
            else
                issue_extra = "staged=1 group=" +
                              std::to_string(group_id);
        }
    }

    Trace("DTE_async_issue", "B", token, xfer, issue_extra);
    Trace("DTE_async_issue", "E", token, xfer, issue_extra);
    return xfer;
}

void DteAsyncTracker::WaitForCompletion(uint32_t token, bool trace_wait) {
    auto it = records_.find(token);
    if (it == records_.end())
        throw std::invalid_argument(
            "DTE async wait references an unknown token");
    if (it->second.staged)
        FlushGroup(it->second.group_id, "dependency");

    DteAsyncRecord &record = records_.at(token);
    if (record.context == nullptr ||
        record.context->xfer_id != record.xfer_id)
        throw std::logic_error("DTE async token/context mapping is corrupt");
    if (trace_wait)
        Trace("DTE_async_wait", "B", token, record.xfer_id);
    if (record.context->state != DteTransferState::COMPLETED)
        wait(record.context->done);
    if (record.context->state != DteTransferState::COMPLETED)
        throw std::logic_error(
            "DTE async wait woke without a completed transfer");
}

void DteAsyncTracker::WaitAndRelease(uint32_t token, bool trace_wait) {
    auto it = records_.find(token);
    if (it == records_.end())
        throw std::invalid_argument(
            "DTE async wait references an unknown token");
    WaitForCompletion(token, trace_wait);
    const uint64_t xfer_id = records_.at(token).xfer_id;
    auto batch_it = physical_batches_.find(xfer_id);
    if (batch_it == physical_batches_.end() ||
        batch_it->second.remaining_tokens == 0)
        throw std::logic_error("DTE async physical batch accounting is corrupt");

    records_.erase(token);
    --batch_it->second.remaining_tokens;
    if (batch_it->second.remaining_tokens == 0) {
        if (!unit_.Release(xfer_id))
            throw std::logic_error(
                "DTE async completed transfer context could not be released");
        physical_batches_.erase(batch_it);
    }
    if (trace_wait)
        Trace("DTE_async_wait", "E", token, xfer_id);
}

void DteAsyncTracker::WaitToken(uint32_t token) {
    WaitAndRelease(token, true);
}

bool DteAsyncTracker::PollToken(uint32_t token) {
    auto it = records_.find(token);
    if (it == records_.end())
        throw std::invalid_argument(
            "DTE async poll references an unknown token");
    const DteAsyncRecord &record = it->second;
    if (record.staged) {
        const std::string extra =
            "complete=0 staged=1 group=" +
            std::to_string(record.group_id);
        Trace("DTE_async_poll", "B", token,
              DTE_ASYNC_INVALID_XFER_ID, extra);
        Trace("DTE_async_poll", "E", token,
              DTE_ASYNC_INVALID_XFER_ID, extra);
        return false;
    }
    if (record.context == nullptr ||
        record.context->xfer_id != record.xfer_id)
        throw std::logic_error("DTE async token/context mapping is corrupt");
    const bool complete =
        record.context->state == DteTransferState::COMPLETED;
    const std::string extra = std::string("complete=") +
                              (complete ? "1" : "0");
    Trace("DTE_async_poll", "B", token, record.xfer_id, extra);
    Trace("DTE_async_poll", "E", token, record.xfer_id, extra);
    return complete;
}

void DteAsyncTracker::Fence() {
    Trace("DTE_async_fence", "B", std::numeric_limits<uint32_t>::max(), 0,
          "count=" + std::to_string(records_.size()));
    FlushAllOpenGroups("fence");
    std::vector<std::pair<uint64_t, uint32_t>> ordered;
    ordered.reserve(records_.size());
    for (const auto &[token, record] : records_)
        ordered.emplace_back(record.issue_sequence, token);
    std::sort(ordered.begin(), ordered.end());
    for (const auto &[sequence, token] : ordered) {
        (void)sequence;
        if (records_.count(token))
            WaitAndRelease(token, true);
    }
    Trace("DTE_async_fence", "E", std::numeric_limits<uint32_t>::max(), 0,
          "count=0");
}

void DteAsyncTracker::CancelStagedToken(uint32_t token) {
    DteAsyncRecord record = records_.at(token);
    auto group_it = open_groups_.find(record.group_id);
    if (group_it == open_groups_.end())
        throw std::logic_error("DTE staged token has no open group");
    OpenGroup &group = group_it->second;
    if (group.tokens.empty() || group.tokens.back() != token)
        throw std::runtime_error(
            "DTE V3b can cancel only the tail of an open aggregation group");

    group.tokens.pop_back();
    group.total_payload_bits -= record.payload_bits;
    group.total_payload_bytes -= record.payload_bits / 8;
    records_.erase(token);
    if (group.tokens.empty()) {
        group_by_key_.erase(group.key);
        open_groups_.erase(group_it);
    } else {
        const DteAsyncRecord &tail = records_.at(group.tokens.back());
        group.next_spm_addr = tail.spm_addr + tail.spm_size;
        group.next_remote_addr =
            tail.remote_addr + tail.payload_bits / 8;
    }
    aggregation_changed_.notify(SC_ZERO_TIME);
}

void DteAsyncTracker::CancelToken(uint32_t token) {
    auto it = records_.find(token);
    if (it == records_.end())
        throw std::invalid_argument(
            "DTE async cancel references an unknown token");
    const uint64_t xfer_id = it->second.xfer_id;
    Trace("DTE_async_cancel", "B", token, xfer_id);
    if (it->second.staged) {
        CancelStagedToken(token);
        Trace("DTE_async_cancel", "E", token, xfer_id,
              "staged=1");
        return;
    }

    auto batch_it = physical_batches_.find(xfer_id);
    if (batch_it == physical_batches_.end())
        throw std::logic_error("DTE async physical batch is absent");
    if (batch_it->second.member_count != 1)
        throw std::runtime_error(
            "DTE V3b cannot cancel one token from an issued compound "
            "descriptor");
    if (!unit_.Cancel(xfer_id))
        throw std::runtime_error(
            "DTE async can cancel only a pending transfer");
    if (!unit_.Release(xfer_id))
        throw std::logic_error(
            "DTE async cancelled transfer context could not be released");
    physical_batches_.erase(batch_it);
    records_.erase(it);
    Trace("DTE_async_cancel", "E", token, xfer_id);
}

const DteAsyncRecord &DteAsyncTracker::Record(uint32_t token) const {
    auto it = records_.find(token);
    if (it == records_.end())
        throw std::invalid_argument(
            "DTE async record references an unknown token");
    return it->second;
}

std::vector<uint32_t> DteAsyncTracker::OutstandingTokens() const {
    std::vector<std::pair<uint64_t, uint32_t>> ordered;
    ordered.reserve(records_.size());
    for (const auto &[token, record] : records_)
        ordered.emplace_back(record.issue_sequence, token);
    std::sort(ordered.begin(), ordered.end());
    std::vector<uint32_t> tokens;
    tokens.reserve(ordered.size());
    for (const auto &[sequence, token] : ordered) {
        (void)sequence;
        tokens.push_back(token);
    }
    return tokens;
}

void DteAsyncTracker::timeoutWorker() {
    while (true) {
        if (!aggregation_.enabled || open_groups_.empty()) {
            wait(aggregation_changed_);
            continue;
        }

        sc_time earliest = open_groups_.begin()->second.deadline;
        for (const auto &[group_id, group] : open_groups_) {
            (void)group_id;
            earliest = std::min(earliest, group.deadline);
        }
        if (sc_time_stamp() < earliest) {
            wait(earliest - sc_time_stamp(), aggregation_changed_);
            continue;
        }

        std::vector<std::pair<uint64_t, uint64_t>> due;
        for (const auto &[group_id, group] : open_groups_) {
            if (group.deadline <= sc_time_stamp())
                due.emplace_back(group.first_issue_sequence, group_id);
        }
        std::sort(due.begin(), due.end());
        for (const auto &[sequence, group_id] : due) {
            (void)sequence;
            FlushGroup(group_id, "timeout");
        }
    }
}

void DteAsyncTracker::Trace(const char *stage, const char *phase,
                            uint32_t token, uint64_t xfer_id,
                            const std::string &extra) const {
    if (event_engine_ == nullptr)
        return;
    std::ostringstream detail;
    detail << stage << " core=" << core_id_ << " token=" << token
           << " xfer=" << xfer_id << " outstanding=" << records_.size();
    if (!extra.empty())
        detail << " " << extra;
    const std::string object =
        "DTE_async_" + std::to_string(core_id_) + "_" +
        std::to_string(token);
    const unsigned trace_id =
        xfer_id == DTE_ASYNC_INVALID_XFER_ID
            ? 0U
            : static_cast<unsigned>(xfer_id);
    event_engine_->add_event(object, stage, phase,
                             Trace_event_util(detail.str()), SC_ZERO_TIME,
                             trace_id);
}
