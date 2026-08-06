#include "memory/sram/sram_access_unit.h"

#include "macros/macros.h"
#include "trace/Event_engine.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sram {
namespace {

size_t InitiatorIndex(Initiator initiator) {
    return static_cast<size_t>(initiator);
}

size_t CommandIndex(Command command) {
    return static_cast<size_t>(command);
}

uint64_t CyclesBetween(const sc_time &begin, const sc_time &end) {
    if (end <= begin) return 0;
    const double cycle_ns = static_cast<double>(CYCLE);
    if (cycle_ns <= 0.0) return 0;
    return static_cast<uint64_t>(
        std::llround((end - begin).to_seconds() * 1.0e9 / cycle_ns));
}

bool IsWriteLike(Command command) {
    return command == Command::kWrite || command == Command::kClear;
}

} // namespace

AccessUnit::AccessUnit(sc_module_name name, RegionTable &regions,
                       Storage &storage, Event_engine *event_engine,
                       int core_id)
    : sc_module(name), regions_(regions), storage_(storage),
      bank_busy_(regions.config().bank_count, false),
      event_engine_(event_engine), core_id_(core_id) {
    stats_.bank_requests.resize(regions_.config().bank_count, 0);
    stats_.banks.resize(regions_.config().bank_count);
    regions_.SetRangeBusyProbe(
        [this](ByteRange range) { return IsRangeBusy(range); });
}

AccessUnit::~AccessUnit() {
    regions_.SetRangeBusyProbe({});
}

const InitiatorPortConfig &AccessUnit::Ports(Initiator initiator) const {
    const auto &config = regions_.config();
    switch (initiator) {
    case Initiator::kCompute: return config.compute;
    case Initiator::kDte: return config.dte;
    case Initiator::kLsu: return config.lsu;
    case Initiator::kNocRx: return config.noc_rx;
    case Initiator::kLegacy: return config.legacy;
    }
    throw std::invalid_argument("invalid SRAM initiator");
}

const PortConfig &AccessUnit::Port(Initiator initiator,
                                   Command command) const {
    const auto &ports = Ports(initiator);
    return command == Command::kRead ? ports.read : ports.write;
}

uint32_t AccessUnit::BankForAddress(uint64_t address) const {
    const auto &config = regions_.config();
    return static_cast<uint32_t>(
        (address / config.bank_interleave_bytes) % config.bank_count);
}

std::vector<uint32_t> AccessUnit::BanksForRange(uint64_t address,
                                                uint64_t size_bytes) const {
    if (size_bytes == 0)
        throw std::invalid_argument("SRAM request size must be non-zero");
    const uint64_t end = ByteRange{address, size_bytes}.End();
    std::vector<uint32_t> banks;
    uint64_t cursor = address;
    while (cursor < end) {
        const uint32_t bank = BankForAddress(cursor);
        if (std::find(banks.begin(), banks.end(), bank) == banks.end())
            banks.push_back(bank);
        const uint64_t stripe_end =
            ((cursor / regions_.config().bank_interleave_bytes) + 1) *
            regions_.config().bank_interleave_bytes;
        cursor = std::min(end, stripe_end);
    }
    return banks;
}

bool AccessUnit::HasHazard(const Request &request,
                           const ActiveAccess &active) const {
    if (!Overlaps({request.address, request.size_bytes},
                  {active.request.address, active.request.size_bytes}))
        return false;
    return IsWriteLike(request.command) ||
           IsWriteLike(active.request.command);
}

bool AccessUnit::IsRangeBusy(ByteRange range) const {
    if (range.Empty()) return false;
    for (const auto &item : active_) {
        if (Overlaps(range,
                     {item->request.address, item->request.size_bytes}))
            return true;
    }
    return false;
}

std::vector<AccessUnit::Beat>
AccessUnit::BuildBeats(const Request &request) const {
    const uint64_t width_bytes = Port(request.initiator, request.command)
                                     .width_bits /
                                 8;
    const uint64_t end = ByteRange{request.address, request.size_bytes}.End();
    std::vector<Beat> beats;
    uint64_t cursor = request.address;
    while (cursor < end) {
        const uint64_t stripe_end =
            ((cursor / regions_.config().bank_interleave_bytes) + 1) *
            regions_.config().bank_interleave_bytes;
        const uint64_t beat_end =
            std::min(end, std::min(stripe_end, cursor + width_bytes));
        beats.push_back(
            {cursor, beat_end - cursor, BankForAddress(cursor)});
        cursor = beat_end;
    }
    return beats;
}

void AccessUnit::TryGrantBeats() {
    while (!pending_beats_.empty()) {
        std::shared_ptr<BeatWaiter> selected;
        size_t selected_pos = 0;
        size_t selected_initiator = 0;

        for (auto &waiter : pending_beats_) {
            const size_t ii = InitiatorIndex(waiter->initiator);
            const size_t ci = CommandIndex(waiter->command);
            const bool bank_block = bank_busy_.at(waiter->bank);
            const bool port_block =
                port_users_[ii][ci] >=
                Port(waiter->initiator, waiter->command).count;
            waiter->saw_bank_block |= bank_block;
            waiter->saw_port_block |= port_block;
        }

        for (size_t offset = 0; offset < 5 && !selected; ++offset) {
            const size_t wanted = (rr_next_initiator_ + offset) % 5;
            for (size_t pos = 0; pos < pending_beats_.size(); ++pos) {
                const auto &candidate = pending_beats_[pos];
                if (InitiatorIndex(candidate->initiator) != wanted)
                    continue;
                const size_t ci = CommandIndex(candidate->command);
                if (bank_busy_.at(candidate->bank) ||
                    port_users_[wanted][ci] >=
                        Port(candidate->initiator, candidate->command).count)
                    continue;
                selected = candidate;
                selected_pos = pos;
                selected_initiator = wanted;
                break;
            }
        }
        if (!selected) return;

        pending_beats_.erase(pending_beats_.begin() + selected_pos);
        const size_t ci = CommandIndex(selected->command);
        bank_busy_.at(selected->bank) = true;
        ++port_users_[selected_initiator][ci];
        selected->granted = true;
        rr_next_initiator_ = (selected_initiator + 1) % 5;
        selected->granted_event.notify(SC_ZERO_TIME);
    }
}

std::shared_ptr<AccessUnit::BeatWaiter>
AccessUnit::AcquireBeat(const Beat &beat, const Request &request,
                        uint64_t trace_request_id) {
    auto waiter = std::make_shared<BeatWaiter>();
    waiter->initiator = request.initiator;
    waiter->command = request.command;
    waiter->bank = beat.bank;
    waiter->size_bytes = beat.size_bytes;
    waiter->sequence = next_beat_sequence_++;
    waiter->enqueue_time = sc_time_stamp();
    pending_beats_.push_back(waiter);
    TryGrantBeats();
    if (!waiter->granted) {
        TraceStage("SRAM_bank_wait", "B", trace_request_id,
                   "bank=" + std::to_string(beat.bank));
        while (!waiter->granted) wait(waiter->granted_event);
        TraceStage("SRAM_bank_wait", "E", trace_request_id,
                   "bank=" + std::to_string(beat.bank));
    }

    const uint64_t stall =
        CyclesBetween(waiter->enqueue_time, sc_time_stamp());
    auto &counter =
        stats_.by_initiator_command[InitiatorIndex(request.initiator)]
                                   [CommandIndex(request.command)];
    if (waiter->saw_bank_block) ++counter.bank_stalls;
    if (waiter->saw_port_block) ++counter.port_stalls;
    stats_.banks.at(beat.bank).stall_cycles += stall;
    stats_.ports[InitiatorIndex(request.initiator)]
                [CommandIndex(request.command)]
                    .stall_cycles += stall;
    return waiter;
}

void AccessUnit::ReleaseBeat(
    const std::shared_ptr<BeatWaiter> &waiter) {
    const size_t ii = InitiatorIndex(waiter->initiator);
    const size_t ci = CommandIndex(waiter->command);
    if (!bank_busy_.at(waiter->bank) || port_users_[ii][ci] == 0)
        throw std::logic_error("SRAM beat resource accounting underflow");
    bank_busy_.at(waiter->bank) = false;
    --port_users_[ii][ci];

    auto &bank = stats_.banks.at(waiter->bank);
    ++bank.beats;
    bank.bytes += waiter->size_bytes;
    bank.service_cycles += waiter->service_cycles;
    auto &port = stats_.ports[ii][ci];
    ++port.beats;
    port.bytes += waiter->size_bytes;
    port.service_cycles += waiter->service_cycles;
    ++stats_.bank_requests.at(waiter->bank);
    TryGrantBeats();
}

void AccessUnit::RemoveActive(
    const std::shared_ptr<ActiveAccess> &active) {
    const auto it = std::find(active_.begin(), active_.end(), active);
    if (it != active_.end()) active_.erase(it);
    if (admitted_ != 0) --admitted_;
    active->done.notify(SC_ZERO_TIME);
    capacity_available_.notify(SC_ZERO_TIME);
}

uint64_t AccessUnit::DeclareRangeLease(Initiator initiator, Command command,
                                       uint64_t address,
                                       uint64_t size_bytes,
                                       uint64_t group_id) {
    regions_.ResolveAbsolute(address, size_bytes, initiator, command);
    auto lease = std::make_shared<ActiveAccess>();
    lease->request.initiator = initiator;
    lease->request.command = command;
    lease->request.address = address;
    lease->request.size_bytes = size_bytes;
    lease->hazard_only = true;
    lease->group_id = group_id;
    while (next_lease_id_ == 0 || leases_.count(next_lease_id_) != 0)
        ++next_lease_id_;
    lease->lease_id = next_lease_id_++;
    for (const auto &candidate : active_) {
        if (group_id != 0 && candidate->group_id == group_id)
            continue;
        if (HasHazard(lease->request, *candidate))
            lease->dependencies.push_back(candidate);
    }
    active_.push_back(lease);
    leases_.emplace(lease->lease_id, lease);
    return lease->lease_id;
}

void AccessUnit::WaitRangeLease(uint64_t lease_id) {
    const auto it = leases_.find(lease_id);
    if (it == leases_.end())
        throw std::out_of_range("unknown SRAM range lease");
    for (const auto &dependency : it->second->dependencies) {
        while (std::find(active_.begin(), active_.end(), dependency) !=
               active_.end())
            wait(dependency->done);
    }
}

void AccessUnit::ReleaseRangeLease(uint64_t lease_id) {
    const auto it = leases_.find(lease_id);
    if (it == leases_.end())
        throw std::out_of_range("unknown SRAM range lease");
    const auto lease = it->second;
    const auto active_it = std::find(active_.begin(), active_.end(), lease);
    if (active_it == active_.end())
        throw std::logic_error("SRAM range lease is absent from active set");
    active_.erase(active_it);
    leases_.erase(it);
    lease->done.notify(SC_ZERO_TIME);
}

void AccessUnit::TraceStage(const char *stage, const char *phase,
                            uint64_t trace_request_id,
                            const std::string &extra) const {
    if (event_engine_ == nullptr) return;
    std::string detail = "request=" + std::to_string(trace_request_id);
    if (!extra.empty()) detail += " " + extra;
    event_engine_->add_event(
        "SRAM_" + std::to_string(core_id_), stage, phase,
        Trace_event_util(detail), SC_ZERO_TIME,
        static_cast<unsigned>(trace_request_id));
}

Response AccessUnit::Access(const Request &request) {
    Response response;
    response.issue_time = sc_time_stamp();
    const uint64_t trace_request_id = next_trace_id_++;
    if (request.size_bytes == 0)
        throw std::invalid_argument("SRAM request size must be non-zero");
    if (request.command == Command::kWrite &&
        request.payload.size() != request.size_bytes)
        throw std::invalid_argument(
            "SRAM write payload length must equal request size");
    if (request.command != Command::kWrite && !request.payload.empty())
        throw std::invalid_argument(
            "SRAM read/clear must not carry a payload");
    if (!request.byte_enable.empty() &&
        (request.command != Command::kWrite ||
         request.byte_enable.size() != request.size_bytes))
        throw std::invalid_argument(
            "SRAM byte-enable is valid only for a full-sized write");
    if (request.command == Command::kClear)
        regions_.LocateAbsolute(request.address, request.size_bytes);
    else
        regions_.ResolveAbsolute(request.address, request.size_bytes,
                                 request.initiator, request.command);

    TraceStage("SRAM_queue", "B", trace_request_id);
    while (admitted_ >= regions_.config().queue_depth)
        wait(capacity_available_);
    ++admitted_;
    stats_.peak_queued = std::max(stats_.peak_queued, admitted_);

    std::shared_ptr<ActiveAccess> active;
    while (!active) {
        std::shared_ptr<ActiveAccess> blocker;
        for (const auto &candidate : active_) {
            if (request.hazard_lease != 0 && candidate->hazard_only)
                continue;
            if (HasHazard(request, *candidate)) {
                blocker = candidate;
                break;
            }
        }
        if (blocker) {
            auto &counter =
                stats_.by_initiator_command[InitiatorIndex(request.initiator)]
                                           [CommandIndex(request.command)];
            ++counter.hazard_stalls;
            wait(blocker->done);
            continue;
        }
        active = std::make_shared<ActiveAccess>();
        active->request = request;
        active_.push_back(active);
    }

    response.service_begin = sc_time_stamp();
    TraceStage("SRAM_queue", "E", trace_request_id);
    const char *data_stage = request.command == Command::kRead
                                 ? "SRAM_read"
                                 : "SRAM_write";
    TraceStage(data_stage, "B", trace_request_id);
    const uint64_t base =
        request.command == Command::kRead
            ? regions_.config().read_base_latency_cycles
            : regions_.config().write_base_latency_cycles;
    const auto beats = BuildBeats(request);
    try {
        for (size_t index = 0; index < beats.size(); ++index) {
            auto waiter = AcquireBeat(beats[index], request, trace_request_id);
            waiter->service_cycles = 1 + (index == 0 ? base : 0);
            wait(sc_time(
                static_cast<double>(waiter->service_cycles * CYCLE), SC_NS));
            ReleaseBeat(waiter);
        }

        if (request.command == Command::kRead)
            response.payload =
                storage_.Read(request.address, request.size_bytes);
        else if (request.command == Command::kWrite)
            storage_.Write(request.address, request.payload,
                           request.byte_enable);
        else
            storage_.Clear(request.address, request.size_bytes);
    } catch (...) {
        TraceStage(data_stage, "E", trace_request_id, "failed=1");
        RemoveActive(active);
        throw;
    }

    response.completion_time = sc_time_stamp();
    TraceStage(data_stage, "E", trace_request_id);
    auto &counter =
        stats_.by_initiator_command[InitiatorIndex(request.initiator)]
                                   [CommandIndex(request.command)];
    ++counter.requests;
    counter.bytes += request.size_bytes;
    counter.queue_wait_cycles +=
        CyclesBetween(response.issue_time, response.service_begin);
    counter.service_cycles += base + beats.size();

    SramTraceRecord trace;
    trace.request_id = trace_request_id;
    trace.initiator = request.initiator;
    trace.command = request.command;
    trace.address = request.address;
    trace.size_bytes = request.size_bytes;
    trace.issue_time = response.issue_time;
    trace.service_begin = response.service_begin;
    trace.completion_time = response.completion_time;
    trace_.push_back(trace);
    RemoveActive(active);
    return response;
}

} // namespace sram
