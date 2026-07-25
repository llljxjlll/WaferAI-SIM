#include "dte/dte_unit.h"

#include "macros/macros.h"
#include "trace/Event_engine.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
sc_time CyclesToTime(uint64_t cycles) {
    const double ns = static_cast<double>(cycles) * static_cast<double>(CYCLE);
    return sc_time(ns, SC_NS);
}

bool IsKnownDirection(DteDir dir) {
    return dir >= DteDir::SPM_TO_REMOTE && dir <= DteDir::DRAM_TO_REMOTE;
}
} // namespace

DTEConfig MakeDTEConfig(uint32_t channel_count, uint32_t bit_width_bits,
                        int64_t gamma_ns, int64_t tau_launch_ns) {
    if (gamma_ns < 0)
        throw std::invalid_argument("DTE gamma_ns must be >= 0");
    if (tau_launch_ns < 0)
        throw std::invalid_argument("DTE tau_launch_ns must be >= 0");
    DTEConfig config;
    config.channel_count = channel_count;
    config.bit_width_bits = bit_width_bits;
    config.spm_read_width_bits = bit_width_bits;
    config.spm_write_width_bits = bit_width_bits;
    config.axi_read_width_bits = bit_width_bits;
    config.axi_write_width_bits = bit_width_bits;
    config.gamma_cycles = NanosecondsToDteCycles(uint64_t(gamma_ns));
    config.tau_launch_cycles = NanosecondsToDteCycles(uint64_t(tau_launch_ns));
    DTEUnit::ValidateConfig(config);
    return config;
}

DTEUnit::DTEUnit(const sc_module_name &name, const DTEConfig &config,
                 int core_id, Event_engine *event_engine)
    : sc_module(name), config_(config), core_id_(core_id),
      event_engine_(event_engine) {
    ValidateConfig(config_);
    const uint64_t slot_count =
        uint64_t(config_.channel_count) * config_.command_slots_per_channel;
    active_.assign(static_cast<size_t>(slot_count), nullptr);
    resource_owners_.fill(nullptr);
    statistics_.area_um2 = computeAreaUm2();
    SC_THREAD(scheduler);
}

void DTEUnit::ValidateConfig(const DTEConfig &config) {
    if (config.channel_count == 0)
        throw std::invalid_argument("DTE channel_count must be > 0");
    if (config.bit_width_bits == 0)
        throw std::invalid_argument("DTE bit_width_bits must be > 0");
    if (config.gamma_cycles >
        std::numeric_limits<uint64_t>::max() - config.tau_launch_cycles)
        throw std::overflow_error("DTE launch cycle count overflows");
    if (config.command_slots_per_channel == 0)
        throw std::invalid_argument(
            "DTE command_slots_per_channel must be > 0");
    const uint64_t slots =
        uint64_t(config.channel_count) * config.command_slots_per_channel;
    if (slots > std::numeric_limits<size_t>::max())
        throw std::overflow_error("DTE command slot count overflows");
    if (config.fine_grained_resources) {
        if (config.command_slots_per_channel != 2)
            throw std::invalid_argument(
                "DTE V4 requires exactly two command slots per channel");
        if (config.pending_queue_depth == 0)
            throw std::invalid_argument(
                "DTE V4 pending_queue_depth must be > 0");
        if (config.spm_read_width_bits == 0 ||
            config.spm_write_width_bits == 0 ||
            config.axi_read_width_bits == 0 ||
            config.axi_write_width_bits == 0)
            throw std::invalid_argument(
                "DTE V4 SPM/AXI port widths must all be > 0");
    }
    const double coefficients[] = {
        config.launch_energy_pj, config.spm_energy_pj_per_bit,
        config.axi_energy_pj_per_bit, config.base_area_um2,
        config.channel_area_um2, config.command_slot_area_um2,
        config.port_bit_area_um2};
    for (double value : coefficients)
        if (!std::isfinite(value) || value < 0.0)
            throw std::invalid_argument(
                "DTE V4 energy/area coefficients must be finite and >= 0");
}

uint32_t DTEUnit::RequiredPorts(DteDir dir, bool fine_grained) {
    if (!IsKnownDirection(dir))
        throw std::invalid_argument("DTE transfer direction is invalid");
    if (!fine_grained)
        return DtePortBit(DtePort::LEGACY_BUS);
    switch (dir) {
    case DteDir::SPM_TO_REMOTE:
        return DtePortBit(DtePort::SPM_READ);
    case DteDir::REMOTE_TO_SPM:
        return DtePortBit(DtePort::SPM_WRITE);
    case DteDir::SPM_TO_SPM:
        return DtePortBit(DtePort::SPM_READ) |
               DtePortBit(DtePort::SPM_WRITE);
    case DteDir::SPM_TO_DRAM:
        return DtePortBit(DtePort::SPM_READ) |
               DtePortBit(DtePort::AXI_WRITE);
    case DteDir::DRAM_TO_SPM:
        return DtePortBit(DtePort::AXI_READ) |
               DtePortBit(DtePort::SPM_WRITE);
    case DteDir::DRAM_TO_REMOTE:
        return DtePortBit(DtePort::AXI_READ);
    }
    throw std::invalid_argument("DTE transfer direction is invalid");
}

size_t DTEUnit::descriptorCapacity() const {
    if (!config_.fine_grained_resources)
        return std::numeric_limits<size_t>::max();
    if (active_.size() >
        std::numeric_limits<size_t>::max() - config_.pending_queue_depth)
        return std::numeric_limits<size_t>::max();
    return active_.size() + config_.pending_queue_depth;
}

bool DTEUnit::CanAccept() const {
    return inflight_count_ < descriptorCapacity();
}

void DTEUnit::WaitForCredit() {
    while (!CanAccept()) {
        ++statistics_.backpressure_stalls;
        wait(credit_available_);
    }
}

DteTransferContext *DTEUnit::TryIssue(uint64_t payload_bits, DteDir dir) {
    if (payload_bits == 0)
        throw std::invalid_argument("DTE payload_bits must be > 0");
    if (!IsKnownDirection(dir))
        throw std::invalid_argument("DTE transfer direction is invalid");
    if (!config_.fine_grained_resources &&
        dir != DteDir::SPM_TO_REMOTE && dir != DteDir::REMOTE_TO_SPM)
        throw std::invalid_argument(
            "DTE V4 direction requires fine_grained_resources=true");
    if (!CanAccept())
        return nullptr;
    if (next_xfer_id_ == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("DTE transfer id space exhausted");

    auto context = std::make_unique<DteTransferContext>();
    context->xfer_id = next_xfer_id_++;
    context->payload_bits = payload_bits;
    context->dir = dir;
    context->state = DteTransferState::PENDING;
    context->issue_time = sc_time_stamp();
    context->resource_mask = RequiredPorts(dir, config_.fine_grained_resources);

    DteTransferContext *raw = context.get();
    contexts_.push_back(std::move(context));
    pending_.push_back(raw);
    ++inflight_count_;
    ++statistics_.physical_issued;
    if (!have_issue_time_) {
        first_issue_time_ = sc_time_stamp();
        have_issue_time_ = true;
    }
    traceStage(*raw, "DTE_pending", "B");
    state_changed_.notify(SC_ZERO_TIME);
    return raw;
}

DteTransferContext &DTEUnit::Issue(uint64_t payload_bits, DteDir dir) {
    DteTransferContext *context = TryIssue(payload_bits, dir);
    if (context == nullptr)
        throw std::runtime_error(
            "DTE descriptor credits exhausted; call WaitForCredit first");
    return *context;
}

bool DTEUnit::Cancel(uint64_t xfer_id) {
    for (auto &owned : contexts_) {
        DteTransferContext *ctx = owned.get();
        if (ctx->xfer_id != xfer_id)
            continue;
        if (ctx->state != DteTransferState::PENDING)
            return false;

        auto pending = std::find(pending_.begin(), pending_.end(), ctx);
        if (pending == pending_.end())
            throw std::logic_error(
                "DTE pending context is absent from the pending queue");
        pending_.erase(pending);
        traceStage(*ctx, "DTE_pending", "E");
        traceStage(*ctx, "DTE_cancel", "B");
        ctx->state = DteTransferState::CANCELLED;
        ctx->completion_time = sc_time_stamp();
        --inflight_count_;
        ++statistics_.cancelled;
        traceStage(*ctx, "DTE_cancel", "E");
        credit_available_.notify(SC_ZERO_TIME);
        state_changed_.notify(SC_ZERO_TIME);
        return true;
    }
    return false;
}

bool DTEUnit::Release(uint64_t xfer_id) {
    for (auto it = contexts_.begin(); it != contexts_.end(); ++it) {
        if ((*it)->xfer_id != xfer_id)
            continue;
        if ((*it)->state != DteTransferState::COMPLETED &&
            (*it)->state != DteTransferState::CANCELLED)
            return false;
        contexts_.erase(it);
        return true;
    }
    return false;
}

sc_time DTEUnit::launchTime() const {
    return CyclesToTime(config_.gamma_cycles + config_.tau_launch_cycles);
}

uint32_t DTEUnit::portWidth(DtePort port) const {
    switch (port) {
    case DtePort::LEGACY_BUS: return config_.bit_width_bits;
    case DtePort::SPM_READ: return config_.spm_read_width_bits;
    case DtePort::SPM_WRITE: return config_.spm_write_width_bits;
    case DtePort::AXI_READ: return config_.axi_read_width_bits;
    case DtePort::AXI_WRITE: return config_.axi_write_width_bits;
    case DtePort::COUNT: break;
    }
    throw std::invalid_argument("DTE port is invalid");
}

sc_time DTEUnit::serviceTime(uint64_t payload_bits, DtePort port) const {
    return CyclesToTime(CeilDivU64(payload_bits, portWidth(port)));
}

double DTEUnit::portEnergyPerBit(DtePort port) const {
    switch (port) {
    case DtePort::SPM_READ:
    case DtePort::SPM_WRITE:
        return config_.spm_energy_pj_per_bit;
    case DtePort::AXI_READ:
    case DtePort::AXI_WRITE:
        return config_.axi_energy_pj_per_bit;
    case DtePort::LEGACY_BUS:
    case DtePort::COUNT:
        return 0.0;
    }
    return 0.0;
}

double DTEUnit::computeAreaUm2() const {
    if (!config_.fine_grained_resources)
        return config_.base_area_um2 +
               config_.channel_count * config_.channel_area_um2;
    const uint64_t slot_count =
        uint64_t(config_.channel_count) * config_.command_slots_per_channel;
    const uint64_t total_port_width =
        uint64_t(config_.spm_read_width_bits) +
        config_.spm_write_width_bits + config_.axi_read_width_bits +
        config_.axi_write_width_bits;
    return config_.base_area_um2 +
           config_.channel_count * config_.channel_area_um2 +
           slot_count * config_.command_slot_area_um2 +
           total_port_width * config_.port_bit_area_um2;
}

int DTEUnit::findFreeSlot() const {
    for (size_t i = 0; i < active_.size(); ++i)
        if (active_[i] == nullptr)
            return static_cast<int>(i);
    return -1;
}

bool DTEUnit::resourcesAvailable(uint32_t mask) const {
    for (size_t i = 0; i < resource_owners_.size(); ++i)
        if ((mask & (uint32_t(1) << i)) && resource_owners_[i] != nullptr)
            return false;
    return true;
}

int DTEUnit::chooseReadySlot() const {
    const int count = static_cast<int>(active_.size());
    for (int offset = 1; offset <= count; ++offset) {
        const int slot = (last_served_slot_ + offset + count) % count;
        DteTransferContext *ctx = active_[slot];
        if (ctx != nullptr && ctx->state == DteTransferState::BUS_WAIT &&
            resourcesAvailable(ctx->resource_mask))
            return slot;
    }
    return -1;
}

void DTEUnit::admitPending() {
    while (!pending_.empty() && active_count_ < active_.size()) {
        const int slot = findFreeSlot();
        if (slot < 0)
            throw std::logic_error("DTE active count/free-slot mismatch");

        DteTransferContext *ctx = pending_.front();
        pending_.pop_front();
        traceStage(*ctx, "DTE_pending", "E");
        ctx->active_slot = slot;
        ctx->channel_id =
            slot / static_cast<int>(config_.command_slots_per_channel);
        ctx->command_slot =
            slot % static_cast<int>(config_.command_slots_per_channel);
        ctx->state = DteTransferState::LAUNCHING;
        ctx->admitted_time = sc_time_stamp();
        ctx->launch_done_time = sc_time_stamp() + launchTime();
        active_[slot] = ctx;
        ++active_count_;
        max_active_count_ = std::max(max_active_count_, active_count_);
        statistics_.launch_energy_pj += config_.launch_energy_pj;
        traceStage(*ctx, "DTE_launch", "B");
    }
}

bool DTEUnit::finishLaunches() {
    bool progressed = false;
    const sc_time now = sc_time_stamp();
    for (DteTransferContext *ctx : active_) {
        if (ctx == nullptr || ctx->state != DteTransferState::LAUNCHING ||
            ctx->launch_done_time > now)
            continue;
        traceStage(*ctx, "DTE_launch", "E");
        ctx->state = DteTransferState::BUS_WAIT;
        traceStage(*ctx, "DTE_bus_wait", "B");
        progressed = true;
    }
    return progressed;
}

bool DTEUnit::startReadyTransfers() {
    bool progressed = false;
    while (true) {
        const int slot = chooseReadySlot();
        if (slot < 0)
            break;
        DteTransferContext *ctx = active_[slot];
        if (!resourcesAvailable(ctx->resource_mask))
            throw std::logic_error("DTE resource availability changed");

        traceStage(*ctx, "DTE_bus_wait", "E");
        ctx->state = DteTransferState::TRANSMITTING;
        ctx->transmit_start_time = sc_time_stamp();
        ctx->active_resource_mask = ctx->resource_mask;
        ctx->scheduled_completion_time = SC_ZERO_TIME;
        for (size_t i = 0; i < resource_owners_.size(); ++i) {
            const uint32_t bit = uint32_t(1) << i;
            if (!(ctx->resource_mask & bit))
                continue;
            const DtePort port = static_cast<DtePort>(i);
            resource_owners_[i] = ctx;
            ctx->resource_done_times[i] =
                sc_time_stamp() + serviceTime(ctx->payload_bits, port);
            ctx->scheduled_completion_time =
                std::max(ctx->scheduled_completion_time,
                         ctx->resource_done_times[i]);
            statistics_.data_energy_pj +=
                static_cast<double>(ctx->payload_bits) *
                portEnergyPerBit(port);
            tracePort(*ctx, port, "B");
        }
        last_served_slot_ = slot;
        traceStage(*ctx, "DTE_transmit", "B");
        ctx->transmit_started.notify(SC_ZERO_TIME);
        progressed = true;
    }
    return progressed;
}

bool DTEUnit::finishPortServices() {
    bool progressed = false;
    const sc_time now = sc_time_stamp();
    for (size_t slot = 0; slot < active_.size(); ++slot) {
        DteTransferContext *ctx = active_[slot];
        if (ctx == nullptr || ctx->state != DteTransferState::TRANSMITTING)
            continue;
        for (size_t i = 0; i < resource_owners_.size(); ++i) {
            const uint32_t bit = uint32_t(1) << i;
            if (!(ctx->active_resource_mask & bit) ||
                ctx->resource_done_times[i] > now)
                continue;
            if (resource_owners_[i] != ctx)
                throw std::logic_error("DTE resource owner mismatch");
            resource_owners_[i] = nullptr;
            ctx->active_resource_mask &= ~bit;
            tracePort(*ctx, static_cast<DtePort>(i), "E");
            progressed = true;
        }
        if (ctx->active_resource_mask != 0)
            continue;

        traceStage(*ctx, "DTE_transmit", "E");
        ctx->state = DteTransferState::COMPLETED;
        ctx->completion_time = now;
        ctx->done.notify(SC_ZERO_TIME);
        ++completed_count_;
        ++statistics_.completed;
        --active_count_;
        --inflight_count_;
        active_[slot] = nullptr;
        last_completion_time_ = now;
        traceStatistics(*ctx);
        credit_available_.notify(SC_ZERO_TIME);
        progressed = true;
    }
    return progressed;
}

void DTEUnit::scheduler() {
    while (true) {
        bool progressed;
        do {
            progressed = false;
            progressed |= finishPortServices();
            progressed |= finishLaunches();
            if (!pending_.empty() && active_count_ < active_.size()) {
                admitPending();
                progressed = true;
            }
            progressed |= startReadyTransfers();
        } while (progressed);

        bool have_deadline = false;
        sc_time deadline = SC_ZERO_TIME;
        for (DteTransferContext *ctx : active_) {
            if (ctx == nullptr)
                continue;
            if (ctx->state == DteTransferState::LAUNCHING) {
                if (!have_deadline || ctx->launch_done_time < deadline) {
                    deadline = ctx->launch_done_time;
                    have_deadline = true;
                }
            } else if (ctx->state == DteTransferState::TRANSMITTING) {
                for (size_t i = 0; i < resource_owners_.size(); ++i) {
                    if (!(ctx->active_resource_mask & (uint32_t(1) << i)))
                        continue;
                    if (!have_deadline ||
                        ctx->resource_done_times[i] < deadline) {
                        deadline = ctx->resource_done_times[i];
                        have_deadline = true;
                    }
                }
            }
        }

        if (have_deadline) {
            if (deadline <= sc_time_stamp())
                continue;
            wait(deadline - sc_time_stamp(), state_changed_);
        } else {
            wait(state_changed_);
        }
    }
}

bool DTEUnit::PortBusy(DtePort port) const {
    const size_t index = static_cast<size_t>(port);
    if (index >= resource_owners_.size())
        return false;
    return resource_owners_[index] != nullptr;
}

bool DTEUnit::BusBusy() const {
    for (DteTransferContext *owner : resource_owners_)
        if (owner != nullptr)
            return true;
    return false;
}

double DTEUnit::AveragePowerMw() const {
    if (!have_issue_time_)
        return 0.0;
    const sc_time end = inflight_count_ == 0
                            ? last_completion_time_
                            : sc_time_stamp();
    const double elapsed_ns = (end - first_issue_time_).to_seconds() * 1e9;
    if (elapsed_ns <= 0.0)
        return 0.0;
    // Numerically, pJ/ns equals mW.
    return statistics_.TotalDynamicEnergyPj() / elapsed_ns;
}

void DTEUnit::traceStage(const DteTransferContext &ctx, const char *stage,
                         const char *phase) {
    if (event_engine_ == nullptr)
        return;
    std::ostringstream detail;
    detail << stage << " xfer=" << ctx.xfer_id << " core=" << core_id_
           << " channel=" << ctx.channel_id << " dir=" << DteDirName(ctx.dir)
           << " bits=" << ctx.payload_bits;
    event_engine_->add_event(name(), stage, phase, Trace_event_util(detail.str()),
                             SC_ZERO_TIME,
                             static_cast<unsigned>(ctx.xfer_id));
}

void DTEUnit::tracePort(const DteTransferContext &ctx, DtePort port,
                        const char *phase) {
    if (event_engine_ == nullptr || !config_.fine_grained_resources)
        return;
    std::ostringstream detail;
    detail << "DTE_port_service xfer=" << ctx.xfer_id
           << " core=" << core_id_ << " channel=" << ctx.channel_id
           << " slot=" << ctx.command_slot << " port=" << DtePortName(port)
           << " dir=" << DteDirName(ctx.dir) << " bits=" << ctx.payload_bits
           << " width=" << portWidth(port);
    event_engine_->add_event(name(), "DTE_port_service", phase,
                             Trace_event_util(detail.str()), SC_ZERO_TIME,
                             static_cast<unsigned>(ctx.xfer_id));
}

void DTEUnit::traceStatistics(const DteTransferContext &ctx) {
    if (event_engine_ == nullptr || !config_.fine_grained_resources)
        return;
    std::ostringstream detail;
    detail << std::fixed << std::setprecision(3)
           << "DTE_stats core=" << core_id_
           << " completed=" << statistics_.completed
           << " issued=" << statistics_.physical_issued
           << " energy_pj=" << statistics_.TotalDynamicEnergyPj()
           << " area_um2=" << statistics_.area_um2
           << " average_power_mw=" << AveragePowerMw()
           << " backpressure_stalls=" << statistics_.backpressure_stalls;
    event_engine_->add_event(name(), "DTE_stats", "B",
                             Trace_event_util(detail.str()), SC_ZERO_TIME,
                             static_cast<unsigned>(ctx.xfer_id));
    event_engine_->add_event(name(), "DTE_stats", "E",
                             Trace_event_util(detail.str()), SC_ZERO_TIME,
                             static_cast<unsigned>(ctx.xfer_id));
}
