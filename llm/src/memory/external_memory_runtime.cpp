#include "memory/external_memory_runtime.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace external_memory {
namespace {

uint64_t CheckedAdd(uint64_t lhs, uint64_t rhs, const char *context) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs)
        throw std::overflow_error(std::string(context) + " overflows uint64");
    return lhs + rhs;
}

uint64_t CeilDiv(uint64_t numerator, uint64_t denominator) {
    return 1 + (numerator - 1) / denominator;
}

void ValidateRange(uint64_t address, uint64_t size_bytes,
                   uint64_t base_address, uint64_t capacity_bytes,
                   const char *context) {
    if (size_bytes == 0)
        throw std::invalid_argument(std::string(context) + " size must be > 0");
    const uint64_t end = CheckedAdd(address, size_bytes, context);
    const uint64_t capacity_end = CheckedAdd(
        base_address, capacity_bytes, "external runtime capacity");
    if (address < base_address || end > capacity_end)
        throw std::out_of_range(
            std::string(context) + " exceeds configured capacity");
}

} // namespace

struct ExternalMemoryRuntimeBridge::Impl {
    struct Entry {
        TransferRequest request;
        sc_core::sc_time submitted_at = sc_core::SC_ZERO_TIME;
        std::vector<uint8_t> external_payload;
        std::optional<RuntimeCompletion> completion;
        sc_core::sc_event done;
        std::shared_ptr<HBMBackendTransaction> backend_tx;
    };

    class LinkWorker : public sc_core::sc_module {
    public:
        SC_HAS_PROCESS(LinkWorker);
        LinkWorker(const sc_core::sc_module_name &name, Impl &owner,
                   const LinkConfig &link)
            : sc_core::sc_module(name), owner_(owner), link_(link) {
            SC_THREAD(Run);
        }

        void Enqueue(const std::shared_ptr<Entry> &entry) {
            queue_.push_back(entry);
            wake_.notify(sc_core::SC_ZERO_TIME);
        }

        uint64_t Outstanding() const {
            return queue_.size() + (active_ ? 1U : 0U);
        }

        uint64_t Waiting() const {
            if (active_) return queue_.size();
            return queue_.empty() ? 0 : queue_.size() - 1;
        }

    private:
        void Run();

        Impl &owner_;
        const LinkConfig &link_;
        std::deque<std::shared_ptr<Entry>> queue_;
        bool active_ = false;
        sc_core::sc_event wake_;
    };

    Impl(FabricConfig value,
         std::map<std::string, HBMBackend *> backend_bindings,
         sc_core::sc_time configured_cycle_time)
        : config(std::move(value)), hbm_backends(std::move(backend_bindings)),
          cycle_time(configured_cycle_time) {
        ValidateFabricConfig(config);
        if (cycle_time <= sc_core::SC_ZERO_TIME)
            throw std::invalid_argument(
                "external runtime cycle time must be > 0");

        std::set<std::string> expected_hbm;
        for (const auto &capacity : config.external_capacities) {
            external_capacities.emplace(capacity.id, &capacity);
            external_backings.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(capacity.id),
                std::forward_as_tuple(capacity.base_address,
                                      capacity.capacity_bytes));
        }
        for (const auto &capacity : config.hbm_capacities) {
            hbm_capacities.emplace(capacity.id, &capacity);
            expected_hbm.insert(capacity.id);
        }
        if (hbm_backends.size() != expected_hbm.size())
            throw std::invalid_argument(
                "HBM backend bindings must exactly cover capacities");
        for (const auto &binding : hbm_backends)
            if (expected_hbm.count(binding.first) == 0 ||
                binding.second == nullptr)
                throw std::invalid_argument(
                    "HBM backend binding is unknown or null");

        for (const auto &link : config.links) {
            links.emplace(link.id, &link);
            workers.emplace(
                link.id,
                std::make_unique<LinkWorker>(
                    sc_core::sc_gen_unique_name(
                        "external_memory_link_worker"),
                    *this, link));
        }
        for (const auto &connection : config.connections)
            connections.emplace(connection.id, &connection);
    }

    uint64_t CurrentCycle() const {
        const auto now = sc_core::sc_time_stamp().value();
        const auto quantum = cycle_time.value();
        if (now % quantum != 0)
            throw std::logic_error(
                "external DMA submission is not cycle aligned");
        return now / quantum;
    }

    sc_core::sc_time Cycles(uint64_t cycles) const {
        if (cycles > std::numeric_limits<uint64_t>::max() /
                         cycle_time.value())
            throw std::overflow_error(
                "external runtime service time overflows");
        return sc_core::sc_time::from_value(cycles * cycle_time.value());
    }

    const ExternalCapacityConfig &ExternalCapacity(
        const ConnectionConfig &connection) const {
        const LinkConfig &link = *links.at(connection.link_ref);
        return *external_capacities.at(link.external_capacity_ref);
    }

    const HbmCapacityConfig &HbmCapacity(
        const ConnectionConfig &connection) const {
        return *hbm_capacities.at(connection.hbm_capacity_ref);
    }

    FabricConfig config;
    std::map<std::string, const ExternalCapacityConfig *> external_capacities;
    std::map<std::string, const HbmCapacityConfig *> hbm_capacities;
    std::map<std::string, const LinkConfig *> links;
    std::map<std::string, const ConnectionConfig *> connections;
    std::map<std::string, SparseMemoryBacking> external_backings;
    std::map<std::string, HBMBackend *> hbm_backends;
    std::map<std::string, std::unique_ptr<LinkWorker>> workers;
    std::map<std::string, std::shared_ptr<Entry>> entries;
    sc_core::sc_time cycle_time;
    sc_core::sc_event completion_event;
    RuntimeStats stats;
};

void ExternalMemoryRuntimeBridge::Impl::LinkWorker::Run() {
    while (true) {
        while (queue_.empty()) sc_core::wait(wake_);
        const auto entry = queue_.front();
        queue_.pop_front();
        active_ = true;

        const auto connection =
            owner_.connections.at(entry->request.connection_ref);
        const auto &hbm_capacity = owner_.HbmCapacity(*connection);
        RuntimeCompletion completion;
        completion.request_ref = entry->request.id;
        completion.submitted_at = entry->submitted_at;
        completion.started_at = sc_core::sc_time_stamp();
        completion.payload_bytes = entry->request.size_bytes;
        owner_.stats.queue_stall_time +=
            completion.started_at - completion.submitted_at;

        const uint64_t external_cycles = CheckedAdd(
            link_.latency_cycles,
            CeilDiv(entry->request.size_bytes, link_.bytes_per_cycle),
            "external runtime link cycles");
        uint64_t route_cycles = 0;
        if (connection->route_die_ids.size() > 1)
            route_cycles = CheckedAdd(
                connection->route_latency_cycles,
                CeilDiv(entry->request.size_bytes,
                        *connection->route_bytes_per_cycle),
                "external runtime route cycles");
        completion.external_service_time = owner_.Cycles(
            CheckedAdd(external_cycles, route_cycles,
                       "external runtime service cycles"));
        sc_core::wait(completion.external_service_time);

        auto tx = std::make_shared<HBMBackendTransaction>();
        tx->command = entry->request.direction ==
                              TransferDirection::kExternalToHbm
                          ? MemCommand::kWrite
                          : MemCommand::kRead;
        tx->address = entry->request.hbm_address -
                      hbm_capacity.base_address;
        tx->payload = entry->request.direction ==
                              TransferDirection::kExternalToHbm
                          ? entry->external_payload
                          : std::vector<uint8_t>(
                                static_cast<size_t>(entry->request.size_bytes));
        if (tx->command == MemCommand::kWrite)
            tx->byte_enable.assign(tx->payload.size(), 0xff);
        tx->submitted = sc_core::sc_time_stamp();
        sc_core::sc_event backend_done;
        bool callback_received = false;
        sc_core::sc_time completion_delay = sc_core::SC_ZERO_TIME;
        tx->complete = [&](sc_core::sc_time delay,
                           sc_core::sc_time service_time, int status,
                           const std::string &error) {
            completion.hbm_service_time = service_time;
            completion.status = status;
            completion.error = error;
            completion_delay = delay;
            callback_received = true;
            backend_done.notify(delay);
        };
        entry->backend_tx = tx;
        try {
            owner_.hbm_backends.at(hbm_capacity.id)->Submit(tx);
            if (!callback_received ||
                completion_delay > sc_core::SC_ZERO_TIME)
                sc_core::wait(backend_done);
        } catch (const std::exception &error) {
            completion.status = 1;
            completion.error = error.what();
        }

        completion.completed_at = sc_core::sc_time_stamp();
        owner_.stats.external_service_time +=
            completion.external_service_time;
        owner_.stats.hbm_service_time += completion.hbm_service_time;
        if (completion.status == 0) {
            if (entry->request.direction ==
                TransferDirection::kExternalToHbm) {
                owner_.stats.external_read_bytes += entry->request.size_bytes;
                owner_.stats.hbm_write_bytes += entry->request.size_bytes;
            } else {
                owner_.external_backings
                    .at(owner_.ExternalCapacity(*connection).id)
                    .Write(entry->request.external_address, tx->payload);
                owner_.stats.hbm_read_bytes += entry->request.size_bytes;
                owner_.stats.external_write_bytes += entry->request.size_bytes;
            }
        } else {
            ++owner_.stats.failed_requests;
        }
        ++owner_.stats.completed_requests;
        tx->complete = {};
        entry->backend_tx.reset();
        entry->completion = completion;
        active_ = false;
        owner_.completion_event.notify(sc_core::SC_ZERO_TIME);
        entry->done.notify(sc_core::SC_ZERO_TIME);
    }
}

ExternalMemoryRuntimeBridge::ExternalMemoryRuntimeBridge(
    const sc_core::sc_module_name &name, FabricConfig config,
    std::map<std::string, HBMBackend *> hbm_backends,
    sc_core::sc_time cycle_time)
    : sc_core::sc_module(name),
      impl_(std::make_unique<Impl>(std::move(config),
                                  std::move(hbm_backends), cycle_time)) {}

ExternalMemoryRuntimeBridge::~ExternalMemoryRuntimeBridge() = default;

void ExternalMemoryRuntimeBridge::SeedExternal(
    const std::string &capacity_ref, uint64_t address,
    const std::vector<uint8_t> &payload) {
    impl_->external_backings.at(capacity_ref).Write(address, payload);
}

std::vector<uint8_t> ExternalMemoryRuntimeBridge::ProbeExternal(
    const std::string &capacity_ref, uint64_t address,
    uint64_t size_bytes) const {
    return impl_->external_backings.at(capacity_ref).Read(
        address, size_bytes);
}

void ExternalMemoryRuntimeBridge::Submit(const TransferRequest &request) {
    if (request.schema_version != kExternalDmaRequestSchemaVersion)
        throw std::invalid_argument(
            "unsupported external DMA request schema version");
    if (request.id.empty() || impl_->entries.count(request.id) != 0)
        throw std::invalid_argument(
            "external DMA request id is empty or duplicated");
    if (request.direction != TransferDirection::kExternalToHbm &&
        request.direction != TransferDirection::kHbmToExternal)
        throw std::invalid_argument("external DMA direction is invalid");
    if (request.issue_cycle != impl_->CurrentCycle())
        throw std::invalid_argument(
            "external DMA issue cycle differs from runtime cycle");
    const auto connection_it = impl_->connections.find(
        request.connection_ref);
    if (connection_it == impl_->connections.end())
        throw std::invalid_argument(
            "external DMA request has no declared connection");
    const auto &connection = *connection_it->second;
    const auto &external_capacity = impl_->ExternalCapacity(connection);
    const auto &hbm_capacity = impl_->HbmCapacity(connection);
    ValidateRange(request.external_address, request.size_bytes,
                  external_capacity.base_address,
                  external_capacity.capacity_bytes,
                  "external DMA external range");
    ValidateRange(request.hbm_address, request.size_bytes,
                  hbm_capacity.base_address, hbm_capacity.capacity_bytes,
                  "external DMA HBM range");

    auto &worker = *impl_->workers.at(connection.link_ref);
    const auto &link = *impl_->links.at(connection.link_ref);
    const uint64_t outstanding = worker.Outstanding();
    if (outstanding >= link.max_outstanding)
        throw std::runtime_error(
            "external DMA max_outstanding exhausted");
    if (outstanding > 0 && worker.Waiting() >= link.queue_depth)
        throw std::runtime_error("external DMA queue depth exhausted");

    auto entry = std::make_shared<Impl::Entry>();
    entry->request = request;
    entry->submitted_at = sc_core::sc_time_stamp();
    if (request.direction == TransferDirection::kExternalToHbm)
        entry->external_payload = impl_->external_backings
                                      .at(external_capacity.id)
                                      .Read(request.external_address,
                                            request.size_bytes);
    impl_->entries.emplace(request.id, entry);
    ++impl_->stats.submitted_requests;
    impl_->stats.max_outstanding = std::max(
        impl_->stats.max_outstanding, outstanding + 1);
    worker.Enqueue(entry);
}

std::optional<RuntimeCompletion> ExternalMemoryRuntimeBridge::Poll(
    const std::string &request_ref) const {
    const auto found = impl_->entries.find(request_ref);
    if (found == impl_->entries.end())
        throw std::invalid_argument("unknown external DMA request id");
    return found->second->completion;
}

RuntimeCompletion ExternalMemoryRuntimeBridge::Wait(
    const std::string &request_ref) {
    const auto found = impl_->entries.find(request_ref);
    if (found == impl_->entries.end())
        throw std::invalid_argument("unknown external DMA request id");
    while (!found->second->completion.has_value())
        sc_core::wait(found->second->done);
    return *found->second->completion;
}

const sc_core::sc_event &ExternalMemoryRuntimeBridge::CompletionEvent() const {
    return impl_->completion_event;
}

const RuntimeStats &ExternalMemoryRuntimeBridge::Stats() const {
    return impl_->stats;
}

uint64_t ExternalMemoryRuntimeBridge::Outstanding() const {
    return impl_->stats.submitted_requests -
           impl_->stats.completed_requests;
}

} // namespace external_memory
