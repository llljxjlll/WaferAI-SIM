#include "memory/external_memory_service.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace external_memory {
namespace {

uint64_t CheckedAdd(uint64_t lhs, uint64_t rhs, const char *context) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs)
        throw std::overflow_error(std::string(context) + " overflows uint64");
    return lhs + rhs;
}

uint64_t RangeEnd(uint64_t base, uint64_t size, const char *context) {
    if (size == 0)
        throw std::invalid_argument(std::string(context) + " size must be > 0");
    return CheckedAdd(base, size, context);
}

uint64_t CeilDiv(uint64_t numerator, uint64_t denominator) {
    if (denominator == 0)
        throw std::invalid_argument("external memory divisor must be > 0");
    return 1 + (numerator - 1) / denominator;
}

template <typename T>
std::map<std::string, const T *> IndexById(
    const std::vector<T> &items, const char *context) {
    std::map<std::string, const T *> result;
    for (const auto &item : items) {
        if (item.id.empty())
            throw std::invalid_argument(std::string(context) +
                                        " id must not be empty");
        if (!result.emplace(item.id, &item).second)
            throw std::invalid_argument(std::string(context) +
                                        " contains duplicate id " + item.id);
    }
    return result;
}

template <typename Owner, typename Capacity, typename OwnerOf>
void ValidateNonoverlap(const std::vector<Capacity> &capacities,
                        OwnerOf owner_of, const char *context) {
    std::map<Owner, std::vector<std::pair<uint64_t, uint64_t>>> ranges;
    for (const auto &capacity : capacities) {
        const uint64_t end = RangeEnd(
            capacity.base_address, capacity.capacity_bytes, context);
        auto &owned = ranges[owner_of(capacity)];
        for (const auto &prior : owned)
            if (capacity.base_address < prior.second &&
                prior.first < end)
                throw std::invalid_argument(
                    std::string(context) +
                    " address spaces overlap for one typed owner");
        owned.emplace_back(capacity.base_address, end);
    }
}

void ValidateFabricImpl(const FabricConfig &config) {
    if (config.external_capacities.empty() ||
        config.hbm_capacities.empty() || config.links.empty() ||
        config.connections.empty())
        throw std::invalid_argument(
            "external memory fabric collections must not be empty");

    const auto external = IndexById(
        config.external_capacities, "external capacities");
    const auto hbm = IndexById(config.hbm_capacities, "HBM capacities");
    const auto links = IndexById(config.links, "external links");
    IndexById(config.connections, "external connections");

    ValidateNonoverlap<std::string>(
        config.external_capacities,
        [](const ExternalCapacityConfig &item) {
            if (item.owner_ref.empty())
                throw std::invalid_argument(
                    "external capacity owner must not be empty");
            return item.owner_ref;
        },
        "external capacity");
    ValidateNonoverlap<uint64_t>(
        config.hbm_capacities,
        [](const HbmCapacityConfig &item) { return item.owner_die_id; },
        "HBM capacity");

    std::map<std::string, std::string> link_by_external;
    for (const auto &link : config.links) {
        if (external.count(link.external_capacity_ref) == 0)
            throw std::invalid_argument(
                "external link references unknown external capacity");
        if (link.bytes_per_cycle == 0 || link.queue_depth == 0 ||
            link.max_outstanding == 0)
            throw std::invalid_argument(
                "external link bandwidth and queue limits must be > 0");
        if (link.max_outstanding - 1 > link.queue_depth)
            throw std::invalid_argument(
                "external link max_outstanding exceeds queue_depth + 1");
        if (!link_by_external
                 .emplace(link.external_capacity_ref, link.id)
                 .second)
            throw std::invalid_argument(
                "external capacity has multiple service links");
    }
    if (link_by_external.size() != external.size())
        throw std::invalid_argument(
            "every external capacity requires one service link");

    std::set<std::pair<std::string, uint64_t>> link_targets;
    std::set<std::string> direct_links;
    std::set<std::string> connected_hbm;
    for (const auto &connection : config.connections) {
        const auto link_it = links.find(connection.link_ref);
        const auto hbm_it = hbm.find(connection.hbm_capacity_ref);
        if (link_it == links.end())
            throw std::invalid_argument(
                "external connection references unknown link");
        if (hbm_it == hbm.end())
            throw std::invalid_argument(
                "external connection references unknown HBM capacity");
        const auto &link = *link_it->second;
        const auto &capacity = *hbm_it->second;
        if (connection.target_die_id != capacity.owner_die_id)
            throw std::invalid_argument(
                "external connection target does not own HBM capacity");
        if (!link_targets
                 .emplace(connection.link_ref, connection.target_die_id)
                 .second)
            throw std::invalid_argument(
                "duplicate external link and target connection");
        if (connection.route_die_ids.empty() ||
            connection.route_die_ids.back() != connection.target_die_id)
            throw std::invalid_argument(
                "external connection route must end at target die");
        if (std::set<uint64_t>(connection.route_die_ids.begin(),
                               connection.route_die_ids.end())
                .size() != connection.route_die_ids.size())
            throw std::invalid_argument(
                "external connection route must be simple");

        if (connection.target_die_id == link.ingress_die_id) {
            if (connection.route_die_ids !=
                    std::vector<uint64_t>{link.ingress_die_id} ||
                connection.route_latency_cycles != 0 ||
                connection.route_bytes_per_cycle.has_value())
                throw std::invalid_argument(
                    "direct external connection must use zero-cost route");
            direct_links.insert(link.id);
        } else {
            if (connection.route_die_ids.size() < 2 ||
                connection.route_die_ids.front() != link.ingress_die_id ||
                connection.route_latency_cycles == 0 ||
                !connection.route_bytes_per_cycle.has_value() ||
                *connection.route_bytes_per_cycle == 0)
                throw std::invalid_argument(
                    "remote external connection requires explicit route cost");
        }
        connected_hbm.insert(capacity.id);
    }
    if (direct_links.size() != links.size())
        throw std::invalid_argument(
            "every external link requires its ingress-die connection");
    if (connected_hbm.size() != hbm.size())
        throw std::invalid_argument(
            "every HBM capacity requires an external connection");
}

} // namespace

void ValidateFabricConfig(const FabricConfig &config) {
    ValidateFabricImpl(config);
}

SparseMemoryBacking::SparseMemoryBacking(uint64_t base_address,
                                         uint64_t capacity_bytes)
    : base_address_(base_address), capacity_bytes_(capacity_bytes) {
    RangeEnd(base_address_, capacity_bytes_, "sparse memory capacity");
}

void SparseMemoryBacking::ValidateRange(uint64_t address,
                                        uint64_t size_bytes) const {
    const uint64_t end = RangeEnd(address, size_bytes, "sparse memory access");
    const uint64_t capacity_end = CheckedAdd(
        base_address_, capacity_bytes_, "sparse memory capacity");
    if (address < base_address_ || end > capacity_end)
        throw std::out_of_range(
            "external memory access exceeds configured capacity");
}

void SparseMemoryBacking::Write(uint64_t address,
                                const std::vector<uint8_t> &payload) {
    ValidateRange(address, payload.size());
    for (size_t index = 0; index < payload.size(); ++index) {
        const uint64_t byte_address = address + index;
        if (payload[index] == 0)
            nonzero_bytes_.erase(byte_address);
        else
            nonzero_bytes_[byte_address] = payload[index];
    }
}

std::vector<uint8_t> SparseMemoryBacking::Read(
    uint64_t address, uint64_t size_bytes) const {
    ValidateRange(address, size_bytes);
    if (size_bytes > std::numeric_limits<size_t>::max())
        throw std::length_error("external memory read is too large");
    std::vector<uint8_t> payload(static_cast<size_t>(size_bytes), 0);
    for (size_t index = 0; index < payload.size(); ++index) {
        const auto found = nonzero_bytes_.find(address + index);
        if (found != nonzero_bytes_.end()) payload[index] = found->second;
    }
    return payload;
}

uint64_t SparseMemoryBacking::NonzeroByteCount() const {
    return nonzero_bytes_.size();
}

ExternalMemoryService::ExternalMemoryService(FabricConfig config)
    : config_(std::move(config)) {
    ValidateFabricConfig(config_);
    for (const auto &capacity : config_.external_capacities)
        external_backings_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(capacity.id),
            std::forward_as_tuple(capacity.base_address,
                                  capacity.capacity_bytes));
    for (const auto &capacity : config_.hbm_capacities)
        hbm_backings_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(capacity.id),
            std::forward_as_tuple(capacity.base_address,
                                  capacity.capacity_bytes));
}

void ExternalMemoryService::SeedExternal(
    const std::string &capacity_ref, uint64_t address,
    const std::vector<uint8_t> &payload) {
    external_backings_.at(capacity_ref).Write(address, payload);
}

void ExternalMemoryService::SeedHbm(
    const std::string &capacity_ref, uint64_t address,
    const std::vector<uint8_t> &payload) {
    hbm_backings_.at(capacity_ref).Write(address, payload);
}

std::vector<uint8_t> ExternalMemoryService::PeekExternal(
    const std::string &capacity_ref, uint64_t address,
    uint64_t size_bytes) const {
    return external_backings_.at(capacity_ref).Read(address, size_bytes);
}

std::vector<uint8_t> ExternalMemoryService::PeekHbm(
    const std::string &capacity_ref, uint64_t address,
    uint64_t size_bytes) const {
    return hbm_backings_.at(capacity_ref).Read(address, size_bytes);
}

TransferReport ExternalMemoryService::Execute(
    const std::vector<TransferRequest> &requests) {
    const auto links = IndexById(config_.links, "external links");
    const auto connections = IndexById(
        config_.connections, "external connections");
    const auto external = IndexById(
        config_.external_capacities, "external capacities");
    const auto hbm = IndexById(config_.hbm_capacities, "HBM capacities");

    TransferReport report;
    report.requests = requests;
    std::sort(report.requests.begin(), report.requests.end(),
              [](const TransferRequest &lhs, const TransferRequest &rhs) {
                  return std::tie(lhs.issue_cycle, lhs.id) <
                         std::tie(rhs.issue_cycle, rhs.id);
              });
    std::set<std::string> request_ids;
    std::map<std::string, std::vector<TransferCompletion>> by_link;
    for (const auto &request : report.requests) {
        if (request.schema_version != kExternalDmaRequestSchemaVersion)
            throw std::invalid_argument(
                "unsupported external DMA request schema version");
        if (request.id.empty() || !request_ids.insert(request.id).second)
            throw std::invalid_argument(
                "external transfer request id is empty or duplicated");
        if (request.size_bytes == 0)
            throw std::invalid_argument(
                "external transfer size must be > 0");
        if (request.direction != TransferDirection::kExternalToHbm &&
            request.direction != TransferDirection::kHbmToExternal)
            throw std::invalid_argument(
                "external transfer direction is invalid");
        const auto connection_it = connections.find(request.connection_ref);
        if (connection_it == connections.end())
            throw std::invalid_argument(
                "external transfer request has no declared connection");
        const auto &connection = *connection_it->second;
        const auto &link = *links.at(connection.link_ref);
        const auto &external_capacity =
            *external.at(link.external_capacity_ref);
        const auto &hbm_capacity = *hbm.at(connection.hbm_capacity_ref);
        auto &external_backing = external_backings_.at(external_capacity.id);
        auto &hbm_backing = hbm_backings_.at(hbm_capacity.id);

        external_backing.Read(request.external_address, request.size_bytes);
        hbm_backing.Read(request.hbm_address, request.size_bytes);

        auto &prior = by_link[link.id];
        uint64_t outstanding_before = 0;
        uint64_t queued_before = 0;
        for (const auto &completion : prior) {
            if (completion.completion_cycle > request.issue_cycle)
                ++outstanding_before;
            if (completion.start_cycle > request.issue_cycle)
                ++queued_before;
        }
        if (outstanding_before >= link.max_outstanding)
            throw std::runtime_error(
                "external link max_outstanding exhausted");
        const uint64_t free_cycle = prior.empty()
                                        ? request.issue_cycle
                                        : prior.back().completion_cycle;
        const uint64_t start_cycle =
            std::max(request.issue_cycle, free_cycle);
        if (start_cycle > request.issue_cycle &&
            queued_before >= link.queue_depth)
            throw std::runtime_error("external link queue depth exhausted");

        const uint64_t external_cycles = CheckedAdd(
            link.latency_cycles,
            CeilDiv(request.size_bytes, link.bytes_per_cycle),
            "external service cycles");
        uint64_t route_cycles = 0;
        if (connection.route_die_ids.size() > 1)
            route_cycles = CheckedAdd(
                connection.route_latency_cycles,
                CeilDiv(request.size_bytes,
                        *connection.route_bytes_per_cycle),
                "external route cycles");
        const uint64_t service_cycles = CheckedAdd(
            external_cycles, route_cycles, "external total service cycles");
        const uint64_t completion_cycle = CheckedAdd(
            start_cycle, service_cycles, "external completion cycle");

        std::vector<uint8_t> payload;
        if (request.direction == TransferDirection::kExternalToHbm) {
            payload = external_backing.Read(
                request.external_address, request.size_bytes);
            hbm_backing.Write(request.hbm_address, payload);
            report.stats.external_read_bytes = CheckedAdd(
                report.stats.external_read_bytes, request.size_bytes,
                "external read bytes");
            report.stats.hbm_write_bytes = CheckedAdd(
                report.stats.hbm_write_bytes, request.size_bytes,
                "HBM write bytes");
        } else {
            payload = hbm_backing.Read(request.hbm_address,
                                       request.size_bytes);
            external_backing.Write(request.external_address, payload);
            report.stats.hbm_read_bytes = CheckedAdd(
                report.stats.hbm_read_bytes, request.size_bytes,
                "HBM read bytes");
            report.stats.external_write_bytes = CheckedAdd(
                report.stats.external_write_bytes, request.size_bytes,
                "external write bytes");
        }
        TransferCompletion completion{
            request.id,
            link.id,
            request.direction,
            start_cycle,
            completion_cycle,
            external_cycles,
            route_cycles,
            start_cycle - request.issue_cycle,
            request.size_bytes,
        };
        prior.push_back(completion);
        report.completions.push_back(completion);
        report.stats.link_busy_cycles = CheckedAdd(
            report.stats.link_busy_cycles, service_cycles,
            "external link busy cycles");
        report.stats.queue_stall_cycles = CheckedAdd(
            report.stats.queue_stall_cycles,
            completion.queue_stall_cycles,
            "external queue stall cycles");
        report.stats.max_outstanding = std::max(
            report.stats.max_outstanding, outstanding_before + 1);
        report.stats.max_queue_occupancy = std::max(
            report.stats.max_queue_occupancy,
            queued_before + (start_cycle > request.issue_cycle ? 1U : 0U));
        report.stats.makespan_cycles = std::max(
            report.stats.makespan_cycles, completion_cycle);
    }

    report.stats.submitted_requests = report.requests.size();
    report.stats.completed_requests = report.completions.size();
    if (!report.requests.empty()) {
        const uint64_t first_issue = report.requests.front().issue_cycle;
        report.stats.makespan_cycles -= first_issue;
    }
    report.stats.pending_requests =
        report.stats.submitted_requests - report.stats.completed_requests;
    return report;
}

} // namespace external_memory
