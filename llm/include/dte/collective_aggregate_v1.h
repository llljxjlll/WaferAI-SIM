#pragma once

#include "dte/coll_types.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

struct IsaV1CollectiveArtifactLowering;

struct CollectiveAggregateCapacity {
    size_t max_aggregates = 0;
    size_t max_child_tokens = 0;
    size_t max_local_work_items = 0;
    size_t max_reserved_tokens = 0;
};

enum class CollectiveAggregatePhase : uint8_t {
    REGISTERED = 0,
    ACTIVE = 1,
    READY = 2,
};

enum class CollectiveAggregateLocalWorkKind : uint8_t {
    LOCAL_COPY = 0,
    REDUCE_COMPUTE = 1,
};

struct CollectiveAggregateLocalWorkHandle {
    uint64_t id = 0;
    uint32_t public_token = 0;
    CollectiveKey key;
    CollectiveAggregateLocalWorkKind kind =
        CollectiveAggregateLocalWorkKind::LOCAL_COPY;
    size_t plan_index = 0;
    uint32_t item_index = 0;

    bool operator==(
        const CollectiveAggregateLocalWorkHandle &other) const noexcept;
};

struct CollectiveAggregateResidual {
    size_t aggregates = 0;
    size_t child_tokens = 0;
    size_t local_work_items = 0;
    size_t reserved_tokens = 0;

    bool ActiveEmpty() const noexcept {
        return aggregates == 0 && child_tokens == 0 &&
               local_work_items == 0;
    }
};

// One instance belongs to exactly one core. Registration is transactional:
// plan/action/token membership is validated into candidate maps and becomes
// visible only after every aggregate in the artifact has passed validation.
class CollectiveAggregateRuntime final {
public:
    CollectiveAggregateRuntime(uint16_t local_core,
                               CollectiveAggregateCapacity capacity);

    // reserved_tokens is the local core's ordinary DTE/P2P token namespace.
    // Token zero is synchronous/no-token and is ignored.
    void RegisterArtifact(
        const IsaV1CollectiveArtifactLowering &lowering,
        const std::set<uint32_t> &reserved_tokens = {});

    // BEGIN starts every child and local-work gate in one aggregate. CANCEL is
    // therefore legal only while the aggregate remains REGISTERED.
    void Begin(uint32_t public_token);
    void MarkChildLocalComplete(uint32_t internal_token);
    void MarkChildTransportRetired(uint32_t internal_token);
    void MarkLocalWorkComplete(
        const CollectiveAggregateLocalWorkHandle &handle);

    // Successful WAIT consumes one public token and all of its private child
    // and local-work state. FENCE is all-or-nothing over every aggregate.
    bool TryWait(uint32_t public_token);
    bool TryFence();
    void Cancel(uint32_t public_token);

    CollectiveAggregatePhase Poll(uint32_t public_token) const;
    std::vector<uint32_t> ChildTokens(uint32_t public_token) const;
    std::vector<CollectiveAggregateLocalWorkHandle>
    LocalWorks(uint32_t public_token) const;

    bool HasPublicToken(uint32_t public_token) const noexcept;
    bool HasChildToken(uint32_t internal_token) const noexcept;
    CollectiveAggregateResidual Residual() const noexcept;
    uint16_t LocalCore() const noexcept { return local_core_; }
    const CollectiveAggregateCapacity &Capacity() const noexcept {
        return capacity_;
    }

private:
    struct Aggregate {
        CollectiveKey key;
        bool begun = false;
        std::set<uint32_t> child_tokens;
        std::set<uint64_t> local_work_ids;
    };

    struct Child {
        uint32_t public_token = 0;
        bool started = false;
        bool local_complete = false;
        bool transport_retired = false;
    };

    struct LocalWork {
        CollectiveAggregateLocalWorkHandle handle;
        bool started = false;
        bool complete = false;
    };

    using AggregateMap = std::map<uint32_t, Aggregate>;

    AggregateMap::iterator RequireAggregate(uint32_t public_token);
    AggregateMap::const_iterator RequireAggregate(
        uint32_t public_token) const;
    bool Ready(const Aggregate &aggregate) const;
    void Retire(AggregateMap::iterator aggregate);

    uint16_t local_core_;
    CollectiveAggregateCapacity capacity_;
    uint64_t next_local_work_id_ = 1;
    AggregateMap aggregates_;
    std::map<uint32_t, Child> children_;
    std::map<uint64_t, LocalWork> local_work_;
    std::set<uint32_t> reserved_tokens_;
};
