#include "dte/sync_runtime.h"

#include "dte/coll_runtime.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

CoreGroupRegistry::CoreGroupRegistry(
    std::vector<CoreGroupDefinition> definitions, uint32_t total_cores,
    uint32_t cores_per_die) {
    if (total_cores == 0 || cores_per_die == 0 ||
        total_cores % cores_per_die != 0 || total_cores > UINT16_MAX + 1u)
        throw std::invalid_argument("core group topology is invalid");

    std::map<uint32_t, std::vector<uint16_t>> candidate;
    for (auto &definition : definitions) {
        if (definition.group_id == 0)
            throw std::invalid_argument("core group ID 0 is reserved");
        if (definition.members.empty())
            throw std::invalid_argument("core group must not be empty");
        if (candidate.count(definition.group_id) != 0)
            throw std::invalid_argument("duplicate core group ID");

        std::set<uint16_t> unique;
        const uint32_t first_die = definition.members.front() / cores_per_die;
        for (uint16_t member : definition.members) {
            if (member >= total_cores)
                throw std::invalid_argument("core group member is out of range");
            if (!unique.insert(member).second)
                throw std::invalid_argument("duplicate core group member");
            if (member / cores_per_die != first_die)
                throw std::invalid_argument("cross-die core group is unsupported");
        }
        candidate.emplace(definition.group_id, std::move(definition.members));
    }
    groups_ = std::move(candidate);
}

const std::vector<uint16_t> &
CoreGroupRegistry::Members(uint32_t group_id) const {
    const auto found = groups_.find(group_id);
    if (found == groups_.end())
        throw std::out_of_range("unknown core group ID");
    return found->second;
}

uint16_t CoreGroupRegistry::RankOf(uint32_t group_id,
                                   uint16_t core_id) const {
    const auto &members = Members(group_id);
    const auto found = std::find(members.begin(), members.end(), core_id);
    if (found == members.end())
        throw std::invalid_argument("GROUP_SYNC core is not a group member");
    return static_cast<uint16_t>(found - members.begin());
}

bool CoreGroupRegistry::Contains(uint32_t group_id) const noexcept {
    return groups_.count(group_id) != 0;
}

GroupSyncRuntime::GroupSyncRuntime(
    std::shared_ptr<const CoreGroupRegistry> registry)
    : registry_(std::move(registry)) {
    if (registry_ == nullptr)
        throw std::invalid_argument("GROUP_SYNC registry is null");
}

void GroupSyncRuntime::Wait(uint16_t core_id, uint32_t group_id,
                            uint32_t sync_seq) {
    const auto &members = registry_->Members(group_id);
    if (members.size() > UINT16_MAX)
        throw std::overflow_error("GROUP_SYNC group exceeds rank capacity");
    const uint16_t rank = registry_->RankOf(group_id, core_id);
    const auto state_key = std::make_pair(group_id, core_id);
    uint64_t &next = next_sequence_[state_key];
    if (next > UINT32_MAX || sync_seq != next)
        throw std::invalid_argument("GROUP_SYNC sequence is out of order");

    const uint64_t previous = next;
    ++next;
    try {
        WaitCollectiveBarrier(
            {group_id, RESERVED_GROUP_SYNC_ID, sync_seq}, 0, rank,
            static_cast<uint16_t>(members.size()), 0);
    } catch (...) {
        next = previous;
        throw;
    }
}

uint64_t GroupSyncRuntime::NextSequence(uint32_t group_id,
                                        uint16_t core_id) const {
    const auto found = next_sequence_.find({group_id, core_id});
    return found == next_sequence_.end() ? 0 : found->second;
}

EventControlQueue::EventControlQueue(size_t capacity) : capacity_(capacity) {
    if (capacity == 0)
        throw std::invalid_argument("EVENT control queue capacity is zero");
}

void EventControlQueue::Push(const EventControlMessage &message) {
    if (Full())
        throw std::overflow_error("EVENT control queue is full");
    queue_.push_back(message);
}

EventControlMessage EventControlQueue::Pop() {
    if (queue_.empty())
        throw std::underflow_error("EVENT control queue is empty");
    EventControlMessage message = queue_.front();
    queue_.pop_front();
    return message;
}

const EventControlMessage &EventControlQueue::Front() const {
    if (queue_.empty())
        throw std::underflow_error("EVENT control queue is empty");
    return queue_.front();
}

EventMailbox::EventMailbox(size_t capacity) : capacity_(capacity) {
    if (capacity == 0)
        throw std::invalid_argument("EVENT mailbox capacity is zero");
}

void EventMailbox::Deliver(const EventControlMessage &message,
                           uint16_t endpoint_core, uint32_t total_cores) {
    if (total_cores == 0 || total_cores > UINT16_MAX + 1u ||
        message.source >= total_cores || message.destination >= total_cores)
        throw std::invalid_argument("EVENT endpoint is out of range");
    if (message.destination != endpoint_core)
        throw std::invalid_argument("EVENT delivered to the wrong endpoint");
    if (total_credits_ == capacity_)
        throw std::overflow_error("EVENT mailbox capacity exhausted");

    const EventKey key{message.source, message.destination, message.tag};
    uint64_t &credit = credits_[key];
    if (credit == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("EVENT credit counter overflow");
    ++credit;
    ++total_credits_;
}

bool EventMailbox::TryConsume(const EventKey &key, uint32_t count) {
    if (count == 0)
        throw std::invalid_argument("EVENT_WAIT count must be non-zero");
    const auto found = credits_.find(key);
    if (found == credits_.end() || found->second < count) return false;

    found->second -= count;
    total_credits_ -= count;
    if (found->second == 0) credits_.erase(found);
    return true;
}

uint64_t EventMailbox::Credit(const EventKey &key) const noexcept {
    const auto found = credits_.find(key);
    return found == credits_.end() ? 0 : found->second;
}
