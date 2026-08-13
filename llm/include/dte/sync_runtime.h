#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <utility>
#include <vector>

inline constexpr uint32_t RESERVED_GROUP_SYNC_ID = UINT32_MAX;

struct CoreGroupDefinition {
    uint32_t group_id = 0;
    std::vector<uint16_t> members;
};

// Immutable after construction. Validation is transactional: the public
// object is assigned only after every definition has passed validation.
class CoreGroupRegistry final {
public:
    CoreGroupRegistry(std::vector<CoreGroupDefinition> definitions,
                      uint32_t total_cores, uint32_t cores_per_die);

    const std::vector<uint16_t> &Members(uint32_t group_id) const;
    uint16_t RankOf(uint32_t group_id, uint16_t core_id) const;
    bool Contains(uint32_t group_id) const noexcept;
    size_t GroupCount() const noexcept { return groups_.size(); }

private:
    std::map<uint32_t, std::vector<uint16_t>> groups_;
};

class GroupSyncRuntime final {
public:
    explicit GroupSyncRuntime(
        std::shared_ptr<const CoreGroupRegistry> registry);

    void Wait(uint16_t core_id, uint32_t group_id, uint32_t sync_seq);
    uint64_t NextSequence(uint32_t group_id, uint16_t core_id) const;
    size_t SequenceStateCount() const noexcept { return next_sequence_.size(); }

private:
    std::shared_ptr<const CoreGroupRegistry> registry_;
    std::map<std::pair<uint32_t, uint16_t>, uint64_t> next_sequence_;
};

struct EventControlMessage {
    uint16_t source = 0;
    uint16_t destination = 0;
    uint32_t tag = 0;

    bool operator==(const EventControlMessage &other) const noexcept {
        return source == other.source && destination == other.destination &&
               tag == other.tag;
    }
};

class EventControlQueue final {
public:
    explicit EventControlQueue(size_t capacity);

    void Push(const EventControlMessage &message);
    const EventControlMessage &Front() const;
    EventControlMessage Pop();
    bool Empty() const noexcept { return queue_.empty(); }
    bool Full() const noexcept { return queue_.size() == capacity_; }
    size_t Residual() const noexcept { return queue_.size(); }
    size_t Capacity() const noexcept { return capacity_; }

private:
    size_t capacity_;
    std::deque<EventControlMessage> queue_;
};

struct EventKey {
    uint16_t source = 0;
    uint16_t destination = 0;
    uint32_t tag = 0;

    bool operator<(const EventKey &other) const noexcept {
        if (source != other.source) return source < other.source;
        if (destination != other.destination)
            return destination < other.destination;
        return tag < other.tag;
    }
};

class EventMailbox final {
public:
    explicit EventMailbox(size_t capacity);

    void Deliver(const EventControlMessage &message, uint16_t endpoint_core,
                 uint32_t total_cores);
    bool TryConsume(const EventKey &key, uint32_t count);
    uint64_t Credit(const EventKey &key) const noexcept;
    size_t KeyCount() const noexcept { return credits_.size(); }
    size_t Residual() const noexcept { return total_credits_; }
    bool Empty() const noexcept { return total_credits_ == 0; }

private:
    size_t capacity_;
    size_t total_credits_ = 0;
    std::map<EventKey, uint64_t> credits_;
};
