#pragma once

#include "systemc.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>

struct SerializedWireItem {
    sc_bv<256> wire;
    bool control = false;
    uint64_t ticket = 0;
};

// Value-owning handoff between every Worker producer and the sole physical
// channel writer. Keeping this type free of SystemC process state makes the
// ownership and exactly-once rules independently testable.
class SerializedWireQueue {
  public:
    explicit SerializedWireQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity_ == 0)
            throw std::invalid_argument(
                "serialized wire queue capacity must be positive");
    }

    bool Empty() const noexcept { return items_.empty(); }
    bool Full() const noexcept { return items_.size() == capacity_; }
    std::size_t Size() const noexcept { return items_.size(); }
    std::size_t Capacity() const noexcept { return capacity_; }

    uint64_t Enqueue(const sc_bv<256> &wire, bool control) {
        if (Full())
            throw std::overflow_error("serialized wire queue is full");
        if (next_ticket_ == 0)
            throw std::overflow_error("serialized wire ticket overflows u64");
        const uint64_t ticket = next_ticket_++;
        items_.push_back({wire, control, ticket});
        return ticket;
    }

    const SerializedWireItem &Front() const {
        if (Empty())
            throw std::logic_error("serialized wire queue is empty");
        return items_.front();
    }

    SerializedWireItem CompleteFront() {
        if (Empty())
            throw std::logic_error("serialized wire queue is empty");
        SerializedWireItem item = items_.front();
        if (item.ticket != completed_ticket_ + 1)
            throw std::logic_error(
                "serialized wire tickets are not completed in FIFO order");
        items_.pop_front();
        completed_ticket_ = item.ticket;
        return item;
    }

    bool Completed(uint64_t ticket) const noexcept {
        return ticket != 0 && completed_ticket_ >= ticket;
    }

  private:
    std::size_t capacity_;
    std::deque<SerializedWireItem> items_;
    uint64_t next_ticket_ = 1;
    uint64_t completed_ticket_ = 0;
};
