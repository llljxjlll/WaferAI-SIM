#pragma once

#include <cstddef>
#include <stdexcept>

// P2P endpoint DATA is reassembled as one strict byte stream.  A Router
// output therefore belongs to exactly one complete endpoint flow from seq=1
// through its tail; tag alone is not an identity because multiple sources in
// a collective can legitimately use the same destination transport tag.
struct EndpointOutputFlowKey {
    int source = -1;
    int destination = -1;
    int transport_tag = -1;
    int subflow = 0;

    bool operator==(const EndpointOutputFlowKey &other) const {
        return source == other.source && destination == other.destination &&
            transport_tag == other.transport_tag &&
            subflow == other.subflow;
    }
    bool operator!=(const EndpointOutputFlowKey &other) const {
        return !(*this == other);
    }
};

class EndpointOutputFlowLock {
public:
    bool Active() const { return active_; }
    std::size_t Residual() const { return active_ ? 1U : 0U; }

    bool OwnedBy(const EndpointOutputFlowKey &key) const {
        return active_ && owner_ == key;
    }

    const EndpointOutputFlowKey &Owner() const {
        if (!active_)
            throw std::logic_error("endpoint output flow lock has no owner");
        return owner_;
    }

    void Acquire(const EndpointOutputFlowKey &key) {
        if (active_)
            throw std::logic_error(
                "endpoint output flow lock acquired while active");
        owner_ = key;
        active_ = true;
    }

    void Release(const EndpointOutputFlowKey &key) {
        if (!OwnedBy(key))
            throw std::logic_error(
                "endpoint output flow lock released by non-owner");
        owner_ = EndpointOutputFlowKey{};
        active_ = false;
    }

private:
    EndpointOutputFlowKey owner_;
    bool active_ = false;
};
