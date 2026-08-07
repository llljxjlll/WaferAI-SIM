#include "monitor/start_data_tracker.h"
#include <sstream>
#include <stdexcept>

namespace {
uint64_t &Counter(StartDataStageCounts &counts, StartDataStage stage) {
    switch (stage) {
    case StartDataStage::ENQUEUED:
        return counts.enqueued;
    case StartDataStage::INJECTED:
        return counts.injected;
    case StartDataStage::ROUTER_ACCEPTED:
        return counts.router_accepted;
    case StartDataStage::DELIVERED:
        return counts.delivered;
    case StartDataStage::CONSUMED:
        return counts.consumed;
    case StartDataStage::COMPLETED:
        return counts.completed;
    }
    throw std::logic_error("unknown S_DATA lifecycle stage");
}
} // namespace

StartDataTracker &StartDataTracker::Instance() {
    static StartDataTracker tracker;
    return tracker;
}

void StartDataTracker::Reset() {
    totals_ = {};
    per_message_.clear();
}

void StartDataTracker::Record(StartDataStage stage, const Msg &msg) {
    if (msg.msg_type_ != MSG_TYPE::S_DATA)
        throw std::logic_error("S_DATA tracker received a non-S_DATA message");
    const Key key{msg.des_, msg.tag_id_, msg.seq_id_};
    auto &counts = per_message_[key];
    auto require_predecessor = [&](uint64_t predecessor, uint64_t current,
                                   const char *name) {
        if (current >= predecessor)
            throw std::logic_error(std::string("S_DATA ") + name +
                                   " observed before its predecessor");
    };
    switch (stage) {
    case StartDataStage::ENQUEUED:
        break;
    case StartDataStage::INJECTED:
        require_predecessor(counts.enqueued, counts.injected, "injection");
        break;
    case StartDataStage::ROUTER_ACCEPTED:
        require_predecessor(counts.injected, counts.router_accepted,
                            "router acceptance");
        break;
    case StartDataStage::DELIVERED:
        require_predecessor(counts.router_accepted, counts.delivered,
                            "delivery");
        break;
    case StartDataStage::CONSUMED:
        require_predecessor(counts.delivered, counts.consumed, "consumption");
        break;
    case StartDataStage::COMPLETED:
        require_predecessor(counts.consumed, counts.completed, "completion");
        break;
    }
    ++Counter(totals_, stage);
    ++Counter(counts, stage);
    if (stage == StartDataStage::COMPLETED)
        completed_event_.notify(SC_ZERO_TIME);
}

std::string StartDataTracker::Summary() const {
    std::ostringstream os;
    os << "enqueued=" << totals_.enqueued
       << " injected=" << totals_.injected
       << " accepted=" << totals_.router_accepted
       << " delivered=" << totals_.delivered
       << " consumed=" << totals_.consumed
       << " completed=" << totals_.completed;
    return os.str();
}

std::string StartDataTracker::OutstandingSummary() const {
    std::ostringstream os;
    bool first = true;
    for (const auto &entry : per_message_) {
        const auto &counts = entry.second;
        if (counts.completed >= counts.enqueued)
            continue;
        if (!first)
            os << ";";
        first = false;
        os << "dest=" << std::get<0>(entry.first)
           << ",tag=" << std::get<1>(entry.first)
           << ",seq=" << std::get<2>(entry.first)
           << ",q=" << counts.enqueued
           << ",i=" << counts.injected
           << ",r=" << counts.router_accepted
           << ",d=" << counts.delivered
           << ",c=" << counts.consumed
           << ",done=" << counts.completed;
    }
    return first ? "none" : os.str();
}

void ResetStartDataTracking() { StartDataTracker::Instance().Reset(); }

void RecordStartDataStage(StartDataStage stage, const Msg &msg) {
    StartDataTracker::Instance().Record(stage, msg);
}
