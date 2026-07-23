#pragma once
#include <map>
#include <stdexcept>
#include <string>
// flow 标识：(source_global_id, tag, subflow)。
//
// **重要（V1 审查结论）**：本类型**不用于 router 的 output_lock**。output_lock 有意按 `tag` 锁，
// 因为本工程 `tag == 接收核的 recv_tag == 全局核 id`（唯一），即 tag 就是「接收端聚合槽」：
//   - 同 tag ⟺ 同接收核 ⟺ 同一聚合槽 → 多个源同 tag 发一个核（「多发一」，见 router.cpp 的
//     output_lock_ref 累加）本就该**共享锁、交错通过**，接收端按包内地址重组；
//   - 全局 tag 下不存在「不同接收核撞 tag」的别名，故加 source 维会**错误地把多发一拆成串行**。
// 因此 output_lock 保持 tag-only 是正确的。FlowKey 的 runtime 使用者是 V3 whole-flow SAF
// reservation（必须按发送流分别记账，不能按接收槽合并）；V5 tensor striping 还会使用 subflow
// 做子流重组/跟踪。它仍不用于 router output_lock。
struct FlowKey {
    int source = -1;  // source_global_id（真实全局发送核 id）
    int tag = -1;     // tag_id
    int subflow = 0;  // V5 striping 子流号；V1 恒 0

    bool operator==(const FlowKey &o) const {
        return source == o.source && tag == o.tag && subflow == o.subflow;
    }
    bool operator!=(const FlowKey &o) const { return !(*this == o); }
    bool operator<(const FlowKey &o) const {
        if (source != o.source)
            return source < o.source;
        if (tag != o.tag)
            return tag < o.tag;
        return subflow < o.subflow;
    }
};


enum class WholeFlowSafStatus {
    ADMITTED = 0,
    INVALID_FLOW_SIZE,
    DUPLICATE_FLOW,
    INSUFFICIENT_CAPACITY,
};

inline const char *WholeFlowSafStatusName(WholeFlowSafStatus s) {
    switch (s) {
    case WholeFlowSafStatus::ADMITTED:
        return "admitted";
    case WholeFlowSafStatus::INVALID_FLOW_SIZE:
        return "invalid_flow_size";
    case WholeFlowSafStatus::DUPLICATE_FLOW:
        return "duplicate_flow";
    case WholeFlowSafStatus::INSUFFICIENT_CAPACITY:
        return "insufficient_capacity";
    }
    return "unknown";
}

struct WholeFlowSafResult {
    WholeFlowSafStatus status = WholeFlowSafStatus::INVALID_FLOW_SIZE;
    int requested = 0;
    int available_before = 0;
    bool admitted() const { return status == WholeFlowSafStatus::ADMITTED; }
};

// V3-c：单个接收 buffer 的 whole-flow SAF 原子预留账本。
// Reserve() 先完整检查再一次性记账；失败路径绝不改变 reserved_/reservations_，因而不会出现
// 「先注入部分包，再持有上游资源等待剩余空间」的 hold-and-wait。实际有限 FIFO 接线属 V3-d。
class WholeFlowSafAdmission {
public:
    explicit WholeFlowSafAdmission(int capacity = 0) { Reset(capacity); }

    void Reset(int capacity) {
        if (capacity < 0)
            throw std::runtime_error("whole-flow SAF capacity must be >= 0");
        capacity_ = capacity;
        reserved_ = 0;
        reservations_.clear();
    }

    WholeFlowSafResult Reserve(const FlowKey &key, int flow_packets) {
        WholeFlowSafResult r;
        r.requested = flow_packets;
        r.available_before = Available();
        if (flow_packets <= 0) {
            r.status = WholeFlowSafStatus::INVALID_FLOW_SIZE;
            return r;
        }
        if (reservations_.count(key)) {
            r.status = WholeFlowSafStatus::DUPLICATE_FLOW;
            return r;
        }
        if (flow_packets > r.available_before) {
            r.status = WholeFlowSafStatus::INSUFFICIENT_CAPACITY;
            return r;
        }
        reservations_[key] = flow_packets;
        reserved_ += flow_packets;
        r.status = WholeFlowSafStatus::ADMITTED;
        return r;
    }

    int Release(const FlowKey &key) {
        auto it = reservations_.find(key);
        if (it == reservations_.end())
            throw std::runtime_error("whole-flow SAF release without reservation");
        int n = it->second;
        reservations_.erase(it);
        reserved_ -= n;
        if (reserved_ < 0)
            throw std::runtime_error("whole-flow SAF reservation accounting underflow");
        return n;
    }

    int Capacity() const { return capacity_; }
    int Reserved() const { return reserved_; }
    int Available() const { return capacity_ - reserved_; }
    int ActiveFlows() const { return (int)reservations_.size(); }
    bool Has(const FlowKey &key) const { return reservations_.count(key) != 0; }

private:
    int capacity_ = 0;
    int reserved_ = 0;
    std::map<FlowKey, int> reservations_;
};
