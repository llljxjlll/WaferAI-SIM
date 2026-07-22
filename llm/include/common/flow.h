#pragma once
// flow 标识：(source_global_id, tag, subflow)。
//
// **重要（V1 审查结论）**：本类型**不用于 router 的 output_lock**。output_lock 有意按 `tag` 锁，
// 因为本工程 `tag == 接收核的 recv_tag == 全局核 id`（唯一），即 tag 就是「接收端聚合槽」：
//   - 同 tag ⟺ 同接收核 ⟺ 同一聚合槽 → 多个源同 tag 发一个核（「多发一」，见 router.cpp 的
//     output_lock_ref 累加）本就该**共享锁、交错通过**，接收端按包内地址重组；
//   - 全局 tag 下不存在「不同接收核撞 tag」的别名，故加 source 维会**错误地把多发一拆成串行**。
// 因此 output_lock 保持 tag-only 是正确的。FlowKey 现**专门预留给 V5 tensor striping 的
// subflow 重组 / 子流跟踪**（那里同一 (source,tag) 会拆成多条 subflow，需要三元组区分），
// 以及需要「按发送流」而非「按接收槽」聚合的未来场景。V1 不消费它（保留类型 + 单测即可）。
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
