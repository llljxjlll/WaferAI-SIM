#pragma once
// V1 起的 flow 标识：(source_global_id, tag[, subflow])。
// router/端口的流跟踪与 output_lock 从 V1 起以此为 key，**不得只按 tag 区分**——否则不同 die
// 同 local-id（同 tag）的并发流会互相别名。V1 单流不撞 tag，但结构此版就位，供 V2+ 并发兜底。
// subflow 预留给 V5 tensor striping。
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
