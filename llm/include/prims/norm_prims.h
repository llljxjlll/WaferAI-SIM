#pragma once
#include "systemc.h"

#include "common/memory.h"
#include "common/pd.h"
#include "dte/dte_async_types.h"
#include "dte/coll_types.h"
#include "memory/core_lsu_unit.h"
#include "prims/base.h"

class Clear_sram : public PrimBase {
public:
    int taskCoreDefault(TaskCoreContext &context);

    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> buffer);
    void printSelf();

    Clear_sram() { name = "Clear_sram"; }
};


class Load_prim : public PrimBase {
public:
    int dram_addr = 0;
    int sram_addr = 0;
    int size = 0;

    int taskCoreDefault(TaskCoreContext &context);

    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> buffer);
    void printSelf();

    Load_prim() { name = "Load_prim"; }
};


enum class LsuMemOp : uint8_t {
    ISSUE = 0, WAIT, POLL, FENCE, CANCEL, LOAD_BLOCKING, STORE_BLOCKING
};

class Lsu_mem_prim : public PrimBase {
public:
    LsuMemOp op = LsuMemOp::ISSUE;
    uint64_t token = 0;
    sram::LsuDirection direction = sram::LsuDirection::kHbmToSram;
    uint64_t hbm_addr = 0;
    uint64_t sram_addr = 0;
    uint64_t sram_offset = 0;
    uint64_t size_bytes = 0;
    std::string sram_region;
    bool absolute_sram = false;
    bool poll_complete = false;

    int taskCoreDefault(TaskCoreContext &context);
    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> segments);
    void parseJson(json j);
    void printSelf();

    Lsu_mem_prim() { name = "Lsu_mem"; }
};

enum class SramPipelineEngine : uint8_t { kLsu = 0, kDte = 1 };

class Sram_pipeline_prim : public PrimBase {
public:
    SramPipelineEngine engine = SramPipelineEngine::kLsu;
    bool double_buffer = true;
    uint32_t tile_count = 4;
    uint64_t tile_bytes = 256;
    uint64_t compute_cycles = 32;
    uint64_t input_hbm_base = 0x10000;
    uint64_t output_hbm_base = 0x20000;
    uint32_t token_base = 1000;
    uint8_t transform_xor = 0x5a;
    std::string region_a = "double_a";
    std::string region_b = "double_b";

    int taskCoreDefault(TaskCoreContext &context);
    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> segments);
    void parseJson(json j);
    void printSelf();

    Sram_pipeline_prim() { name = "Sram_pipeline"; }
};

class Dte_async_prim : public PrimBase {
public:
    DteAsyncOp op = DteAsyncOp::ISSUE;
    uint32_t token = 0;
    uint64_t payload_bits = 0;
    DteDir direction = DteDir::SPM_TO_REMOTE;
    uint64_t spm_addr = 0;
    uint64_t spm_size = 0; // byte
    std::string sram_region;
    uint64_t sram_offset = 0;
    // V3b aggregation compatibility metadata. Legacy V3a workloads may omit it
    // while aggregation is disabled.
    uint32_t remote_peer = DTE_ASYNC_INVALID_REMOTE_PEER;
    uint64_t remote_addr = 0;
    uint32_t address_block = 0;
    // Runtime-only poll result; the repository has no branch primitive yet.
    bool poll_complete = false;

    int taskCoreDefault(TaskCoreContext &context);
    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> segments);
    void parseJson(json j);
    void printSelf();

    Dte_async_prim() { name = "Dte_async"; }
};

class Collective_prim : public PrimBase {
public:
    enum class MarkerKind : uint8_t {
        BARRIER = 0, GATHER_ARRIVAL = 1, REDUCE_ARRIVAL = 2
    };
    CollDescriptor descriptor;
    uint16_t phase_id = 0;
    MarkerKind marker_kind = MarkerKind::BARRIER;
    uint16_t release_tree_id = 0;

    int taskCoreDefault(TaskCoreContext &context);
    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> segments);
    void parseJson(json j);
    void printSelf();

    Collective_prim() { name = "Collective_prim"; }
};

class Reduce_compute_prim : public PrimBase {
public:
    CollDescriptor descriptor;

    int taskCoreDefault(TaskCoreContext &context);
    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> segments);
    void printSelf();

    Reduce_compute_prim() { name = "Reduce_compute_prim"; }
};

class Collective_data_prim : public PrimBase {
public:
    enum class Mode : uint8_t {
        BROADCAST_TX = 0, BROADCAST_RX = 1,
        REDUCE_TX = 2, REDUCE_RX = 3,
        REDUCE_STREAM_RX_START = 4,
        REDUCE_STREAM_TX = 5,
        REDUCE_STREAM_RX_WAIT = 6,
        CORE_VECTOR_START = 7,
        CORE_VECTOR_WAIT = 8
    };
    CollDescriptor descriptor;
    uint16_t tree_id = 0;
    uint32_t core_vector_beats = 0;
    Mode mode = Mode::BROADCAST_RX;

    int taskCoreDefault(TaskCoreContext &context);
    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> segments);
    void printSelf();

    Collective_data_prim() { name = "Collective_data_prim"; }
};


class Recv_prim : public PrimBase {
public:
    // 类内默认值：避免 1-arg/默认构造留下未初始化字段（tag_id 曾被 RECV_CONF 的 CONFIG ACK
    // 读到未初始化值 → 非确定 tag，属 UB）。RECV_CONF 的 CONFIG ACK tag 契约固定为 0。
    RECV_TYPE type = RECV_CONF;
    int tag_id = 0;   // 和send原语对应的tag（RECV_CONF 默认 0 = CONFIG ACK tag 契约）
    int recv_cnt = 0; // 需要接收到的end包数量（用于多发一）
    int stripe_count = 1; // V5 grouped-recv：每个逻辑 sender 的 subflow 数

    int taskCoreDefault(TaskCoreContext &context);

    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> buffer);
    void printSelf();

    Recv_prim() { name = "Recv_prim"; }
    Recv_prim(RECV_TYPE type) : type(type) { name = "Recv_prim"; }
    Recv_prim(RECV_TYPE type, int tag, int cnt)
        : type(type), tag_id(tag), recv_cnt(cnt) {
        name = "Recv_prim";
    }
};


class Send_prim : public PrimBase {
public:
    SEND_TYPE type = SEND_REQ;
    int des_id = -1;                   // 目标id
    string output_label = UNSET_LABEL; // 需要从哪一个数据块标签获取结果，并发送
    // SEND_DATA：聚合后的模拟 DATA 包数；SEND_REQ：tagged-union 为随后 DATA flow 的总包数（V3-c）。
    int max_packet = 0;
    int tag_id = 0;                    // send_tag，用于与recv原语对应
    int end_length = 0;                // 尾包有效长度（bit），范围 1..M_D_DATA
    // max_packet 是按 HW_NOC_PAYLOAD_PER_CYCLE 聚合后的模拟包数：
    // raw_packets = (max_packet - 1) * packet_scale + packets_in_last_group。
    int packet_scale = 1;
    int packets_in_last_group = 1;
    int stripe_count = 1;              // V5：1/2/4 条 subflow

    // V1-c2 运行时路由状态（不进入 prim 配置序列化）：一条 SEND_DATA 原语只选一次
    // source-die C2C 出口，所有 DATA 包复制同一 pinned port。same-die 为 -1。
    int d2d_exit_port = -1;
    bool d2d_exit_selected = false;

    // V5 运行态（不序列化）：每条 subflow 的包数、已发包数和固定出口。
    vector<int> stripe_packets;
    vector<int> stripe_sent;
    vector<int> stripe_exit_ports;
    int next_subflow = 0;
    bool stripe_saf_reserved = false;

    int data_packet_id = 0; // 已经发送的包裹数量

    int taskCoreDefault(TaskCoreContext &context);

    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> buffer);

    void printSelf();

    Send_prim() { name = "Send_prim"; }
    Send_prim(SEND_TYPE type) : type(type) { name = "Send_prim"; }
    Send_prim(SEND_TYPE type, int des, int tag)
        : type(type), des_id(des), tag_id(tag) {
        name = "Send_prim";
    } // 用于SEND_ACK
    Send_prim(SEND_TYPE type, int des, int max_packet, int tag)
        : des_id(des), type(type), max_packet(max_packet), tag_id(tag) {
        name = "Send_prim";
    }
};


class Set_addr : public PrimBase {
public:
    AddrDatapassLabel datapass_label;

    int taskCoreDefault(TaskCoreContext &context);

    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> buffer);

    void printSelf();
    Set_addr() { name = "Set_addr"; }
};

class Set_batch : public PrimBase {
public:
    vector<Stage> batch_info;
    int auto_pd;

    int taskCoreDefault(TaskCoreContext &context);

    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> buffer);

    void printSelf();

    Set_batch() {
        name = "Set_batch";
        auto_pd = 0;
    }

    Set_batch(vector<Stage> batchInfo) {
        name = "Set_batch";
        this->batch_info = batchInfo;
        auto_pd = 0;
    }

    Set_batch(vector<Stage> batchInfo, int auto_pd) {
        name = "Set_batch";
        this->batch_info = batchInfo;
        this->auto_pd = auto_pd;
    }
};

class Store_prim : public PrimBase {
public:
    int dram_addr;
    int sram_addr;
    int size;

    int taskCoreDefault(TaskCoreContext &context);

    void deserialize(vector<sc_bv<128>> buffer);
    vector<sc_bv<128>> serialize();

    void printSelf();

    Store_prim() { name = "Store_prim"; }
};
