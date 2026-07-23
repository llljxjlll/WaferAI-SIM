#pragma once
#include "systemc.h"

#include "common/memory.h"
#include "common/pd.h"
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
    int taskCoreDefault(TaskCoreContext &context);

    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> buffer);
    void printSelf();

    Load_prim() { name = "Load_prim"; }
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
    // SEND_DATA：实际 DATA 包数；SEND_REQ：tagged-union 为随后 DATA flow 的总包数（V3-c）。
    int max_packet = 0;
    int tag_id = 0;                    // send_tag，用于与recv原语对应
    int end_length = 0;                // 尾包长度，避免覆盖
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