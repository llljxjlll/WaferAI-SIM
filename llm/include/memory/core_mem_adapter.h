#pragma once
// R1：CoreMemAdapter —— 核侧的分布式 HBM 访存适配器。接收一次逻辑访存调用，做地址
// 解码、打包成 MemMsg、分配/回收 txid，转交给对应 (stack_id, channel_id) 的
// MemEndpointUnit，阻塞等待完成。详见 HBM建模计划.md（修订版）四.1。
//
// R0-R3 范围：本类用于合成请求 testbench 和 HBMRuntime 集成验证，尚未接入
// memory_utils.cpp 里真实算子的 sram_*_generic 调用链；生产迁移属于 R4。
#include "memory/hbm_mem_wire.h"
#include "memory/hbm_address_map.h"
#include "memory/mem_endpoint_unit.h"

#include <map>
#include <set>
#include <systemc>
#include <utility>

class MemTransport {
public:
    virtual ~MemTransport() = default;
    virtual MemMsg Exchange(const MemMsg &request) = 0;
};

class CoreMemAdapter : public sc_core::sc_module {
public:
    int core_id;

    // txid_wrap 缺省用生产级大周期；测试可以传一个很小的值，快速证明 txid 会被
    // 回收复用，而不用真的发几十万个请求去等自然回绕。
    CoreMemAdapter(const sc_core::sc_module_name &n, int core_id,
                  int txid_wrap = 1 << 20);

    // 把某个 (stack_id, channel_id) 的落地端点接进来；DecodeAddress 解出的
    // (stack_id, channel_id) 必须能在这张表里查到，否则 Access() 会抛错。
    void BindEndpoint(int stack_id, int channel_id, MemEndpointUnit *ep);
    // R4 production path: once bound, every request is serialized and carried by
    // the shared Router/NoC/C2C fabric. Direct endpoint binding remains only for
    // the isolated R1-R3 compatibility tests.
    void BindTransport(MemTransport *transport);

    // 阻塞调用：必须在调用方自己的 SC_THREAD 上下文里调用。addr 是物理地址（会经
    // DecodeAddress 解码，不是已经解码过的 channel 内地址）。写请求 data.size() 必须
    // 等于 length_bytes；读请求返回值的 payload 就是读到的数据。
    MemMsg Access(MemCommand cmd, uint64_t phys_addr, int length_bytes,
                 const std::vector<uint8_t> &write_data = {},
                 const std::vector<uint8_t> &byte_enable = {});

    // 观测：本适配器已发出、尚未收到响应的请求数（R1 用单适配器单线程顺序发起，
    // 恒为 0 或 1；多适配器并发发起时，系统级的多 outstanding 由多个 adapter
    // 实例各自的 in-flight 状态共同体现，可在 MemEndpointUnit::InFlightCount 观测）。
    bool HasOutstanding() const { return !outstanding_txids_.empty(); }
    int OutstandingCount() const { return (int)outstanding_txids_.size(); }
    int LastTxid() const { return last_txid_; }

private:
    int next_txid_ = 0;
    int txid_wrap_;
    int last_txid_ = -1;
    std::set<int> outstanding_txids_;
    std::map<std::pair<int, int>, MemEndpointUnit *> endpoints_;
    MemTransport *transport_ = nullptr;

    int AllocateTxid();
    MemMsg IssueChunk(MemCommand cmd, const AddressDecodeResult &decoded,
                      const std::vector<uint8_t> &data,
                      const std::vector<uint8_t> &byte_enable);
};
