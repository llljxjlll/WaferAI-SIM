#include "memory/core_mem_adapter.h"

#include "defs/spec.h"

#include <algorithm>
#include <stdexcept>

using namespace sc_core;

CoreMemAdapter::CoreMemAdapter(const sc_module_name &n, int core_id,
                               int txid_wrap)
    : sc_module(n), core_id(core_id), txid_wrap_(txid_wrap) {
    if (txid_wrap_ <= 0)
        throw std::runtime_error("CoreMemAdapter: txid_wrap must be > 0");
}

void CoreMemAdapter::BindEndpoint(int stack_id, int channel_id,
                                  MemEndpointUnit *ep) {
    if (!ep)
        throw std::runtime_error("CoreMemAdapter: cannot bind null endpoint");
    auto key = std::make_pair(stack_id, channel_id);
    if (endpoints_.count(key))
        throw std::runtime_error("CoreMemAdapter: duplicate endpoint binding");
    endpoints_[key] = ep;
}

void CoreMemAdapter::BindTransport(MemTransport *transport) {
    if (!transport)
        throw std::runtime_error("CoreMemAdapter: cannot bind null transport");
    if (transport_)
        throw std::runtime_error("CoreMemAdapter: transport already bound");
    transport_ = transport;
}

int CoreMemAdapter::AllocateTxid() {
    for (int attempts = 0; attempts < txid_wrap_; ++attempts) {
        int candidate = next_txid_;
        next_txid_ = (next_txid_ + 1) % txid_wrap_;
        if (outstanding_txids_.insert(candidate).second) {
            last_txid_ = candidate;
            return candidate;
        }
    }
    throw std::runtime_error(
        "CoreMemAdapter: txid space exhausted by outstanding transactions");
}

MemMsg CoreMemAdapter::IssueChunk(MemCommand cmd,
                                  const AddressDecodeResult &decoded,
                                  const std::vector<uint8_t> &data,
                                  const std::vector<uint8_t> &byte_enable) {
    auto it = endpoints_.find({decoded.stack_id, decoded.channel_id});
    if (!transport_ && it == endpoints_.end())
        throw std::runtime_error(
            "CoreMemAdapter::Access: no MemEndpointUnit bound for decoded "
            "stack/channel");

    MemMsg req;
    req.txid = AllocateTxid();
    req.message_type = MemMessageType::kRequest;
    req.source_core = core_id;
    req.home_die = decoded.home_die;
    req.stack_id = decoded.stack_id;
    req.channel_id = decoded.channel_id;
    req.address = decoded.local_address;
    req.command = cmd;
    req.length_bytes = (int)(cmd == MemCommand::kWrite ? data.size()
                                                       : byte_enable.size());
    req.payload = cmd == MemCommand::kWrite ? data : std::vector<uint8_t>{};
    req.byte_enable = byte_enable;

    try {
        // 即使 R1 的 synthetic fabric 仍是函数调用，也强制经过与真实 NoC 相同的
        // 256-bit codec，避免 codec 与功能路径长期分叉。
        MemMsg resp;
        if (transport_) {
            resp = transport_->Exchange(req);
        } else {
            MemMsg on_wire_req = DeserializeMemMsg(SerializeMemMsg(req));
            MemMsg endpoint_resp = it->second->HandleRequest(on_wire_req);
            resp = DeserializeMemMsg(SerializeMemMsg(endpoint_resp));
        }
        outstanding_txids_.erase(req.txid);

        if (resp.txid != req.txid ||
            resp.message_type != MemMessageType::kResponse ||
            resp.source_core != req.source_core || resp.status != 0 ||
            resp.command != req.command)
            throw std::runtime_error(
                "CoreMemAdapter::Access: malformed or mismatched response");
        return resp;
    } catch (...) {
        outstanding_txids_.erase(req.txid);
        throw;
    }
}

MemMsg CoreMemAdapter::Access(MemCommand cmd, uint64_t phys_addr,
                              int length_bytes,
                              const std::vector<uint8_t> &write_data,
                              const std::vector<uint8_t> &byte_enable) {
    if (length_bytes <= 0)
        throw std::runtime_error(
            "CoreMemAdapter::Access: length_bytes out of range");
    if (phys_addr > UINT64_MAX - (uint64_t)length_bytes)
        throw std::runtime_error("CoreMemAdapter::Access: address range overflows");
    if (cmd == MemCommand::kWrite && (int)write_data.size() != length_bytes)
        throw std::runtime_error(
            "CoreMemAdapter::Access: write_data size must equal length_bytes");
    if (!byte_enable.empty() && (int)byte_enable.size() != length_bytes)
        throw std::runtime_error(
            "CoreMemAdapter::Access: byte_enable size must equal length_bytes");

    std::vector<uint8_t> enables = byte_enable;
    if (enables.empty())
        enables.assign(length_bytes, 0xff);
    int current_die = core_id >= 0 && CORES_PER_DIE > 0
                          ? core_id / CORES_PER_DIE
                          : -1;

    MemMsg aggregate;
    aggregate.message_type = MemMessageType::kResponse;
    aggregate.source_core = core_id;
    aggregate.command = cmd;
    aggregate.length_bytes = cmd == MemCommand::kRead ? length_bytes : 0;
    aggregate.status = 0;

    int offset = 0;
    while (offset < length_bytes) {
        AddressDecodeResult first =
            DecodeAddress(phys_addr + (uint64_t)offset, current_die);
        int chunk = 1;
        while (offset + chunk < length_bytes &&
               chunk < kMemMsgMaxPayloadBytes) {
            AddressDecodeResult next = DecodeAddress(
                phys_addr + (uint64_t)(offset + chunk), current_die);
            if (next.home_die != first.home_die ||
                next.stack_id != first.stack_id ||
                next.channel_id != first.channel_id ||
                next.local_address != first.local_address + (uint64_t)chunk)
                break;
            ++chunk;
        }

        std::vector<uint8_t> part_data;
        if (cmd == MemCommand::kWrite)
            part_data.assign(write_data.begin() + offset,
                             write_data.begin() + offset + chunk);
        std::vector<uint8_t> part_enable(enables.begin() + offset,
                                         enables.begin() + offset + chunk);
        MemMsg resp = IssueChunk(cmd, first, part_data, part_enable);
        if (aggregate.txid < 0)
            aggregate.txid = resp.txid;
        if (cmd == MemCommand::kRead)
            aggregate.payload.insert(aggregate.payload.end(), resp.payload.begin(),
                                     resp.payload.end());
        offset += chunk;
    }
    aggregate.byte_enable = enables;
    return aggregate;
}
