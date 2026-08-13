#include "memory/dramsys_hbm_backend.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

using namespace sc_core;

DRAMSysHBMBackend::DRAMSysHBMBackend(const sc_module_name &n,
                                     const std::string &dram_config_path,
                                     std::string_view resource_directory)
    : sc_module(n),
      config_(::DRAMSys::Config::from_path(dram_config_path,
                                           resource_directory)),
      initiator_socket_("initiatorSocket"),
      peq_(this, &DRAMSysHBMBackend::peqCallback) {
    config_.simconfig.StoreMode = ::DRAMSys::Config::StoreModeType::Store;
    dram_sys_wrapper_ =
        new gem5::memory::DRAMSysWrapper("DRAMSysWrapper", config_, false);
    initiator_socket_.register_nb_transport_bw(
        this, &DRAMSysHBMBackend::nb_transport_bw);
    initiator_socket_.bind(dram_sys_wrapper_->tSocket);
    SC_THREAD(CleanupLoop);
}

DRAMSysHBMBackend::~DRAMSysHBMBackend() { delete dram_sys_wrapper_; }

uint64_t DRAMSysHBMBackend::CapacityBytes() const {
    return dram_sys_wrapper_->dramsys->getMemSpec().memorySizeBytes;
}

DRAMSys::DecodedAddress
DRAMSysHBMBackend::DecodeBackendAddress(uint64_t address) const {
    return dram_sys_wrapper_->dramsys->getAddressDecoder().decodeAddress(address);
}

uint64_t DRAMSysHBMBackend::EncodeBackendAddress(
    DRAMSys::DecodedAddress address) const {
    return dram_sys_wrapper_->dramsys->getAddressDecoder().encodeAddress(address);
}

void DRAMSysHBMBackend::Submit(
    const std::shared_ptr<HBMBackendTransaction> &tx) {
    if (!tx || tx->payload.empty() || !tx->complete)
        throw std::runtime_error(
            "DRAMSysHBMBackend::Submit: malformed transaction");
    if (tx->address >= CapacityBytes() ||
        tx->payload.size() > CapacityBytes() - tx->address)
        throw std::runtime_error(
            "DRAMSysHBMBackend::Submit: address exceeds backend capacity");
    if (!tx->byte_enable.empty() &&
        tx->byte_enable.size() != tx->payload.size())
        throw std::runtime_error(
            "DRAMSysHBMBackend::Submit: byte-enable size mismatch");
    if (active_logical_.count(tx.get()))
        throw std::runtime_error(
            "DRAMSysHBMBackend::Submit: transaction object already pending");

    const auto &mem_spec = dram_sys_wrapper_->dramsys->getMemSpec();
    const uint64_t burst_bytes = mem_spec.defaultBytesPerBurst;
    if (burst_bytes == 0 ||
        burst_bytes >
            static_cast<uint64_t>(std::numeric_limits<unsigned>::max()))
        throw std::runtime_error(
            "DRAMSysHBMBackend::Submit: invalid physical burst size");

    const uint64_t logical_end = tx->address + tx->payload.size();
    const uint64_t first_burst = tx->address - (tx->address % burst_bytes);
    const uint64_t last_burst =
        (logical_end - 1) - ((logical_end - 1) % burst_bytes);
    const uint64_t capacity_bytes = CapacityBytes();
    if (capacity_bytes < burst_bytes ||
        first_burst > capacity_bytes - burst_bytes ||
        last_burst > capacity_bytes - burst_bytes)
        throw std::runtime_error(
            "DRAMSysHBMBackend::Submit: aligned burst exceeds backend capacity");

    DRAMSys::DecodedAddress decoded = DecodeBackendAddress(tx->address);
    DRAMSysHBMAccessRecord record;
    record.command = tx->command;
    record.address = tx->address;
    record.channel = decoded.channel;
    record.rank = decoded.rank;
    record.bankgroup = decoded.bankgroup;
    record.bank = decoded.bank;
    record.row = decoded.row;
    record.column = decoded.column;
    record.submitted = sc_time_stamp();
    auto logical = std::make_shared<LogicalRequest>(tx);
    std::list<std::shared_ptr<PhysicalRequest>> physical_requests;
    uint64_t burst_address = first_burst;
    while (burst_address < logical_end) {
        const uint64_t covered_begin = std::max(burst_address, tx->address);
        const uint64_t burst_end = burst_address + burst_bytes;
        const uint64_t covered_end = std::min(burst_end, logical_end);

        auto physical = std::make_shared<PhysicalRequest>();
        physical->logical = logical;
        physical->address = burst_address;
        physical->payload.assign(static_cast<size_t>(burst_bytes), 0);
        physical->logical_offset =
            static_cast<size_t>(covered_begin - tx->address);
        physical->burst_offset =
            static_cast<size_t>(covered_begin - burst_address);
        physical->copy_length =
            static_cast<size_t>(covered_end - covered_begin);
        if (tx->command == MemCommand::kWrite) {
            physical->byte_enable.assign(static_cast<size_t>(burst_bytes), 0);
            for (size_t i = 0; i < physical->copy_length; ++i) {
                const size_t logical_index = physical->logical_offset + i;
                const size_t burst_index = physical->burst_offset + i;
                physical->payload[burst_index] = tx->payload[logical_index];
                if (tx->byte_enable.empty() || tx->byte_enable[logical_index])
                    physical->byte_enable[burst_index] = 0xff;
            }
        }
        physical_requests.push_back(std::move(physical));
        ++logical->remaining;
        burst_address = burst_end;
    }

    logical->trace_index = access_trace_.size();
    access_trace_.push_back(record);
    try {
        const auto inserted = active_logical_.emplace(tx.get(), logical);
        if (!inserted.second)
            throw std::logic_error(
                "DRAMSysHBMBackend: duplicate active logical request");
    } catch (...) {
        access_trace_.pop_back();
        throw;
    }
    pending_.splice(pending_.end(), physical_requests);
    stats_.requests++;
    stats_.bytes += tx->payload.size();
    if (tx->command == MemCommand::kRead)
        stats_.reads++;
    else
        stats_.writes++;
    try {
        TryIssue();
    } catch (...) {
        if (!logical->issued) {
            pending_.remove_if([&](const auto &request) {
                return request->logical == logical;
            });
            active_logical_.erase(tx.get());
            access_trace_.pop_back();
            --stats_.requests;
            stats_.bytes -= tx->payload.size();
            if (tx->command == MemCommand::kRead)
                --stats_.reads;
            else
                --stats_.writes;
        }
        throw;
    }
}

void DRAMSysHBMBackend::TryIssue() {
    if (begin_req_in_progress_ || pending_.empty())
        return;

    auto holder = std::make_unique<Inflight>(pending_.front(), this);
    pending_.pop_front();
    Inflight *h = holder.get();
    h->issued = sc_time_stamp();
    auto &logical = *h->request->logical;
    if (!logical.issued) {
        logical.issued = true;
        logical.issued_at = h->issued;
        access_trace_[logical.trace_index].issued = h->issued;
    }
    h->payload.set_command(logical.tx->command == MemCommand::kWrite
                               ? tlm::TLM_WRITE_COMMAND
                               : tlm::TLM_READ_COMMAND);
    h->payload.set_address(h->request->address);
    h->payload.set_data_ptr(h->request->payload.data());
    h->payload.set_data_length((unsigned)h->request->payload.size());
    h->payload.set_streaming_width((unsigned)h->request->payload.size());
    if (h->request->byte_enable.empty()) {
        h->payload.set_byte_enable_ptr(nullptr);
        h->payload.set_byte_enable_length(0);
    } else {
        h->payload.set_byte_enable_ptr(h->request->byte_enable.data());
        h->payload.set_byte_enable_length(
            (unsigned)h->request->byte_enable.size());
    }
    h->payload.set_dmi_allowed(false);
    h->payload.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
    h->payload.acquire(); // initiator holds through END_RESP

    tlm::tlm_generic_payload *payload = &h->payload;
    inflight_[payload] = std::move(holder);
    peak_inflight_ = std::max(peak_inflight_, inflight_.size());
    begin_req_in_progress_ = true;
    begin_req_payload_ = payload;

    tlm::tlm_phase phase = tlm::BEGIN_REQ;
    sc_time delay = SC_ZERO_TIME;
    tlm::tlm_sync_enum result =
        initiator_socket_->nb_transport_fw(*payload, phase, delay);
    if (result == tlm::TLM_UPDATED)
        peq_.notify(*payload, phase, delay);
    else if (result == tlm::TLM_COMPLETED)
        throw std::runtime_error(
            "DRAMSysHBMBackend: unexpected TLM_COMPLETED on BEGIN_REQ");
}

tlm::tlm_sync_enum DRAMSysHBMBackend::nb_transport_bw(
    tlm::tlm_generic_payload &payload, tlm::tlm_phase &phase,
    sc_time &delay) {
    peq_.notify(payload, phase, delay);
    return tlm::TLM_ACCEPTED;
}

void DRAMSysHBMBackend::peqCallback(tlm::tlm_generic_payload &payload,
                                    const tlm::tlm_phase &phase) {
    auto it = inflight_.find(&payload);
    if (it == inflight_.end())
        throw std::runtime_error(
            "DRAMSysHBMBackend: phase for unknown transaction");

    if (phase == tlm::END_REQ) {
        if (begin_req_payload_ != &payload)
            throw std::runtime_error(
                "DRAMSysHBMBackend: END_REQ does not match active BEGIN_REQ");
        begin_req_in_progress_ = false;
        begin_req_payload_ = nullptr;
        TryIssue();
        return;
    }
    if (phase != tlm::BEGIN_RESP)
        throw std::runtime_error(
            "DRAMSysHBMBackend: unexpected backward TLM phase");

    // Base protocol 允许 BEGIN_RESP 隐式结束 request phase（没有单独 END_REQ）。
    // 只在响应属于当前 active BEGIN_REQ 时释放 exclusion，不能被更早事务的响应
    // 错误清除一个较新请求的 exclusion。
    bool response_ends_request = begin_req_payload_ == &payload;
    if (response_ends_request) {
        begin_req_in_progress_ = false;
        begin_req_payload_ = nullptr;
    }

    tlm::tlm_phase end_phase = tlm::END_RESP;
    sc_time end_delay = SC_ZERO_TIME;
    initiator_socket_->nb_transport_fw(payload, end_phase, end_delay);

    Inflight *holder = it->second.get();
    auto physical = holder->request;
    auto logical = physical->logical;
    const int status =
        payload.get_response_status() == tlm::TLM_OK_RESPONSE ? 0 : 1;
    if (status && logical->status == 0) {
        logical->status = status;
        logical->error = "DRAMSys transaction returned non-OK status";
    }
    if (status == 0 && logical->tx->command == MemCommand::kRead) {
        std::copy_n(physical->payload.begin() + physical->burst_offset,
                    physical->copy_length,
                    logical->tx->payload.begin() + physical->logical_offset);
    }
    if (logical->remaining == 0)
        throw std::logic_error(
            "DRAMSysHBMBackend: physical completion underflow");
    --logical->remaining;
    if (logical->remaining == 0) {
        const sc_time service = sc_time_stamp() - logical->issued_at;
        stats_.service_time += service;
        stats_.completed++;
        if (logical->status)
            stats_.failed++;
        DRAMSysHBMAccessRecord &record = access_trace_[logical->trace_index];
        record.completed = sc_time_stamp();
        record.service = service;
        record.status = logical->status;
        active_logical_.erase(logical->tx.get());
        logical->tx->complete(SC_ZERO_TIME, service, logical->status,
                              logical->error);
    }
    // 这里只释放 initiator 自己的引用。Arbiter 在稍后消费 END_RESP 后才释放其引用；
    // 最终 refcount==0 会回调 free()，下一 delta 才真正销毁 payload。
    payload.release();
    if (response_ends_request)
        TryIssue();
}

void DRAMSysHBMBackend::free(tlm::tlm_generic_payload *payload) {
    retired_.push_back(payload);
    cleanup_event_.notify(SC_ZERO_TIME);
}

void DRAMSysHBMBackend::CleanupLoop() {
    while (true) {
        wait(cleanup_event_);
        while (!retired_.empty()) {
            auto *payload = retired_.front();
            retired_.pop_front();
            inflight_.erase(payload);
        }
    }
}
