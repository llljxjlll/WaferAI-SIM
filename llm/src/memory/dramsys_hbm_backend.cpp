#include "memory/dramsys_hbm_backend.h"

#include <algorithm>
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
    if (trace_index_.count(tx.get()))
        throw std::runtime_error(
            "DRAMSysHBMBackend::Submit: transaction object already pending");
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
    trace_index_[tx.get()] = access_trace_.size();
    access_trace_.push_back(record);
    pending_.push_back(tx);
    TryIssue();
}

void DRAMSysHBMBackend::TryIssue() {
    if (begin_req_in_progress_ || pending_.empty())
        return;

    auto holder = std::make_unique<Inflight>(pending_.front(), this);
    pending_.pop_front();
    Inflight *h = holder.get();
    h->issued = sc_time_stamp();
    auto trace_it = trace_index_.find(h->tx.get());
    if (trace_it == trace_index_.end())
        throw std::runtime_error(
            "DRAMSysHBMBackend: transaction has no trace record");
    h->trace_index = trace_it->second;
    access_trace_[h->trace_index].issued = h->issued;
    h->payload.set_command(h->tx->command == MemCommand::kWrite
                               ? tlm::TLM_WRITE_COMMAND
                               : tlm::TLM_READ_COMMAND);
    h->payload.set_address(h->tx->address);
    h->payload.set_data_ptr(h->tx->payload.data());
    h->payload.set_data_length((unsigned)h->tx->payload.size());
    h->payload.set_streaming_width((unsigned)h->tx->payload.size());
    if (h->tx->byte_enable.empty()) {
        h->payload.set_byte_enable_ptr(nullptr);
        h->payload.set_byte_enable_length(0);
    } else {
        h->payload.set_byte_enable_ptr(h->tx->byte_enable.data());
        h->payload.set_byte_enable_length((unsigned)h->tx->byte_enable.size());
    }
    h->payload.set_dmi_allowed(false);
    h->payload.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
    h->payload.acquire(); // initiator 持有至 END_RESP；DRAMSys 可在内部继续 acquire

    tlm::tlm_generic_payload *payload = &h->payload;
    inflight_[payload] = std::move(holder);
    peak_inflight_ = std::max(peak_inflight_, inflight_.size());
    begin_req_in_progress_ = true;
    begin_req_payload_ = payload;
    stats_.requests++;
    stats_.bytes += payload->get_data_length();
    if (payload->is_read())
        stats_.reads++;
    else
        stats_.writes++;

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
    sc_time service = sc_time_stamp() - holder->issued;
    int status = payload.get_response_status() == tlm::TLM_OK_RESPONSE ? 0 : 1;
    stats_.service_time += service;
    stats_.completed++;
    if (status)
        stats_.failed++;
    DRAMSysHBMAccessRecord &record = access_trace_[holder->trace_index];
    record.completed = sc_time_stamp();
    record.service = service;
    record.status = status;
    holder->tx->complete(
        SC_ZERO_TIME, service, status,
        status ? "DRAMSys transaction returned non-OK status" : "");
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
            auto it = inflight_.find(payload);
            if (it != inflight_.end())
                trace_index_.erase(it->second->tx.get());
            inflight_.erase(payload);
        }
    }
}
