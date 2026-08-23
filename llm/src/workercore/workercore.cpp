#include "common/system.h"
#include "systemc.h"
#include <algorithm>
#include <deque>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <typeinfo>

#include "defs/const.h"
#include "defs/global.h"
#include "die/port.h"
#include "dte/coll_latency.h"
#include "dte/coll_multicast.h"
#include "dte/coll_innetwork_reduce.h"
#pragma push_macro("DUMMY")
#undef DUMMY
#include "dte/coll_program_profile_v1.h"
#include "dte/coll_accel_runtime_v1.h"
#include "dte/coll_tree_registry_bridge_v1.h"
#include "dte/collective_executor_v1.h"
#pragma pop_macro("DUMMY")
#include "prims/collective_data_v1_prim.h"
#include "prims/collective_launch_v1_prim.h"
#include "prims/collective_phase_barrier_v1_prim.h"
#include "dte/coll_dca_payload_v1.h"
#include "dte/coll_stream_engine.h"
#include "link/nb_global_memif_v2.h"
#include "memory/dram/GPUNB_DcacheIF.h"
#include "memory/gpu/GPU_L1L2_Cache.h"
#include "memory/sram/Mem_access_unit.h"
#include "monitor/start_data_tracker.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "prims/moe_prims.h"
#include "prims/sram_lifecycle_prim.h"
#include "prims/norm_prims.h"
#include "prims/dte_endpoint_prims.h"
#include "prims/pd_prims.h"
#include "prims/sync_prims.h"
#include "trace/Event_engine.h"
#include "utils/memory_utils.h"
#include "utils/msg_utils.h"
#include "utils/prim_utils.h"
#include "utils/print_utils.h"
#include "utils/router_utils.h"
#include "utils/system_utils.h"
#include "workercore/workercore.h"
#include "workercore/prim_refill_policy.h"
#include "memory/hbm_network.h"
#include "memory/hbm_address_map.h"

using namespace std;

namespace {
bool PrimMayRefill(const PrimBase *prim) {
    if (typeid(*prim) == typeid(Collective_data_prim) ||
        typeid(*prim) == typeid(Collective_prim) ||
        typeid(*prim) == typeid(Reduce_compute_prim))
        return false;
    if (const auto *send = dynamic_cast<const Send_prim *>(prim))
        return SendPrimMayRefill(send->type, send->tag_id);
    if (const auto *recv = dynamic_cast<const Recv_prim *>(prim))
        return RecvPrimMayRefill(recv->type, recv->tag_id);
    return true;
}

void CheckedAccumulate(uint64_t &counter, uint64_t delta,
                       const char *name) {
    if (delta > UINT64_MAX - counter)
        throw std::overflow_error(std::string("P2P endpoint ") + name +
                                  " counter overflow");
    counter += delta;
}

bool IsP2pEndpointMessage(const Msg &message) {
    return message.p2p_endpoint_ &&
           message.offset_ == P2P_ENDPOINT_MSG_MARKER &&
           (message.msg_type_ == MSG_TYPE::REQUEST ||
            message.msg_type_ == MSG_TYPE::DATA ||
            message.msg_type_ == MSG_TYPE::ACK);
}

uint64_t ResolveEndpointSramAddress(
    WorkerCoreExecutor &worker, const DteEndpointSramAddress &address,
    uint64_t size_bytes, sram::Initiator initiator, sram::Command command) {
    if (worker.sram_regions == nullptr)
        throw std::runtime_error(
            "P2P endpoint requires a configured SRAM region table");
    if (address.kind == DteEndpointAddressKind::ABSOLUTE)
        return worker.sram_regions->ResolveAbsolute(
            address.absolute_address_bytes, size_bytes, initiator, command)
            .address;
    return worker.sram_regions->Resolve(
        address.region, address.region_offset_bytes, size_bytes, initiator,
        command).address;
}

void TraceP2pEndpoint(Event_engine *engine, int cid, const char *stage,
                      const char *phase, uint32_t fsm_id, uint64_t bytes,
                      uint64_t fragments, uint32_t checksum) {
    if (engine == nullptr) return;
    engine->add_event(
        "Core " + ToHexString(cid), stage, phase,
        Trace_event_util(
            std::string(stage),
            {{"fsm_id", fsm_id}, {"bytes", bytes},
             {"fragments", fragments}, {"checksum", checksum}}));
}

uint64_t CurrentP2pNanoseconds() {
    return static_cast<uint64_t>(
        sc_time_stamp().value() / sc_time(1, SC_NS).value());
}

void TraceP2pStreamTiming(Event_engine *engine, int cid, uint32_t fsm_id,
                          const P2pTimingMetadata &timing) {
    if (engine == nullptr) return;
    engine->add_event(
        "Core " + ToHexString(cid), "P2P_stream_timing", "i",
        Trace_event_util(
            "P2P_stream_timing",
            {{"fsm_id", fsm_id},
             {"source_first_ns", timing.source_first_ns},
             {"source_done_ns", timing.source_done_ns},
             {"network_tail_cycles", timing.network_tail_cycles},
             {"timing_residual",
              P2pSharedTimingSidebandRuntime::Residual()}}));
}
} // namespace

// workercore
WorkerCore::WorkerCore(const sc_module_name &n, int s_cid,
                       Event_engine *event_engine, string dram_config_name)
    : sc_module(n), cid(s_cid), event_engine(event_engine), dcache(nullptr),
      hbm_adapter(nullptr) {
    const bool distributed = g_memory_system_active &&
        g_memory_topology == MemoryTopology::kDistributedHbm;
    // systolic_config = new HardwareTaskConfig();
    // other_config = new HardwareTaskConfig();
    if (!distributed)
        dcache = new DCache(sc_gen_unique_name("dcache"), cid,
                            (int)cid / GRID_X, (int)cid % GRID_X,
                            this->event_engine, dram_config_name,
                            "../DRAMSys/configs");

    LOG_DEBUG(SYSTEM) << "Core " << cid << " dram config path "
                      << dram_config_name;
    if (dcache)
        LOG_DEBUG(SYSTEM)
            << " max address "
            << dcache->dramSysWrapper->dramsys->getAddressDecoder().maxAddress();

    const auto &sram_config =
        sram::ConfigRegistry::Instance().ForCore(cid);
    auto sram_bitw = GetCoreHWConfig(cid)->sram_bitwidth;
    const uint64_t legacy_sram_rows =
        sram_config.real_data_path
            ? 1
            : HW_SRAM_SIZE * 8 / sram_bitw / SRAM_BANKS;
    ram_array = new DynamicBandwidthRamRow<sc_bv<SRAM_BITWIDTH>, SRAM_BANKS>(
        sc_gen_unique_name("ram_array"), 0,
        legacy_sram_rows, SIMU_READ_PORT,
        SIMU_WRITE_PORT, BANK_PORT_NUM + SRAM_BANKS, BANK_PORT_NUM,
        BANK_HIGH_READ_PORT_NUM, event_engine);
    temp_ram_array =
        new DynamicBandwidthRamRow<sc_bv<SRAM_BITWIDTH>, SRAM_BANKS>(
            sc_gen_unique_name("temp_ram_array"), 0, legacy_sram_rows,
            SIMU_READ_PORT, SIMU_WRITE_PORT, BANK_PORT_NUM + SRAM_BANKS,
            BANK_PORT_NUM, BANK_HIGH_READ_PORT_NUM, event_engine);
    sram_regions = std::make_unique<sram::RegionTable>(
        sram_config, event_engine, cid);

    executor = new WorkerCoreExecutor(sc_gen_unique_name("workercore-exec"),
                                      cid, this->event_engine);
    // executor->MaxDramAddr =
    //     dcache->dramSysWrapper->dramsys->getAddressDecoder().maxAddress();
    if (distributed) {
        if (!ActiveHBMNetwork())
            throw std::runtime_error(
                "distributed_hbm WorkerCore requires an elaborated HBMNetwork");
        hbm_adapter = new CoreMemAdapter(sc_gen_unique_name("core-hbm-adapter"),
                                         cid);
        hbm_adapter->BindTransport(ActiveHBMNetwork());
        executor->hbm_adapter = hbm_adapter;
        uint64_t exposed = 0;
        for (const auto &range : g_address_policy.home_ranges)
            exposed = std::max<uint64_t>(exposed, range.base + range.size_bytes);
        if (exposed == 0)
            for (const auto &stack : g_hbm_stacks)
                exposed += stack.capacity_bytes;
        executor->MaxDramAddr = exposed;
        executor->defaultDataLength = 32;
    } else {
        executor->MaxDramAddr =
            dcache->dramSysWrapper->dramsys->getMemSpec().memorySizeBytes;
        executor->defaultDataLength =
            dcache->dramSysWrapper->dramsys->getMemSpec().defaultBytesPerBurst;
    }

    executor->sram_regions = sram_regions.get();
    executor->core_context->sram_pos_locator_->BindRegionTable(
        sram_regions.get());
    if (sram_config.real_data_path) {
        if (distributed && !hbm_adapter)
            throw std::runtime_error(
                "distributed real SRAM data path requires an HBM adapter");
        if (!distributed && !dcache)
            throw std::runtime_error(
                "legacy_private real SRAM data path requires a DCache");
        sram_storage = std::make_unique<sram::Storage>(
            sram_config.capacity_bytes, true);
        sram_access = std::make_unique<sram::AccessUnit>(
            sc_gen_unique_name("sram-access"), *sram_regions, *sram_storage,
            event_engine, cid);
        compute_timeline =
            std::make_unique<sram::ComputeTimeline>(*sram_access, event_engine, cid);
        if (distributed)
            hbm_byte_transport =
                std::make_unique<sram::CoreMemByteTransport>(*hbm_adapter);
        else
            hbm_byte_transport =
                std::make_unique<sram::LegacyPrivateByteTransport>(*dcache);
        lsu_memory = std::make_unique<sram::CoreLsuUnit>(
            sc_gen_unique_name("core-lsu"), *sram_regions, *sram_access,
            *hbm_byte_transport, sram_config.lsu_queue_depth,
            sram_config.lsu_max_outstanding,
            sram_config.lsu_issue_latency_ns, event_engine, cid);
        dte_memory_bridge = std::make_unique<DteMemoryBridge>(
            sc_gen_unique_name("dte-memory-bridge"), *sram_regions,
            *sram_access, *hbm_byte_transport,
            sram_config.dte_memory_queue_depth,
            sram_config.dte_memory_workers, event_engine, cid);
        executor->dte_async->BindMemoryBridge(dte_memory_bridge.get());
        executor->sram_storage = sram_storage.get();
        executor->sram_access = sram_access.get();
        executor->hbm_byte_transport = hbm_byte_transport.get();
        executor->compute_timeline = compute_timeline.get();
        executor->lsu_memory = lsu_memory.get();
    }
    if (SYSTEM_MODE != SIM_GPU && SYSTEM_MODE != SIM_GPU_PD) {
        GPU_DRAM_ALIGNED = executor->defaultDataLength;
    }
    if (distributed) {
        // The distributed real-memory path is backed by HBMNetwork and its
        // address policy. The legacy per-core KV table uses a different
        // address space (and dataset_words_per_tile is measured in words), so
        // constructing it from the distributed byte range could underflow its
        // reserved-address calculation for small exposed ranges.
        g_dram_kvtable[cid] = nullptr;
    } else {
        if (dataset_words_per_tile >
            executor->MaxDramAddr / sizeof(uint32_t))
            throw std::invalid_argument(
                "legacy private DRAM is smaller than dataset_words_per_tile");
        g_dram_kvtable[cid] = new DramKVTable(
            executor->MaxDramAddr, (uint64_t)50 * 1024 * 1024, 20);
    }
    if (!distributed) {
#if USE_NB_DRAMSYS == 1
        executor->nb_dcache_socket->socket.bind(dcache->socket);
#else
        executor->dcache_socket->isocket.bind(dcache->socket);
#endif
    }
    executor->mem_access_port->mem_read_port(*ram_array);
    executor->mem_access_port->mem_write_port(*ram_array);
    executor->high_bw_mem_access_port->mem_read_port(*ram_array);

    auto *temp_port_target = temp_ram_array ? temp_ram_array : ram_array;
    executor->temp_mem_access_port->mem_read_port(*temp_port_target);
    executor->temp_mem_access_port->mem_write_port(*temp_port_target);
    executor->high_bw_temp_mem_access_port->mem_read_port(*temp_port_target);
}

WorkerCore::~WorkerCore() {
    delete executor;
    delete dcache;
    delete hbm_adapter;
    delete ram_array;
    delete temp_ram_array;
}

// workercore executor
WorkerCoreExecutor::WorkerCoreExecutor(const sc_module_name &n, int s_cid,
                                       Event_engine *event_engine)
    : sc_module(n), cid(s_cid), event_control_queue(MAX_BUFFER_PACKET_SIZE),
      event_mailbox(65536), event_engine(event_engine) {
    const bool distributed = g_memory_system_active &&
        g_memory_topology == MemoryTopology::kDistributedHbm;
    const CoreHWConfig *hw = GetCoreHWConfig(cid);
    DTEConfig dte_config =
        MakeDTEConfig(static_cast<uint32_t>(hw->dte_channel_count),
                      static_cast<uint32_t>(hw->dte_bit_width),
                      HW_DTE_GAMMA_NS, HW_DTE_TAU_LAUNCH_NS);
    dte_config.fine_grained_resources = SPEC_DTE_V4_RESOURCES;
    if (SPEC_DTE_V4_RESOURCES) {
        dte_config.command_slots_per_channel =
            static_cast<uint32_t>(HW_DTE_COMMAND_SLOTS_PER_CHANNEL);
        dte_config.pending_queue_depth =
            static_cast<uint32_t>(HW_DTE_PENDING_QUEUE_DEPTH);
        dte_config.spm_read_width_bits =
            static_cast<uint32_t>(HW_DTE_SPM_READ_WIDTH_BITS);
        dte_config.spm_write_width_bits =
            static_cast<uint32_t>(HW_DTE_SPM_WRITE_WIDTH_BITS);
        dte_config.axi_read_width_bits =
            static_cast<uint32_t>(HW_DTE_AXI_READ_WIDTH_BITS);
        dte_config.axi_write_width_bits =
            static_cast<uint32_t>(HW_DTE_AXI_WRITE_WIDTH_BITS);
        dte_config.launch_energy_pj = HW_DTE_LAUNCH_ENERGY_PJ;
        dte_config.spm_energy_pj_per_bit =
            HW_DTE_SPM_ENERGY_PJ_PER_BIT;
        dte_config.axi_energy_pj_per_bit =
            HW_DTE_AXI_ENERGY_PJ_PER_BIT;
        dte_config.base_area_um2 = HW_DTE_BASE_AREA_UM2;
        dte_config.channel_area_um2 = HW_DTE_CHANNEL_AREA_UM2;
        dte_config.command_slot_area_um2 =
            HW_DTE_COMMAND_SLOT_AREA_UM2;
        dte_config.port_bit_area_um2 = HW_DTE_PORT_BIT_AREA_UM2;
    }
    DTEUnit::ValidateConfig(dte_config);
    const std::string dte_name = "dte_core_" + std::to_string(cid);
    dte = std::make_unique<DTEUnit>(dte_name.c_str(), dte_config, cid,
                                    event_engine);
    DteAggregationConfig aggregation;
    aggregation.enabled = SPEC_DTE_AGGREGATION;
    aggregation.max_descriptors =
        static_cast<uint32_t>(DTE_AGGREGATION_MAX_DESCRIPTORS);
    aggregation.max_payload_bytes =
        static_cast<uint64_t>(DTE_AGGREGATION_MAX_BYTES);
    aggregation.timeout_cycles =
        (static_cast<uint64_t>(DTE_AGGREGATION_TIMEOUT_NS) +
         static_cast<uint64_t>(CYCLE) - 1) /
        static_cast<uint64_t>(CYCLE);
    aggregation.address_block_bytes =
        static_cast<uint64_t>(DTE_AGGREGATION_ADDRESS_BLOCK_BYTES);
    const std::string async_name =
        "dte_async_core_" + std::to_string(cid);
    dte_async = std::make_unique<DteAsyncTracker>(
        async_name.c_str(), *dte, aggregation, cid, event_engine);
    if (cid < 0 || static_cast<uint64_t>(cid) > UINT16_MAX)
        throw std::out_of_range("P2P endpoint core id exceeds transport width");
    p2p_endpoint = std::make_unique<P2pEndpointSessionRuntime>(
        static_cast<uint16_t>(cid), MAX_BUFFER_PACKET_SIZE,
        static_cast<size_t>(kDteEndpointP2pMaxBytes), UINT16_MAX,
        static_cast<uint32_t>(TOTAL_CORES));
    multicast_byte_reassembler =
        std::make_unique<IsaV1CollectiveByteReassembler>(
            static_cast<size_t>(kDteEndpointP2pMaxBytes) *
                MAX_BUFFER_PACKET_SIZE,
            MAX_BUFFER_PACKET_SIZE);
    if (TOTAL_CORES <= 0)
        throw std::invalid_argument(
            "P2P pending REQUEST capacity requires positive TOTAL_CORES");
    const size_t topology_cores = static_cast<size_t>(TOTAL_CORES);
    // Global upper bound when every TX session targets this core. If full, no
    // legal local TX remains that could need an inbound ACK, so upstream busy
    // cannot create ACK head-of-line blocking.
    p2p_pending_request_capacity = CheckedP2pPendingRequestCapacity(
        topology_cores, p2p_endpoint->MaxSessions());

    prim_refill = false;
    SC_THREAD(catch_channel_avail_i);
    sensitive << channel_avail_i.pos();
    dont_initialize();

    SC_THREAD(catch_data_sent_i);
    sensitive << data_sent_i.pos();
    dont_initialize();

    //  控制信道相关线程
    SC_THREAD(catch_ctrl_channel_avail_i);
    sensitive << ctrl_channel_avail_i.pos();
    dont_initialize();

    SC_THREAD(catch_ctrl_sent_i);
    sensitive << ctrl_sent_i.pos();
    dont_initialize();

    // This thread has no static sensitivity and must run once at time zero
    // so it can arm its explicit ev_ctrl_sent_i wait.
    SC_THREAD(poll_ctrl_buffer_i);

    SC_THREAD(switch_prim_block);
    sensitive << ev_block;

    SC_THREAD(worker_core_execute);

    SC_THREAD(p2p_tx_worker);
    SC_THREAD(p2p_request_admission_worker);
    SC_THREAD(collective_program_worker);
    SC_THREAD(collective_acceleration_worker);

    SC_THREAD(send_logic);
    sensitive << ev_send;
    dont_initialize();

    SC_THREAD(send_para_logic);
    sensitive << ev_para_send;
    dont_initialize();

    SC_THREAD(recv_logic);
    sensitive << ev_recv;
    dont_initialize();

    SC_THREAD(send_helper);
    sensitive << ev_send_helper;
    dont_initialize();

    SC_THREAD(task_logic);
    sensitive << ev_comp;
    dont_initialize();

    SC_THREAD(req_logic);
    // req_logic 只处理 REQUEST 消息，只需要监听：
    // 1. ev_recv_msg_type_[REQUEST] - 收到 REQUEST 时触发（数据信道或控制信道都会触发）
    // 2. ev_prim_recv_notice - 执行 recv_data 原语时触发，需要检查是否有匹配的 REQUEST
    sensitive << ev_recv_msg_type_[MSG_TYPE::REQUEST] << ev_prim_recv_notice;
    dont_initialize();

    SC_THREAD(poll_buffer_i);
    sram_addr = new int(0);

    // 初始化PrimCoreContext
    core_context = make_shared<PrimCoreContext>(cid);

    send_done = true;
    send_last_packet = false;

    start_global_mem_event = new sc_event();
    end_global_mem_event = new sc_event();
    start_nb_dram_event = new sc_event();
    start_nb_gpu_dram_event = new sc_event();
    start_sram_event = new sc_event();
    end_sram_event = new sc_event();

    end_nb_dram_event = new sc_event();
    end_nb_gpu_dram_event = new sc_event();

    sram_writer = new SRAMWriteModule("sram_writer", end_sram_event);
    if (!distributed) {
#if USE_NB_DRAMSYS == 1
        nb_dcache_socket =
            new NB_DcacheIF(cid, sc_gen_unique_name("nb_dcache"),
                            start_nb_dram_event, end_nb_dram_event, event_engine);
#else
        dcache_socket = new DcacheCore(sc_gen_unique_name("dcache"), event_engine);
#endif
    }
#if USE_L1L2_CACHE == 1
    if (!distributed) {
        core_lv1_cache = new L1Cache(("l1_cache_" + to_string(cid)).c_str(), cid,
                                     L1CACHESIZE, L1CACHELINESIZE, 4, 8);
        gpunb_dcache_if = new GPUNB_dcacheIF(sc_gen_unique_name("nb_dcache_if"),
                                             cid, start_nb_gpu_dram_event,
                                             end_nb_gpu_dram_event, event_engine);
    }
#else
#endif
    mem_access_port = new mem_access_unit(sc_gen_unique_name("mem_access_unit"),
                                          event_engine);
    high_bw_mem_access_port = new high_bw_mem_access_unit(
        sc_gen_unique_name("high_bw_mem_access_unit"), event_engine);
    temp_mem_access_port = new mem_access_unit(
        sc_gen_unique_name("temp_mem_access_unit"), event_engine);
    high_bw_temp_mem_access_port = new high_bw_mem_access_unit(
        sc_gen_unique_name("high_bw_temp_mem_access_unit"), event_engine);
}

void WorkerCoreExecutor::init_global_mem() {
    nb_global_mem_socket = new NB_GlobalMemIF(
        sc_gen_unique_name("nb_global_mem"), start_global_mem_event,
        end_global_mem_event, event_engine);
}

void WorkerCoreExecutor::end_of_elaboration() {
    // 在构造函数之后设置信号的初始值
    data_sent_o.write(false);
    core_busy_o.write(false);
    ctrl_sent_o.write(false);
    ctrl_core_busy_o.write(false);
}

bool WorkerCoreExecutor::send_p2p_message(
    Msg message, bool request_transition,
    const P2pEndpointHandle *handle) {
    if (!IsP2pEndpointMessage(message))
        throw std::invalid_argument(
            "P2P endpoint dispatch received a non-marker message");
    const bool control = message.IsControlMsg();
    while (true) {
        if (request_transition &&
            (handle == nullptr || !p2p_endpoint->HasFsm(handle->fsm_id)))
            return false;
        const bool available = control ? ctrl_channel_avail_i.read()
                                       : channel_avail_i.read();
        if (available && atomic_helper_lock(sc_time_stamp(), 3)) break;
        wait(CYCLE, SC_NS);
    }

    try {
        if (request_transition) {
            if (handle == nullptr || !p2p_endpoint->HasFsm(handle->fsm_id)) {
                send_helper_write = 0;
                ev_send_helper.notify(SC_ZERO_TIME);
                return false;
            }
            PinControlMsgExit(message);
            p2p_endpoint->MarkRequestSent(*handle);
        } else if (control) {
            PinControlMsgExit(message);
        } else {
            message.exit_port_ = SelectCoreMsgExit(
                message.source_, message.des_, message.tag_id_,
                message.subflow_);
        }
        send_serialized_wire(SerializeMsg(message), control);
        return true;
    } catch (...) {
        send_helper_write = 0;
        ev_send_helper.notify(SC_ZERO_TIME);
        throw;
    }
}

void WorkerCoreExecutor::send_serialized_wire(
    const sc_bv<256> &wire, bool control) {
    if (send_helper_write != 3)
        throw std::logic_error(
            "serialized wire enqueue does not own the send helper");
    while (serialized_wire_queue.Full())
        wait(ev_serialized_wire_progress);
    const uint64_t ticket = serialized_wire_queue.Enqueue(wire, control);
    ev_send_helper.notify(SC_ZERO_TIME);
    while (!serialized_wire_queue.Completed(ticket))
        wait(ev_serialized_wire_progress);
}

void WorkerCoreExecutor::p2p_tx_worker() {
    while (true) {
        while (p2p_tx_queue.empty()) wait(ev_p2p_tx);
        P2pTxJob job = std::move(p2p_tx_queue.front());
        p2p_tx_queue.pop_front();
        const auto handle = job.issue.handle;
        if (!p2p_endpoint->HasFsm(handle.fsm_id)) {
            ev_p2p_progress.notify(SC_ZERO_TIME);
            continue;
        }

        const P2pPayloadDeclaration declaration =
            ParseP2pPayloadRequest(job.issue.messages.request);
        const P2pTimingKey timing_key{declaration.flow, handle.round};
        bool timing_published = false;
        try {
            P2pSharedTimingSidebandRuntime::Publish(
                timing_key, handle.fsm_id, job.source_first_ns,
                job.source_done_ns);
            timing_published = true;

            const uint64_t bytes = declaration.total_bytes;
            const uint64_t fragments =
                job.issue.messages.fragments.size();
            const uint32_t checksum = declaration.checksum;
            TraceP2pEndpoint(event_engine, cid, "P2P_wire", "B",
                             handle.fsm_id, bytes, fragments, checksum);
            if (!send_p2p_message(
                    job.issue.messages.request, true, &handle)) {
                P2pSharedTimingSidebandRuntime::Abort(
                    declaration.flow, handle.fsm_id);
                timing_published = false;
                ev_p2p_progress.notify(SC_ZERO_TIME);
                continue;
            }
            CheckedAccumulate(p2p_stats.admission_requests_sent, 1,
                              "admission REQUEST sent");
            TraceP2pEndpoint(event_engine, cid,
                             "P2P_admission_request", "i",
                             handle.fsm_id, bytes, fragments, checksum);
            while (p2p_endpoint->HasFsm(handle.fsm_id) &&
                   p2p_endpoint->Phase(handle) ==
                       P2pEndpointPhase::REQUEST_SENT)
                wait(ev_p2p_progress);
            if (!p2p_endpoint->HasFsm(handle.fsm_id)) {
                if (P2pSharedTimingSidebandRuntime::Contains(
                        declaration.flow))
                    P2pSharedTimingSidebandRuntime::Abort(
                        declaration.flow, handle.fsm_id);
                timing_published = false;
                ev_p2p_progress.notify(SC_ZERO_TIME);
                continue;
            }
            if (!p2p_endpoint->IsAdmitted(handle))
                throw std::logic_error(
                    "P2P TX left REQUEST_SENT without admission");
            for (Msg &fragment : job.issue.messages.fragments) {
                if (!send_p2p_message(std::move(fragment), false))
                    throw std::logic_error(
                        "P2P DATA dispatch unexpectedly reported cancellation");
            }
            p2p_endpoint->CompleteSend(handle);
            CheckedAccumulate(p2p_stats.wire_bytes, bytes, "wire byte");
            CheckedAccumulate(p2p_stats.wire_fragments, fragments,
                              "wire fragment");
            CheckedAccumulate(p2p_stats.tx_local_completions, 1,
                              "TX local completion");
            p2p_stats.last_wire_checksum = checksum;
            TraceP2pEndpoint(event_engine, cid, "P2P_wire", "E",
                             handle.fsm_id, bytes, fragments, checksum);
            TraceP2pEndpoint(event_engine, cid,
                             "P2P_tx_local_complete", "i",
                             handle.fsm_id, bytes, fragments, checksum);
            ev_p2p_progress.notify(SC_ZERO_TIME);
        } catch (...) {
            (void)p2p_endpoint->Abort(handle);
            if (handle.completion == DteEndpointCompletion::ASYNC) {
                auto tracked = p2p_async_handles.find(handle.token);
                if (tracked != p2p_async_handles.end() &&
                    tracked->second == handle)
                    p2p_async_handles.erase(tracked);
            }
            if (timing_published &&
                P2pSharedTimingSidebandRuntime::Contains(declaration.flow)) {
                try {
                    P2pSharedTimingSidebandRuntime::Abort(
                        declaration.flow, handle.fsm_id);
                } catch (...) {
                }
            }
            ev_p2p_progress.notify(SC_ZERO_TIME);
            throw;
        }
    }
}

void WorkerCoreExecutor::p2p_request_admission_worker() {
    while (true) {
        while (p2p_pending_requests.empty()) wait(ev_p2p_request);
        const P2pPayloadDeclaration declaration =
            p2p_pending_requests.front().declaration;
        const P2pRequestDisposition disposition =
            p2p_endpoint->ClassifyRequest(declaration);
        if (disposition == P2pRequestDisposition::CONFLICT) {
            p2p_pending_requests.pop_front();
            CheckedAccumulate(p2p_stats.request_conflicts_rejected, 1,
                              "REQUEST conflict rejected");
            TraceP2pEndpoint(event_engine, cid,
                             "P2P_request_conflict", "i",
                             declaration.fsm_id, declaration.total_bytes,
                             declaration.fragment_count, declaration.checksum);
            ev_p2p_progress.notify(SC_ZERO_TIME);
            continue;
        }
        auto existing = p2p_rx_fsm_by_flow.find(declaration.flow);
        if (disposition == P2pRequestDisposition::NEW &&
            existing == p2p_rx_fsm_by_flow.end() &&
            !p2p_endpoint->CanReceiveRequest(
                declaration.flow, declaration.total_bytes)) {
            wait(ev_p2p_progress);
            continue;
        }

        P2pPendingRequest pending =
            std::move(p2p_pending_requests.front());
        p2p_pending_requests.pop_front();
        ev_p2p_progress.notify(SC_ZERO_TIME);

        if (disposition != P2pRequestDisposition::NEW) {
            const bool active =
                disposition == P2pRequestDisposition::ACTIVE_DUPLICATE;
            if (active != (existing != p2p_rx_fsm_by_flow.end()) ||
                (active && existing->second != declaration.fsm_id))
                throw std::logic_error(
                    "P2P duplicate REQUEST runtime/Worker index mismatch");
            CheckedAccumulate(p2p_stats.duplicate_requests_suppressed, 1,
                              "duplicate REQUEST suppressed");
            TraceP2pEndpoint(event_engine, cid,
                             "P2P_request_duplicate", "i",
                             declaration.fsm_id, declaration.total_bytes,
                             declaration.fragment_count, declaration.checksum);
            continue;
        }

        const auto existing_fsm = std::find_if(
            p2p_rx_fsm_by_flow.begin(), p2p_rx_fsm_by_flow.end(),
            [&](const auto &entry) {
                return entry.second == declaration.fsm_id;
            });
        if (existing_fsm != p2p_rx_fsm_by_flow.end() &&
            !(existing_fsm->first == declaration.flow))
            throw std::invalid_argument(
                "conflicting duplicate P2P REQUEST fsm index");

        if (existing != p2p_rx_fsm_by_flow.end()) {
            // Identical retransmission is idempotent. Conflicting duplicates
            // fail before mutating the active runtime/index/timing tuple.
            if (existing->second != declaration.fsm_id)
                throw std::invalid_argument(
                    "conflicting duplicate P2P REQUEST timing flow");
            try {
                if (p2p_endpoint->ReceiveRequest(pending.message).has_value())
                    throw std::logic_error(
                        "duplicate P2P REQUEST unexpectedly produced DATA");
            } catch (...) {
                RethrowP2pEndpointProtocolFailure(
                    "conflicting duplicate REQUEST",
                    std::current_exception());
            }
            // The transport does not retry. An identical duplicate is an
            // exactly-once no-op and cannot emit or count a second CTS.
            continue;
        }

        bool runtime_admitted = false;
        try {
            std::optional<P2pRxDelivery> delivery =
                p2p_endpoint->ReceiveRequest(pending.message);
            runtime_admitted = true;
            const auto indexed = p2p_rx_fsm_by_flow.emplace(
                declaration.flow, declaration.fsm_id);
            if (!indexed.second)
                throw std::logic_error(
                    "P2P REQUEST timing flow insertion raced");
            TraceP2pEndpoint(event_engine, cid,
                             "P2P_admission_accept", "i",
                             declaration.fsm_id,
                             declaration.total_bytes,
                             declaration.fragment_count,
                             declaration.checksum);
            Msg ack = MakeP2pAdmissionAck(
                declaration.flow, declaration.fsm_id);
            if (!send_p2p_message(std::move(ack), false))
                throw std::logic_error(
                    "admission ACK unexpectedly reported cancellation");
            CheckedAccumulate(p2p_stats.admission_acks_sent, 1,
                              "admission ACK sent");
            TraceP2pEndpoint(event_engine, cid,
                             "P2P_admission_ack", "i",
                             declaration.fsm_id,
                             declaration.total_bytes,
                             declaration.fragment_count,
                             declaration.checksum);
            if (delivery.has_value())
                commit_p2p_delivery(std::move(*delivery));
            ev_p2p_progress.notify(SC_ZERO_TIME);
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            if (runtime_admitted)
                (void)p2p_endpoint->AbortInbound(declaration.flow);
            (void)p2p_endpoint->AbortReceiveFsm(declaration.fsm_id);
            p2p_rx_fsm_by_flow.erase(declaration.flow);
            p2p_rx_addresses.erase(declaration.fsm_id);
            for (auto handle = p2p_async_handles.begin();
                 handle != p2p_async_handles.end();) {
                if (handle->second.fsm_id == declaration.fsm_id)
                    handle = p2p_async_handles.erase(handle);
                else
                    ++handle;
            }
            (void)P2pSharedTimingSidebandRuntime::AbortFlow(
                declaration.flow);
            CheckedAccumulate(p2p_stats.request_aborts, 1,
                              "REQUEST admission abort");
            TraceP2pEndpoint(event_engine, cid, "P2P_request_abort", "i",
                             declaration.fsm_id, declaration.total_bytes,
                             declaration.fragment_count, declaration.checksum);
            ev_p2p_progress.notify(SC_ZERO_TIME);
            RethrowP2pEndpointProtocolFailure(
                "REQUEST admission/ACK send", failure);
        }
    }
}

void WorkerCoreExecutor::commit_p2p_delivery(P2pRxDelivery delivery) {
    const uint32_t fsm_id = delivery.handle.fsm_id;
    const P2pFlowKey flow = delivery.flow;
    try {
        auto destination = p2p_rx_addresses.find(fsm_id);
        if (destination == p2p_rx_addresses.end())
            throw std::logic_error(
                "P2P receive commit has no resolved destination");
        if (sram_access == nullptr)
            throw std::runtime_error(
                "P2P receive requires the real SRAM data path");
        auto timing_flow = p2p_rx_fsm_by_flow.find(flow);
        if (timing_flow == p2p_rx_fsm_by_flow.end() ||
            timing_flow->second != fsm_id)
            throw std::logic_error(
                "P2P receive commit lost its timing flow index");
        const uint32_t checksum = P2pPayloadChecksum(delivery.bytes);
        TraceP2pEndpoint(event_engine, cid, "P2P_noc_rx_write", "B",
                         fsm_id, delivery.bytes.size(), 0, checksum);
        sram::Request write;
        write.initiator = sram::Initiator::kNocRx;
        write.command = sram::Command::kWrite;
        write.address = destination->second;
        write.size_bytes = delivery.bytes.size();
        write.payload = delivery.bytes;
        (void)sram_access->Access(write);
        const P2pTimingMetadata timing =
            P2pSharedTimingSidebandRuntime::Consume(flow, fsm_id);
        Msg ack = p2p_endpoint->CompleteReceive(delivery.handle);
        if (!send_p2p_message(std::move(ack), false))
            throw std::logic_error(
                "P2P completion ACK unexpectedly reported cancellation");

        p2p_rx_addresses.erase(destination);
        p2p_rx_fsm_by_flow.erase(timing_flow);
        TraceP2pStreamTiming(event_engine, cid, fsm_id, timing);
        CheckedAccumulate(p2p_stats.noc_rx_write_bytes,
                          delivery.bytes.size(), "NoC RX write byte");
        CheckedAccumulate(p2p_stats.rx_local_completions, 1,
                          "RX local completion");
        CheckedAccumulate(p2p_stats.completion_acks_sent, 1,
                          "completion ACK sent");
        p2p_stats.last_destination_checksum = checksum;
        TraceP2pEndpoint(event_engine, cid, "P2P_noc_rx_write", "E",
                         fsm_id, delivery.bytes.size(), 0, checksum);
        TraceP2pEndpoint(event_engine, cid, "P2P_rx_local_complete", "i",
                         fsm_id, delivery.bytes.size(), 0, checksum);
        TraceP2pEndpoint(event_engine, cid, "P2P_completion_ack", "i",
                         fsm_id, delivery.bytes.size(), 0, checksum);
        ev_p2p_progress.notify(SC_ZERO_TIME);
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        // Runtime owns the session/reassembly/token state and must be torn
        // down before Worker indexes or timing metadata are released.
        (void)p2p_endpoint->Abort(delivery.handle);
        p2p_rx_addresses.erase(fsm_id);
        p2p_rx_fsm_by_flow.erase(flow);
        for (auto tracked = p2p_async_handles.begin();
             tracked != p2p_async_handles.end();) {
            if (tracked->second == delivery.handle)
                tracked = p2p_async_handles.erase(tracked);
            else
                ++tracked;
        }
        if (P2pSharedTimingSidebandRuntime::Contains(flow)) {
            try {
                P2pSharedTimingSidebandRuntime::Abort(flow, fsm_id);
            } catch (...) {
            }
        }
        ev_p2p_progress.notify(SC_ZERO_TIME);
        RethrowP2pEndpointProtocolFailure(
            "destination commit/completion ACK send", failure);
    }
}

void WorkerCoreExecutor::execute_dte_send_endpoint(
    Dte_send_endpoint_prim *prim) {
    if (prim == nullptr)
        throw std::invalid_argument("P2P send received a null primitive");
    prim->Validate();
    if (prim->mode != DteEndpointSendMode::P2P)
        throw std::invalid_argument(
            "P5 production dispatch only accepts P2P DTE_SEND");
    if (p2p_endpoint == nullptr)
        throw std::runtime_error("P2P endpoint runtime is unavailable");
    if (p2p_endpoint->HasFsm(prim->fsm_id))
        throw std::invalid_argument(
            "P2P endpoint fsm_id is already active");
    if (p2p_endpoint->Residual().sessions >= p2p_endpoint->MaxSessions())
        throw std::length_error(
            "P2P endpoint session capacity exhausted before source read");
    if (prim->completion == DteEndpointCompletion::ASYNC &&
        (dte_async->HasToken(prim->token) ||
         p2p_endpoint->HasToken(prim->token) ||
         p2p_async_handles.count(prim->token) != 0))
        throw std::invalid_argument(
            "P2P endpoint token collides with an active token");

    std::vector<uint8_t> bytes;
    const bool hbm_source =
        prim->source_space == DteEndpointSourceSpace::HBM;
    const char *source_stage = hbm_source ? "P2P_source_read_HBM"
                                          : "P2P_source_read_SRAM";
    const uint64_t source_first_ns = CurrentP2pNanoseconds();
    TraceP2pEndpoint(event_engine, cid, source_stage, "B",
                     prim->fsm_id, prim->length_bytes, 0, 0);
    if (hbm_source) {
        if (hbm_byte_transport == nullptr)
            throw std::runtime_error(
                "P2P HBM source requires the real HBM byte transport");
        bytes = hbm_byte_transport->Read(
            prim->source.absolute_address_bytes, prim->length_bytes);
    } else {
        if (sram_access == nullptr)
            throw std::runtime_error(
                "P2P SRAM source requires the real SRAM data path");
        const uint64_t address = ResolveEndpointSramAddress(
            *this, prim->source, prim->length_bytes,
            sram::Initiator::kDte, sram::Command::kRead);
        sram::Request read;
        read.initiator = sram::Initiator::kDte;
        read.command = sram::Command::kRead;
        read.address = address;
        read.size_bytes = prim->length_bytes;
        bytes = sram_access->Access(read).payload;
    }
    const uint64_t source_done_ns = CurrentP2pNanoseconds();
    if (bytes.size() != prim->length_bytes)
        throw std::runtime_error(
            "P2P source read returned an unexpected byte count");
    const uint32_t checksum = P2pPayloadChecksum(bytes);
    CheckedAccumulate(p2p_stats.source_read_bytes, bytes.size(),
                      "source read byte");
    if (hbm_source)
        CheckedAccumulate(p2p_stats.hbm_source_read_bytes, bytes.size(),
                          "HBM source read byte");
    else
        CheckedAccumulate(p2p_stats.sram_source_read_bytes, bytes.size(),
                          "SRAM source read byte");
    p2p_stats.last_source_checksum = checksum;
    TraceP2pEndpoint(event_engine, cid, source_stage, "E",
                     prim->fsm_id, bytes.size(), 0, checksum);

    P2pTxIssue issue = p2p_endpoint->IssueSend(*prim, bytes);
    const P2pEndpointHandle handle = issue.handle;
    try {
        if (prim->completion == DteEndpointCompletion::ASYNC) {
            const auto inserted =
                p2p_async_handles.emplace(prim->token, handle);
            if (!inserted.second)
                throw std::logic_error(
                    "P2P async handle insertion unexpectedly failed");
        }
        p2p_tx_queue.push_back(P2pTxJob{
            std::move(issue), source_first_ns, source_done_ns});
    } catch (...) {
        (void)p2p_endpoint->Abort(handle);
        auto tracked = p2p_async_handles.find(handle.token);
        if (tracked != p2p_async_handles.end() &&
            tracked->second == handle)
            p2p_async_handles.erase(tracked);
        throw;
    }
    ev_p2p_tx.notify(SC_ZERO_TIME);

    if (prim->completion == DteEndpointCompletion::SYNC) {
        while (!p2p_endpoint->TryRetireSync(handle))
            wait(ev_p2p_progress);
    }
}

void WorkerCoreExecutor::execute_dte_recv_endpoint(
    Dte_recv_endpoint_prim *prim) {
    if (prim == nullptr)
        throw std::invalid_argument("P2P receive received a null primitive");
    prim->Validate();
    if (prim->mode != DteEndpointRecvMode::P2P)
        throw std::invalid_argument(
            "P5 production dispatch only accepts P2P DTE_RECV");
    if (p2p_endpoint == nullptr || sram_access == nullptr)
        throw std::runtime_error(
            "P2P receive requires endpoint runtime and real SRAM data path");
    if (p2p_endpoint->HasFsm(prim->fsm_id) ||
        p2p_rx_addresses.count(prim->fsm_id) != 0)
        throw std::invalid_argument(
            "P2P receive fsm_id is already active");
    if (p2p_endpoint->Residual().sessions >= p2p_endpoint->MaxSessions())
        throw std::length_error(
            "P2P endpoint receive session capacity exhausted");
    if (prim->completion == DteEndpointCompletion::ASYNC &&
        (dte_async->HasToken(prim->token) ||
         p2p_endpoint->HasToken(prim->token) ||
         p2p_async_handles.count(prim->token) != 0))
        throw std::invalid_argument(
            "P2P endpoint token collides with an active token");

    const uint64_t address = ResolveEndpointSramAddress(
        *this, prim->destination, prim->length_bytes,
        sram::Initiator::kNocRx, sram::Command::kWrite);
    P2pRxPostResult posted;
    bool runtime_posted = false;
    try {
        posted = p2p_endpoint->PostReceive(*prim);
        runtime_posted = true;
        const auto destination =
            p2p_rx_addresses.emplace(prim->fsm_id, address);
        if (!destination.second)
            throw std::logic_error(
                "P2P receive destination insertion unexpectedly failed");
        if (prim->completion == DteEndpointCompletion::ASYNC) {
            const auto inserted =
                p2p_async_handles.emplace(prim->token, posted.handle);
            if (!inserted.second)
                throw std::logic_error(
                    "P2P async receive handle insertion unexpectedly failed");
        }
        if (posted.ready.has_value())
            commit_p2p_delivery(std::move(*posted.ready));
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        const std::optional<P2pFlowKey> aborted_flow =
            p2p_endpoint->AbortInboundFsm(prim->fsm_id);
        if (!aborted_flow.has_value() && runtime_posted)
            (void)p2p_endpoint->Abort(posted.handle);
        if (aborted_flow.has_value()) {
            p2p_rx_fsm_by_flow.erase(*aborted_flow);
            (void)P2pSharedTimingSidebandRuntime::AbortFlow(*aborted_flow);
            CheckedAccumulate(p2p_stats.request_aborts, 1,
                              "PostReceive REQUEST abort");
            TraceP2pEndpoint(event_engine, cid, "P2P_request_abort", "i",
                             prim->fsm_id, prim->length_bytes, 0, 0);
        }
        p2p_rx_addresses.erase(prim->fsm_id);
        for (auto tracked = p2p_async_handles.begin();
             tracked != p2p_async_handles.end();) {
            if (tracked->second.fsm_id == prim->fsm_id)
                tracked = p2p_async_handles.erase(tracked);
            else
                ++tracked;
        }
        ev_p2p_progress.notify(SC_ZERO_TIME);
        RethrowP2pEndpointProtocolFailure(
            "PostReceive setup/match", failure);
    }


    if (prim->completion == DteEndpointCompletion::SYNC) {
        while (!p2p_endpoint->TryRetireSync(posted.handle))
            wait(ev_p2p_progress);
    }
}

void WorkerCoreExecutor::execute_dte_async(Dte_async_prim *prim) {
    if (prim == nullptr)
        throw std::invalid_argument("DTE V3a received a null primitive");

    if (collective_executor_v1) {
        if (prim->op == DteAsyncOp::WAIT &&
            collective_executor_v1->HasPublicToken(prim->token)) {
            while (!collective_executor_v1->TryWait(prim->token))
                wait(CYCLE, SC_NS);
            maybe_retire_collective_wave_image();
            ev_collective_program.notify(SC_ZERO_TIME);
            return;
        }
        if (prim->op == DteAsyncOp::CANCEL &&
            collective_executor_v1->HasPublicToken(prim->token)) {
            collective_executor_v1->Cancel(prim->token);
            collective_p2p_handles.clear();
            ev_collective_program.notify(SC_ZERO_TIME);
            return;
        }
        if (prim->op == DteAsyncOp::FENCE &&
            collective_executor_v1->Configured() &&
            !collective_executor_v1->Drained()) {
            while (!collective_executor_v1->TryFence())
                wait(CYCLE, SC_NS);
            maybe_retire_collective_wave_image();
            ev_collective_program.notify(SC_ZERO_TIME);
        }
    }
    if (!SPEC_DTE_ASYNC)
        throw std::runtime_error(
            "Dte_async primitive requires dte.async=true");

    switch (prim->op) {
    case DteAsyncOp::ISSUE: {
        if (p2p_endpoint->HasToken(prim->token))
            throw std::invalid_argument(
                "local DTE token collides with P2P endpoint token");
        uint64_t spm_addr = prim->spm_addr;
        if (!prim->sram_region.empty()) {
            if (sram_regions == nullptr)
                throw std::runtime_error(
                    "Dte_async named region requires SRAM region table");
            const auto command = prim->direction == DteDir::DRAM_TO_SPM
                                     ? sram::Command::kWrite
                                     : sram::Command::kRead;
            spm_addr = sram_regions->Resolve(
                prim->sram_region, prim->sram_offset, prim->spm_size,
                sram::Initiator::kDte, command).address;
        }
        uint64_t remote_addr = prim->remote_addr;
        if (!prim->destination_sram_region.empty()) {
            if (sram_regions == nullptr)
                throw std::runtime_error(
                    "Dte_async named destination requires SRAM region table");
            remote_addr = sram_regions->Resolve(
                prim->destination_sram_region,
                prim->destination_sram_offset, prim->spm_size,
                sram::Initiator::kDte, sram::Command::kWrite).address;
        }
        dte_async->IssueToken(
            prim->token, prim->payload_bits, prim->direction,
            spm_addr, prim->spm_size, prim->remote_peer,
            remote_addr, prim->address_block);
        break;
    }
    case DteAsyncOp::WAIT:
        if (p2p_endpoint->HasToken(prim->token)) {
            auto tracked = p2p_async_handles.find(prim->token);
            if (tracked == p2p_async_handles.end())
                throw std::logic_error(
                    "P2P token is absent from Worker handle index");
            while (!p2p_endpoint->TryWait(prim->token))
                wait(ev_p2p_progress);
            p2p_async_handles.erase(tracked);
        } else {
            dte_async->WaitToken(prim->token);
        }
        break;
    case DteAsyncOp::POLL:
        if (p2p_endpoint->HasToken(prim->token))
            prim->poll_complete =
                p2p_endpoint->Poll(prim->token) ==
                P2pEndpointPhase::COMPLETE;
        else
            prim->poll_complete = dte_async->PollToken(prim->token);
        break;
    case DteAsyncOp::FENCE: {
        dte_async->Fence();
        std::vector<uint32_t> tokens;
        tokens.reserve(p2p_async_handles.size());
        for (const auto &entry : p2p_async_handles)
            tokens.push_back(entry.first);
        for (uint32_t token : tokens) {
            while (p2p_endpoint->HasToken(token) &&
                   !p2p_endpoint->TryWait(token))
                wait(ev_p2p_progress);
            p2p_async_handles.erase(token);
        }
        break;
    }
    case DteAsyncOp::CANCEL:
        if (p2p_endpoint->HasToken(prim->token)) {
            auto tracked = p2p_async_handles.find(prim->token);
            if (tracked == p2p_async_handles.end())
                throw std::logic_error(
                    "P2P token is absent from Worker handle index");
            const P2pEndpointHandle handle = tracked->second;
            if (handle.direction == P2pEndpointDirection::RX)
                throw std::invalid_argument(
                    "P2P CANCEL does not support RX descriptors");
            p2p_endpoint->Cancel(prim->token);
            p2p_async_handles.erase(tracked);
            ev_p2p_progress.notify(SC_ZERO_TIME);
        } else {
            dte_async->CancelToken(prim->token);
        }
        break;
    }
}

void WorkerCoreExecutor::ConfigureCollectiveProgram(
    std::shared_ptr<const IsaV1CollectiveProgramImage> image,
    std::shared_ptr<CollectiveWaveAdmissionCoordinatorV1> coordinator,
    std::shared_ptr<const IsaV1CollectiveProfileProgramImage> profile_image,
    std::shared_ptr<IsaV1CollectiveTreeRegistryBridge> tree_bridge,
    std::shared_ptr<IsaV1CollectiveAccelerationRuntime>
        acceleration_runtime) {
    if (image == nullptr || coordinator == nullptr)
        throw std::invalid_argument(
            "collective program requires an image and coordinator");
    if ((profile_image == nullptr) != (tree_bridge == nullptr) ||
        (profile_image == nullptr) != (acceleration_runtime == nullptr))
        throw std::invalid_argument(
            "collective profile image, tree bridge and acceleration runtime "
            "must be injected together");
    if (profile_image != nullptr &&
        (profile_image->BaseGeneration() != image->Generation() ||
         profile_image->BaseCookie() != image->Cookie()))
        throw std::invalid_argument(
            "collective profile image does not match the P6 base image");
    if (cid < 0 || static_cast<uint64_t>(cid) > UINT16_MAX)
        throw std::out_of_range(
            "collective program core exceeds the image core width");
    auto candidate = std::make_unique<CollectiveExecutorV1>();
    candidate->Configure(image, static_cast<uint16_t>(cid),
                         coordinator.get());
    uint64_t trace_base = 0;
    bool found_core = false;
    for (const IsaV1CollectiveCoreProgramImage &core : image->Cores()) {
        if (core.core_id == static_cast<uint16_t>(cid)) {
            found_core = true;
            break;
        }
        if (trace_base > UINT64_MAX - core.actions.size())
            throw std::overflow_error(
                "collective action trace index overflows u64");
        trace_base += core.actions.size();
    }
    if (!found_core)
        throw std::logic_error(
            "collective executor core is absent from the program image");
    collective_program_image_v1 = image;
    collective_profile_program_image_v1 = std::move(profile_image);
    collective_tree_registry_bridge_v1 = std::move(tree_bridge);
    collective_acceleration_runtime_v1 = std::move(acceleration_runtime);
    p7_active_plans.clear();
    p7_observed_plans.clear();
    p7_ready_chunks.clear();
    p7_tree_states.clear();
    collective_action_trace_base = trace_base;
    collective_executor_v1 = std::move(candidate);
    collective_wave_coordinator_v1 = std::move(coordinator);
    ev_collective_program.notify(SC_ZERO_TIME);
    ev_p7_acceleration.notify(SC_ZERO_TIME);
}

size_t WorkerCoreExecutor::CollectiveProgramResidual() const noexcept {
    size_t result = collective_p2p_handles.size();
    if (collective_executor_v1) {
        const auto residual = collective_executor_v1->Residual();
        result += residual.images + residual.plans +
                  residual.runnable_plans + residual.remaining_actions +
                  residual.inflight_actions +
                  residual.aggregate.aggregates +
                  residual.aggregate.child_tokens +
                  residual.aggregate.local_work_items +
                  residual.final_gate.plans +
                  residual.final_gate.public_tokens;
    }
    if (collective_wave_coordinator_v1 &&
        collective_wave_coordinator_v1->HasImage()) {
        const auto residual =
            collective_wave_coordinator_v1->Residual();
        result += residual.images + residual.forming_waves +
                  residual.pending_waves + residual.active_waves +
                  residual.active_endpoint_sessions +
                  residual.active_receive_bytes;
        if (residual.complete_waves != residual.waves)
            result += residual.waves - residual.complete_waves;
    }
    return result;
}

void WorkerCoreExecutor::maybe_retire_collective_wave_image() {
    if (!collective_wave_coordinator_v1 ||
        !collective_wave_coordinator_v1->HasImage())
        return;
    const auto residual = collective_wave_coordinator_v1->Residual();
    if (residual.waves != 0 &&
        residual.complete_waves == residual.waves &&
        residual.forming_waves == 0 && residual.pending_waves == 0 &&
        residual.active_waves == 0 &&
        residual.arrivals == residual.departures)
        collective_wave_coordinator_v1->RetireImage(
            collective_wave_coordinator_v1->Identity());
}

void WorkerCoreExecutor::trace_collective_action_complete(
    const CollectiveExecutorActionV1 &action) {
    event_engine->add_event(
        "Core " + ToHexString(cid), "Collective_program_v1", "E",
        Trace_event_util(
            "collective action",
            {{"plan_index", action.plan_index},
             {"action_index", action.action_stream_index},
             {"wave", action.wave_index},
             {"phase", action.phase_id}}));
    event_engine->add_event(
        "Core " + ToHexString(cid), "P6_collective_action", "i",
        Trace_event_util(
            "P6_collective_action",
            {{"action_index", collective_action_trace_base +
                                  action.action_stream_index},
             {"plan_index", action.plan_index},
             {"wave", action.wave_index},
             {"phase", action.phase_id}}));
}

const IsaV1LoweredCollectiveAction &
WorkerCoreExecutor::collective_lowered_action(
    uint32_t stream_index) const {
    if (!collective_program_image_v1)
        throw std::logic_error("P7 action lookup has no P6 image");
    for (const auto &stream :
         collective_program_image_v1->Lowering().core_actions)
        if (stream.core_id == static_cast<uint16_t>(cid)) {
            if (stream_index >= stream.actions.size())
                throw std::out_of_range("P7 action index is out of range");
            return stream.actions[stream_index];
        }
    throw std::logic_error("P7 executing core has no action stream");
}

bool WorkerCoreExecutor::execute_p7_action(
    const CollectiveExecutorActionV1 &action) {
    if (!collective_profile_program_image_v1 ||
        !collective_acceleration_runtime_v1)
        return false;
    const auto *profile =
        collective_profile_program_image_v1->FindPlan(action.plan_index);
    if (!profile || profile->trees.empty()) return false;
    const auto &lowered =
        collective_lowered_action(action.action_stream_index);
    const IsaV1CollectivePlan &plan =
        collective_program_image_v1->Lowering().plans.at(action.plan_index);
    const bool endpoint =
        action.canonical_kind == IsaV1ActionKind::POST_RECEIVE ||
        action.canonical_kind == IsaV1ActionKind::ISSUE_SEND;
    const bool wait_action =
        action.canonical_kind == IsaV1ActionKind::WAIT_RECEIVE ||
        action.canonical_kind == IsaV1ActionKind::WAIT_SEND ||
        action.canonical_kind == IsaV1ActionKind::WAIT_TRANSPORT_RETIRE;
    const bool suppress_reduce =
        action.canonical_kind == IsaV1ActionKind::REDUCE_COMPUTE &&
        profile->suppress_endpoint_reduce_compute;
    if (!endpoint && !wait_action && !suppress_reduce) return false;

    // The acceleration batch can finish before the remaining canonical P6
    // actions for this core are dispatched.  Once this Worker has cleaned and
    // observed completion, those late actions must only advance P6; they must
    // never reactivate or poll a globally retired acceleration plan.
    if (p7_observed_plans.count(action.plan_index) != 0) {
        if (action.payload_kind == CollectiveExecutorPayloadKindV1::WAIT) {
            if (!collective_executor_v1->WaitComplete(
                    action.action_id, true, true))
                throw std::logic_error(
                    "P7 late completed WAIT did not advance P6");
        } else {
            collective_executor_v1->ActionComplete(action.action_id);
        }
        return true;
    }

    p7_active_plans.insert(action.plan_index);
    if (endpoint && lowered.action.item_index != kIsaV1NoItem) {
        (void)plan.child_flows.at(lowered.action.item_index);
        const uint64_t chunks =
            plan.length_bytes / kDteEndpointP2pMaxBytes +
            (plan.length_bytes % kDteEndpointP2pMaxBytes != 0);
        for (uint64_t chunk = 0; chunk < chunks; ++chunk)
            p7_ready_chunks[action.plan_index].insert(
                static_cast<uint32_t>(chunk));
    }
    ev_p7_acceleration.notify(SC_ZERO_TIME);
    if (action.payload_kind == CollectiveExecutorPayloadKindV1::WAIT) {
        while (!collective_acceleration_runtime_v1->PlanComplete(plan.key))
            wait(ev_p7_acceleration | ev_multicast_progress |
                 ev_reduce_stream_progress);
        if (!collective_executor_v1->WaitComplete(
                action.action_id, true, true))
            throw std::logic_error("P7 completed WAIT did not advance P6");
    } else {
        collective_executor_v1->ActionComplete(action.action_id);
    }
    return true;
}

void WorkerCoreExecutor::arm_dca_receive(
    const IsaV1CollectivePlan &plan,
    const IsaV1CollectiveAcceleratedTree &tree,
    uint32_t chunk_id) {
    const auto target = std::find_if(
        plan.reduce_targets.begin(), plan.reduce_targets.end(),
        [&](const auto &value) { return value.rank == tree.root_rank; });
    if (target == plan.reduce_targets.end())
        throw std::logic_error("P7 DCA tree has no reduce target");
    const uint64_t offset =
        static_cast<uint64_t>(chunk_id) * kDteEndpointP2pMaxBytes;
    const uint64_t bytes = std::min<uint64_t>(
        kDteEndpointP2pMaxBytes, plan.length_bytes - offset);
    Collective_data_prim prim;
    prim.mode = Collective_data_prim::Mode::REDUCE_STREAM_RX_START;
    prim.tree_id = tree.topology.tree_id;
    prim.descriptor.op = plan.op;
    prim.descriptor.dtype = target->dtype;
    prim.descriptor.reduce_op = target->reduce_op;
    prim.descriptor.key = plan.key;
    prim.descriptor.group = plan.group;
    prim.descriptor.root_rank = tree.root_rank;
    prim.descriptor.self_rank = tree.root_rank;
    prim.descriptor.count = bytes / (CollDTypeBits(target->dtype) / 8);
    prim.descriptor.dst_addr = target->result_address_bytes + offset;
    prim.dca_phase_id = 0;
    prim.dca_stream_id = chunk_id + 1;
    execute_collective_data(&prim);
    collective_acceleration_runtime_v1->RegisterDcaRootReady(
        plan.key, tree.topology.tree_id, chunk_id,
        static_cast<uint16_t>(cid));
}

void WorkerCoreExecutor::send_dca_bytes(
    const IsaV1CollectivePlan &plan,
    const IsaV1CollectiveAcceleratedTree &tree,
    uint32_t chunk_id) {
    const auto target = std::find_if(
        plan.reduce_targets.begin(), plan.reduce_targets.end(),
        [&](const auto &value) { return value.rank == tree.root_rank; });
    if (target == plan.reduce_targets.end())
        throw std::logic_error("P7 DCA tree has no reduce target");
    const auto self = std::find(
        plan.group.begin(), plan.group.end(), static_cast<uint16_t>(cid));
    if (self == plan.group.end())
        throw std::logic_error("P7 DCA sender is outside the group");
    const uint16_t self_rank = static_cast<uint16_t>(
        self - plan.group.begin());
    const uint64_t offset =
        static_cast<uint64_t>(chunk_id) * kDteEndpointP2pMaxBytes;
    const uint64_t bytes = std::min<uint64_t>(
        kDteEndpointP2pMaxBytes, plan.length_bytes - offset);
    uint64_t source_offset = offset;
    if (plan.op == CollOp::REDUCESCATTER)
        source_offset += static_cast<uint64_t>(tree.root_rank) *
                         plan.length_bytes;
    Collective_data_prim prim;
    prim.mode = Collective_data_prim::Mode::REDUCE_STREAM_TX;
    prim.tree_id = tree.topology.tree_id;
    prim.descriptor.op = plan.op;
    prim.descriptor.dtype = target->dtype;
    prim.descriptor.reduce_op = target->reduce_op;
    prim.descriptor.key = plan.key;
    prim.descriptor.group = plan.group;
    prim.descriptor.root_rank = tree.root_rank;
    prim.descriptor.self_rank = self_rank;
    prim.descriptor.count = bytes / (CollDTypeBits(target->dtype) / 8);
    prim.descriptor.src_addr =
        plan.rank_records.at(self_rank).send.base_address_bytes +
        source_offset;
    prim.dca_phase_id = 0;
    prim.dca_stream_id = chunk_id + 1;
    execute_collective_data(&prim);
}

void WorkerCoreExecutor::wait_dca_receive(
    const IsaV1CollectivePlan &plan,
    const IsaV1CollectiveAcceleratedTree &tree,
    uint32_t chunk_id) {
    const auto target = std::find_if(
        plan.reduce_targets.begin(), plan.reduce_targets.end(),
        [&](const auto &value) { return value.rank == tree.root_rank; });
    if (target == plan.reduce_targets.end())
        throw std::logic_error("P7 DCA wait has no reduce target");
    const uint64_t offset =
        static_cast<uint64_t>(chunk_id) * kDteEndpointP2pMaxBytes;
    const uint64_t bytes = std::min<uint64_t>(
        kDteEndpointP2pMaxBytes, plan.length_bytes - offset);
    Collective_data_prim prim;
    prim.mode = Collective_data_prim::Mode::REDUCE_STREAM_RX_WAIT;
    prim.tree_id = tree.topology.tree_id;
    prim.descriptor.op = plan.op;
    prim.descriptor.dtype = target->dtype;
    prim.descriptor.reduce_op = target->reduce_op;
    prim.descriptor.key = plan.key;
    prim.descriptor.group = plan.group;
    prim.descriptor.root_rank = tree.root_rank;
    prim.descriptor.self_rank = tree.root_rank;
    prim.descriptor.count = bytes / (CollDTypeBits(target->dtype) / 8);
    prim.dca_phase_id = 0;
    prim.dca_stream_id = chunk_id + 1;
    execute_collective_data(&prim);
    collective_acceleration_runtime_v1->AcknowledgeDca(
        plan.key, tree.topology.tree_id, chunk_id,
        static_cast<uint16_t>(cid));
    ev_p7_acceleration.notify(SC_ZERO_TIME);
}

void WorkerCoreExecutor::collective_acceleration_worker() {
    while (true) {
        wait(ev_p7_acceleration);
        if (!collective_program_image_v1 ||
            !collective_profile_program_image_v1 ||
            !collective_acceleration_runtime_v1)
            continue;
        bool progress = true;
        while (progress) {
            progress = false;
            for (uint32_t plan_index : p7_active_plans) {
                const auto *profile =
                    collective_profile_program_image_v1->FindPlan(
                        plan_index);
                if (!profile || profile->trees.empty()) continue;
                const IsaV1CollectivePlan &plan =
                    collective_program_image_v1->Lowering().plans.at(
                        plan_index);
                const auto self = std::find(
                    plan.group.begin(), plan.group.end(),
                    static_cast<uint16_t>(cid));
                if (self == plan.group.end()) continue;
                const uint16_t rank = static_cast<uint16_t>(
                    self - plan.group.begin());
                for (const auto &tree : profile->trees) {
                    auto &state = p7_tree_states[
                        {plan_index, tree.topology.tree_id}];
                    for (uint32_t chunk : p7_ready_chunks[plan_index]) {
                        const uint64_t offset =
                            static_cast<uint64_t>(chunk) *
                            kDteEndpointP2pMaxBytes;
                        if (offset >= plan.length_bytes) continue;
                        const uint64_t bytes = std::min<uint64_t>(
                            kDteEndpointP2pMaxBytes,
                            plan.length_bytes - offset);

                        const EndpointMulticastPostKey post_key{
                            tree.topology.tree_id, plan.key};
                        auto live_post = multicast_posts.find(post_key);
                        if (tree.multicast && rank != tree.root_rank &&
                            live_post != multicast_posts.end() &&
                            live_post->second.complete) {
                            const uint32_t completed_chunk =
                                live_post->second.chunk_id;
                            wait_multicast_receive(
                                tree.topology.tree_id, plan.key);
                            state.multicast_received.insert(
                                completed_chunk);
                            progress = true;
                            live_post = multicast_posts.end();
                        }
                        const bool previous_multicast_done =
                            chunk == 0 ||
                            state.multicast_received.count(chunk - 1) != 0;
                        if (tree.multicast && rank != tree.root_rank &&
                            previous_multicast_done &&
                            state.multicast_posted.count(chunk) == 0 &&
                            live_post == multicast_posts.end()) {
                            const uint64_t destination =
                                IsaV1CollectiveMulticastDestinationAddress(
                                    plan, tree, rank, offset);
                            arm_multicast_receive(
                                tree.topology.tree_id, plan.key, chunk,
                                destination);
                            collective_acceleration_runtime_v1->
                                RegisterMulticastPost(
                                    plan.key, tree.topology.tree_id, chunk,
                                    static_cast<uint16_t>(cid));
                            state.multicast_posted.insert(chunk);
                            progress = true;
                        }

                        if (tree.dca_reduce && rank == tree.root_rank &&
                            (chunk == 0 ||
                             state.dca_complete.count(chunk - 1) != 0) &&
                            reduce_stream_sessions.count(
                                tree.topology.tree_id) == 0 &&
                            state.dca_root_armed.count(chunk) == 0) {
                            arm_dca_receive(plan, tree, chunk);
                            state.dca_root_armed.insert(chunk);
                            progress = true;
                        }
                        if (tree.dca_reduce &&
                            state.dca_sent.count(chunk) == 0 &&
                            collective_acceleration_runtime_v1->
                                TryAcquireDca(
                                    plan.key, tree.topology.tree_id, chunk,
                                    static_cast<uint16_t>(cid))) {
                            send_dca_bytes(plan, tree, chunk);
                            state.dca_sent.insert(chunk);
                            progress = true;
                        }
                        if (tree.dca_reduce && rank == tree.root_rank &&
                            state.dca_root_armed.count(chunk) != 0 &&
                            state.dca_complete.count(chunk) == 0 &&
                            collective_acceleration_runtime_v1->
                                DcaSourcesReady(
                                    plan.key, tree.topology.tree_id,
                                    chunk)) {
                            wait_dca_receive(plan, tree, chunk);
                            state.dca_complete.insert(chunk);
                            progress = true;
                        }

                        if (tree.multicast && rank == tree.root_rank &&
                            state.multicast_sent.count(chunk) == 0) {
                            const auto session =
                                collective_acceleration_runtime_v1->
                                    TryAcquireMulticast(
                                        plan.key, tree.topology.tree_id,
                                        chunk);
                            if (session) {
                                const uint64_t source =
                                    IsaV1CollectiveMulticastSourceAddress(
                                        plan, tree, offset);
                                send_multicast_bytes(
                                    tree.topology.tree_id, *session,
                                    plan.key, source, bytes);
                                state.multicast_sent.insert(chunk);
                                progress = true;
                            }
                        }
                    }
                }
            }
            for (auto plan = p7_active_plans.begin();
                 plan != p7_active_plans.end();) {
                const CollectiveKey key =
                    collective_program_image_v1->Lowering().plans.at(*plan)
                        .key;
                if (collective_acceleration_runtime_v1->PlanComplete(key)) {
                    for (auto state = p7_tree_states.begin();
                         state != p7_tree_states.end();) {
                        if (state->first.first == *plan)
                            state = p7_tree_states.erase(state);
                        else
                            ++state;
                    }
                    p7_ready_chunks.erase(*plan);
                    if (!p7_observed_plans.insert(*plan).second)
                        throw std::logic_error(
                            "P7 Worker observed a completed plan twice");
                    collective_acceleration_runtime_v1->ObservePlanComplete(
                        key, static_cast<uint16_t>(cid));
                    plan = p7_active_plans.erase(plan);
                    progress = true;
                    ev_collective_program.notify(SC_ZERO_TIME);
                    ev_p7_acceleration.notify(SC_ZERO_TIME);
                } else {
                    ++plan;
                }
            }
            if (!progress && !p7_active_plans.empty()) {
                wait(CYCLE, SC_NS);
                progress = true;
            }
        }
    }
}

void WorkerCoreExecutor::collective_program_worker() {
    while (true) {
        while (!collective_executor_v1 ||
               !collective_executor_v1->Configured())
            wait(ev_collective_program);
        try {
            if (!collective_executor_v1->HasRunnable() &&
                !collective_executor_v1->HasInflight()) {
                wait(ev_collective_program);
                continue;
            }
            auto action = collective_executor_v1->NextAction();
            if (!action.has_value()) {
                wait(CYCLE, SC_NS);
                continue;
            }
            event_engine->add_event(
                "Core " + ToHexString(cid), "Collective_program_v1", "B",
                Trace_event_util(
                    "collective action",
                    {{"plan_index", action->plan_index},
                     {"action_index", action->action_stream_index},
                     {"wave", action->wave_index},
                     {"phase", action->phase_id}}));

            if (execute_p7_action(*action)) {
                trace_collective_action_complete(*action);
                maybe_retire_collective_wave_image();
                ev_collective_program.notify(SC_ZERO_TIME);
                ev_p7_acceleration.notify(SC_ZERO_TIME);
                ev_p2p_progress.notify(SC_ZERO_TIME);
                continue;
            }

            if (action->payload_kind ==
                CollectiveExecutorPayloadKindV1::ENDPOINT_PRIM) {
                uint32_t token = 0;
                if (auto *send = dynamic_cast<Dte_send_endpoint_prim *>(
                        action->prim.get())) {
                    token = send->token;
                    execute_dte_send_endpoint(send);
                } else if (auto *receive =
                               dynamic_cast<Dte_recv_endpoint_prim *>(
                                   action->prim.get())) {
                    token = receive->token;
                    execute_dte_recv_endpoint(receive);
                } else {
                    throw std::logic_error(
                        "collective endpoint action has an invalid Prim type");
                }
                auto handle = p2p_async_handles.find(token);
                if (handle == p2p_async_handles.end() ||
                    !collective_p2p_handles.emplace(token, handle->second)
                         .second)
                    throw std::logic_error(
                        "collective endpoint handle was not indexed exactly once");
                collective_executor_v1->ActionComplete(action->action_id);
            } else if (action->payload_kind ==
                       CollectiveExecutorPayloadKindV1::LOCAL_DATA_PRIM) {
                auto *data = dynamic_cast<Collective_data_v1_prim *>(
                    action->prim.get());
                if (data == nullptr)
                    throw std::logic_error(
                        "collective local data action has an invalid Prim type");
                TaskCoreContext context = generate_context(this);
                const int delay = data->taskCoreDefault(context);
                if (delay < 0)
                    throw std::logic_error(
                        "collective data action returned a negative delay");
                if (delay != 0) wait(delay, SC_NS);
                collective_executor_v1->ActionComplete(action->action_id);
            } else if (action->payload_kind ==
                       CollectiveExecutorPayloadKindV1::PHASE_BARRIER_PRIM) {
                auto *barrier =
                    dynamic_cast<Collective_phase_barrier_v1_prim *>(
                        action->prim.get());
                if (barrier == nullptr)
                    throw std::logic_error(
                        "collective phase action has an invalid Prim type");
                TaskCoreContext context = generate_context(this);
                (void)barrier->taskCoreDefault(context);
                const size_t complete_waves_before =
                    collective_wave_coordinator_v1->Residual().complete_waves;
                collective_executor_v1->ActionComplete(action->action_id);
                const size_t complete_waves_after =
                    collective_wave_coordinator_v1->Residual().complete_waves;
                if (action->canonical_kind ==
                        IsaV1ActionKind::COMPLETE_BARRIER &&
                    complete_waves_after == complete_waves_before + 1) {
                    event_engine->add_event(
                        "Core " + ToHexString(cid),
                        "P6_collective_wave_complete", "i",
                        Trace_event_util(
                            "P6_collective_wave_complete",
                            {{"plan_index", action->plan_index},
                             {"wave", action->wave_index}}));
                }
            } else {
                if (!action->wait.has_value())
                    throw std::logic_error(
                        "collective WAIT action lost its completion contract");
                bool completed = false;
                while (!completed) {
                    const uint32_t token = action->wait->internal_token;
                    auto tracked = collective_p2p_handles.find(token);
                    if (tracked == collective_p2p_handles.end())
                        throw std::logic_error(
                            "collective WAIT lost its endpoint handle");
                    bool local_complete = false;
                    bool transport_retired = false;
                    if (action->wait->kind ==
                        CollectiveExecutorWaitKindV1::TRANSPORT_RETIRE) {
                        transport_retired =
                            !p2p_endpoint->HasHandle(tracked->second);
                        if (transport_retired) {
                            p2p_async_handles.erase(token);
                            collective_p2p_handles.erase(tracked);
                        }
                    } else if (p2p_endpoint->HasHandle(tracked->second) &&
                               p2p_endpoint->Phase(tracked->second) ==
                                   P2pEndpointPhase::COMPLETE) {
                        if (!p2p_endpoint->TryWait(token))
                            throw std::logic_error(
                                "collective endpoint COMPLETE did not retire");
                        p2p_async_handles.erase(token);
                        local_complete = true;
                        transport_retired =
                            !p2p_endpoint->HasHandle(tracked->second);
                        if (action->wait->kind ==
                                CollectiveExecutorWaitKindV1::
                                    RECEIVE_LOCAL_AND_TRANSPORT &&
                            transport_retired)
                            collective_p2p_handles.erase(tracked);
                    }
                    completed = collective_executor_v1->WaitComplete(
                        action->action_id, local_complete,
                        transport_retired);
                    if (!completed) wait(CYCLE, SC_NS);
                }
            }

            trace_collective_action_complete(*action);
            maybe_retire_collective_wave_image();
            ev_collective_program.notify(SC_ZERO_TIME);
            ev_p2p_progress.notify(SC_ZERO_TIME);
        } catch (...) {
            if (collective_executor_v1)
                collective_executor_v1->Abort();
            collective_p2p_handles.clear();
            ev_p2p_progress.notify(SC_ZERO_TIME);
            throw;
        }
    }
}

void WorkerCoreExecutor::worker_core_execute() {
    while (true) {
        if (!moe_swizzle_pending_fixed_interval_kinds.empty())
            FinalizeMoeSwizzlePendingFixedIntervals(
                static_cast<uint16_t>(cid),
                moe_swizzle_pending_fixed_interval_start,
                sc_time_stamp().value(),
                &moe_swizzle_pending_fixed_interval_kinds,
                &moe_swizzle_runtime_intervals);
        PrimBase *p = nullptr;    // 下一个要执行的原语
        bool conf_delete = false; // 是否自动填充了一个recv_conf原语

        if (prim_queue.size() == 0) {
            if (SPEC_DTE_ASYNC && dte_async->OutstandingCount() != 0)
                throw std::runtime_error(
                    "DTE V3a primitive queue drained with outstanding tokens; "
                    "an explicit wait/fence is required");
            if (!p2p_async_handles.empty())
                throw std::runtime_error(
                    "primitive queue drained with outstanding P2P endpoint "
                    "tokens; an explicit wait/fence is required");
            if (lsu_memory && lsu_memory->OutstandingCount() != 0)
                throw std::runtime_error(
                    "primitive queue drained with outstanding LSU tokens; "
                    "an explicit Lsu_mem wait/fence is required");
            // 队列中没有指令，意味着现在是初始状态或者所有原语都被执行完了（假设所有原语只做一轮），默认作recv，直到config发进来
            // 显式 tag=0、recv_cnt=0：CONFIG ACK tag 契约固定为 0（不依赖未初始化值）。
            p = new Recv_prim(RECV_TYPE::RECV_CONF, /*tag=*/0, /*recv_cnt=*/0);
            prim_queue.emplace_front(p);
            conf_delete = true;
        } else {
            p = prim_queue.front();
        }

        std::vector<MoeSwizzleRuntimeIntervalKind> fixed_kinds;
        if (auto *endpoint = dynamic_cast<Dte_endpoint_prim_base *>(p)) {
            fixed_kinds.push_back(
                MoeSwizzleRuntimeIntervalKind::DTE_LAUNCH);
            fixed_kinds.push_back(
                MoeSwizzleRuntimeIntervalKind::SESSION_OPEN);
            if (endpoint->completion == DteEndpointCompletion::SYNC) {
                fixed_kinds.push_back(
                    MoeSwizzleRuntimeIntervalKind::DTE_SYNC);
                fixed_kinds.push_back(
                    MoeSwizzleRuntimeIntervalKind::SESSION_RETIRE);
            }
        } else if (auto *async_prim = dynamic_cast<Dte_async_prim *>(p)) {
            if (async_prim->op == DteAsyncOp::ISSUE) {
                fixed_kinds.push_back(
                    MoeSwizzleRuntimeIntervalKind::DTE_LAUNCH);
                fixed_kinds.push_back(
                    MoeSwizzleRuntimeIntervalKind::LOCAL_COPY_FIXED);
            } else if (async_prim->op == DteAsyncOp::WAIT ||
                       async_prim->op == DteAsyncOp::FENCE) {
                fixed_kinds.push_back(
                    MoeSwizzleRuntimeIntervalKind::DTE_SYNC);
                if (!p2p_async_handles.empty())
                    fixed_kinds.push_back(
                        MoeSwizzleRuntimeIntervalKind::SESSION_RETIRE);
            }
        } else if (auto *lifecycle = dynamic_cast<Sram_lifecycle *>(p)) {
            if (lifecycle->op == SramLifecycleOp::ALLOC ||
                lifecycle->op == SramLifecycleOp::ALLOC_AT)
                fixed_kinds.push_back(
                    MoeSwizzleRuntimeIntervalKind::SRAM_ALLOC_FIXED);
            else if (lifecycle->op == SramLifecycleOp::FREE)
                fixed_kinds.push_back(
                    MoeSwizzleRuntimeIntervalKind::SRAM_FREE_FIXED);
        } else if (dynamic_cast<Sram_bind_oneshot *>(p) != nullptr) {
            fixed_kinds.push_back(MoeSwizzleRuntimeIntervalKind::SRAM_BIND);
        } else if (auto *event = dynamic_cast<Event_control_prim *>(p)) {
            fixed_kinds.push_back(
                event->op == EventControlOp::SET
                    ? MoeSwizzleRuntimeIntervalKind::EVENT_SET_FIXED
                    : MoeSwizzleRuntimeIntervalKind::EVENT_WAIT_FIXED);
        } else if (auto *send = dynamic_cast<Send_prim *>(p);
                   send != nullptr && send->type == SEND_DONE) {
            fixed_kinds.push_back(
                MoeSwizzleRuntimeIntervalKind::TERMINAL_DONE_FIXED);
        }
        if (!fixed_kinds.empty()) {
            moe_swizzle_pending_fixed_interval_kinds = std::move(fixed_kinds);
            moe_swizzle_pending_fixed_interval_start =
                sc_time_stamp().value();
        }

        // NOTE:
        // send原语和recv原语和其他计算原语不同，需要涉及core中信号的处理，所以需要在core这个文件内部处理相关逻辑，否则会出现依赖问题。
        // 需要等待 switch_prim_block 将 prim_block 置为 false，然后再执行
        // switch_prim_block 收到 ev_block 触发 ev_block 在 send_logic 和
        // recv_logic 中触发

        if (typeid(*p) == typeid(Group_sync_prim)) {
            auto *sync = static_cast<Group_sync_prim *>(p);
            event_engine->add_event("Core " + ToHexString(cid),
                                    "Group_sync_prim", "B",
                                    Trace_event_util("group sync"));
            execute_group_sync(sync);
            event_engine->add_event("Core " + ToHexString(cid),
                                    "Group_sync_prim", "E",
                                    Trace_event_util("group sync"));
        } else if (typeid(*p) == typeid(Event_control_prim)) {
            auto *event = static_cast<Event_control_prim *>(p);
            const uint64_t marker_start = sc_time_stamp().value();
            const char *op = event->op == EventControlOp::SET
                ? "event set" : "event wait";
            event_engine->add_event("Core " + ToHexString(cid),
                                    "Event_control_prim", "B",
                                    Trace_event_util(op));
            execute_event_control(event);
            moe_swizzle_runtime_intervals.push_back(
                {static_cast<uint16_t>(cid),
                 MoeSwizzleRuntimeIntervalKind::EVENT_CONTROL,
                 marker_start, sc_time_stamp().value()});
            event_engine->add_event("Core " + ToHexString(cid),
                                    "Event_control_prim", "E",
                                    Trace_event_util(op));
        } else if (typeid(*p) == typeid(Collective_data_prim)) {
            auto *collective = static_cast<Collective_data_prim *>(p);
            event_engine->add_event("Core " + ToHexString(cid),
                                    "Collective_data_prim", "B",
                                    Trace_event_util("router collective data"));
            execute_collective_data(collective);
            event_engine->add_event("Core " + ToHexString(cid),
                                    "Collective_data_prim", "E",
                                    Trace_event_util("router collective data"));
        } else if (typeid(*p) == typeid(Collective_prim)) {
            auto *collective = static_cast<Collective_prim *>(p);
            const bool gather_arrival = collective->marker_kind ==
                Collective_prim::MarkerKind::GATHER_ARRIVAL;
            const bool reduce_arrival = collective->marker_kind ==
                Collective_prim::MarkerKind::REDUCE_ARRIVAL;
            const char *collective_event = gather_arrival ? "Gather reorder" :
                (reduce_arrival ? "Reduce RX" : "Collective barrier");
            event_engine->add_event("Core " + ToHexString(cid),
                                    "Collective_prim", "B",
                                    Trace_event_util(collective_event));
            TaskCoreContext context = generate_context(this);
            collective->taskCoreDefault(context);
            event_engine->add_event("Core " + ToHexString(cid),
                                    "Collective_prim", "E",
                                    Trace_event_util(collective_event));
        } else if (typeid(*p) == typeid(Collective_launch_v1_prim)) {
            auto *launch = static_cast<Collective_launch_v1_prim *>(p);
            if (!collective_executor_v1 ||
                !collective_executor_v1->Configured())
                throw std::runtime_error(
                    "collective launch reached an unconfigured Worker");
            event_engine->add_event(
                "Core " + ToHexString(cid),
                "Collective_launch_v1_prim", "B",
                Trace_event_util("collective launch"));
            (void)collective_executor_v1->AcceptLaunch(*launch);
            ev_collective_program.notify(SC_ZERO_TIME);
            event_engine->add_event(
                "Core " + ToHexString(cid),
                "Collective_launch_v1_prim", "E",
                Trace_event_util("collective launch"));
        } else if (typeid(*p) == typeid(Dte_send_endpoint_prim)) {
            auto *endpoint = static_cast<Dte_send_endpoint_prim *>(p);
            event_engine->add_event(
                "Core " + ToHexString(cid), "Dte_send_endpoint_prim", "B",
                Trace_event_util("P2P endpoint send"));
            execute_dte_send_endpoint(endpoint);
            event_engine->add_event(
                "Core " + ToHexString(cid), "Dte_send_endpoint_prim", "E",
                Trace_event_util("P2P endpoint send"));
        } else if (typeid(*p) == typeid(Dte_recv_endpoint_prim)) {
            auto *endpoint = static_cast<Dte_recv_endpoint_prim *>(p);
            event_engine->add_event(
                "Core " + ToHexString(cid), "Dte_recv_endpoint_prim", "B",
                Trace_event_util("P2P endpoint receive"));
            execute_dte_recv_endpoint(endpoint);
            event_engine->add_event(
                "Core " + ToHexString(cid), "Dte_recv_endpoint_prim", "E",
                Trace_event_util("P2P endpoint receive"));
        } else if (typeid(*p) == typeid(Dte_async_prim)) {
            auto *async_prim = static_cast<Dte_async_prim *>(p);
            const uint64_t marker_start = sc_time_stamp().value();
            const std::string op = DteAsyncOpName(async_prim->op);
            event_engine->add_event(
                "Core " + ToHexString(cid), "Dte_async_prim", "B",
                Trace_event_util("Dte_async_prim " + op));
            execute_dte_async(async_prim);
            const uint64_t marker_end = sc_time_stamp().value();
            if (async_prim->op == DteAsyncOp::ISSUE) {
                if (!moe_swizzle_local_dte_starts
                         .emplace(async_prim->token, marker_start).second)
                    throw std::logic_error(
                        "MoE Swizzle DTE capture observed duplicate token issue");
            } else if (async_prim->op == DteAsyncOp::WAIT ||
                       async_prim->op == DteAsyncOp::CANCEL) {
                const auto tracked = moe_swizzle_local_dte_starts.find(
                    async_prim->token);
                if (tracked != moe_swizzle_local_dte_starts.end()) {
                    moe_swizzle_runtime_intervals.push_back(
                        {static_cast<uint16_t>(cid),
                         MoeSwizzleRuntimeIntervalKind::LOCAL_DTE,
                         tracked->second, marker_end});
                    moe_swizzle_local_dte_starts.erase(tracked);
                }
            } else if (async_prim->op == DteAsyncOp::FENCE) {
                for (const auto &[token, start] :
                     moe_swizzle_local_dte_starts) {
                    (void)token;
                    moe_swizzle_runtime_intervals.push_back(
                        {static_cast<uint16_t>(cid),
                         MoeSwizzleRuntimeIntervalKind::LOCAL_DTE,
                         start, marker_end});
                }
                moe_swizzle_local_dte_starts.clear();
            }
            event_engine->add_event(
                "Core " + ToHexString(cid), "Dte_async_prim", "E",
                Trace_event_util("Dte_async_prim " + op));
        } else if (typeid(*p) == typeid(Send_prim)) {
            auto *send_prim = static_cast<Send_prim *>(p);
            if (SPEC_DTE_ASYNC && send_prim->type == SEND_DONE &&
                dte_async->OutstandingCount() != 0)
                throw std::runtime_error(
                    "DTE V3a SEND_DONE reached with outstanding tokens; "
                    "an explicit wait/fence is required");
            if (send_prim->type == SEND_DONE &&
                !p2p_async_handles.empty())
                throw std::runtime_error(
                    "SEND_DONE reached with outstanding P2P endpoint tokens; "
                    "an explicit wait/fence is required");
            if (send_prim->type == SEND_DONE) {
                maybe_retire_collective_wave_image();
                if ((collective_executor_v1 &&
                     !collective_executor_v1->Drained()) ||
                    !collective_p2p_handles.empty() ||
                    (collective_wave_coordinator_v1 &&
                     collective_wave_coordinator_v1->HasImage()))
                    throw std::runtime_error(
                        "SEND_DONE reached before collective program drain; "
                        "an explicit wait/fence is required");
                while (!p2p_tx_queue.empty() ||
                       !p2p_pending_requests.empty() ||
                       !p2p_endpoint->Drained())
                    wait(ev_p2p_progress);
                if (!p2p_tx_queue.empty() ||
                    !p2p_pending_requests.empty() ||
                    !p2p_async_handles.empty() ||
                    !p2p_rx_addresses.empty() ||
                    !p2p_rx_fsm_by_flow.empty() ||
                    !p2p_endpoint->Drained())
                    throw std::logic_error(
                        "SEND_DONE observed inconsistent P2P drain state");
            }
            // 触发 send_logic
            if (!SPEC_SEND_RECV_PARALLEL) {
                ev_send.notify(CYCLE, SC_NS);
                event_engine->add_event(
                    "Core " + ToHexString(cid), "Send_prim", "B",
                    Trace_event_util(
                        "Send_prim" +
                        GetEnumSendType(dynamic_cast<Send_prim *>(p)->type)));
                wait(prim_block.negedge_event());
                event_engine->add_event(
                    "Core " + ToHexString(cid), "Send_prim", "E",
                    Trace_event_util(
                        "Send_prim" +
                        GetEnumSendType(dynamic_cast<Send_prim *>(p)->type)));
            } else {
                while (!send_done) {
                    wait(CYCLE, SC_NS);
                }

                // send 模块处理的四条指令
                while ((typeid(*p) == typeid(Recv_prim) &&
                        ((Recv_prim *)p)->type == RECV_ACK) ||
                       (typeid(*p) == typeid(Send_prim) &&
                        ((Send_prim *)p)->type == SEND_DATA) ||
                       (typeid(*p) == typeid(Send_prim) &&
                        ((Send_prim *)p)->type == SEND_REQ) ||
                       (typeid(*p) == typeid(Send_prim) &&
                        ((Send_prim *)p)->type == SEND_DONE)) {
                    prim_queue.pop_front();
                    send_para_queue.push(p);
                    if (prim_refill && PrimMayRefill(p)) {
                        prim_queue.emplace_back(p);
                    }
                    if (!prim_queue.size())
                        break;
                    // 这里会pop出来RECV_ACK
                    p = prim_queue.front();
                }

                send_done = false;

                // 触发 send_logic
                ev_para_send.notify(CYCLE, SC_NS);
                continue;
            }
        } else if (typeid(*p) == typeid(Recv_prim)) {
            ev_recv.notify(CYCLE, SC_NS);
            event_engine->add_event(
                "Core " + ToHexString(cid), "Receive_prim", "B",
                Trace_event_util(
                    "Receive_prim" +
                    GetEnumRecvType(dynamic_cast<Recv_prim *>(p)->type)));
            wait(prim_block.negedge_event());
            event_engine->add_event(
                "Core " + ToHexString(cid), "Receive_prim", "E",
                Trace_event_util(
                    "Receive_prim" +
                    GetEnumRecvType(dynamic_cast<Recv_prim *>(p)->type)));
        } else {
            // Execution dispatch and data-pipeline structure are distinct
            // from the ISA statistics category. CompBase-derived memory
            // helpers still belong to the same producer chain.
            if (!HasExactlyOnePrimMainCategory(p->prim_type))
                throw std::runtime_error(
                    "primitive has no unique main category: " + p->name);
            const char *trace_category =
                (p->prim_type & COMP_PRIM) ? "Comp_prim" :
                (p->prim_type & MEM_PRIM) ? "Mem_prim" :
                (p->prim_type & COMM_PRIM) ? "Comm_prim" : "Sync_prim";
            const uint64_t marker_start = sc_time_stamp().value();
            const bool group_gemm =
                dynamic_cast<Matmul_f *>(p) != nullptr ||
                dynamic_cast<matmul_forward_moe *>(p) != nullptr;
            const bool swiglu_group =
                dynamic_cast<swiglu_forward *>(p) != nullptr;
            if (group_gemm) {
                if (moe_swizzle_pending_group_gemm_dispatch_start.has_value())
                    throw std::logic_error(
                        "GroupGEMM dispatch phase is already pending");
                moe_swizzle_pending_group_gemm_dispatch_start = marker_start;
            }
            ev_comp.notify(CYCLE, SC_NS);
            event_engine->add_event("Core " + ToHexString(cid), trace_category,
                                    "B", Trace_event_util(p->name));
            wait(prim_block.negedge_event());
            const uint64_t marker_end = sc_time_stamp().value();

            std::optional<MoeSwizzleRuntimeIntervalKind> marker_kind;
            if (group_gemm)
                marker_kind = MoeSwizzleRuntimeIntervalKind::MATMUL;
            else if (swiglu_group)
                marker_kind = MoeSwizzleRuntimeIntervalKind::SWIGLU_GROUP;
            else if (const auto *collective =
                         dynamic_cast<Collective_data_v1_prim *>(p);
                     collective != nullptr &&
                     collective->mode == CollectiveDataV1PrimMode::REDUCE)
                marker_kind = MoeSwizzleRuntimeIntervalKind::LOCAL_REDUCE;
            else if (dynamic_cast<Sram_lifecycle *>(p) != nullptr)
                marker_kind = MoeSwizzleRuntimeIntervalKind::SRAM_LIFECYCLE;
            if (marker_kind.has_value())
                moe_swizzle_runtime_intervals.push_back(
                    {static_cast<uint16_t>(cid), *marker_kind,
                     marker_start, marker_end});

            // 发送信号让send发送最后一个包
            if (prim_queue.size() >= 2 &&
                dynamic_cast<CompBase *>(prim_queue[1]) == nullptr) {
                send_last_packet = true;
                ev_send_last_packet.notify(CYCLE, SC_NS);
            }

            event_engine->add_event("Core " + ToHexString(cid), trace_category,
                                    "E", Trace_event_util(p->name));
        }

        // 将原语重新填充到队列中
        if (prim_refill && PrimMayRefill(p))
            prim_queue.emplace_back(p);

        if (conf_delete)
            delete p;

        prim_queue.pop_front();
        wait(CYCLE, SC_NS);
    }
}

void WorkerCoreExecutor::execute_group_sync(Group_sync_prim *prim) {
    if (prim == nullptr || group_sync_runtime == nullptr)
        throw std::runtime_error(
            "GROUP_SYNC requires a configured immutable core-group registry");
    if (cid < 0 || static_cast<uint64_t>(cid) > UINT16_MAX)
        throw std::out_of_range("GROUP_SYNC core id exceeds the runtime wire");
    group_sync_runtime->Wait(static_cast<uint16_t>(cid), prim->group_id,
                             prim->sync_seq);
}

void WorkerCoreExecutor::send_event_control(
    const EventControlMessage &message) {
    while (!ctrl_channel_avail_i.read() ||
           !atomic_helper_lock(sc_time_stamp(), 3))
        wait(CYCLE, SC_NS);
    Msg wire_message = MakeEventControlMsg(message);
    PinControlMsgExit(wire_message);
    send_serialized_wire(SerializeMsg(wire_message), true);
}

void WorkerCoreExecutor::drain_event_control_queue() {
    while (!event_control_queue.Empty()) {
        event_mailbox.Deliver(event_control_queue.Front(),
                              static_cast<uint16_t>(cid), TOTAL_CORES);
        (void)event_control_queue.Pop();
        ev_event_queue_space.notify(SC_ZERO_TIME);
    }
}

void WorkerCoreExecutor::execute_event_control(Event_control_prim *prim) {
    if (prim == nullptr)
        throw std::invalid_argument("EVENT received a null primitive");
    if (cid < 0 || static_cast<uint64_t>(cid) > UINT16_MAX ||
        TOTAL_CORES <= 0 ||
        prim->source_core >= static_cast<uint64_t>(TOTAL_CORES) ||
        prim->destination_core >= static_cast<uint64_t>(TOTAL_CORES))
        throw std::out_of_range("EVENT endpoint is outside the core topology");

    if (prim->op == EventControlOp::SET) {
        if (prim->source_core != static_cast<uint16_t>(cid) ||
            prim->count != 1)
            throw std::invalid_argument(
                "EVENT_SET must originate at the executing core with count=1");
        send_event_control({prim->source_core, prim->destination_core,
                            prim->tag});
        return;
    }
    if (prim->op != EventControlOp::WAIT ||
        prim->destination_core != static_cast<uint16_t>(cid) ||
        prim->count == 0)
        throw std::invalid_argument(
            "EVENT_WAIT must target the executing core with nonzero count");

    const EventKey key{prim->source_core, prim->destination_core, prim->tag};
    while (true) {
        drain_event_control_queue();
        if (event_mailbox.TryConsume(key, prim->count)) return;
        wait(ev_recv_msg_type_[MSG_TYPE::EVENT]);
    }
}

void WorkerCoreExecutor::execute_collective_data(Collective_data_prim *prim) {
    if (prim == nullptr || prim->tree_id == 0)
        throw std::invalid_argument("invalid collective data primitive");
    auto send_raw = [&](const sc_bv<256> &wire) {
        while (!channel_avail_i.read() ||
               !atomic_helper_lock(sc_time_stamp(), 3))
            wait(CYCLE, SC_NS);
        send_serialized_wire(wire, false);
    };
    if (prim->mode ==
            Collective_data_prim::Mode::REDUCE_STREAM_RX_START) {
        if (prim->descriptor.self_rank != prim->descriptor.root_rank)
            throw std::invalid_argument(
                "only reduce root may arm a stream RX session");
        if (reduce_stream_sessions.count(prim->tree_id))
            throw std::invalid_argument(
                "duplicate endpoint reduce stream session");
        EndpointReduceStreamSession session;
        session.descriptor = prim->descriptor;
        session.tree_id = prim->tree_id;
        session.phase_id = prim->dca_phase_id;
        session.stream_id = prim->dca_stream_id;
        reduce_stream_sessions.emplace(prim->tree_id,
                                       std::move(session));
        std::cout << "[COLL_STREAM_ARM] tree=" << prim->tree_id
                  << " root=" << cid << std::endl;
        return;
    }
    if (prim->mode == Collective_data_prim::Mode::CORE_VECTOR_START) {
        if (prim->descriptor.self_rank != prim->descriptor.root_rank ||
            prim->core_vector_beats == 0)
            throw std::invalid_argument(
                "shared CORE vector start must execute at reduce root");
        auto *engine = coll_refactor::LookupProductionReduceStreamEngine(
            static_cast<uint16_t>(cid));
        if (!engine)
            throw std::runtime_error(
                "shared CORE vector start has no production tile pool");
        if (core_vector_sessions.count(prim->tree_id))
            throw std::invalid_argument(
                "duplicate shared CORE vector session");
        auto inserted = core_vector_sessions.emplace(
            prim->tree_id, EndpointCoreVectorSession{});
        auto &session = inserted.first->second;
        auto consume_ready = [&] {
            for (uint64_t tag : session.tags) {
                auto result = engine->TakeCoreResult(tag);
                if (!result) continue;
                if (result->source !=
                    coll_refactor::DcaRequestSource::CORE)
                    throw std::logic_error(
                        "shared CORE session received DCA result");
                ++session.completed;
            }
        };
        const uint64_t dtype_bits =
            CollDTypeBits(prim->descriptor.dtype);
        const uint64_t lanes =
            SPEC_NOC_COLL_CONFIG.dca.vector_bits / dtype_bits;
        for (uint32_t beat = 0; beat < prim->core_vector_beats; ++beat) {
            coll_refactor::DcaPoolRequest request;
            request.request.key = {
                {prim->descriptor.key, 0,
                 static_cast<uint32_t>(0x800000u + beat)},
                0, beat};
            request.request.op = prim->descriptor.reduce_op;
            coll_refactor::VectorBeat geometry{
                prim->descriptor.dtype,
                SPEC_NOC_COLL_CONFIG.dca.vector_bits, {lanes, lanes}};
            request.request.operands = {geometry, geometry};
            if (SPEC_NOC_COLL_CONFIG.dca.value_mode !=
                    NocCollValueMode::TIMING_ONLY) {
                const uint64_t one =
                    prim->descriptor.dtype == CollDType::FP32
                        ? coll_refactor::CollFp32Bits(1.0f) : 1;
                request.operand_values[0].assign(lanes, one);
                request.operand_values[1].assign(lanes, one);
            }
            while (true) {
                consume_ready();
                auto tag = engine->TrySubmitCore(request);
                if (tag) {
                    session.tags.push_back(*tag);
                    break;
                }
                wait(engine->CoreProgressEvent());
            }
        }
        consume_ready();
        std::cout << "[COLL_CORE_START] tree=" << prim->tree_id
                  << " beats=" << prim->core_vector_beats
                  << " completed=" << session.completed << std::endl;
        return;
    }
    if (prim->mode == Collective_data_prim::Mode::CORE_VECTOR_WAIT) {
        auto *engine = coll_refactor::LookupProductionReduceStreamEngine(
            static_cast<uint16_t>(cid));
        auto session = core_vector_sessions.find(prim->tree_id);
        if (!engine || session == core_vector_sessions.end() ||
            session->second.tags.size() != prim->core_vector_beats)
            throw std::invalid_argument(
                "shared CORE vector wait session mismatch");
        while (session->second.completed < session->second.tags.size()) {
            bool progress = false;
            for (uint64_t tag : session->second.tags) {
                auto result = engine->TakeCoreResult(tag);
                if (!result) continue;
                ++session->second.completed;
                progress = true;
            }
            if (!progress) wait(engine->CoreProgressEvent());
        }
        std::cout << "[COLL_CORE_DONE] tree=" << prim->tree_id
                  << " beats=" << session->second.completed << std::endl;
        core_vector_sessions.erase(session);
        return;
    }
    if (prim->mode == Collective_data_prim::Mode::REDUCE_STREAM_TX) {
        const auto &dca = SPEC_NOC_COLL_CONFIG.dca;
        const uint64_t vector_bits = dca.vector_bits;
        const uint64_t dtype_bits = CollDTypeBits(prim->descriptor.dtype);
        if (sram_access == nullptr)
            throw std::runtime_error(
                "DCA stream TX requires the real SRAM data path");
        const uint64_t total_bits = CollCheckedMul(
            prim->descriptor.count, dtype_bits);
        if (total_bits % 8 != 0)
            throw std::invalid_argument(
                "DCA stream TX source is not whole bytes");
        const uint64_t total_bytes = total_bits / 8;
        sram::Request read;
        read.initiator = sram::Initiator::kDte;
        read.command = sram::Command::kRead;
        read.address = prim->descriptor.src_addr;
        read.size_bytes = total_bytes;
        const std::vector<uint8_t> bytes =
            sram_access->Access(read).payload;
        if (bytes.size() != total_bytes)
            throw std::runtime_error(
                "DCA stream TX SRAM read returned the wrong byte count");
        const auto stream = coll_refactor::BuildIsaV1DcaByteStream(
            prim->tree_id, prim->descriptor.key, prim->dca_phase_id,
            prim->dca_stream_id,
            static_cast<uint16_t>(cid), prim->descriptor.dtype,
            prim->descriptor.reduce_op, bytes, vector_bits);
        send_raw(coll_refactor::SerializeReduceStreamHeader(
            stream.header, vector_bits));
        for (const auto &beat : stream.beats) {
            for (const auto &data : coll_refactor::SplitReduceVectorBeat(
                     stream.header, beat, 128, vector_bits))
                send_raw(coll_refactor::SerializeReduceStreamData(data));
        }
        std::cout << "[COLL_STREAM_TX] tree=" << prim->tree_id
                  << " rank=" << prim->descriptor.self_rank
                  << " bytes=" << total_bytes
                  << " header=1 data="
                  << stream.header.stream.physical_data_flits
                  << " beats=" << stream.header.stream.vector_beats
                  << " source=SRAM" << std::endl;
        return;
    }
    if (prim->mode ==
            Collective_data_prim::Mode::REDUCE_STREAM_RX_WAIT) {
        auto session = reduce_stream_sessions.find(prim->tree_id);
        if (session == reduce_stream_sessions.end() ||
            session->second.phase_id != prim->dca_phase_id ||
            session->second.stream_id != prim->dca_stream_id)
            throw std::invalid_argument(
                "wait for unknown endpoint reduce stream generation");
        while (!session->second.complete)
            wait(ev_reduce_stream_progress);
        const uint64_t values = session->second.values_seen;
        reduce_stream_sessions.erase(session);
        std::cout << "[COLL_STREAM_RESULT] tree=" << prim->tree_id
                  << " elements=" << values
                  << " value=committed-to-SRAM"
                  << std::endl;
        return;
    }
    if (prim->mode == Collective_data_prim::Mode::REDUCE_TX) {
        const uint64_t width = CollDTypeBits(prim->descriptor.dtype);
        const uint64_t per_chunk = 128 / width;
        const uint64_t chunks = CollCeilDiv(prim->descriptor.count, per_chunk);
        for (uint64_t chunk = 0; chunk < chunks; ++chunk) {
            CollReduceOperand operand;
            operand.tree_id = prim->tree_id;
            operand.collective = prim->descriptor.key;
            operand.phase_id = 0; operand.chunk_id = chunk;
            operand.child_id = CENTER; operand.dtype = prim->descriptor.dtype;
            operand.op = prim->descriptor.reduce_op;
            operand.valid_elements = static_cast<uint16_t>(std::min<uint64_t>(
                per_chunk, prim->descriptor.count - chunk * per_chunk));
            for (uint16_t e = 0; e < operand.valid_elements; ++e)
                operand.payload.range((e + 1) * width - 1, e * width) =
                    static_cast<uint64_t>(prim->descriptor.self_rank + 1);
            const auto wire = SerializeCollReduceOperand(operand);
            send_raw(wire[0]); send_raw(wire[1]);
        }
        std::cout << "[COLL_V5_TX] tree=" << prim->tree_id
                  << " rank=" << prim->descriptor.self_rank
                  << " chunks=" << chunks << std::endl;
        return;
    }
    if (prim->mode == Collective_data_prim::Mode::REDUCE_RX) {
        const uint64_t width = CollDTypeBits(prim->descriptor.dtype);
        const uint64_t chunks = CollCeilDiv(prim->descriptor.count, 128 / width);
        for (uint64_t chunk = 0; chunk < chunks; ++chunk) {
            while (collective_reduce_buffer.size() < 2)
                wait(ev_collective_data);
            std::vector<sc_bv<256>> wire{
                collective_reduce_buffer.front()};
            collective_reduce_buffer.pop();
            wire.push_back(collective_reduce_buffer.front());
            collective_reduce_buffer.pop();
            const auto result = DeserializeCollReduceOperand(wire);
            if (result.tree_id != prim->tree_id ||
                !(result.collective == prim->descriptor.key) ||
                result.chunk_id != chunk)
                throw std::runtime_error("root reduce result identity mismatch");
            const uint64_t expected = prim->descriptor.reduce_op ==
                    CollReduceOp::SUM
                ? uint64_t(prim->descriptor.group.size()) *
                      (prim->descriptor.group.size() + 1) / 2
                : uint64_t(prim->descriptor.group.size());
            const uint64_t mask = width == 64 ? UINT64_MAX
                                               : (uint64_t(1) << width) - 1;
            for (uint16_t e = 0; e < result.valid_elements; ++e)
                if (result.payload.range((e + 1) * width - 1,
                                         e * width).to_uint64() !=
                    (expected & mask))
                    throw std::runtime_error(
                        "root reduce result value mismatch");
            ev_msg_process_end.notify(SC_ZERO_TIME);
        }
        std::cout << "[COLL_V5_RESULT] tree=" << prim->tree_id
                  << " chunks=" << chunks << " value=verified" << std::endl;
        return;
    }
    const uint64_t packets = CollCeilDiv(prim->descriptor.chunk_bits, 128);
    if (packets == 0 || packets > 0xffffffu)
        throw std::overflow_error("collective data packet count exceeds wire");
    if (prim->mode == Collective_data_prim::Mode::BROADCAST_TX) {
        for (uint64_t seq = 1; seq <= packets; ++seq) {
            CollDataHeader h;
            h.tree_id = prim->tree_id;
            h.packet.collective = prim->descriptor.key;
            h.packet.phase_id = 0;
            h.packet.chunk_id = 0;
            h.packet.src_rank = prim->descriptor.root_rank;
            h.packet.dst_rank = 0xffffu;
            h.seq_id = static_cast<uint32_t>(seq);
            const uint64_t remaining =
                prim->descriptor.chunk_bits - (seq - 1) * 128;
            h.length_bits = static_cast<uint8_t>(
                std::min<uint64_t>(128, remaining));
            h.is_end = seq == packets;
            send_raw(SerializeCollData(h));
        }
        std::cout << "[COLL_V4_TX] tree=" << prim->tree_id
                  << " packets=" << packets << std::endl;
        return;
    }
    uint64_t expected_seq = 1;
    while (expected_seq <= packets) {
        while (collective_data_buffer.empty()) wait(ev_collective_data);
        const auto h = DeserializeCollData(collective_data_buffer.front());
        if (h.tree_id != prim->tree_id ||
            !(h.packet.collective == prim->descriptor.key) ||
            h.seq_id != expected_seq)
            throw std::runtime_error(
                "collective RX packet identity/order mismatch");
        collective_data_buffer.pop();
        ev_msg_process_end.notify(SC_ZERO_TIME);
        if (h.is_end != (expected_seq == packets))
            throw std::runtime_error("collective RX tail mismatch");
        ++expected_seq;
    }
    std::cout << "[COLL_V4_RX] tree=" << prim->tree_id
              << " rank=" << prim->descriptor.self_rank
              << " packets=" << packets << std::endl;
}

void WorkerCoreExecutor::arm_multicast_receive(
    uint16_t tree_id, const CollectiveKey &key,
    uint32_t chunk_id, uint64_t destination_address) {
    if (tree_id == 0 || sram_access == nullptr || sram_regions == nullptr)
        throw std::runtime_error(
            "strict multicast receive requires a tree and real SRAM");
    (void)sram_regions->ResolveAbsolute(
        destination_address, 1, sram::Initiator::kNocRx,
        sram::Command::kWrite);
    const EndpointMulticastPostKey post_key{tree_id, key};
    if (!multicast_posts.emplace(
             post_key,
             EndpointMulticastPost{destination_address, chunk_id,
                                   std::nullopt, false})
             .second)
        throw std::invalid_argument(
            "duplicate strict multicast receive post");
}

void WorkerCoreExecutor::wait_multicast_receive(
    uint16_t tree_id, const CollectiveKey &key) {
    const EndpointMulticastPostKey post_key{tree_id, key};
    auto post = multicast_posts.find(post_key);
    if (post == multicast_posts.end())
        throw std::invalid_argument(
            "wait for unknown strict multicast receive post");
    while (!post->second.complete) wait(ev_multicast_progress);
    if (post->second.lock &&
        multicast_routes.count(*post->second.lock) != 0)
        throw std::logic_error(
            "completed strict multicast receive retained a route");
    multicast_posts.erase(post);
}

void WorkerCoreExecutor::send_multicast_bytes(
    uint16_t tree_id, uint16_t session_id, const CollectiveKey &key,
    uint64_t source_address, uint64_t length_bytes) {
    if (tree_id == 0 || session_id == 0 || length_bytes == 0 ||
        length_bytes > kDteEndpointP2pMaxBytes || sram_access == nullptr)
        throw std::invalid_argument(
            "strict multicast send descriptor is invalid");
    sram::Request read;
    read.initiator = sram::Initiator::kDte;
    read.command = sram::Command::kRead;
    read.address = source_address;
    read.size_bytes = length_bytes;
    const std::vector<uint8_t> bytes =
        sram_access->Access(read).payload;
    if (bytes.size() != length_bytes)
        throw std::runtime_error(
            "strict multicast SRAM read returned the wrong byte count");
    const IsaV1CollectiveByteBuiltStream stream =
        BuildIsaV1CollectiveByteStream(
            {IsaV1CollectiveByteKind::MULTICAST, tree_id, session_id, key},
            bytes);
    auto send_raw = [&](const sc_bv<256> &wire) {
        while (!channel_avail_i.read() ||
               !atomic_helper_lock(sc_time_stamp(), 3))
            wait(CYCLE, SC_NS);
        send_serialized_wire(wire, false);
    };
    send_raw(stream.start);
    for (const sc_bv<256> &wire : stream.data) send_raw(wire);
    std::cout << "[P7_MULTICAST_TX] tree=" << tree_id
              << " session=" << session_id
              << " bytes=" << length_bytes << std::endl;
}

void WorkerCoreExecutor::handle_multicast_start(
    const sc_bv<256> &wire) {
    const IsaV1CollectiveByteStart start =
        DeserializeIsaV1CollectiveByteStart(wire);
    if (start.kind != IsaV1CollectiveByteKind::MULTICAST)
        throw std::runtime_error(
            "Worker strict byte START is not multicast");
    const EndpointMulticastPostKey post_key{start.tree_id,
                                             start.collective};
    auto post = multicast_posts.find(post_key);
    if (post == multicast_posts.end() || post->second.lock ||
        post->second.complete)
        throw std::runtime_error(
            "strict multicast START arrived without one receive post");
    // Arm-time validates the first byte so malformed addresses fail before
    // they become visible protocol state. START supplies the authoritative
    // business length; validate the complete destination span before taking
    // a reassembly reservation or accepting any DATA.
    (void)sram_regions->ResolveAbsolute(
        post->second.destination_address, start.total_bytes,
        sram::Initiator::kNocRx, sram::Command::kWrite);
    const IsaV1CollectiveByteLock lock{
        start.tree_id, start.session_id, start.collective.epoch};
    multicast_byte_reassembler->Begin(wire);
    try {
        if (!multicast_routes.emplace(lock, post_key).second)
            throw std::runtime_error(
                "strict multicast route collision at Worker");
        post->second.lock = lock;
    } catch (...) {
        (void)multicast_byte_reassembler->Abort(lock);
        throw;
    }
    ev_multicast_progress.notify(SC_ZERO_TIME);
}

void WorkerCoreExecutor::handle_multicast_data(
    const sc_bv<256> &wire) {
    const IsaV1CollectiveByteData data =
        InspectIsaV1CollectiveByteDataWire(wire);
    auto route = multicast_routes.find(data.lock);
    if (route == multicast_routes.end())
        throw std::runtime_error(
            "strict multicast DATA arrived without Worker START state");
    const auto commit = multicast_byte_reassembler->Accept(wire);
    if (!commit) {
        ev_multicast_progress.notify(SC_ZERO_TIME);
        return;
    }
    auto post = multicast_posts.find(route->second);
    if (post == multicast_posts.end() || post->second.complete ||
        !post->second.lock || !(*post->second.lock == data.lock) ||
        commit->identity.tree_id != route->second.first ||
        !(commit->identity.collective == route->second.second))
        throw std::logic_error(
            "strict multicast commit lost its receive post");
    sram::Request write;
    write.initiator = sram::Initiator::kNocRx;
    write.command = sram::Command::kWrite;
    write.address = post->second.destination_address;
    write.size_bytes = commit->bytes.size();
    write.payload = commit->bytes;
    (void)sram_access->Access(write);
    if (!collective_acceleration_runtime_v1)
        throw std::logic_error(
            "strict multicast commit has no acceleration coordinator");
    collective_acceleration_runtime_v1->AcknowledgeMulticast(
        route->second.second, route->second.first,
        post->second.chunk_id, static_cast<uint16_t>(cid));
    post->second.complete = true;
    std::cout << "[P7_MULTICAST_COMMIT] tree="
              << route->second.first << " core=" << cid
              << " bytes=" << commit->bytes.size() << std::endl;
    multicast_routes.erase(route);
    ev_multicast_progress.notify(SC_ZERO_TIME);
    ev_msg_process_end.notify(SC_ZERO_TIME);
}

void WorkerCoreExecutor::handle_reduce_stream_header(
    const sc_bv<256> &wire) {
    const auto header = coll_refactor::DeserializeReduceStreamHeader(
        wire, SPEC_NOC_COLL_CONFIG.dca.vector_bits);
    auto session = reduce_stream_sessions.find(header.stream.tree_id);
    if (session == reduce_stream_sessions.end())
        throw std::runtime_error(
            "reduce stream result arrived before root RX arm");
    auto &state = session->second;
    if (state.header || state.complete)
        throw std::runtime_error(
            "duplicate reduce stream result header");
    if (!(header.stream.key.collective == state.descriptor.key) ||
        header.stream.key.phase_id != state.phase_id ||
        header.stream.key.stream_id != state.stream_id ||
        header.stream.dtype != state.descriptor.dtype ||
        header.stream.op != state.descriptor.reduce_op ||
        header.stream.total_elements != state.descriptor.count)
        throw std::runtime_error(
            "endpoint reduce stream header identity mismatch");
    state.header = header;
    state.assembler = std::make_unique<
        coll_refactor::ReduceStreamAssembler>(
            header,
            static_cast<size_t>(
                SPEC_NOC_COLL_CONFIG.dca.result_fifo_depth),
            128, SPEC_NOC_COLL_CONFIG.dca.vector_bits);
    if (!reduce_stream_routes.emplace(header.Route(),
                                      header.stream.tree_id).second)
        throw std::runtime_error(
            "endpoint reduce stream compact-route collision");
    ev_reduce_stream_progress.notify(SC_ZERO_TIME);
}

void WorkerCoreExecutor::handle_reduce_stream_data(
    const sc_bv<256> &wire) {
    const auto data = coll_refactor::DeserializeReduceStreamData(wire);
    auto route = reduce_stream_routes.find(data.route);
    if (route == reduce_stream_routes.end())
        throw std::runtime_error(
            "endpoint reduce data arrived without active result header");
    auto session = reduce_stream_sessions.find(route->second);
    if (session == reduce_stream_sessions.end() ||
        !session->second.assembler)
        throw std::logic_error(
            "endpoint reduce stream route lost its session");
    auto &state = session->second;
    const auto status = state.assembler->Accept(data);
    if (status == coll_refactor::ReduceStreamAcceptStatus::BACKPRESSURE)
        throw std::logic_error(
            "endpoint consumes every ready beat and must not backpressure");
    while (auto beat = state.assembler->PopBeat()) {
        state.values_seen += beat->geometry.lane_mask.valid_lanes;
        state.result_beats.push_back(std::move(*beat));
    }
    if (status == coll_refactor::
            ReduceStreamAcceptStatus::STREAM_COMPLETE) {
        state.assembler->Finish();
        if (state.assembler->Residual() != 0 ||
            state.values_seen != state.descriptor.count)
            throw std::runtime_error(
                "endpoint reduce stream completed with residual data");
        if (sram_access == nullptr || !state.header)
            throw std::runtime_error(
                "DCA stream result requires real SRAM and a header");
        const std::vector<uint8_t> bytes =
            coll_refactor::DecodeIsaV1DcaByteStream(
                *state.header, state.result_beats,
                SPEC_NOC_COLL_CONFIG.dca.vector_bits);
        sram::Request write;
        write.initiator = sram::Initiator::kNocRx;
        write.command = sram::Command::kWrite;
        write.address = state.descriptor.dst_addr;
        write.size_bytes = bytes.size();
        write.payload = bytes;
        (void)sram_access->Access(write);
        state.complete = true;
        reduce_stream_routes.erase(route);
        ev_reduce_stream_progress.notify(SC_ZERO_TIME);
    }
    ev_msg_process_end.notify(SC_ZERO_TIME);
}

void WorkerCoreExecutor::switch_prim_block() {
    while (true) {
        prim_block.write(true);
        wait();

        prim_block.write(false);
        wait(CYCLE, SC_NS);
    }
}

// 指令被 RECV_CONF发送过来后，会在本地核实例化对应的指令类
PrimBase *WorkerCoreExecutor::parse_prim(vector<sc_bv<128>> segments) {
    if (segments.empty())
        throw std::invalid_argument("cannot decode an empty Prim wire");

    int type = segments[0].range(7, 0).to_uint64();
    std::unique_ptr<PrimBase> task(
        PrimFactory::getInstance().createPrim(type, false, false));
    task->deserialize(segments);
    task->prim_context = core_context;

    PrimBase *decoded = task.release();
    g_prim_stash.push_back(decoded);
    return decoded;
}

// 数据信道接收
void WorkerCoreExecutor::poll_buffer_i() {
    MSG_TYPE block_mark = MSG_TYPE::MSG_TYPE_NUM;
    bool collective_blocked = false;
    bool endpoint_early_blocked = false;

    while (true) {
        if (endpoint_early_blocked) {
            if (p2p_endpoint->EarlyDataBytes() == 0 ||
                !p2p_endpoint->EarlyDataAtCapacity()) {
                endpoint_early_blocked = false;
                core_busy_o.write(false);
            } else {
                core_busy_o.write(true);
                wait(ev_p2p_progress);
                continue;
            }
        }
        if (!data_sent_i.read()) {
            if (collective_blocked) {
                if (collective_data_buffer.size() < MAX_BUFFER_PACKET_SIZE &&
                    collective_reduce_buffer.size() < MAX_BUFFER_PACKET_SIZE) {
                    collective_blocked = false;
                    core_busy_o.write(false);
                    continue;
                }
                wait(ev_msg_process_end);
                continue;
            } else if (block_mark < MSG_TYPE::MSG_TYPE_NUM) {
                if (msg_buffer_[block_mark].size() < MAX_BUFFER_PACKET_SIZE) {
                    block_mark = MSG_TYPE::MSG_TYPE_NUM;
                    core_busy_o.write(false);
                    continue;
                }

                wait(ev_msg_process_end);
                continue;
            } else {
                wait(ev_data_sent_i);
            }
        }

        const sc_bv<256> wire = channel_i.read();
        // Strict byte wires own a low-magic discriminator that is impossible
        // for valid legacy Msg. Classify them before every payload-sensitive
        // legacy/reduce decoder: their business payload occupies [255:128]
        // and may coincidentally contain any older high-bit magic.
        if (SPEC_NOC_COLL_CONFIG.UsesMulticast() &&
            IsIsaV1CollectiveByteStartWire(wire)) {
            handle_multicast_start(wire);
            core_busy_o.write(false);
            wait(CYCLE, SC_NS);
            continue;
        }
        if (SPEC_NOC_COLL_CONFIG.UsesMulticast() &&
            IsIsaV1CollectiveByteDataWire(wire)) {
            handle_multicast_data(wire);
            core_busy_o.write(false);
            wait(CYCLE, SC_NS);
            continue;
        }
        if (SPEC_NOC_COLL_CONFIG.UsesDcaOffload() &&
            coll_refactor::IsReduceStreamHeaderWire(wire)) {
            handle_reduce_stream_header(wire);
            core_busy_o.write(false);
            wait(CYCLE, SC_NS);
            continue;
        }
        if (SPEC_NOC_COLL_CONFIG.UsesDcaOffload() &&
            coll_refactor::IsReduceStreamDataWire(wire)) {
            handle_reduce_stream_data(wire);
            core_busy_o.write(false);
            wait(CYCLE, SC_NS);
            continue;
        }
        if (IsCollDataWire(wire)) {
            (void)DeserializeCollData(wire);
            collective_data_buffer.push(wire);
            ev_collective_data.notify(SC_ZERO_TIME);
            core_busy_o.write(collective_data_buffer.size() >=
                              MAX_BUFFER_PACKET_SIZE);
            collective_blocked = collective_data_buffer.size() >=
                                 MAX_BUFFER_PACKET_SIZE;
            wait(CYCLE, SC_NS);
            continue;
        }
        if (IsCollReduceHeaderWire(wire) ||
            IsCollReducePayloadWire(wire)) {
            collective_reduce_buffer.push(wire);
            ev_collective_data.notify(SC_ZERO_TIME);
            core_busy_o.write(collective_reduce_buffer.size() >=
                              MAX_BUFFER_PACKET_SIZE);
            collective_blocked = collective_reduce_buffer.size() >=
                                 MAX_BUFFER_PACKET_SIZE;
            wait(CYCLE, SC_NS);
            continue;
        }
        Msg m = DeserializeMsg(wire);
        if (IsP2pEndpointMessage(m)) {
            if (m.msg_type_ != MSG_TYPE::DATA)
                throw std::runtime_error(
                    "P2P endpoint control message arrived on data channel");
            const P2pFlowKey timing_flow{
                static_cast<uint16_t>(m.source_),
                static_cast<uint16_t>(m.des_),
                static_cast<uint16_t>(m.tag_id_),
                static_cast<uint8_t>(m.subflow_)};
            const bool declared = p2p_endpoint->HasInboundFlow(timing_flow);
            try {
                if (m.is_end_)
                    P2pSharedTimingSidebandRuntime::CompleteNetworkAtNs(
                        timing_flow, CurrentP2pNanoseconds(),
                        static_cast<uint64_t>(CYCLE));
                auto delivery = p2p_endpoint->ReceiveData(m);
                if (!declared && p2p_endpoint->EarlyDataBytes() != 0 &&
                    p2p_endpoint->EarlyDataAtCapacity()) {
                    endpoint_early_blocked = true;
                    core_busy_o.write(true);
                }
                if (delivery.has_value()) {
                    core_busy_o.write(true);
                    commit_p2p_delivery(std::move(*delivery));
                }
            } catch (...) {
                auto timing = p2p_rx_fsm_by_flow.find(timing_flow);
                const bool has_fsm = timing != p2p_rx_fsm_by_flow.end();
                const uint32_t failed_fsm =
                    has_fsm ? timing->second : uint32_t{0};
                // Runtime/reassembly/token state owns the failure transition.
                // Clear it before any Worker-side address/handle indexes.
                (void)p2p_endpoint->AbortInbound(timing_flow);
                if (has_fsm) {
                    p2p_rx_fsm_by_flow.erase(timing);
                    p2p_rx_addresses.erase(failed_fsm);
                    for (auto handle = p2p_async_handles.begin();
                         handle != p2p_async_handles.end();) {
                        if (handle->second.fsm_id == failed_fsm)
                            handle = p2p_async_handles.erase(handle);
                        else
                            ++handle;
                    }
                    if (P2pSharedTimingSidebandRuntime::Contains(timing_flow)) {
                        try {
                            P2pSharedTimingSidebandRuntime::Abort(
                                timing_flow, failed_fsm);
                        } catch (...) {
                        }
                    }
                } else {
                    (void)P2pSharedTimingSidebandRuntime::AbortFlow(
                        timing_flow);
                }
                endpoint_early_blocked = false;
                core_busy_o.write(false);
                ev_p2p_progress.notify(SC_ZERO_TIME);
                throw;
            }
            if (!endpoint_early_blocked)
                core_busy_o.write(false);
            ev_p2p_progress.notify(SC_ZERO_TIME);
            wait(CYCLE, SC_NS);
            continue;
        }
        if (m.msg_type_ == MSG_TYPE::EVENT)
            throw std::runtime_error(
                "EVENT control message arrived on the data channel");
        if (m.msg_type_ == MSG_TYPE::S_DATA)
            RecordStartDataStage(StartDataStage::DELIVERED, m);
        msg_buffer_[m.msg_type_].push(m);
        ev_recv_msg_type_[m.msg_type_].notify(0, SC_NS);

        if (IsBlockableMsgType(m.msg_type_) &&
            msg_buffer_[m.msg_type_].size() >= MAX_BUFFER_PACKET_SIZE) {
            core_busy_o.write(true);
            block_mark = m.msg_type_;
        } else
            core_busy_o.write(false);

        wait(CYCLE, SC_NS);
    }
}

// 控制信道接收
void WorkerCoreExecutor::poll_ctrl_buffer_i() {
    while (true) {
        if (!ctrl_sent_i.read()) {
            // 控制信道空闲时，检查当前所有控制消息类型的 buffer 是否有空间
            // 控制消息包括 REQUEST、ACK、DONE、EVENT。
            // 如果任何一个控制消息类型的 buffer 满了，就设置 busy
            if (msg_buffer_[MSG_TYPE::REQUEST].size() >= MAX_BUFFER_PACKET_SIZE ||
                msg_buffer_[MSG_TYPE::ACK].size() >= MAX_BUFFER_PACKET_SIZE ||
                msg_buffer_[MSG_TYPE::DONE].size() >= MAX_BUFFER_PACKET_SIZE ||
                p2p_pending_requests.size() >=
                    p2p_pending_request_capacity ||
                event_control_queue.Full()) {
                ctrl_core_busy_o.write(true);
            } else {
                ctrl_core_busy_o.write(false);
            }

            if (event_control_queue.Full())
                wait(ev_ctrl_sent_i | ev_event_queue_space);
            else if (p2p_pending_requests.size() >=
                     p2p_pending_request_capacity)
                wait(ev_ctrl_sent_i | ev_p2p_progress);
            else
                wait(ev_ctrl_sent_i);
            continue;
        }

        Msg m = DeserializeMsg(ctrl_channel_i.read());
        if (!m.IsControlMsg())
            throw std::runtime_error(
                "non-control message arrived on the control channel");
        if (IsP2pEndpointMessage(m)) {
            if (m.msg_type_ == MSG_TYPE::REQUEST) {
                const P2pPayloadDeclaration declaration =
                    ParseP2pPayloadRequest(m);
                CheckedAccumulate(p2p_stats.admission_requests_received, 1,
                                  "admission REQUEST received");
                const auto queued = std::find_if(
                    p2p_pending_requests.begin(),
                    p2p_pending_requests.end(),
                    [&](const P2pPendingRequest &entry) {
                        return entry.declaration.flow.source ==
                                   declaration.flow.source &&
                               entry.declaration.flow.transport_tag ==
                                   declaration.flow.transport_tag;
                    });
                const P2pPayloadDeclaration *queued_declaration =
                    queued == p2p_pending_requests.end()
                        ? nullptr
                        : &queued->declaration;
                const P2pRequestDisposition lifetime_disposition =
                    queued_declaration == nullptr
                        ? p2p_endpoint->ClassifyRequest(declaration)
                        : P2pRequestDisposition::NEW;
                const P2pRequestIngressAction ingress_action =
                    ClassifyP2pRequestIngress(
                        declaration, queued_declaration,
                        lifetime_disposition);
                if (ingress_action ==
                    P2pRequestIngressAction::REJECT_CONFLICT) {
                    CheckedAccumulate(
                        p2p_stats.request_conflicts_rejected, 1,
                        "REQUEST conflict rejected");
                    TraceP2pEndpoint(event_engine, cid,
                                     "P2P_request_conflict", "i",
                                     declaration.fsm_id,
                                     declaration.total_bytes,
                                     declaration.fragment_count,
                                     declaration.checksum);
                } else if (ingress_action ==
                           P2pRequestIngressAction::DUPLICATE) {
                    CheckedAccumulate(
                        p2p_stats.duplicate_requests_suppressed, 1,
                        "duplicate REQUEST suppressed");
                    TraceP2pEndpoint(event_engine, cid,
                                     "P2P_request_duplicate", "i",
                                     declaration.fsm_id,
                                     declaration.total_bytes,
                                     declaration.fragment_count,
                                     declaration.checksum);
                } else {
                    if (p2p_pending_requests.size() >=
                        p2p_pending_request_capacity)
                        throw std::length_error(
                            "P2P pending REQUEST exceeded global upper bound");
                    p2p_pending_requests.push_back(
                        P2pPendingRequest{m, declaration});
                    ev_p2p_request.notify(SC_ZERO_TIME);
                }
            } else if (m.msg_type_ == MSG_TYPE::ACK) {
                if (m.subflow_ == 1) {
                    const std::optional<P2pCompletionAck> abort_identity =
                        ExtractP2pAckIdentityForAbort(m, 1);
                    std::optional<P2pAdmissionAck> ack;
                    try {
                        ack = ParseP2pAdmissionAck(m);
                        p2p_endpoint->ReceiveAdmissionAck(m);
                        CheckedAccumulate(
                            p2p_stats.admission_acks_received, 1,
                            "admission ACK received");
                        TraceP2pEndpoint(event_engine, cid,
                                         "P2P_admission_ack", "i",
                                         ack->fsm_id, 0, 0, 0);
                    } catch (...) {
                        std::optional<P2pEndpointHandle> aborted;
                        if (abort_identity.has_value()) {
                            aborted =
                                p2p_endpoint->AbortAdmissionAckFlow(
                                    abort_identity->flow,
                                    abort_identity->fsm_id);
                            if (aborted.has_value()) {
                                for (auto handle = p2p_async_handles.begin();
                                     handle != p2p_async_handles.end();) {
                                    if (handle->second == *aborted)
                                        handle = p2p_async_handles.erase(handle);
                                    else
                                        ++handle;
                                }
                                (void)P2pSharedTimingSidebandRuntime::AbortFlow(
                                    abort_identity->flow);
                            }
                        }
                        ev_p2p_progress.notify(SC_ZERO_TIME);
                        RethrowP2pEndpointProtocolFailure(
                            "admission ACK parse/receive",
                            std::current_exception());
                    }
                } else {
                    const std::optional<P2pCompletionAck> abort_identity =
                        ExtractP2pAckIdentityForAbort(m, 0);
                    std::optional<P2pCompletionAck> ack;
                    try {
                        ack = ParseP2pCompletionAck(m);
                        p2p_endpoint->ReceiveAck(m);
                        CheckedAccumulate(
                            p2p_stats.completion_acks_received, 1,
                            "completion ACK received");
                        TraceP2pEndpoint(event_engine, cid,
                                         "P2P_completion_ack", "i",
                                         ack->fsm_id, 0, 0, 0);
                    } catch (...) {
                        std::optional<P2pEndpointHandle> aborted;
                        if (abort_identity.has_value()) {
                            aborted =
                                p2p_endpoint->AbortCompletionAckFlow(
                                    abort_identity->flow,
                                    abort_identity->fsm_id);
                            if (aborted.has_value()) {
                                for (auto handle = p2p_async_handles.begin();
                                     handle != p2p_async_handles.end();) {
                                    if (handle->second == *aborted)
                                        handle = p2p_async_handles.erase(handle);
                                    else
                                        ++handle;
                                }
                                (void)P2pSharedTimingSidebandRuntime::AbortFlow(
                                    abort_identity->flow);
                            }
                        }
                        ev_p2p_progress.notify(SC_ZERO_TIME);
                        RethrowP2pEndpointProtocolFailure(
                            "completion ACK parse/receive",
                            std::current_exception());
                    }
                }
            } else {
                throw std::runtime_error(
                    "P2P endpoint DATA arrived on the control channel");
            }
            ctrl_core_busy_o.write(
                p2p_pending_requests.size() >=
                p2p_pending_request_capacity);
            ev_p2p_progress.notify(SC_ZERO_TIME);
            wait(ctrl_sent_i.negedge_event());
            continue;
        }
        if (m.msg_type_ == MSG_TYPE::EVENT) {
            event_control_queue.Push(ParseEventControlMsg(m));
        } else {
            msg_buffer_[m.msg_type_].push(m);
        }
        ev_ctrl_msg_recv.notify(0, SC_NS);
        // 同时触发对应消息类型的event，兼容原有逻辑
        ev_recv_msg_type_[m.msg_type_].notify(0, SC_NS);

        // 检查所有控制消息类型的 buffer 是否满
        // 如果任何一个满了，就设置 busy，阻止接收更多消息
        if (msg_buffer_[MSG_TYPE::REQUEST].size() >= MAX_BUFFER_PACKET_SIZE ||
            msg_buffer_[MSG_TYPE::ACK].size() >= MAX_BUFFER_PACKET_SIZE ||
            msg_buffer_[MSG_TYPE::DONE].size() >= MAX_BUFFER_PACKET_SIZE ||
            p2p_pending_requests.size() >=
                p2p_pending_request_capacity ||
            event_control_queue.Full()) {
            ctrl_core_busy_o.write(true);
        } else {
            ctrl_core_busy_o.write(false);
        }

        wait(ctrl_sent_i.negedge_event());
    }
}

// 捕获控制信道空闲信号
void WorkerCoreExecutor::catch_ctrl_channel_avail_i() {
    while (true) {
        ev_ctrl_channel_avail_i.notify(CYCLE, SC_NS);
        wait();
    }
}

// 捕获控制信道发送信号
void WorkerCoreExecutor::catch_ctrl_sent_i() {
    while (true) {
        // Capture the asserted pulse in the same timestamp; Router supplies
        // an explicit low cycle before the next core-directed control wire.
        ev_ctrl_sent_i.notify(SC_ZERO_TIME);
        wait();
    }
}

/*
 在workercore executor中添加了一把锁，用于lock住write helper，
 因为同时运行send和recv原语会在同一个时钟周期内access write helper函数
*/

/*
send_helper_write >= 2, that data_sent_o = true send a msg to router
send_helper_write < 2, that data_sent_o = false, reset signal

present_time the most recent time that the helper is try to lock

try = present_time some one has try to lock the helper before in the same
time

status = 0, If a new cycle begins and no other module requires the helper,
reset send_helper_write to 0. pool down data_sent_o status = 1, 表示
准备执行send taskCoreDefault （会有delay） 一般在status 0 之后
同一个周期内，行为和 0 一致

status = 2 表示send 从 sram 里面已经拿到数据了，可以开始发送了

status = 1 2 都只出现一次



*/

bool WorkerCoreExecutor::atomic_helper_lock(sc_time try_time, int status,
                                            bool force) {
    bool res;

    if (try_time < present_time)
        res = false;

    if (try_time == present_time) {
        if (status == 0) {
            if (force == true) {
                send_helper_write = status;
                return true;
            }
            return false;
        }

        if (status == 1) { // send prepare
            // status 1 只会出现在这里
            if (send_helper_write == 0) {
                send_helper_write = 1;
                res = true;
            } else {
                res = false;
            }
        }
        if (status == 2) { // send ready
            if (send_helper_write == 1) {
                send_helper_write = 2;
                res = true;
            } else {
                res = false;
            }
        }
        if (status == 3) { // other pass cond.
            if (send_helper_write == 0) {
                send_helper_write = 3;
                res = true;
            } else {
                res = false;
            }
        }
    }

    if (try_time > present_time) {
        if (try_time - present_time < sc_time(CYCLE, SC_NS))
            return false;

        present_time = try_time;
        // status 1 不会进入到这里，因为status 1 之前肯定会有 status 0
        // 修改了 present_time
        if (status == 2) { // send ready
            if (send_helper_write == 1) {
                send_helper_write = 2;
                res = true;
            } else {
                res = false;
            }
        } else {
            // 这里应该只会进 status 0,
            // 1 和 3 ( 3 除了返回给host ack 因为while循环)
            // 都会在0后处理，且不会有延迟，所以pre=try_time
            // status 1 后面只能紧跟status 2
            // 防止status 3 把 原本 status 2 抢占 了
            if (send_helper_write == 1 && force == false)
                res = false;
            else {
                // 0 或者 3 是当前周期第一来的状态
                // 2 只会出现上面一种情况 成为当前周期第一来的状态

                send_helper_write = status;
                res = true;
            }
        }
    }

    return res;
}
// data_sent_o pos trigger router && later router can self trigger if
// data_sent_o is true 是否拉低不重要，只要 data_sent_o 是高就能发送
// 根据消息类型选择数据信道或控制信道
void WorkerCoreExecutor::send_helper() {
    while (true) {
        if (!serialized_wire_queue.Empty()) {
            const SerializedWireItem item = serialized_wire_queue.Front();
            const bool available = item.control
                ? ctrl_channel_avail_i.read() : channel_avail_i.read();
            if (!available) {
                wait(CYCLE, SC_NS);
                continue;
            }
            if (item.control) {
                ctrl_channel_o.write(item.wire);
                ctrl_sent_o.write(true);
                data_sent_o.write(false);
            } else {
                channel_o.write(item.wire);
                data_sent_o.write(true);
                ctrl_sent_o.write(false);
            }
            wait(CYCLE, SC_NS);
            data_sent_o.write(false);
            ctrl_sent_o.write(false);
            (void)serialized_wire_queue.CompleteFront();
            send_helper_write = 0;
            ev_serialized_wire_progress.notify(SC_ZERO_TIME);
            // Make the low phase observable before the next FIFO item can
            // raise the same signal. This is a wire-level protocol gap, not
            // an event-delivery convention.
            wait(CYCLE, SC_NS);
            continue;
        }
        // Status 3 exclusively reserves the helper for a value-owned
        // serialized FIFO item. A producer can acquire the reservation one
        // SystemC delta before it enqueues and notifies the item, so retain
        // that reservation until the queue owns the exact wire value.
        if (send_helper_write == 3) {
            data_sent_o.write(false);
            ctrl_sent_o.write(false);
            wait();
            continue;
        }
        if (send_helper_write != 0)
            throw std::logic_error(
                "send helper retained a legacy single-slot reservation");
        data_sent_o.write(false);
        ctrl_sent_o.write(false);

        wait();
    }
}

void WorkerCoreExecutor::catch_channel_avail_i() {
    while (true) {
        ev_channel_avail_i.notify(CYCLE, SC_NS);

        wait();
    }
}

void WorkerCoreExecutor::catch_data_sent_i() {
    while (true) {
        ev_data_sent_i.notify(CYCLE, SC_NS);

        wait();
    }
}

WorkerCoreExecutor::~WorkerCoreExecutor() {
    delete sram_addr;
#if USE_NB_DRAMSYS == 1
    delete nb_dcache_socket;
#else
    delete dcache_socket;
#endif
    delete mem_access_port;
    delete high_bw_mem_access_port;
    delete temp_mem_access_port;
    delete high_bw_temp_mem_access_port;
    delete start_nb_dram_event;
    delete end_nb_dram_event;
    delete start_sram_event;
    delete end_sram_event;
    delete start_global_mem_event;
    delete end_global_mem_event;
    delete sram_writer;
    // Monitor normally releases these entries because WorkerCore teardown is
    // intentionally deferred.  Keep the executor path safe for tests or a
    // future explicit WorkerCore teardown.
    if (g_dram_kvtable != nullptr && g_dram_kvtable[cid] != nullptr) {
        delete g_dram_kvtable[cid];
        g_dram_kvtable[cid] = nullptr;
    }
}
