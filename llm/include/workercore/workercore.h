#pragma once
#include "systemc.h"

#include "common/memory.h"
#include "common/msg.h"
#include "common/pd.h"
#include "common/system.h"
#include "defs/const.h"
#include "dte/dte_async.h"
#include "dte/dte_unit.h"
#include "dte/coll_byte_wire_v1.h"
#include "dte/coll_reduce_stream.h"
#include "dte/p2p_session_runtime.h"
#include "workercore/moe_swizzle_runtime_capture.h"
#include "dte/sync_runtime.h"
#include "link/nb_global_memif_v2.h"
#include "macros/macros.h"
#include "memory/dram/Dcache.h"
#include "memory/dram/DummyDcache.h"
#include "memory/core_mem_adapter.h"
#include "memory/core_lsu_unit.h"
#include "memory/hbm_byte_transport.h"
#include "memory/sram/sram_access_unit.h"
#include "memory/sram/compute_timeline.h"
#include "memory/gpu/GPU_L1L2_Cache.h"
#include "memory/sram/dynamic_bandwidth_ram_row.h"
#include "memory/sram_writer.h"
#include "trace/Event_engine.h"
#include "unit_module/sram_manager/sram_manager.h"
#include "workercore/serialized_wire_queue.h"
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>

struct DteFlowPayloadRound {
    uint64_t payload_bits = 0;
    uint8_t subflow_mask = 0;
    int stripe_count = 1;
};

struct CollectiveEndpointResidualSnapshot {
    size_t collective_data = 0;
    size_t collective_reduce = 0;
    size_t stream_routes = 0;
    size_t stream_sessions = 0;
    size_t stream_assembler = 0;
    size_t multicast_posts = 0;
    size_t multicast_routes = 0;
    size_t serialized_wires = 0;
    size_t multicast_reassembly = 0;
    size_t core_vector_sessions = 0;
    size_t p2p = 0;

    size_t Total() const noexcept {
        return collective_data + collective_reduce + stream_routes +
               stream_sessions + stream_assembler + multicast_posts +
               multicast_routes + serialized_wires + multicast_reassembly +
               core_vector_sessions + p2p;
    }
};

class WorkerCoreExecutor;
class Group_sync_prim;
class Event_control_prim;
class Dte_send_endpoint_prim;
class Dte_recv_endpoint_prim;
class CollectiveExecutorV1;
class CollectiveWaveAdmissionCoordinatorV1;
class IsaV1CollectiveProgramImage;
class IsaV1CollectiveProfileProgramImage;
class IsaV1CollectiveTreeRegistryBridge;
class IsaV1CollectiveAccelerationRuntime;
struct IsaV1CollectiveAcceleratedTree;
struct IsaV1CollectivePlan;
struct IsaV1LoweredCollectiveAction;
struct CollectiveExecutorActionV1;

struct P2pEndpointProductionStats {
    uint64_t source_read_bytes = 0;
    uint64_t sram_source_read_bytes = 0;
    uint64_t hbm_source_read_bytes = 0;
    uint64_t wire_bytes = 0;
    uint64_t wire_fragments = 0;
    uint64_t noc_rx_write_bytes = 0;
    uint64_t tx_local_completions = 0;
    uint64_t rx_local_completions = 0;
    uint64_t admission_requests_sent = 0;
    uint64_t admission_requests_received = 0;
    uint64_t admission_acks_sent = 0;
    uint64_t admission_acks_received = 0;
    uint64_t duplicate_requests_suppressed = 0;
    uint64_t request_conflicts_rejected = 0;
    uint64_t request_aborts = 0;
    uint64_t completion_acks_sent = 0;
    uint64_t completion_acks_received = 0;
    uint32_t last_source_checksum = 0;
    uint32_t last_wire_checksum = 0;
    uint32_t last_destination_checksum = 0;
};

class WorkerCore : public sc_module {
public:
    int cid;
    Event_engine *event_engine;

    WorkerCoreExecutor *executor;
    DCache *dcache;
    CoreMemAdapter *hbm_adapter;
    // SystolicArray *systolic;
    // DynamicBandwidthRamRow<sc_bv<256>, column_num> ram_array("ram_array", 0,
    // bank_depth, 2, 1, port_num + column_num * high_bw_port_num, port_num,
    // high_bw_port_num, event_engine_test); DynamicBandwidthRamRow<sc_bv<128>,
    // 4> ram_array(sc_gen_unique_name("ram_array"), 0, BANK_DEPTH,
    // SIMU_READ_PORT, SIMU_WRITE_PORT, BANK_PORT_NUM + SRAM_BANKS,
    // BANK_PORT_NUM, BANK_HIGH_READ_PORT_NUM, event_engine);
    DynamicBandwidthRamRow<sc_bv<SRAM_BITWIDTH>, SRAM_BANKS> *ram_array;
    DynamicBandwidthRamRow<sc_bv<SRAM_BITWIDTH>, SRAM_BANKS> *temp_ram_array;
    std::unique_ptr<sram::RegionTable> sram_regions;
    std::unique_ptr<sram::Storage> sram_storage;
    std::unique_ptr<sram::AccessUnit> sram_access;
    std::unique_ptr<sram::ComputeTimeline> compute_timeline;
    std::unique_ptr<sram::HbmByteTransport> hbm_byte_transport;
    std::unique_ptr<sram::CoreLsuUnit> lsu_memory;
    std::unique_ptr<DteMemoryBridge> dte_memory_bridge;

    // HardwareTaskConfig *systolic_config;
    // HardwareTaskConfig *other_config;
    DummyDCache *dummy_dcache;

    sc_signal<bool> systolic_done;
    sc_signal<bool> systolic_start;

    SC_HAS_PROCESS(WorkerCore);
    WorkerCore(const sc_module_name &n, int s_cid, Event_engine *event_engine,
               string dram_config_name);
    ~WorkerCore();
};

class WorkerCoreExecutor : public sc_module {
public:
    shared_ptr<PrimCoreContext> core_context; // 存储元数据

    uint64_t MaxDramAddr; // 当前核最大的 dram 地址
    unsigned int defaultDataLength;
    CoreMemAdapter *hbm_adapter = nullptr;
    sram::CoreLsuUnit *lsu_memory = nullptr;
    sram::RegionTable *sram_regions = nullptr;
    sram::AccessUnit *sram_access = nullptr;
    sram::Storage *sram_storage = nullptr;
    sram::HbmByteTransport *hbm_byte_transport = nullptr;
    sram::ComputeTimeline *compute_timeline = nullptr;
    int cid;
    bool prim_refill;    // 是否通过原语重填的方式实现循环
    int loop_cnt;        // 如果开启prim_refill，表明现在是第几个循环
    int send_global_mem; // [yicheng] todo

    /* ------------------PRIM----------------------- */
    sc_signal<bool> prim_block; // 用于指示当前运行的原语是否正在执行

    // 原语执行完毕之后，利用该event将prim_block切换回unblock状态，使执行流继续
    sc_event ev_block;
    sc_event ev_send;
    sc_event ev_para_send;
    sc_event ev_recv;
    sc_event ev_comp;
    sc_event ev_send_helper;
    static constexpr size_t kSerializedWireQueueCapacity = 64;
    SerializedWireQueue serialized_wire_queue{kSerializedWireQueueCapacity};
    sc_event ev_serialized_wire_progress;
    sc_event ev_systolic;

    sc_event ev_recv_msg_type_
        [MSG_TYPE::MSG_TYPE_NUM]; // 使用统一数组存储接收数据包后触发的event
    queue<Msg> msg_buffer_[MSG_TYPE::MSG_TYPE_NUM]; // 使用统一数组存储数据包
    queue<sc_bv<256>> collective_data_buffer;
    queue<sc_bv<256>> collective_reduce_buffer;
    sc_event ev_collective_data;
    struct EndpointMulticastPost {
        uint64_t destination_address = 0;
        uint32_t chunk_id = 0;
        std::optional<IsaV1CollectiveByteLock> lock;
        bool complete = false;
    };
    using EndpointMulticastPostKey =
        std::pair<uint16_t, CollectiveKey>;
    std::unique_ptr<IsaV1CollectiveByteReassembler>
        multicast_byte_reassembler;
    std::map<EndpointMulticastPostKey, EndpointMulticastPost>
        multicast_posts;
    std::map<IsaV1CollectiveByteLock, EndpointMulticastPostKey>
        multicast_routes;
    sc_event ev_multicast_progress;
    struct EndpointReduceStreamSession {
        CollDescriptor descriptor;
        uint16_t tree_id = 0;
        uint16_t phase_id = 0;
        uint32_t stream_id = 0;
        std::optional<coll_refactor::ReduceStreamWireHeader> header;
        std::unique_ptr<coll_refactor::ReduceStreamAssembler> assembler;
        std::vector<coll_refactor::ReduceVectorBeat> result_beats;
        uint64_t values_seen = 0;
        bool complete = false;
    };
    std::map<uint16_t, EndpointReduceStreamSession>
        reduce_stream_sessions;
    std::map<coll_refactor::ReduceStreamRouteKey, uint16_t>
        reduce_stream_routes;
    sc_event ev_reduce_stream_progress;
    struct EndpointCoreVectorSession {
        std::vector<uint64_t> tags;
        size_t completed = 0;
    };
    std::map<uint16_t, EndpointCoreVectorSession> core_vector_sessions;

    std::shared_ptr<GroupSyncRuntime> group_sync_runtime;
    EventControlQueue event_control_queue;
    EventMailbox event_mailbox;
    sc_event ev_event_queue_space;

    sc_event ev_prim_recv_notice; // 当执行recv_data时触发

    sc_event
        ev_msg_process_end; // 当单个数据包处理结束之后触发，避免每个周期轮询

    bool send_done; // 并行策略：send和recv并行
    bool send_last_packet;
    sc_event
        ev_send_last_packet; // send和recv并行，只有在comp执行完毕之后，send才能发送最后一个数据包。

    bool comp_done;                    // 并行策略：comp和send并行
    deque<PrimBase *> prim_queue;      // 用于存储所有需要依次执行的原语
    queue<PrimBase *> send_para_queue; // 并行策略：send和recv并行


    /* ----------------SendHelper------------------- */
    sc_time present_time = sc_time(0, SC_NS);
    int send_helper_write; // 用于指示send
                           // helper是要向data_sent_o写入true还是false

    // 向router传递：是否可以向core传递信息（数据信道）
    sc_out<bool> core_busy_o;
    // 向router传递：控制信道是否可以向core传递信息
    sc_out<bool> ctrl_core_busy_o;

    // 传递数据的真正信道
    sc_in<sc_bv<256>> channel_i;
    sc_out<sc_bv<256>> channel_o;

    // 告知数据已经发送，通道使能信号
    sc_in<bool> data_sent_i;
    sc_event ev_data_sent_i;
    sc_out<bool> data_sent_o;

    // 通道未满的握手信号
    sc_in<bool> channel_avail_i;
    sc_event
        ev_channel_avail_i; // 当channel_avail_i的电平由低改为高，则触发这个event

    /* ---------------Control Channel------------------- */
    // 控制信道 - 用于传输 ACK/REQ/DONE 信号
    sc_in<sc_bv<256>> ctrl_channel_i;
    sc_out<sc_bv<256>> ctrl_channel_o;

    // 控制信道发送使能信号
    sc_in<bool> ctrl_sent_i;
    sc_event ev_ctrl_sent_i;
    sc_out<bool> ctrl_sent_o;

    // 控制信道空闲信号
    sc_in<bool> ctrl_channel_avail_i;
    sc_event ev_ctrl_channel_avail_i;

    sc_event ev_ctrl_msg_recv;  // 收到控制消息时触发
    /* ------------------------------------------------- */

    Event_engine *event_engine;
    std::unique_ptr<DTEUnit> dte;
    std::unique_ptr<DteAsyncTracker> dte_async;
    std::unique_ptr<P2pEndpointSessionRuntime> p2p_endpoint;
    struct P2pTxJob {
        P2pTxIssue issue;
        uint64_t source_first_ns = 0;
        uint64_t source_done_ns = 0;
    };
    struct P2pPendingRequest {
        Msg message;
        P2pPayloadDeclaration declaration;
    };
    std::deque<P2pTxJob> p2p_tx_queue;
    std::deque<P2pPendingRequest> p2p_pending_requests;
    size_t p2p_pending_request_capacity = 0;
    std::map<uint32_t, P2pEndpointHandle> p2p_async_handles;
    std::map<uint32_t, P2pEndpointHandle> collective_p2p_handles;
    std::map<uint32_t, uint64_t> p2p_rx_addresses;
    std::map<P2pFlowKey, uint32_t> p2p_rx_fsm_by_flow;
    sc_event ev_p2p_tx;
    sc_event ev_p2p_request;
    sc_event ev_p2p_progress;
    sc_event ev_collective_program;
    std::unique_ptr<CollectiveExecutorV1> collective_executor_v1;
    std::shared_ptr<const IsaV1CollectiveProgramImage>
        collective_program_image_v1;
    std::shared_ptr<const IsaV1CollectiveProfileProgramImage>
        collective_profile_program_image_v1;
    std::shared_ptr<IsaV1CollectiveTreeRegistryBridge>
        collective_tree_registry_bridge_v1;
    std::shared_ptr<IsaV1CollectiveAccelerationRuntime>
        collective_acceleration_runtime_v1;
    struct P7TreeWorkerState {
        std::set<uint32_t> multicast_posted;
        std::set<uint32_t> multicast_received;
        std::set<uint32_t> dca_root_armed;
        std::set<uint32_t> dca_sent;
        std::set<uint32_t> dca_complete;
        std::set<uint32_t> multicast_sent;
    };
    std::set<uint32_t> p7_active_plans;
    std::set<uint32_t> p7_observed_plans;
    std::map<uint32_t, std::set<uint32_t>> p7_ready_chunks;
    std::map<std::pair<uint32_t, uint16_t>, P7TreeWorkerState>
        p7_tree_states;
    sc_event ev_p7_acceleration;
    uint64_t collective_action_trace_base = 0;
    std::shared_ptr<CollectiveWaveAdmissionCoordinatorV1>
        collective_wave_coordinator_v1;
    P2pEndpointProductionStats p2p_stats;
    std::vector<MoeSwizzleRuntimeInterval> moe_swizzle_runtime_intervals;
    std::optional<uint64_t> moe_swizzle_pending_group_gemm_dispatch_start;
    std::vector<MoeSwizzleRuntimeIntervalKind>
        moe_swizzle_pending_fixed_interval_kinds;
    uint64_t moe_swizzle_pending_fixed_interval_start = 0;
    std::map<uint32_t, uint64_t> moe_swizzle_local_dte_starts;
    // REQUEST 可能跨迭代提前到达；每个 (source, tag) 按逻辑轮次排队，
    // 每轮用 subflow_mask 聚合 stripe 声明，RECV_DATA 每次只消费队首一轮。
    std::map<std::pair<int, int>, std::deque<DteFlowPayloadRound>>
        dte_flow_payload_rounds;

    NB_GlobalMemIF *nb_global_mem_socket = nullptr;

#if USE_NB_DRAMSYS == 1
    NB_DcacheIF *nb_dcache_socket = nullptr;
#else
    DcacheCore *dcache_socket = nullptr;
#endif
#if USE_L1L2_CACHE == 1
    L1Cache *core_lv1_cache = nullptr;
    // Processor *cache_processor;
    GPUNB_dcacheIF *gpunb_dcache_if = nullptr;
    GpuPosLocator *gpu_pos_locator = nullptr;
#else
#endif
    mem_access_unit *mem_access_port;
    high_bw_mem_access_unit *high_bw_mem_access_port;
    mem_access_unit *temp_mem_access_port;
    high_bw_mem_access_unit *high_bw_temp_mem_access_port;

    // sram相关
    int *sram_addr;                    // 用于记录当前sram可分配的起始地址
    sc_event *start_nb_dram_event;     // 用于启动非阻塞dram访存
    sc_event *end_nb_dram_event;       // 非阻塞sram访存结束标志
    sc_event *start_nb_gpu_dram_event; // 用于启动非阻塞gpu dram访存
    sc_event *end_nb_gpu_dram_event;   // 非阻塞gpu dram访存结束标志
    sc_event *start_global_mem_event;  // 用于启动global memory访存
    sc_event *end_global_mem_event;    // global memory访存结束标志
    sc_event *start_sram_event;
    sc_event *end_sram_event;
    SRAMWriteModule *sram_writer;

    SC_HAS_PROCESS(WorkerCoreExecutor);
    WorkerCoreExecutor(const sc_module_name &n, int s_cid,
                       Event_engine *event_engine);
    ~WorkerCoreExecutor();

    void init_global_mem();

    void catch_channel_avail_i();
    void catch_data_sent_i();
    
    // 控制信道相关方法
    void catch_ctrl_channel_avail_i();
    void catch_ctrl_sent_i();
    void poll_ctrl_buffer_i();    // 轮询控制信道输入
    void ctrl_send_helper();      // 控制信道发送辅助

    void worker_core_execute();
    void switch_prim_block();
    void poll_buffer_i(); // 每个时钟周期，将发送进core的数据包统一转移到input
                          // buffer中，实现发送和处理逻辑的解耦

    void send_logic();
    void send_para_logic();
    void recv_logic();
    void task_logic();
    void req_logic();
    void execute_dte_async(Dte_async_prim *prim);
    void execute_dte_send_endpoint(Dte_send_endpoint_prim *prim);
    void execute_dte_recv_endpoint(Dte_recv_endpoint_prim *prim);
    void p2p_tx_worker();
    void p2p_request_admission_worker();
    void collective_program_worker();
    void collective_acceleration_worker();
    void maybe_retire_collective_wave_image();
    void trace_collective_action_complete(
        const CollectiveExecutorActionV1 &action);
    bool send_p2p_message(Msg message, bool request_transition,
                          const P2pEndpointHandle *handle = nullptr);
    void send_serialized_wire(const sc_bv<256> &wire, bool control);
    void commit_p2p_delivery(P2pRxDelivery delivery);
    void execute_collective_data(Collective_data_prim *prim);
    void execute_group_sync(Group_sync_prim *prim);
    void execute_event_control(Event_control_prim *prim);
    void send_event_control(const EventControlMessage &message);
    void drain_event_control_queue();
    void handle_reduce_stream_header(const sc_bv<256> &wire);
    void handle_reduce_stream_data(const sc_bv<256> &wire);
    void handle_multicast_start(const sc_bv<256> &wire);
    void handle_multicast_data(const sc_bv<256> &wire);
    void arm_multicast_receive(uint16_t tree_id, const CollectiveKey &key,
                               uint32_t chunk_id,
                               uint64_t destination_address);
    void wait_multicast_receive(uint16_t tree_id, const CollectiveKey &key);
    void send_multicast_bytes(uint16_t tree_id, uint16_t session_id,
                              const CollectiveKey &key,
                              uint64_t source_address,
                              uint64_t length_bytes);
    void arm_dca_receive(const IsaV1CollectivePlan &plan,
                         const IsaV1CollectiveAcceleratedTree &tree,
                         uint32_t chunk_id);
    void send_dca_bytes(const IsaV1CollectivePlan &plan,
                        const IsaV1CollectiveAcceleratedTree &tree,
                        uint32_t chunk_id);
    void wait_dca_receive(const IsaV1CollectivePlan &plan,
                          const IsaV1CollectiveAcceleratedTree &tree,
                          uint32_t chunk_id);
    const IsaV1LoweredCollectiveAction &collective_lowered_action(
        uint32_t stream_index) const;
    bool execute_p7_action(const CollectiveExecutorActionV1 &action);

    void send_helper(); // 同时在send和recv中被调用
    void call_systolic_array();

    bool atomic_helper_lock(sc_time try_time, int status, bool force = false);

    PrimBase *parse_prim(vector<sc_bv<128>> buffer);

    void end_of_elaboration();

    void ConfigureCoreGroups(
        std::shared_ptr<const CoreGroupRegistry> registry) {
        group_sync_runtime =
            std::make_shared<GroupSyncRuntime>(std::move(registry));
    }

    void ConfigureCollectiveProgram(
        std::shared_ptr<const IsaV1CollectiveProgramImage> image,
        std::shared_ptr<CollectiveWaveAdmissionCoordinatorV1> coordinator,
        std::shared_ptr<const IsaV1CollectiveProfileProgramImage>
            profile_image = nullptr,
        std::shared_ptr<IsaV1CollectiveTreeRegistryBridge>
            tree_bridge = nullptr,
        std::shared_ptr<IsaV1CollectiveAccelerationRuntime>
            acceleration_runtime = nullptr);

    size_t CollectiveProgramResidual() const noexcept;

    size_t EventResidual() const noexcept {
        return event_control_queue.Residual() + event_mailbox.Residual();
    }

    size_t P2pEndpointResidual() const {
        if (!p2p_endpoint) return 0;
        const auto residual = p2p_endpoint->Residual();
        return residual.sessions + residual.allocated_transport_tags +
               residual.inbound_flows + residual.pending_requests +
               residual.inflight_reassemblies +
               residual.completed_unposted + residual.commit_ready +
               residual.completed_sessions +
               residual.tx_awaiting_admission + residual.tx_awaiting_ack +
               residual.tx_awaiting_local_retire +
               residual.reserved_rx_bytes + residual.early_data_flows +
               residual.early_data_fragments + residual.early_data_bytes +
               p2p_tx_queue.size() + p2p_pending_requests.size() +
               p2p_async_handles.size() + p2p_rx_addresses.size() +
               p2p_rx_fsm_by_flow.size();
    }

    CollectiveEndpointResidualSnapshot
    CollectiveEndpointResidualState() const {
        CollectiveEndpointResidualSnapshot result;
        result.collective_data = collective_data_buffer.size();
        result.collective_reduce = collective_reduce_buffer.size();
        result.stream_routes = reduce_stream_routes.size();
        result.stream_sessions = reduce_stream_sessions.size();
        for (const auto &entry : reduce_stream_sessions)
            result.stream_assembler += entry.second.assembler
                ? entry.second.assembler->Residual() : 0;
        result.multicast_posts = multicast_posts.size();
        result.multicast_routes = multicast_routes.size();
        result.serialized_wires = serialized_wire_queue.Size();
        result.multicast_reassembly = multicast_byte_reassembler
            ? multicast_byte_reassembler->InflightStreams() : 0;
        result.core_vector_sessions = core_vector_sessions.size();
        result.p2p = P2pEndpointResidual();
        return result;
    }

    size_t CollectiveEndpointResidual() const {
        return CollectiveEndpointResidualState().Total();
    }
    size_t DteOutstandingCount() const {
        return (dte_async ? dte_async->OutstandingCount() : 0) +
               (p2p_endpoint ? p2p_endpoint->Residual().async_tokens : 0);
    }

    const P2pEndpointProductionStats &P2pStats() const noexcept {
        return p2p_stats;
    }
    const std::vector<P2pEndpointLifetimeEvent> &
    P2pLifetimeEvents() const noexcept {
        static const std::vector<P2pEndpointLifetimeEvent> empty;
        return p2p_endpoint ? p2p_endpoint->LifetimeEvents() : empty;
    }
    bool P2pLifetimeEventsComplete() const noexcept {
        return !p2p_endpoint || p2p_endpoint->LifetimeEventsComplete();
    }
    size_t P2pMaxSessions() const noexcept {
        return p2p_endpoint ? p2p_endpoint->MaxSessions() : 0;
    }
    const std::vector<MoeSwizzleRuntimeInterval> &
    MoeSwizzleRuntimeIntervals() const noexcept {
        return moe_swizzle_runtime_intervals;
    }
    bool MoeSwizzleRuntimeIntervalsComplete() const noexcept {
        return moe_swizzle_local_dte_starts.empty() &&
               !moe_swizzle_pending_group_gemm_dispatch_start.has_value();
    }
    bool MoeSwizzleCalibrationIntervalsComplete() const noexcept {
        return MoeSwizzleRuntimeIntervalsComplete() &&
               moe_swizzle_pending_fixed_interval_kinds.empty();
    }
};
