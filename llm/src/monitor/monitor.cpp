#include "monitor/monitor.h"
#include "defs/global.h"
#include "die/d2d_link.h"
#include "die/port.h"
#include "monitor/config_helper_gpu.h"
#include "monitor/config_helper_gpu_pd.h"
#include "utils/system_utils.h"

// 独立统计 SystemC 层级中的 RouterUnit / WorkerCore 实例数（按类型 dynamic_cast，
// 不依赖自报计数），供多 die 实例化验收独立核对模块数量。
static void CountInstantiatedModules(const std::vector<sc_object *> &objs,
                                     int &routers, int &workers, int &links) {
    for (auto *o : objs) {
        if (dynamic_cast<RouterUnit *>(o))
            routers++;
        if (dynamic_cast<WorkerCore *>(o))
            workers++;
        if (dynamic_cast<D2DLinkUnit *>(o))
            links++;
        CountInstantiatedModules(o->get_child_objects(), routers, workers,
                                 links);
    }
}

Monitor::Monitor(const sc_module_name &n, Event_engine *event_engine,
                 const char *config_name)
    : sc_module(n), event_engine(event_engine), config_name(config_name) {
    memInterface =
        new MemInterface("mem-interface", this->event_engine, config_name);
    globalMemInterface = new GlobalMemInterface(
        "global-mem-interface", this->event_engine, config_name);

    init();
}

Monitor::Monitor(const sc_module_name &n, Event_engine *event_engine,
                 config_helper_base *input_config)
    : sc_module(n), event_engine(event_engine), config_name(nullptr) {
    memInterface =
        new MemInterface("mem-interface", this->event_engine, input_config);

    globalMemInterface = new GlobalMemInterface(
        "global-mem-interface", this->event_engine, input_config);
    // globalMemInterface = new GlobalMemInterface();
    init();
}

Monitor::~Monitor() {
    LOG_INFO(SYSTEM) << "Cleanup monitor components";
    delete[] core_busy;
    delete[] rc_channel;
    delete[] rc_data_sent;
    for (int i = 0; i < DIRECTIONS; i++) {
        delete[] channel[i];
        delete[] channel_avail[i];
        delete[] data_sent[i];
    }

    delete[] host_channel_avail;
    delete[] host_data_sent_i;
    delete[] host_data_sent_o;
    delete[] host_channel_i;
    delete[] host_channel_o;

    // 清理控制信道信号
    delete[] rc_ctrl_channel;
    delete[] rc_ctrl_sent;
    delete[] ctrl_core_busy;
    for (int i = 0; i < DIRECTIONS; i++) {
        delete[] ctrl_channel[i];
        delete[] ctrl_channel_avail[i];
        delete[] ctrl_sent[i];
    }


    delete[] host_ctrl_sent_i;
    delete[] host_ctrl_channel_i;

    delete routerMonitor;
    // WorkerCore 对象当前有意“泄漏”（不逐个析构）：其析构链存在既有 teardown 隐患
    // （SystemC 拆解顺序 / 成员释放），启用逐个 delete 会在退出时段错误（已实测）。
    // 故仅用 delete[] 释放指针数组本身；g_dram_kvtable 数组同理由 Monitor 用 delete[]
    // 释放（其元素随 Worker 一起泄漏，但不产生 double-free / use-after-free）。
    // 说明：不析构 Worker 时，扩容到 TOTAL_CORES 也不会触发 g_dram_kvtable 的多核 double-free。
    delete[] workerCores;
    delete[] g_dram_kvtable;
    delete memInterface;
}

void Monitor::init() {
    routerMonitor = new RouterMonitor("router-monitor", this->event_engine);
    workerCores = new WorkerCore *[TOTAL_CORES];
    g_dram_kvtable = new DramKVTable *[TOTAL_CORES];

    // globalMemInterface = new GlobalMemInterface();

    // //[yicheng] 初始化global memory
    // // chipGlobalMemory = new
    // ChipGlobalMemory(sc_gen_unique_name("chip-global-memory"),
    // "../DRAMSys/configs/ddr4-example.json", "../DRAMSys/configs");

    // // dcache = new DCache(sc_gen_unique_name("dcache"), (int)cid / GRID_X,
    // //                     (int)cid % GRID_X, this->event_engine,
    // //                     "../DRAMSys/configs/ddr4-example.json",
    // //                     "../DRAMSys/configs");

    // Initialize global memory interface with config parameters
    // globalMemInterface = new GlobalMemInterface(
    //     sc_gen_unique_name("global-mem-interface"), this->event_engine,
    //     config_name);

    for (int i = 0; i < TOTAL_CORES; i++) {
        workerCores[i] =
            new WorkerCore(sc_gen_unique_name("workercore"), i,
                           this->event_engine, GetCoreHWConfigForGlobal(i)->dram_config);
    }

    // 根据Config的设置连接到Globalmem
    assert(memInterface->has_global_mem.size() <= 1 &&
           "only allow one global mem");
    if (memInterface->has_global_mem.size() == 1) {
        for (auto i : memInterface->has_global_mem) {
            LOG_INFO(GLOBAL_MEMORY) << "Global link initialized " << i;
            // instantiate the NB_GlobalMemIF for this executor
            workerCores[i]->executor->init_global_mem();
            // bind the NB_GlobalMemIF initiator socket to the ChipGlobalMemory
            // target socket
            workerCores[i]->executor->nb_global_mem_socket->socket.bind(
                globalMemInterface->chipGlobalMemory->socket);
        }
    } else { // 如果谁都没有连接，直接绑定到第0个Core上
        LOG_INFO(GLOBAL_MEMORY) << "Global link not initialized";
        workerCores[0]->executor->init_global_mem();
        workerCores[0]->executor->nb_global_mem_socket->socket.bind(
            globalMemInterface->chipGlobalMemory->socket);
    }

#if USE_L1L2_CACHE == 1
    // GPU
    vector<L1Cache *> l1caches;
    vector<GPUNB_dcacheIF *> processors;
    for (int i = 0; i < TOTAL_CORES; i++) {
        l1caches.push_back(workerCores[i]->executor->core_lv1_cache);
        processors.push_back(workerCores[i]->executor->gpunb_dcache_if);
    }

    cacheSystem = new L1L2CacheSystem("l1l2-cache_system", TOTAL_CORES, l1caches,
                                      processors, GPU_DRAM_CONFIG_FILE,
                                      "../DRAMSys/configs");

    if (SYSTEM_MODE == SIM_GPU) {
        gpu_pos_locator = new GpuPosLocator();
        ((config_helper_gpu *)memInterface->config_helper)->gpu_pos_locator =
            gpu_pos_locator;
        for (int i = 0; i < TOTAL_CORES; i++) {
            workerCores[i]->executor->gpu_pos_locator = gpu_pos_locator;
            workerCores[i]->executor->core_context->gpu_pos_locator_ =
                gpu_pos_locator;
        }
    } else if (SYSTEM_MODE == SIM_GPU_PD) {
        gpu_pos_locator = new GpuPosLocator();
        ((config_helper_gpu_pd *)memInterface->config_helper)->gpu_pos_locator =
            gpu_pos_locator;
        for (int i = 0; i < TOTAL_CORES; i++) {
            workerCores[i]->executor->gpu_pos_locator = gpu_pos_locator;
            workerCores[i]->executor->core_context->gpu_pos_locator_ =
                gpu_pos_locator;
        }
    }
#endif

    memInterface->start_i(star);
    memInterface->preparations_done_o(config_done);
    preparations_done_i(config_done);
    start_o(star);

    // bind ports to signals
    core_busy = new sc_signal<bool>[TOTAL_CORES];
    rc_channel = new sc_signal<sc_bv<256>>[TOTAL_CORES];
    rc_data_sent = new sc_signal<bool>[TOTAL_CORES];

    host_channel_avail = new sc_signal<bool>[HOST_LANES];
    host_data_sent_i = new sc_signal<bool>[HOST_LANES];
    host_data_sent_o = new sc_signal<bool>[HOST_LANES];
    host_channel_i = new sc_signal<sc_bv<256>>[HOST_LANES];
    host_channel_o = new sc_signal<sc_bv<256>>[HOST_LANES];

    for (int i = 0; i < DIRECTIONS; i++) {
        channel[i] = new sc_signal<sc_bv<256>>[TOTAL_CORES];
        channel_avail[i] = new sc_signal<bool>[TOTAL_CORES];
        data_sent[i] = new sc_signal<bool>[TOTAL_CORES];
    }

    // 初始化控制信道信号
    rc_ctrl_channel = new sc_signal<sc_bv<256>>[TOTAL_CORES];
    rc_ctrl_sent = new sc_signal<bool>[TOTAL_CORES];
    ctrl_core_busy = new sc_signal<bool>[TOTAL_CORES];

   
    host_ctrl_sent_i = new sc_signal<bool>[HOST_LANES];
    host_ctrl_channel_i = new sc_signal<sc_bv<256>>[HOST_LANES];

    for (int i = 0; i < DIRECTIONS; i++) {
        ctrl_channel[i] = new sc_signal<sc_bv<256>>[TOTAL_CORES];
        ctrl_channel_avail[i] = new sc_signal<bool>[TOTAL_CORES];
        ctrl_sent[i] = new sc_signal<bool>[TOTAL_CORES];
    }

    // host & router —— HOST lane 经挂载表绑定到其 tile(router)。
    // legacy: lane i ↔ 全局行 i ↔ 西边缘 router i*GRID_X ↔ write_buffer[i]，逐位不变。
    for (int i = 0; i < HOST_LANES; i++) {
        int rid = HostTileOfLane(i); // lane i 绑定的全局 router id
        RouterUnit *ru = routerMonitor->routers[rid];

        // 数据信道连接
        memInterface->host_channel_avail_i[i](host_channel_avail[i]);
        (*ru->host_channel_avail_o)(host_channel_avail[i]);
        memInterface->host_data_sent_i[i](host_data_sent_i[i]);
        (*ru->host_data_sent_o)(host_data_sent_i[i]);
        memInterface->host_data_sent_o[i](host_data_sent_o[i]);
        (*ru->host_data_sent_i)(host_data_sent_o[i]);
        memInterface->host_channel_i[i](host_channel_i[i]);
        (*ru->host_channel_o)(host_channel_i[i]);
        memInterface->host_channel_o[i](host_channel_o[i]);
        (*ru->host_channel_i)(host_channel_o[i]);

        // 控制信道连接
        memInterface->host_ctrl_sent_i[i](host_ctrl_sent_i[i]);
        (*ru->host_ctrl_sent_o)(host_ctrl_sent_i[i]);
        memInterface->host_ctrl_channel_i[i](host_ctrl_channel_i[i]);
        (*ru->host_ctrl_channel_o)(host_ctrl_channel_i[i]);
    }

    // （2B1：die>0 西边缘 HOST 端口现由上面的 HOST_LANES 循环统一接入 MemInterface，
    //  不再终结。）

    // core & router
    for (int i = 0; i < TOTAL_CORES; i++) {
        RouterUnit *ru = routerMonitor->routers[i];
        WorkerCoreExecutor *wc = workerCores[i]->executor;

        // 数据信道 core busy 信号
        ru->core_busy_i(core_busy[i]);
        wc->core_busy_o(core_busy[i]);

        // 控制信道 core busy 信号（独立于数据信道）
        ru->ctrl_core_busy_i(ctrl_core_busy[i]);
        wc->ctrl_core_busy_o(ctrl_core_busy[i]);

        // 数据信道连接
        ru->channel_avail_o[CENTER](channel_avail[CENTER][i]);
        wc->channel_avail_i(channel_avail[CENTER][i]);
        ru->data_sent_o[CENTER](data_sent[CENTER][i]);
        wc->data_sent_i(data_sent[CENTER][i]);
        ru->data_sent_i[CENTER](rc_data_sent[i]);
        wc->data_sent_o(rc_data_sent[i]);

        ru->channel_o[CENTER](channel[CENTER][i]);
        wc->channel_i(channel[CENTER][i]);
        ru->channel_i[CENTER](rc_channel[i]);
        wc->channel_o(rc_channel[i]);

        // 控制信道连接
        ru->ctrl_channel_avail_o[CENTER](ctrl_channel_avail[CENTER][i]);
        wc->ctrl_channel_avail_i(ctrl_channel_avail[CENTER][i]);
        ru->ctrl_sent_o[CENTER](ctrl_sent[CENTER][i]);
        wc->ctrl_sent_i(ctrl_sent[CENTER][i]);
        ru->ctrl_sent_i[CENTER](rc_ctrl_sent[i]);
        wc->ctrl_sent_o(rc_ctrl_sent[i]);

        ru->ctrl_channel_o[CENTER](ctrl_channel[CENTER][i]);
        wc->ctrl_channel_i(ctrl_channel[CENTER][i]);
        ru->ctrl_channel_i[CENTER](rc_ctrl_channel[i]);
        wc->ctrl_channel_o(rc_ctrl_channel[i]);
    }

    // V1 runtime 前置配置校验（生产路径，非仅自测）：一旦存在 D2D link，就强制 V1 MVP 契约
    //（每方向 ≤1 个 C2C 端口、邻 die 方向恰好 1 个、link_bw==1）——在任何 deferred binding /
    // D2DLinkUnit 创建之前。非法配置在进入仿真前明确失败（异常 → 非零退出），不静默降级。
    if (!g_d2d_links.empty())
        ValidateV1MvpTopology();

    // router & router —— 开边 mesh（无 torus 环绕）。die 边缘方向无 die 内邻居，
    // 输入侧绑定共享终结通道（永不驱动），彻底移除运行时取模环绕连接。
    // 终结通道（只读、永不写）——多个边缘输入可共享。
    sc_signal<sc_bv<256>> *term_channel = new sc_signal<sc_bv<256>>;
    sc_signal<bool> *term_avail = new sc_signal<bool>;
    sc_signal<bool> *term_sent = new sc_signal<bool>;
    sc_signal<sc_bv<256>> *term_ctrl_channel = new sc_signal<sc_bv<256>>;
    sc_signal<bool> *term_ctrl_avail = new sc_signal<bool>;
    sc_signal<bool> *term_ctrl_sent = new sc_signal<bool>;

    int d2d_link_sites = 0; // V1-b：peer-connected C2C 出口边计数（b1 仍终结，b2 接 link）
    for (int j = 0; j < TOTAL_CORES; j++) {
        RouterUnit *pos = routerMonitor->routers[j];

        for (int i = 0; i < DIRECTIONS - 1; i++) {
            Directions input_dir = GetOpposeDirection(Directions(i));
            int nb = OpenMeshNeighbor(j, Directions(i)); // 开边邻居，-1=边缘
            // V1-b seam：die 边界(nb<0)且是 C2C 出口边 → 记为 D2D link site（b1 暂仍终结）
            bool c2c_edge = (nb < 0) && IsC2CEgressEdge(j, Directions(i));
            if (c2c_edge)
                d2d_link_sites++;

            // 输出侧始终绑定自身信号（边缘输出无人读取，无害）
            pos->channel_o[i](channel[i][j]);
            pos->channel_avail_o[i](channel_avail[i][j]);
            pos->data_sent_o[i](data_sent[i][j]);
            pos->ctrl_channel_o[i](ctrl_channel[i][j]);
            pos->ctrl_channel_avail_o[i](ctrl_channel_avail[i][j]);
            pos->ctrl_sent_o[i](ctrl_sent[i][j]);

            // 输入侧：有邻居→读邻居输出；C2C 出口边→延后由 D2D link pass 绑定；否则接终结通道。
            if (nb >= 0) {
                pos->channel_i[i](channel[input_dir][nb]);
                pos->channel_avail_i[i](channel_avail[input_dir][nb]);
                pos->data_sent_i[i](data_sent[input_dir][nb]);
                pos->ctrl_channel_i[i](ctrl_channel[input_dir][nb]);
                pos->ctrl_channel_avail_i[i](ctrl_channel_avail[input_dir][nb]);
                pos->ctrl_sent_i[i](ctrl_sent[input_dir][nb]);
            } else if (c2c_edge) {
                // 延后：channel_i / data_sent_i / channel_avail_i（+ctrl）由下面的 D2D link
                // pass 绑定到 link 单元输出。此处不绑定（避免与 link 重复绑定）。
            } else {
                pos->channel_i[i](*term_channel);
                pos->channel_avail_i[i](*term_avail);
                pos->data_sent_i[i](*term_sent);
                pos->ctrl_channel_i[i](*term_ctrl_channel);
                pos->ctrl_channel_avail_i[i](*term_ctrl_avail);
                pos->ctrl_sent_i[i](*term_ctrl_sent);
            }
        }
    }

    // ==================== V1-b2：D2D Link pass ====================
    // 对每条有向 D2D link（g_d2d_links 含 A→B 与 B→A 两条），插入一个 D2DLinkUnit（latency FIFO），
    // 取代 C2C 出口边的终结：link 读上游 A 的边缘输出、延迟后驱动下游 B 的边缘输入。
    // 两条有向 link 一起覆盖两端所有被延后的输入端口（每个恰绑定一次）。
    ResetD2DLinkStats();
    for (const auto &l : g_d2d_links) {
        const D2DPort &pa = g_die_ports.ports[l.local_port];
        const D2DPort &pb = g_die_ports.ports[l.remote_port];
        Directions SA = pa.side, SB = pb.side; // A 出口侧、B 接收侧（互反）
        int Ta = l.local_die * CORES_PER_DIE + pa.tile;
        int Tb = l.remote_die * CORES_PER_DIE + pb.tile;
        RouterUnit *A = routerMonitor->routers[Ta];
        RouterUnit *B = routerMonitor->routers[Tb];

        // link 驱动的 router 输入信号（新建，进程退出回收）
        auto *sA_avail = new sc_signal<bool>;
        auto *sA_ctrl_avail = new sc_signal<bool>;
        auto *sB_channel = new sc_signal<sc_bv<256>>;
        auto *sB_sent = new sc_signal<bool>;
        auto *sB_ctrl_channel = new sc_signal<sc_bv<256>>;
        auto *sB_ctrl_sent = new sc_signal<bool>;

        auto *link = new D2DLinkUnit(sc_gen_unique_name("d2d_link"), pa.latency);
        // 上游 A：读其边缘输出（channel/sent = channel[SA][Ta]），驱动其 avail 输入
        link->in_channel(channel[SA][Ta]);
        link->in_sent(data_sent[SA][Ta]);
        link->in_avail(*sA_avail);
        link->in_ctrl_channel(ctrl_channel[SA][Ta]);
        link->in_ctrl_sent(ctrl_sent[SA][Ta]);
        link->in_ctrl_avail(*sA_ctrl_avail);
        // 下游 B：驱动其边缘输入（channel/sent），读其 avail 输出（channel_avail[SB][Tb]）
        link->out_channel(*sB_channel);
        link->out_sent(*sB_sent);
        link->out_avail(channel_avail[SB][Tb]);
        link->out_ctrl_channel(*sB_ctrl_channel);
        link->out_ctrl_sent(*sB_ctrl_sent);
        link->out_ctrl_avail(ctrl_channel_avail[SB][Tb]);

        // 绑定被延后的 router 输入：A 的 avail_i[SA]、B 的 channel_i/sent_i[SB]（+ctrl）
        A->channel_avail_i[SA](*sA_avail);
        A->ctrl_channel_avail_i[SA](*sA_ctrl_avail);
        B->channel_i[SB](*sB_channel);
        B->data_sent_i[SB](*sB_sent);
        B->ctrl_channel_i[SB](*sB_ctrl_channel);
        B->ctrl_sent_i[SB](*sB_ctrl_sent);
    }

    // 独立层级计数（非自报）：routers/workers 应等于 TOTAL_CORES；links 为实例化的 D2DLinkUnit 数
    int hier_routers = 0, hier_workers = 0, hier_links = 0;
    CountInstantiatedModules(sc_get_top_level_objects(), hier_routers,
                             hier_workers, hier_links);
    LOG_INFO(SYSTEM) << "Instantiated cores=" << TOTAL_CORES
                     << " routers=" << hier_routers << " workers=" << hier_workers
                     << " dies=" << DIE_COUNT
                     << " (CORES_PER_DIE=" << CORES_PER_DIE << ", GRID=" << GRID_X
                     << "x" << GRID_Y << ", expect " << TOTAL_CORES << ")";
    // V1-b：peer-connected C2C 出口边数（每条相邻 die 双向 link = 2 个有向出口边）。
    // link_units 为**独立层级计数**的 D2DLinkUnit 实例数（非自报 g_d2d_links.size()）。
    // 单 die / 无 C2C 恒为 0。
    LOG_INFO(SYSTEM) << "[D2D] link_sites=" << d2d_link_sites
                     << " link_units=" << hier_links;
    LOG_INFO(SYSTEM) << "Components initialize complete, prepare to start.";

    SC_THREAD(start_simu);
}

void Monitor::start_simu() {
    // 开始分发配置
    // Msg t;
    // t.des = 0;
    // t.msg_type = DATA;
    // t.is_end = true;


    start_o.write(true);
    wait(preparations_done_i.posedge_event());
    // 开始发送数据
    // memInterface->clear_write_buffer();
    // memInterface->write_buffer[0].push(Msg(START, 0, 0));
    // memInterface->ev_write.notify(CYCLE, SC_NS);


    // Execute any global_interface primitives (e.g., Print_msg)
    // if (globalMemInterface) {
    //     globalMemInterface->execute_prims();
    // }
}