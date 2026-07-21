#include "monitor/monitor.h"
#include "defs/global.h"
#include "die/port.h"
#include "monitor/config_helper_gpu.h"
#include "monitor/config_helper_gpu_pd.h"
#include "utils/system_utils.h"

// 独立统计 SystemC 层级中的 RouterUnit / WorkerCore 实例数（按类型 dynamic_cast，
// 不依赖自报计数），供多 die 实例化验收独立核对模块数量。
static void CountInstantiatedModules(const std::vector<sc_object *> &objs,
                                     int &routers, int &workers) {
    for (auto *o : objs) {
        if (dynamic_cast<RouterUnit *>(o))
            routers++;
        if (dynamic_cast<WorkerCore *>(o))
            workers++;
        CountInstantiatedModules(o->get_child_objects(), routers, workers);
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

    // router & router —— 开边 mesh（无 torus 环绕）。die 边缘方向无 die 内邻居，
    // 输入侧绑定共享终结通道（永不驱动），彻底移除运行时取模环绕连接。
    // 终结通道（只读、永不写）——多个边缘输入可共享。
    sc_signal<sc_bv<256>> *term_channel = new sc_signal<sc_bv<256>>;
    sc_signal<bool> *term_avail = new sc_signal<bool>;
    sc_signal<bool> *term_sent = new sc_signal<bool>;
    sc_signal<sc_bv<256>> *term_ctrl_channel = new sc_signal<sc_bv<256>>;
    sc_signal<bool> *term_ctrl_avail = new sc_signal<bool>;
    sc_signal<bool> *term_ctrl_sent = new sc_signal<bool>;

    for (int j = 0; j < TOTAL_CORES; j++) {
        RouterUnit *pos = routerMonitor->routers[j];

        for (int i = 0; i < DIRECTIONS - 1; i++) {
            Directions input_dir = GetOpposeDirection(Directions(i));
            int nb = OpenMeshNeighbor(j, Directions(i)); // 开边邻居，-1=边缘

            // 输出侧始终绑定自身信号（边缘输出无人读取，无害）
            pos->channel_o[i](channel[i][j]);
            pos->channel_avail_o[i](channel_avail[i][j]);
            pos->data_sent_o[i](data_sent[i][j]);
            pos->ctrl_channel_o[i](ctrl_channel[i][j]);
            pos->ctrl_channel_avail_o[i](ctrl_channel_avail[i][j]);
            pos->ctrl_sent_o[i](ctrl_sent[i][j]);

            // 输入侧：有邻居则读邻居输出，边缘则接终结通道（不再环绕）
            if (nb >= 0) {
                pos->channel_i[i](channel[input_dir][nb]);
                pos->channel_avail_i[i](channel_avail[input_dir][nb]);
                pos->data_sent_i[i](data_sent[input_dir][nb]);
                pos->ctrl_channel_i[i](ctrl_channel[input_dir][nb]);
                pos->ctrl_channel_avail_i[i](ctrl_channel_avail[input_dir][nb]);
                pos->ctrl_sent_i[i](ctrl_sent[input_dir][nb]);
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

    // 独立层级计数（非自报）：应等于 TOTAL_CORES
    int hier_routers = 0, hier_workers = 0;
    CountInstantiatedModules(sc_get_top_level_objects(), hier_routers,
                             hier_workers);
    LOG_INFO(SYSTEM) << "Instantiated cores=" << TOTAL_CORES
                     << " routers=" << hier_routers << " workers=" << hier_workers
                     << " dies=" << DIE_COUNT
                     << " (CORES_PER_DIE=" << CORES_PER_DIE << ", GRID=" << GRID_X
                     << "x" << GRID_Y << ", expect " << TOTAL_CORES << ")";
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