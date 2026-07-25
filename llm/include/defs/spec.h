#pragma once
#include "defs/enums.h"
#include <string>

// 模拟模式
extern SIM_MODE SYSTEM_MODE;

// 计算核心维度（die 内 mesh）
extern int GRID_X;
extern int GRID_Y;
extern int GRID_SIZE; // == CORES_PER_DIE，保留作单-die 兼容别名
extern int CORE_PER_SM;

// ---- D2D / 多 die 维度（V0 引入；单 die 时 DIE_COUNT=1，各量退化为旧 GRID_SIZE）----
extern int DIE_X;         // die 级 mesh 宽
extern int DIE_Y;         // die 级 mesh 高
extern int DIE_COUNT;     // = DIE_X * DIE_Y
extern int CORES_PER_DIE; // = GRID_X * GRID_Y（每 die 核数）
extern int TOTAL_CORES;   // = CORES_PER_DIE * DIE_COUNT（核级数组维度）
extern int HOST_ENDPOINT_ID; // = TOTAL_CORES，保留 endpoint 区段起点，替代所有 == GRID_SIZE 的 HOST 判定
extern int HOST_LANES;    // = GRID_Y * DIE_COUNT（所有 die 西边缘 HOST lane 总数；单 die 方阵 = GRID_X）

// simulation config中包含的参数
extern bool SPEC_USE_BEHA_NOC;        // 是否使用行为级NoC
extern bool SPEC_USE_BEHA_SRAM;       // 是否使用行为级SRAM
extern bool SPEC_USE_BEHA_DRAM;       // 是否使用行为级DRAM
extern bool SPEC_USE_PERF_GEMM;       // 是否使用GEMM性能模式
extern bool SPEC_KVCACHE_SPILL;       // 在SRAM满时，是否优先溢出kvcache
extern std::string SPEC_LOAD_STATIC; // 向SRAM加载数据时，加载的策略
extern std::string SPEC_TTF_FILE;     // 字体文件路径
extern bool SPEC_USE_DRAMSYS;         // 是否使用DRAMSys
extern bool SPEC_FAST_WARMUP;         // 是否跳过初始数据发送
extern bool SPEC_ROUTER_PIPE;         // 是否开启路由并行
extern bool SPEC_SEND_RECV_PARALLEL;  // 发送与接收原语是否同时进行
extern bool SPEC_USE_BEHA_DTE;        // 是否启用行为级 DTE（V1 接入 workload）
extern bool SPEC_DTE_STREAMING;       // V2b：端点 DTE 与网络数据阶段流水重叠
extern bool SPEC_DTE_ASYNC;           // V3a：显式 issue/wait/poll/fence/cancel
extern bool SPEC_DTE_AGGREGATION;     // V3b：在线 compound descriptor 聚合
extern bool SPEC_DTE_V4_RESOURCES;    // V4：双 command、独立 SPM/AXI 端口与 credit
extern int DTE_AGGREGATION_MAX_DESCRIPTORS;
extern int DTE_AGGREGATION_MAX_BYTES;
extern int DTE_AGGREGATION_TIMEOUT_NS;
extern int DTE_AGGREGATION_ADDRESS_BLOCK_BYTES;

// hardware config中包含的参数
extern int HW_CORE_CREDIT; //  在PD模式中，单核每一拍可安排的任务量
extern int HW_PD_RATIO;    // 在PD模式中，单次Prefill所占任务量（Decode默认为1）
extern int HW_DRAM_DEFAULT_BITWIDTH; // DRAM的默认位宽
extern int HW_SRAM_SIZE;             // SRAM的大小
extern int HW_NOC_PAYLOAD_PER_CYCLE; // 在1cycle内，NoC单信道可传输的数据包数量
extern float HW_COMP_UTIL;
extern float HW_BEHA_DRAM_UTIL;
extern int HW_DTE_GAMMA_NS;      // COMET 评估默认固定启动项，可配置
extern int HW_DTE_TAU_LAUNCH_NS; // COMET 评估默认 descriptor launch 项，可配置
extern int HW_DTE_COMMAND_SLOTS_PER_CHANNEL;
extern int HW_DTE_PENDING_QUEUE_DEPTH;
extern int HW_DTE_SPM_READ_WIDTH_BITS;
extern int HW_DTE_SPM_WRITE_WIDTH_BITS;
extern int HW_DTE_AXI_READ_WIDTH_BITS;
extern int HW_DTE_AXI_WRITE_WIDTH_BITS;
extern double HW_DTE_LAUNCH_ENERGY_PJ;
extern double HW_DTE_SPM_ENERGY_PJ_PER_BIT;
extern double HW_DTE_AXI_ENERGY_PJ_PER_BIT;
extern double HW_DTE_BASE_AREA_UM2;
extern double HW_DTE_CHANNEL_AREA_UM2;
extern double HW_DTE_COMMAND_SLOT_AREA_UM2;
extern double HW_DTE_PORT_BIT_AREA_UM2;

// gpu相关参数
extern bool GPU_USE_INNER_MM;
extern int GPU_DRAM_BANDWIDTH;
extern bool GPU_CACHE_LOG;
extern std::string GPU_DRAM_CONFIG_FILE;
extern int GPU_DRAM_BURST_SIZE;
extern int GPU_DRAM_ALIGNED;

// log相关参数
extern int LOG_LEVEL;