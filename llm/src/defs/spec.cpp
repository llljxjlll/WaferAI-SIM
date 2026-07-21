#include "defs/spec.h"
#include "defs/enums.h"

SIM_MODE SYSTEM_MODE = SIM_DATAFLOW;

int GRID_X = 4;
int GRID_Y;
int GRID_SIZE;
int CORE_PER_SM;

// D2D / 多 die 维度（默认单 die；ParseHardwareConfig 会覆写）
int DIE_X = 1;
int DIE_Y = 1;
int DIE_COUNT = 1;
int CORES_PER_DIE = 0;
int TOTAL_CORES = 0;
int HOST_ENDPOINT_ID = 0;
int HOST_LANES = 0;

bool SPEC_USE_BEHA_NOC = true;
bool SPEC_USE_BEHA_SRAM = true;
bool SPEC_USE_BEHA_DRAM = true;
bool SPEC_USE_PERF_GEMM = false;
bool SPEC_KVCACHE_SPILL = false;
std::string SPEC_LOAD_STATIC = "layer";
std::string SPEC_TTF_FILE = "../font/NotoSansDisplay-Bold.ttf";
bool SPEC_USE_DRAMSYS = true;
bool SPEC_FAST_WARMUP = true;
bool SPEC_ROUTER_PIPE = false;
bool SPEC_SEND_RECV_PARALLEL = false;

int HW_CORE_CREDIT = 5;
int HW_PD_RATIO = 4;
int HW_DRAM_DEFAULT_BITWIDTH = 8;
int HW_SRAM_SIZE = 8388608;
int HW_NOC_PAYLOAD_PER_CYCLE = 1;
float HW_COMP_UTIL = 0.7;
float HW_BEHA_DRAM_UTIL = 0.7;

bool GPU_USE_INNER_MM = true;
int GPU_DRAM_BANDWIDTH = 512;
bool GPU_CACHE_LOG = false;
std::string GPU_DRAM_CONFIG_FILE = "../DRAMSys/configs/hbm3-example.json";
int GPU_DRAM_BURST_SIZE = 2048;
int GPU_DRAM_ALIGNED = 64;

int LOG_LEVEL = 1;