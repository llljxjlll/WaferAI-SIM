#pragma once
#include "defs/enums.h"
#include "defs/spec.h"

int GetInputSource(Directions dir, int pos);
bool IsMarginCore(int id);

// des 为 HOST 端点时，egress anchor 必须由 anchor_core 显式给出（消息原始 source core），
// 路由**不隐式用 pos** 推导挂载 tile——否则多非西侧 HOST 端口间会沿途重选 tile。
// 校验：anchor 为合法 core 且与目标 HOST 同 die。非 HOST 路由忽略 anchor_core。
Directions GetNextHop(int des, int pos, int anchor_core = -1);
Directions GetNextHopReverse(int des, int pos, int anchor_core = -1);

// die 级第一跳方向（die 级 XY，先 X 后 Y，与片内 XY 约定一致；NORTH=y+）。
// my_die==des_die 返回 CENTER。用于跨 die 路由选出口方向。
inline Directions DieFirstHopDir(int my_die, int des_die) {
    int mx = my_die % DIE_X, myy = my_die / DIE_X;
    int dx = des_die % DIE_X, dy = des_die / DIE_X;
    if (dx != mx)
        return dx > mx ? EAST : WEST;
    if (dy != myy)
        return dy > myy ? NORTH : SOUTH;
    return CENTER;
}
// die 级 Manhattan 距离（相邻=1；V1 只支持相邻，>1 由 V1-c preflight 拒绝）。
inline int DieManhattan(int a, int b) {
    int ax = a % DIE_X, ay = a / DIE_X, bx = b % DIE_X, by = b / DIE_X;
    return (ax > bx ? ax - bx : bx - ax) + (ay > by ? ay - by : by - ay);
}

// 跨 die 路由，拆成两步以坐实 V1「每 die 选一次出口、离开前不重选」（per-flow pinning）：
//   1) CrossDieSelectExit：进入一个 die 时（在 src 核 / 中间 die 入口 tile）**选一次**出口 C2C
//      端口并钉死。die 级首跳方向 D，查 port_for[local][D]，完整校验（范围/ROLE_C2C/dir==D/tile）。
//      同 die 目的返回 -1（无需出口）；无该方向 C2C（邻 die 不可达）抛 std::runtime_error。
//   2) CrossDieStep：携**固定** exit_port 逐跳收敛（不再按 pos 重查）——同 die 目的→片内 XY；
//      否则朝 exit_port 的端口 tile 走，pos 到该 tile 返回 egress 方向（side==dir）。
// 纯函数，不改运行时。非法 core id 抛错。
int CrossDieSelectExit(int at_core, int des_global);
Directions CrossDieStep(int des_global, int pos, int exit_port);
Directions GetOpposeDirection(Directions dir);

// ---- 全局核编址 helper（V0）：global_core_id <-> (die_id, local_id) ----
// 约定：global_id = die_id * CORES_PER_DIE + local_id
//       die_id    = die_y * DIE_X + die_x
//       local_id  = ly * GRID_X + lx   （die 内 row-major，GRID_X 列）
inline int GlobalId(int die_id, int local_id) {
    return die_id * CORES_PER_DIE + local_id;
}
inline int DieOfGlobal(int gid) { return gid / CORES_PER_DIE; }
inline int LocalOfGlobal(int gid) { return gid % CORES_PER_DIE; }
inline int DieXOf(int die_id) { return die_id % DIE_X; }
inline int DieYOf(int die_id) { return die_id / DIE_X; }
inline int LocalXOf(int local_id) { return local_id % GRID_X; }
inline int LocalYOf(int local_id) { return local_id / GRID_X; }

// die 内边界判断：core 是否位于给定方向的 die 边缘（用于开边 mesh / D2D 端口挂载）
// 坐标约定与现有一致：EAST=x+，NORTH=y+；故 S=y0、N=y(GRID_Y-1)、W=x0、E=x(GRID_X-1)
bool IsDieEdge(int local_id, Directions dir);

// 开边 mesh 邻居（无 torus 环绕）：从全局核 global 沿 dir 的 die 内邻居全局 id；
// 若 global 处于该方向的 die 边缘（无 die 内邻居）返回 -1（=无连接，绑定终结通道）。
// 不跨 die（die 间连接由 D2D 端口负责，V1+）。
int OpenMeshNeighbor(int global_core, Directions dir);

// 端点类型（V0）：解析 des_/source_ 落在核区还是保留 endpoint 区
enum EndpointType { EP_CORE = 0, EP_HOST, EP_MEM, EP_INVALID };
EndpointType DecodeEndpointType(int endpoint_id);

// 保留 endpoint 区段：host 区 [TOTAL_CORES, +DIE_COUNT)，mem 区其后 DIE_COUNT 个。
// 每 die 一个 host / mem 端点；die0 的 host == HOST_ENDPOINT_ID（== 旧 GRID_SIZE，单 die）。
inline int HostEndpointOfDie(int die_id) { return TOTAL_CORES + die_id; }
inline int MemEndpointOfDie(int die_id) { return TOTAL_CORES + DIE_COUNT + die_id; }
inline int DieOfHostEndpoint(int ep) { return ep - TOTAL_CORES; }
// 是否落在 HOST endpoint 区间 [TOTAL_CORES, TOTAL_CORES+DIE_COUNT)（每 die 一个）。
// 单 die 时该区间只含 HOST_ENDPOINT_ID，与旧 == 判断逐位等价。
inline bool IsHostEndpoint(int ep) {
    return ep >= TOTAL_CORES && ep < TOTAL_CORES + DIE_COUNT;
}