#include "utils/router_utils.h"
#include "defs/global.h"
#include "defs/spec.h"
#include "die/port.h"
#include "macros/macros.h"
#include <queue>
#include <fstream>
#include <stdexcept>

int GetInputSource(Directions dir, int pos) {
    int x = pos % GRID_X;
    int y = pos / GRID_X;

    switch (dir) {
    case WEST:
        x = (x - 1 + GRID_X) % GRID_X;
        break;
    case NORTH:
        y = (y + 1) % GRID_Y; // 修正：Y 轴用 GRID_Y（方阵时 GRID_Y==GRID_X，行为不变）
        break;
    case EAST:
        x = (x + 1) % GRID_X;
        break;
    case SOUTH:
        y = (y - 1 + GRID_Y) % GRID_Y; // 修正：Y 轴用 GRID_Y
        break;
    default:
        return 0;
    }

    return y * GRID_X + x;
}

bool IsMarginCore(int id) { return id % GRID_X == 0; }

// die 内边界：core 是否在给定方向的 die 边缘（EAST=x+，NORTH=y+）
bool IsDieEdge(int local_id, Directions dir) {
    int x = local_id % GRID_X;
    int y = local_id / GRID_X;
    switch (dir) {
    case WEST:
        return x == 0;
    case EAST:
        return x == GRID_X - 1;
    case SOUTH:
        return y == 0;
    case NORTH:
        return y == GRID_Y - 1;
    default:
        return false;
    }
}

// 开边 mesh 邻居（无环绕、不跨 die）；边缘返回 -1。
int OpenMeshNeighbor(int global_core, Directions dir) {
    int die = global_core / CORES_PER_DIE;
    int local = global_core % CORES_PER_DIE;
    if (IsDieEdge(local, dir))
        return -1; // die 边缘，无 die 内邻居（不 wrap、不跨 die）
    int lx = local % GRID_X, ly = local / GRID_X;
    switch (dir) {
    case WEST:
        lx -= 1;
        break;
    case EAST:
        lx += 1;
        break;
    case SOUTH:
        ly -= 1;
        break;
    case NORTH:
        ly += 1;
        break;
    default:
        return -1;
    }
    return die * CORES_PER_DIE + (ly * GRID_X + lx);
}

// endpoint 地址空间（V0 定义，provisional）：
//   core : [0, TOTAL_CORES)
//   host : [TOTAL_CORES, TOTAL_CORES + DIE_COUNT)          每 die 一个 host 端点
//   mem  : [TOTAL_CORES + DIE_COUNT, TOTAL_CORES + 2*DIE_COUNT)  每 die 预留一个 mem 端点
// HOST_ENDPOINT_ID == TOTAL_CORES 为 host 区基址（die0 / 全局 host）。
EndpointType DecodeEndpointType(int endpoint_id) {
    if (endpoint_id < 0)
        return EP_INVALID;
    if (endpoint_id < TOTAL_CORES)
        return EP_CORE;
    int host_base = TOTAL_CORES;
    int mem_base = TOTAL_CORES + DIE_COUNT;
    int mem_end = TOTAL_CORES + 2 * DIE_COUNT;
    if (endpoint_id >= host_base && endpoint_id < mem_base)
        return EP_HOST;
    if (endpoint_id >= mem_base && endpoint_id < mem_end)
        return EP_MEM;
    return EP_INVALID;
}

// HOST 目的的挂载 tile 解析：egress anchor 取自 anchor_core（消息 source core），
// 不用 pos。校验 anchor 合法且与目标 HOST 同 die；跨 die HOST 拒绝（需 D2D，未实现）。
// 返回：pos 已到挂载 tile 时置 emit_host=true；否则把 des 改写为挂载 tile 供 XY 收敛。
static void ResolveHostAnchor(int &des, int pos, int anchor_core,
                              bool &emit_host) {
    emit_host = false;
    if (anchor_core < 0 || anchor_core >= TOTAL_CORES)
        throw std::runtime_error(
            "HOST routing needs a legal anchor core (message source)");
    if (DieOfGlobal(anchor_core) != DieOfHostEndpoint(des))
        throw std::runtime_error(
            "cross-die HOST: anchor core die != HOST endpoint die");
    if (DieOfHostEndpoint(des) != DieOfGlobal(pos))
        throw std::runtime_error(
            "cross-die HOST endpoint requires D2D routing (unsupported)");
    int lane = HostLaneOfCore(anchor_core);
    int tile = HostTileOfLane(lane);
    if (lane < 0 || tile < 0)
        throw std::runtime_error(
            "HOST anchor core has no valid attach lane/tile (illegal mapping)");
    if (pos == tile)
        emit_host = true;
    else
        des = tile; // 后续 XY 朝 anchor 的挂载 tile 收敛
}

Directions GetNextHop(int des, int pos, int anchor_core) {
    // 从pos发往des的下一个方向, 先X后Y
    if (IsHostEndpoint(des)) {
        bool emit_host = false;
        ResolveHostAnchor(des, pos, anchor_core, emit_host);
        if (emit_host)
            return HOST;
    }

    int dx = des % GRID_X;
    int dy = des / GRID_X;

    int xx = pos % GRID_X;
    int yy = pos / GRID_X;

    if (dx == xx && dy == yy)
        return CENTER;
    else if (dx != xx) {
        if (dx > xx) {
            return EAST;
        } else {
            return WEST;
        }
    } else {
        if (dy > yy) {
            return NORTH;
        } else {
            return SOUTH;
        }
    }
}

Directions GetNextHopReverse(int des, int pos, int anchor_core) {
    // 从pos发往des的下一个方向, 先Y后X
    // HOST 目的：同 GetNextHop——朝 anchor(消息 source) 的 HOST 挂载 tile 收敛。
    if (IsHostEndpoint(des)) {
        bool emit_host = false;
        ResolveHostAnchor(des, pos, anchor_core, emit_host);
        if (emit_host)
            return HOST;
    }

    int dx = des % GRID_X;
    int dy = des / GRID_X;

    int xx = pos % GRID_X;
    int yy = pos / GRID_X;

    if (dx == xx && dy == yy)
        return CENTER;
    else if (dy != yy) {
        if (dy > yy) {
            return NORTH;
        } else {
            return SOUTH;
        }
    } else {
        if (dx > xx) {
            return EAST;
        } else {
            return WEST;
        }
    }
}

Directions GetOpposeDirection(Directions dir) {
    switch (dir) {
    case WEST:
        return EAST;
    case EAST:
        return WEST;
    case NORTH:
        return SOUTH;
    case SOUTH:
        return NORTH;
    default:
        return CENTER;
    }
}