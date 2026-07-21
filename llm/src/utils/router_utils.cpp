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

// 校验携带的 pinned exit_port 对「从 md 去 dd」仍是合法 C2C 出口——不信任 flow 状态里带来的值：
//   范围合法、ROLE_C2C（非 HOST/MEM 冒充）、dir == die 首跳方向、MVP 下 side==dir、tile 合法。
// 返回该端口引用，或抛 std::runtime_error。
static const D2DPort &ValidatePinnedExit(int exit_port, int md, int dd) {
    if (exit_port < 0 || exit_port >= (int)g_die_ports.ports.size())
        throw std::runtime_error("cross-die: pinned exit port out of range");
    const D2DPort &p = g_die_ports.ports[exit_port];
    Directions D = DieFirstHopDir(md, dd);
    if (p.role != ROLE_C2C)
        throw std::runtime_error(
            "cross-die: pinned exit is not a C2C port (HOST/MEM impersonation)");
    if (p.dir != D)
        throw std::runtime_error(
            "cross-die: pinned exit dir != die first-hop direction");
    if (p.side != p.dir) // MVP：side==dir
        throw std::runtime_error("cross-die: pinned exit violates MVP side==dir");
    if (p.tile < 0 || p.tile >= CORES_PER_DIE)
        throw std::runtime_error("cross-die: pinned exit tile invalid");
    return p;
}

int CrossDieSelectExit(int at_core, int des_global) {
    if (at_core < 0 || at_core >= TOTAL_CORES || des_global < 0 ||
        des_global >= TOTAL_CORES)
        throw std::runtime_error("cross-die: illegal core id");
    int md = DieOfGlobal(at_core), dd = DieOfGlobal(des_global);
    if (md == dd)
        return -1; // 同 die，无需出口端口
    Directions D = DieFirstHopDir(md, dd); // die 级首跳方向
    int port = PortForDir(LocalOfGlobal(at_core), D);
    if (port < 0)
        throw std::runtime_error(
            "cross-die routing: no C2C port toward die-direction (unreachable)");
    ValidatePinnedExit(port, md, dd); // 完整校验（role/dir/side/tile）
    return port;
}

Directions CrossDieStep(int des_global, int pos, int exit_port) {
    if (pos < 0 || pos >= TOTAL_CORES || des_global < 0 ||
        des_global >= TOTAL_CORES)
        throw std::runtime_error("cross-die: illegal core id");
    int md = DieOfGlobal(pos), dd = DieOfGlobal(des_global);
    if (md == dd)
        return GetNextHop(des_global, pos); // 同 die：片内 XY 到目的核
    // 不信任携带的 exit_port：完整校验它对「从 md 去 dd」仍合法
    const D2DPort &p = ValidatePinnedExit(exit_port, md, dd);
    int port_tile = md * CORES_PER_DIE + p.tile; // 本 die 的固定出口端口 tile
    if (pos == port_tile)
        return p.dir; // 到端口 tile：egress 出 C2C 链路（side==dir）
    return GetNextHop(port_tile, pos); // 片内 XY 朝固定出口 tile 收敛
}

int SelectCoreMsgExit(int source_core, int des_core) {
    if (source_core < 0 || source_core >= TOTAL_CORES || des_core < 0 ||
        des_core >= TOTAL_CORES)
        throw std::runtime_error(
            "pinned core-message routing requires legal source/destination cores");
    int sd = DieOfGlobal(source_core), dd = DieOfGlobal(des_core);
    if (sd == dd)
        return -1;
    if (DieManhattan(sd, dd) != 1)
        throw std::runtime_error(
            "multi-hop core-message routing is not supported in V1");
    return CrossDieSelectExit(source_core, des_core);
}

void PinControlMsgExit(Msg &msg) {
    if (!msg.IsControlMsg())
        throw std::runtime_error(
            "cross-die control pinning requires a control message");

    EndpointType des_type = DecodeEndpointType(msg.des_);
    if (des_type == EP_HOST) {
        msg.exit_port_ = -1; // DONE/host ACK 仍走同 die HOST attachment
        return;
    }
    if (des_type != EP_CORE)
        throw std::runtime_error(
            "control message destination is neither a core nor HOST endpoint");
    msg.exit_port_ = SelectCoreMsgExit(msg.source_, msg.des_);
}

Directions ControlMsgNextHop(const Msg &msg, int pos) {
    if (!msg.IsControlMsg())
        throw std::runtime_error(
            "ControlMsgNextHop requires a control message");
    if (pos < 0 || pos >= TOTAL_CORES)
        throw std::runtime_error("control routing: illegal router position");

    EndpointType des_type = DecodeEndpointType(msg.des_);
    if (des_type == EP_CORE)
        return CrossDieStep(msg.des_, pos, msg.exit_port_);
    if (des_type == EP_HOST)
        return GetNextHop(msg.des_, pos, msg.source_);
    throw std::runtime_error(
        "control message destination is neither a core nor HOST endpoint");
}

Directions DataMsgNextHop(const Msg &msg, int pos) {
    if (msg.IsControlMsg())
        throw std::runtime_error("DataMsgNextHop rejects control messages");
    if (pos < 0 || pos >= TOTAL_CORES)
        throw std::runtime_error("data routing: illegal router position");

    EndpointType des_type = DecodeEndpointType(msg.des_);
    if (des_type == EP_CORE) {
        if (DieOfGlobal(pos) != DieOfGlobal(msg.des_) && msg.msg_type_ != DATA)
            throw std::runtime_error(
                "only DATA may use cross-die data-channel routing in V1");
        return CrossDieStep(msg.des_, pos, msg.exit_port_);
    }
    if (des_type == EP_HOST)
        return GetNextHop(msg.des_, pos, msg.source_);
    throw std::runtime_error(
        "data-channel message destination is neither a core nor HOST endpoint");
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