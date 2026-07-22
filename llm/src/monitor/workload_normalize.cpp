#include "monitor/workload_normalize.h"
#include "defs/spec.h"
#include "die/port.h"
#include "utils/router_utils.h"
#include <map>
#include <stdexcept>
#include <string>

void NormalizeWorkloadJson(WLJson &j, int die_id) {
    if (die_id == 0)
        return; // 恒等：单 die / die0，不改（旧配置兼容）
    int off = die_id * CORES_PER_DIE;
    auto shift = [&](WLJson &v) {
        if (v.is_number_integer()) {
            int id = v.get<int>();
            if (id >= 0)
                v = id + off; // 只平移非负核 id；-1（host/loopout）保持
        }
    };
    if (j.contains("chips")) {
        for (auto &chip : j.at("chips")) {
            if (!chip.contains("cores"))
                continue;
            for (auto &core : chip.at("cores")) {
                if (core.contains("id"))
                    shift(core.at("id"));
                if (core.contains("prim_copy"))
                    shift(core.at("prim_copy"));
                if (core.contains("send_global_mem"))
                    shift(core.at("send_global_mem"));
                if (!core.contains("worklist"))
                    continue;
                for (auto &work : core.at("worklist")) {
                    if (!work.contains("cast"))
                        continue;
                    for (auto &cast : work.at("cast"))
                        if (cast.contains("dest"))
                            shift(cast.at("dest"));
                }
            }
        }
    }
    if (j.contains("source"))
        for (auto &s : j.at("source"))
            if (s.contains("dest"))
                shift(s.at("dest"));
    j["id_space"] = "global";
}

void ValidateWorkloadStructure(const WLJson &j, int chip_id,
                               bool allow_d2d) {
    bool global = j.contains("id_space") &&
                  j.at("id_space").get<std::string>() == "global";

    auto check_bounds = [&](int id, const std::string &what) {
        if (id < 0)
            return;
        if (id >= TOTAL_CORES)
            throw std::runtime_error(what + " id " + std::to_string(id) +
                                     " out of range [0," +
                                     std::to_string(TOTAL_CORES) + ")");
        if (!global && id >= CORES_PER_DIE)
            throw std::runtime_error(
                what + " references core " + std::to_string(id) +
                " on die>0 but id_space is die0-local; set "
                "\"id_space\":\"global\"");
    };

    // tag → dest 唯一性：output_lock 按 tag 锁 = 接收端聚合槽。同一 tag 必须只指向一个接收核，
    // 否则两个接收核共享 tag 会在 output_lock 别名（见 common/flow.h）。cast.tag 缺省=cast.dest。
    std::map<int, int> tag2dest;

    if (!j.contains("chips") || chip_id >= (int)j.at("chips").size())
        return;
    const WLJson &cores = j.at("chips").at(chip_id).at("cores");
    for (const auto &core : cores) {
        int cid = core.contains("id") && core.at("id").is_number_integer()
                      ? core.at("id").get<int>()
                      : -1;
        check_bounds(cid, "core");
        int src_die = (cid >= 0) ? cid / CORES_PER_DIE : 0;
        if (core.contains("prim_copy") &&
            core.at("prim_copy").is_number_integer())
            check_bounds(core.at("prim_copy").get<int>(), "prim_copy");
        if (core.contains("send_global_mem") &&
            core.at("send_global_mem").is_number_integer())
            check_bounds(core.at("send_global_mem").get<int>(),
                         "send_global_mem");
        if (!core.contains("worklist"))
            continue;
        for (const auto &work : core.at("worklist")) {
            if (!work.contains("cast"))
                continue;
            for (const auto &cast : work.at("cast")) {
                if (!cast.contains("dest") ||
                    !cast.at("dest").is_number_integer())
                    continue;
                int dest = cast.at("dest").get<int>();
                if (dest < 0)
                    continue;
                check_bounds(dest, "cast.dest");
                // tag → dest 唯一性（cast.tag 缺省=dest）
                int tag = (cast.contains("tag") &&
                           cast.at("tag").is_number_integer())
                              ? cast.at("tag").get<int>()
                              : dest;
                auto it = tag2dest.find(tag);
                if (it != tag2dest.end() && it->second != dest)
                    throw std::runtime_error(
                        "tag " + std::to_string(tag) +
                        " maps to multiple receiver cores (" +
                        std::to_string(it->second) + " and " +
                        std::to_string(dest) +
                        "); tag must uniquely identify a receiver "
                        "(output_lock aggregation slot)");
                tag2dest[tag] = dest;
                int dest_die = dest / CORES_PER_DIE;
                if (dest_die != src_die) {
                    // 精确验证路径上每一跳的实际双向 link；生产 dataflow 路径在
                    // REQUEST/ACK/DATA 闭环后显式传 allow_d2d=true。
                    std::string edge = "core " + std::to_string(cid) + " (die " +
                                       std::to_string(src_die) + ") -> core " +
                                       std::to_string(dest) + " (die " +
                                       std::to_string(dest_die) + ")";
                    // V2-b：多跳放行。沿 die 级维序（X 先于 Y）路径逐跳校验——每一跳的相邻
                    // die 对都必须有双向 peer link，否则该跳不可达。路径长度==DieManhattan，
                    // 每跳距离严格 -1，故循环必然终止（guard 仅防御非预期拓扑）。
                    int cur = src_die;
                    int guard = 0;
                    while (cur != dest_die) {
                        Directions D = DieFirstHopDir(cur, dest_die);
                        int cx = cur % DIE_X, cy = cur / DIE_X;
                        if (D == EAST)
                            cx++;
                        else if (D == WEST)
                            cx--;
                        else if (D == NORTH)
                            cy++;
                        else if (D == SOUTH)
                            cy--;
                        else
                            throw std::runtime_error(
                                "cross-die path: no die-level direction: " + edge);
                        if (cx < 0 || cx >= DIE_X || cy < 0 || cy >= DIE_Y)
                            throw std::runtime_error(
                                "cross-die path leaves the die mesh: " + edge);
                        int nxt = cy * DIE_X + cx;
                        if (!HasD2DLink(cur, nxt) || !HasD2DLink(nxt, cur))
                            throw std::runtime_error(
                                "cross-die traffic requires D2D Link: " + edge +
                                "; no bidirectional peer link on hop die " +
                                std::to_string(cur) + " -> die " +
                                std::to_string(nxt));
                        cur = nxt;
                        if (++guard > DIE_X + DIE_Y + 2)
                            throw std::runtime_error(
                                "cross-die path did not converge: " + edge);
                    }
                    if (!allow_d2d)
                        throw std::runtime_error(
                            "cross-die runtime is disabled for this "
                            "validation path: " +
                            edge);
                    // 路径每跳均有双向 link 且已显式放行：允许（V2 起含多跳）。
                }
            }
        }
    }
    if (j.contains("source"))
        for (const auto &s : j.at("source"))
            if (s.contains("dest") && s.at("dest").is_number_integer())
                check_bounds(s.at("dest").get<int>(), "source.dest");
}
