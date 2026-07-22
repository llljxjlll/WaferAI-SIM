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
                               bool allow_adjacent_d2d) {
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
                    // V1-c3：精确验证相邻 die 的实际双向 link；生产 dataflow 路径在
                    // REQUEST/ACK/DATA 闭环后显式传 allow_adjacent_d2d=true。
                    std::string edge = "core " + std::to_string(cid) + " (die " +
                                       std::to_string(src_die) + ") -> core " +
                                       std::to_string(dest) + " (die " +
                                       std::to_string(dest_die) + ")";
                    if (DieManhattan(src_die, dest_die) != 1)
                        throw std::runtime_error(
                            "multi-hop cross-die not supported in V1 "
                            "(adjacent-die only; V2): " +
                            edge);
                    if (!HasD2DLink(src_die, dest_die) ||
                        !HasD2DLink(dest_die, src_die))
                        throw std::runtime_error(
                            "cross-die traffic requires D2D Link (V1): " + edge +
                            "; no bidirectional peer link between the dies");
                    if (!allow_adjacent_d2d)
                        throw std::runtime_error(
                            "adjacent cross-die runtime is disabled for this "
                            "validation path: " +
                            edge);
                    // 相邻且实际双向 link 存在：允许。
                }
            }
        }
    }
    if (j.contains("source"))
        for (const auto &s : j.at("source"))
            if (s.contains("dest") && s.at("dest").is_number_integer())
                check_bounds(s.at("dest").get<int>(), "source.dest");
}
