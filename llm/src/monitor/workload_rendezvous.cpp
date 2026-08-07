#include "monitor/workload_normalize.h"
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
int IntegerOr(const WLJson &object, const char *field, int fallback) {
    if (!object.contains(field))
        return fallback;
    if (!object.at(field).is_number_integer())
        throw std::runtime_error(std::string(field) + " must be an integer");
    return object.at(field).get<int>();
}
} // namespace

void ValidateWorkloadRendezvous(const WLJson &j, int chip_id) {
    if (!j.contains("chips") || !j.at("chips").is_array() ||
        chip_id < 0 || chip_id >= static_cast<int>(j.at("chips").size()))
        throw std::runtime_error("workload has no selected chip");
    if (!j.contains("source") || !j.at("source").is_array())
        throw std::runtime_error(
            "dataflow workload must declare a host source array");

    const auto &chip = j.at("chips").at(chip_id);
    const auto &cores = chip.at("cores");
    const bool has_collectives =
        chip.contains("collectives") && chip.at("collectives").is_array() &&
        !chip.at("collectives").empty();
    bool has_runnable_root = !j.at("source").empty() || has_collectives;
    if (!cores.is_array() || cores.empty())
        throw std::runtime_error("selected workload chip has no cores");

    std::set<int> active_cores;
    std::map<int, int> external_sources;
    std::map<std::pair<int, int>, int> producers;

    for (const auto &core : cores) {
        const int cid = IntegerOr(core, "id", -1);
        if (cid < 0)
            throw std::runtime_error("workload core id must be non-negative");
        if (!active_cores.insert(cid).second)
            throw std::runtime_error("duplicate workload core id " +
                                     std::to_string(cid));
    }

    for (const auto &source : j.at("source")) {
        const int dest = IntegerOr(source, "dest", -1);
        if (!active_cores.count(dest))
            throw std::runtime_error(
                "source.dest " + std::to_string(dest) +
                " does not identify an active workload core");
        int repetitions = 1;
        if (source.contains("loop") &&
            source.at("loop").is_number_integer())
            repetitions = source.at("loop").get<int>();
        if (repetitions <= 0)
            throw std::runtime_error("source.loop must be positive");
        external_sources[dest] += repetitions;
    }

    for (const auto &core : cores) {
        const int source_core = core.at("id").get<int>();
        if (!core.contains("worklist") || !core.at("worklist").is_array())
            throw std::runtime_error(
                "workload core " + std::to_string(source_core) +
                " has no worklist array");
        const auto &worklist = core.at("worklist");
        if (worklist.empty() && !has_collectives)
            throw std::runtime_error(
                "workload core " + std::to_string(source_core) +
                " has no worklist");
        if (!worklist.empty() &&
            IntegerOr(worklist.at(0), "recv_cnt", -1) == 0)
            has_runnable_root = true;
        for (const auto &work : worklist) {
            if (!work.contains("cast"))
                continue;
            if (!work.at("cast").is_array())
                throw std::runtime_error("work.cast must be an array");
            for (const auto &cast : work.at("cast")) {
                const int dest = IntegerOr(cast, "dest", -1);
                if (dest < 0)
                    continue;
                const int tag = IntegerOr(cast, "tag", dest);
                ++producers[{dest, tag}];
            }
        }
    }

    if (!has_runnable_root)
        throw std::runtime_error(
            "dataflow workload has no runnable root (host source, "
            "zero-recv work, or collective)");

    for (const auto &core : cores) {
        const int cid = core.at("id").get<int>();
        const auto &worklist = core.at("worklist");
        for (size_t index = 0; index < worklist.size(); ++index) {
            const auto &work = worklist.at(index);
            const int recv_count = IntegerOr(work, "recv_cnt", -1);
            if (recv_count < 0)
                throw std::runtime_error(
                    "core " + std::to_string(cid) +
                    " work " + std::to_string(index) +
                    " has invalid or missing recv_cnt");
            if (index == 0 && external_sources.count(cid)) {
                if (external_sources.at(cid) != recv_count)
                    throw std::runtime_error(
                        "core " + std::to_string(cid) +
                        " RECV_START expects " + std::to_string(recv_count) +
                        " source tails but workload declares " +
                        std::to_string(external_sources.at(cid)));
                continue;
            }

            if (recv_count == 0)
                continue;

            const int recv_tag = IntegerOr(work, "recv_tag", cid);
            const auto producer = producers.find({cid, recv_tag});
            const size_t available = producer == producers.end()
                                         ? 0
                                         : static_cast<size_t>(producer->second);
            if (available < static_cast<size_t>(recv_count))
                throw std::runtime_error(
                    "core " + std::to_string(cid) +
                    " work " + std::to_string(index) + " recv(tag=" +
                    std::to_string(recv_tag) + ",cnt=" +
                    std::to_string(recv_count) + ") has only " +
                    std::to_string(available) + " declared producer(s)");
        }
    }
}
