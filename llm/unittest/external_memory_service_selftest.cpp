#include "memory/external_memory_service.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace em = external_memory;

namespace {

struct Checks {
    int failures = 0;

    void Check(bool condition, const std::string &name) {
        if (condition) return;
        ++failures;
        std::cerr << "FAIL: " << name << std::endl;
    }

    void Reject(const std::string &name,
                const std::function<void()> &operation) {
        try {
            operation();
            Check(false, name);
        } catch (const std::exception &) {
            Check(true, name);
        }
    }
};

em::FabricConfig DirectFabric() {
    em::FabricConfig config;
    config.external_capacities.push_back({"external:0", "host:0", 0, 256});
    config.hbm_capacities.push_back({"hbm:0", 0, 0, 128});
    config.links.push_back({"link:0", "external:0", 0, 4, 3, 1, 2});
    config.connections.push_back(
        {"connection:0", "link:0", "hbm:0", 0, {0}, 0,
         std::nullopt});
    return config;
}

void NonzeroRoundTrip(Checks &checks) {
    em::ExternalMemoryService service(DirectFabric());
    const std::vector<uint8_t> payload{1, 7, 0, 9, 13, 0, 255, 2, 3};
    service.SeedExternal("external:0", 0, payload);
    const auto report = service.Execute({
        {"load", "connection:0", em::TransferDirection::kExternalToHbm,
         0, 0, payload.size(), 0},
        {"store", "connection:0", em::TransferDirection::kHbmToExternal,
         64, 0, payload.size(), 6},
    });
    checks.Check(service.PeekHbm("hbm:0", 0, payload.size()) == payload,
                 "nonzero payload reaches HBM");
    checks.Check(
        service.PeekExternal("external:0", 64, payload.size()) == payload,
        "nonzero payload returns to external backing");
    checks.Check(report.completions.size() == 2,
                 "round trip completion count");
    checks.Check(report.completions[0].completion_cycle == 6 &&
                     report.completions[1].completion_cycle == 12,
                 "round trip latency formula");
    checks.Check(report.stats.external_read_bytes == payload.size() &&
                     report.stats.external_write_bytes == payload.size() &&
                     report.stats.hbm_read_bytes == payload.size() &&
                     report.stats.hbm_write_bytes == payload.size(),
                 "round trip directional byte statistics");
    checks.Check(report.stats.completed_requests == 2 &&
                     report.stats.pending_requests == 0,
                 "round trip drains all requests");
}

void SharedHalfDuplexContention(Checks &checks) {
    em::ExternalMemoryService service(DirectFabric());
    const std::vector<uint8_t> payload(9, 0x5a);
    service.SeedExternal("external:0", 0, payload);
    const auto report = service.Execute({
        {"a-load", "connection:0",
         em::TransferDirection::kExternalToHbm,
         0, 0, payload.size(), 0},
        {"b-store", "connection:0",
         em::TransferDirection::kHbmToExternal,
         64, 0, payload.size(), 0},
    });
    checks.Check(report.completions[0].start_cycle == 0 &&
                     report.completions[0].completion_cycle == 6,
                 "first half-duplex request timing");
    checks.Check(report.completions[1].start_cycle == 6 &&
                     report.completions[1].completion_cycle == 12,
                 "opposite direction serializes on shared link");
    checks.Check(report.completions[1].queue_stall_cycles == 6 &&
                     report.stats.queue_stall_cycles == 6,
                 "shared link queue stall is reported");
    checks.Check(report.stats.link_busy_cycles == 12 &&
                     report.stats.max_outstanding == 2 &&
                     report.stats.max_queue_occupancy == 1,
                 "shared link contention statistics");
}

void PerDieLocalAddressSpaces(Checks &checks) {
    auto config = DirectFabric();
    config.hbm_capacities.push_back({"hbm:1", 1, 0, 128});
    config.connections.push_back(
        {"connection:1", "link:0", "hbm:1", 1, {0, 1}, 2, 8});
    em::ExternalMemoryService service(config);
    service.SeedExternal("external:0", 0,
                         std::vector<uint8_t>(16, 0x11));
    service.SeedExternal("external:0", 32,
                         std::vector<uint8_t>(16, 0x22));
    service.Execute({
        {"die:0", "connection:0",
         em::TransferDirection::kExternalToHbm,
         0, 0, 16, 0},
        {"die:1", "connection:1",
         em::TransferDirection::kExternalToHbm,
         32, 0, 16, 0},
    });
    checks.Check(service.PeekHbm("hbm:0", 0, 16) ==
                     std::vector<uint8_t>(16, 0x11),
                 "die zero local HBM address");
    checks.Check(service.PeekHbm("hbm:1", 0, 16) ==
                     std::vector<uint8_t>(16, 0x22),
                 "die one independent local HBM address");
}

void FailClosedValidation(Checks &checks) {
    auto overlapping = DirectFabric();
    overlapping.hbm_capacities.push_back({"hbm:overlap", 0, 64, 128});
    checks.Reject("same-die HBM overlap", [&] {
        em::ExternalMemoryService ignored(overlapping);
    });

    auto wrong_owner = DirectFabric();
    wrong_owner.connections[0].target_die_id = 1;
    wrong_owner.connections[0].route_die_ids = {0, 1};
    wrong_owner.connections[0].route_latency_cycles = 1;
    wrong_owner.connections[0].route_bytes_per_cycle = 4;
    checks.Reject("connection target owner mismatch", [&] {
        em::ExternalMemoryService ignored(wrong_owner);
    });

    checks.Reject("external request capacity overflow", [&] {
        em::ExternalMemoryService service(DirectFabric());
        service.Execute({
            {"range", "connection:0",
             em::TransferDirection::kExternalToHbm,
             250, 0, 16, 0},
        });
    });

    checks.Reject("missing connection", [&] {
        em::ExternalMemoryService service(DirectFabric());
        service.Execute({
            {"missing", "connection:missing",
             em::TransferDirection::kExternalToHbm,
             0, 0, 16, 0},
        });
    });

    checks.Reject("queue and outstanding exhaustion", [&] {
        em::ExternalMemoryService service(DirectFabric());
        service.SeedExternal("external:0", 0,
                             std::vector<uint8_t>(48, 0x33));
        service.Execute({
            {"request:0", "connection:0",
             em::TransferDirection::kExternalToHbm,
             0, 0, 16, 0},
            {"request:1", "connection:0",
             em::TransferDirection::kExternalToHbm,
             16, 16, 16, 0},
            {"request:2", "connection:0",
             em::TransferDirection::kExternalToHbm,
             32, 32, 16, 0},
        });
    });
}

} // namespace

int main() {
    Checks checks;
    NonzeroRoundTrip(checks);
    SharedHalfDuplexContention(checks);
    PerDieLocalAddressSpaces(checks);
    FailClosedValidation(checks);
    if (checks.failures != 0) return 1;
    std::cout << "external memory service selftest: PASS" << std::endl;
    return 0;
}
