#include "memory/behavioral_hbm_backend.h"
#include "memory/external_memory_runtime.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace em = external_memory;
using namespace sc_core;

namespace {

class SelectiveFailBackend : public HBMBackend {
public:
    SelectiveFailBackend(
        const BehavioralHBMBackendConfig &config,
        uint64_t rejected_read_address)
        : backend_(config),
          rejected_read_address_(rejected_read_address) {}

    void Submit(
        const std::shared_ptr<HBMBackendTransaction> &tx) override {
        if (tx && tx->command == MemCommand::kRead &&
            tx->address == rejected_read_address_) {
            tx->complete(SC_ZERO_TIME, SC_ZERO_TIME, 1,
                         "injected HBM read failure");
            return;
        }
        backend_.Submit(tx);
    }

    const HBMBackendStats &Stats() const override {
        return backend_.Stats();
    }

private:
    BehavioralHBMBackend backend_;
    uint64_t rejected_read_address_;
};

em::FabricConfig DirectFabric() {
    em::FabricConfig config;
    config.external_capacities.push_back(
        {"external:0", "host:0", 0, 24576});
    config.hbm_capacities.push_back({"hbm:0", 0, 0, 12288});
    config.links.push_back({"link:0", "external:0", 0, 4, 2, 1, 2});
    config.connections.push_back(
        {"connection:0", "link:0", "hbm:0", 0, {0}, 0,
         std::nullopt});
    return config;
}

struct Driver : sc_module {
    SC_HAS_PROCESS(Driver);

    Driver(sc_module_name name, em::ExternalMemoryRuntimeBridge &runtime,
           HBMBackend &backend)
        : sc_module(name), runtime(runtime), backend(backend) {
        SC_THREAD(Run);
    }

    uint64_t Cycle() const {
        return sc_time_stamp().value() / sc_time(1, SC_NS).value();
    }

    std::vector<uint8_t> HbmAccess(
        MemCommand command, uint64_t address,
        const std::vector<uint8_t> &payload) {
        auto tx = std::make_shared<HBMBackendTransaction>();
        tx->command = command;
        tx->address = address;
        tx->payload = payload;
        if (command == MemCommand::kWrite)
            tx->byte_enable.assign(payload.size(), 0xff);
        tx->submitted = sc_time_stamp();
        sc_event done;
        bool callback = false;
        sc_time delay = SC_ZERO_TIME;
        int status = -1;
        std::string backend_error;
        tx->complete = [&](sc_time value, sc_time, int result,
                           const std::string &message) {
            delay = value;
            status = result;
            backend_error = message;
            callback = true;
            done.notify(value);
        };
        backend.Submit(tx);
        if (!callback || delay > SC_ZERO_TIME) wait(done);
        if (status != 0)
            throw std::runtime_error(
                "HBM access failed: " + backend_error);
        return tx->payload;
    }

    void Check(bool condition, const std::string &message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    template <typename Operation>
    void RejectWithoutAdmission(const std::string &message,
                                Operation operation) {
        const uint64_t submitted = runtime.Stats().submitted_requests;
        try {
            operation();
        } catch (const std::exception &) {
            Check(runtime.Stats().submitted_requests == submitted,
                  message + " changed submitted count");
            return;
        }
        throw std::runtime_error(message + " was accepted");
    }

    void Run() {
        try {
            const std::vector<uint8_t> initial{1, 7, 0, 9, 13, 0, 255, 2};
            runtime.Submit(
                {"load", "connection:0",
                 em::TransferDirection::kExternalToHbm,
                 0, 0, initial.size(), Cycle()});
            Check(!runtime.Poll("load").has_value(),
                  "load completed before runtime event");
            const auto loaded = runtime.Wait("load");
            Check(loaded.status == 0 && loaded.completed_at > loaded.started_at,
                  "load completion is invalid");
            Check(HbmAccess(MemCommand::kRead, 0,
                            std::vector<uint8_t>(initial.size())) == initial,
                  "HBM backend read cannot see external payload");

            const std::vector<uint8_t> dirty{8, 6, 7, 5, 3, 0, 9, 4};
            HbmAccess(MemCommand::kWrite, 0, dirty);
            runtime.Submit(
                {"store", "connection:0",
                 em::TransferDirection::kHbmToExternal,
                 64, 0, dirty.size(), Cycle()});
            const auto stored = runtime.Wait("store");
            Check(stored.status == 0,
                  "dirty HBM store completion failed");
            Check(runtime.ProbeExternal("external:0", 64, dirty.size()) ==
                      dirty,
                  "external probe cannot see dirty HBM payload");
            Check(runtime.Poll("store").has_value(),
                  "completed store is not pollable");

            const std::vector<uint8_t> boundary_block(64, 0x5a);
            runtime.SeedExternal("external:0", 24512, boundary_block);
            runtime.Submit(
                {"exact-hbm-boundary", "connection:0",
                 em::TransferDirection::kExternalToHbm,
                 24512, 12224, boundary_block.size(), Cycle()});
            Check(runtime.Wait("exact-hbm-boundary").status == 0 &&
                      HbmAccess(MemCommand::kRead, 12224,
                                boundary_block) == boundary_block,
                  "exact 12,288-byte HBM boundary transfer failed");
            RejectWithoutAdmission("one-block HBM overflow", [&] {
                runtime.Submit(
                    {"one-block-overflow", "connection:0",
                     em::TransferDirection::kExternalToHbm,
                     24384, 12224, 128, Cycle()});
            });
            RejectWithoutAdmission("capacity failure", [&] {
                runtime.Submit(
                    {"bad-range", "connection:0",
                     em::TransferDirection::kExternalToHbm,
                     24572, 0, 8, Cycle()});
            });
            RejectWithoutAdmission("connection failure", [&] {
                runtime.Submit(
                    {"bad-connection", "missing",
                     em::TransferDirection::kExternalToHbm,
                     0, 0, 8, Cycle()});
            });
            RejectWithoutAdmission("schema version failure", [&] {
                em::TransferRequest request{
                    "bad-version", "connection:0",
                    em::TransferDirection::kExternalToHbm,
                    0, 0, 8, Cycle()};
                request.schema_version = "npusim.external_dma_request/v0";
                runtime.Submit(request);
            });

            const std::vector<uint8_t> preserved(8, 0xa5);
            runtime.SeedExternal("external:0", 160, preserved);
            runtime.Submit(
                {"backend-failure", "connection:0",
                 em::TransferDirection::kHbmToExternal,
                 160, 120, preserved.size(), Cycle()});
            const auto failed = runtime.Wait("backend-failure");
            Check(failed.status != 0 &&
                      runtime.ProbeExternal(
                          "external:0", 160, preserved.size()) == preserved,
                  "failed backend read partially wrote external memory");

            runtime.SeedExternal("external:0", 96,
                                 std::vector<uint8_t>(24, 0x33));
            runtime.Submit(
                {"queue:0", "connection:0",
                 em::TransferDirection::kExternalToHbm,
                 96, 32, 8, Cycle()});
            runtime.Submit(
                {"queue:1", "connection:0",
                 em::TransferDirection::kExternalToHbm,
                 104, 40, 8, Cycle()});
            RejectWithoutAdmission("queue failure", [&] {
                runtime.Submit(
                    {"queue:2", "connection:0",
                     em::TransferDirection::kExternalToHbm,
                     112, 48, 8, Cycle()});
            });
            runtime.Wait("queue:0");
            runtime.Wait("queue:1");
            Check(HbmAccess(MemCommand::kRead, 48,
                            std::vector<uint8_t>(8)) ==
                      std::vector<uint8_t>(8, 0),
                  "rejected queue request partially wrote HBM");
            Check(runtime.Outstanding() == 0 &&
                      runtime.Stats().completed_requests == 6 &&
                      runtime.Stats().failed_requests == 1,
                  "runtime did not drain accepted requests");
            Check(runtime.Stats().external_read_bytes == 88 &&
                      runtime.Stats().external_write_bytes == dirty.size(),
                  "runtime byte statistics are incorrect");
            passed = true;
        } catch (const std::exception &failure) {
            error = failure.what();
        }
        sc_stop();
    }

    em::ExternalMemoryRuntimeBridge &runtime;
    HBMBackend &backend;
    bool passed = false;
    std::string error;
};

} // namespace

int sc_main(int, char **) {
    BehavioralHBMBackendConfig backend_config;
    backend_config.bandwidth_GBps = 8.0;
    backend_config.efficiency = 1.0;
    backend_config.base_latency = sc_time(1, SC_NS);
    SelectiveFailBackend backend(backend_config, 120);
    em::ExternalMemoryRuntimeBridge runtime(
        "external_runtime", DirectFabric(), {{"hbm:0", &backend}},
        sc_time(1, SC_NS));
    runtime.SeedExternal(
        "external:0", 0, {1, 7, 0, 9, 13, 0, 255, 2});
    Driver driver("driver", runtime, backend);
    sc_start(1000, SC_NS);
    if (!driver.passed) {
        std::cerr << "external memory runtime selftest: FAIL: "
                  << driver.error << std::endl;
        return 1;
    }
    std::cout << "external memory runtime selftest: PASS" << std::endl;
    return 0;
}
