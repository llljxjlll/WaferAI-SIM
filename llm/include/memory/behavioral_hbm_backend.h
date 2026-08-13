#pragma once

#include "memory/hbm_backend.h"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>

struct BehavioralHBMBackendConfig {
    double bandwidth_GBps = 0.0;
    double efficiency = 1.0;
    sc_core::sc_time base_latency{0, sc_core::SC_NS};
    sc_core::sc_time read_to_write_turnaround{0, sc_core::SC_NS};
    sc_core::sc_time write_to_read_turnaround{0, sc_core::SC_NS};
};

class BehavioralHBMBackend : public HBMBackend {
public:
    explicit BehavioralHBMBackend(const BehavioralHBMBackendConfig &cfg)
        : cfg_(cfg) {
        if (cfg_.bandwidth_GBps <= 0)
            throw std::runtime_error(
                "BehavioralHBMBackend: bandwidth_GBps must be > 0");
        if (!(cfg_.efficiency > 0.0 && cfg_.efficiency <= 1.0))
            throw std::runtime_error(
                "BehavioralHBMBackend: efficiency must be in (0,1]");
    }

    sc_core::sc_time TransferTime(int length_bytes) const {
        if (length_bytes <= 0)
            throw std::runtime_error(
                "BehavioralHBMBackend::TransferTime: length must be > 0");
        double effective_GBps = cfg_.bandwidth_GBps * cfg_.efficiency;
        return cfg_.base_latency +
               sc_core::sc_time(length_bytes / effective_GBps,
                                sc_core::SC_NS);
    }

    void Submit(const std::shared_ptr<HBMBackendTransaction> &tx) override {
        if (!tx || tx->payload.empty() || !tx->complete)
            throw std::runtime_error(
                "BehavioralHBMBackend::Submit: malformed transaction");
        if (!tx->byte_enable.empty() &&
            tx->byte_enable.size() != tx->payload.size())
            throw std::runtime_error(
                "BehavioralHBMBackend::Submit: byte-enable size mismatch");

        sc_core::sc_time now = sc_core::sc_time_stamp();
        sc_core::sc_time start = next_available_ > now ? next_available_ : now;
        if (has_last_command_ && last_command_ != tx->command)
            start += last_command_ == MemCommand::kRead
                         ? cfg_.read_to_write_turnaround
                         : cfg_.write_to_read_turnaround;
        sc_core::sc_time service = TransferTime((int)tx->payload.size());
        next_available_ = start + service;
        last_command_ = tx->command;
        has_last_command_ = true;

        stats_.requests++;
        stats_.bytes += tx->payload.size();
        stats_.service_time += service;
        if (tx->command == MemCommand::kWrite) {
            stats_.writes++;
            for (size_t i = 0; i < tx->payload.size(); ++i)
                if (tx->byte_enable.empty() || tx->byte_enable[i])
                    backing_[tx->address + i] = tx->payload[i];
        } else {
            stats_.reads++;
            for (size_t i = 0; i < tx->payload.size(); ++i) {
                auto it = backing_.find(tx->address + i);
                tx->payload[i] = it == backing_.end() ? 0 : it->second;
            }
        }
        stats_.completed++;
        tx->complete(next_available_ - now, service, 0, "");
    }


    void DebugSeed(
        uint64_t address,
        const std::vector<uint8_t> &payload) override {
        if (sc_core::sc_is_running())
            throw std::logic_error(
                "HBM DebugSeed is forbidden while simulation is running");
        if (payload.empty())
            throw std::invalid_argument(
                "HBM DebugSeed payload must not be empty");
        if (address > std::numeric_limits<uint64_t>::max() -
                          payload.size())
            throw std::out_of_range("HBM debug seed address overflows");
        for (size_t i = 0; i < payload.size(); ++i)
            backing_[address + i] = payload[i];
    }

    HBMDebugSnapshot DebugPeek(
        uint64_t address, uint64_t size_bytes) const override {
        if (sc_core::sc_is_running())
            throw std::logic_error(
                "HBM DebugPeek is forbidden while simulation is running");
        if (size_bytes == 0 ||
            size_bytes > std::numeric_limits<size_t>::max())
            throw std::invalid_argument(
                "HBM DebugPeek size is invalid");
        if (address > std::numeric_limits<uint64_t>::max() - size_bytes)
            throw std::out_of_range("HBM debug peek address overflows");
        HBMDebugSnapshot snapshot;
        snapshot.address = address;
        snapshot.payload.resize(static_cast<size_t>(size_bytes), 0);
        snapshot.present.resize(static_cast<size_t>(size_bytes), 0);
        for (size_t i = 0; i < snapshot.payload.size(); ++i) {
            const auto it = backing_.find(address + i);
            if (it == backing_.end()) continue;
            snapshot.payload[i] = it->second;
            snapshot.present[i] = 1;
        }
        return snapshot;
    }

    void DebugRestore(
        const HBMDebugSnapshot &snapshot) override {
        if (sc_core::sc_is_running())
            throw std::logic_error(
                "HBM DebugRestore is forbidden while simulation is running");
        if (snapshot.payload.empty() ||
            snapshot.payload.size() != snapshot.present.size())
            throw std::invalid_argument(
                "HBM debug snapshot payload/present size mismatch");
        if (!std::all_of(
                snapshot.present.begin(), snapshot.present.end(),
                [](uint8_t value) { return value <= 1; }))
            throw std::invalid_argument(
                "HBM debug snapshot presence must be zero or one");
        if (snapshot.address >
            std::numeric_limits<uint64_t>::max() -
                snapshot.payload.size())
            throw std::out_of_range("HBM debug restore address overflows");
        for (size_t i = 0; i < snapshot.payload.size(); ++i) {
            const uint64_t address = snapshot.address + i;
            if (snapshot.present[i])
                backing_[address] = snapshot.payload[i];
            else
                backing_.erase(address);
        }
    }

    const HBMBackendStats &Stats() const override { return stats_; }
    double bandwidth_GBps() const { return cfg_.bandwidth_GBps; }
    double efficiency() const { return cfg_.efficiency; }

private:
    BehavioralHBMBackendConfig cfg_;
    std::map<uint64_t, uint8_t> backing_;
    sc_core::sc_time next_available_ = sc_core::SC_ZERO_TIME;
    bool has_last_command_ = false;
    MemCommand last_command_ = MemCommand::kRead;
    HBMBackendStats stats_;
};
