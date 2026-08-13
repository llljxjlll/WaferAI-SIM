#include "router/endpoint_output_flow_lock_selftest.h"

#include "router/control_output_pulse_gate.h"
#include "router/endpoint_output_flow_lock.h"

#include <array>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class Checks {
public:
    void Check(bool ok, const std::string &name) {
        ++checks_;
        if (!ok) {
            ++failures_;
            std::cerr << "endpoint output flow lock selftest failure: "
                      << name << "\n";
        }
    }
    int checks() const { return checks_; }
    int failures() const { return failures_; }

private:
    int checks_ = 0;
    int failures_ = 0;
};

template <typename F> bool Throws(F &&fn) {
    try {
        fn();
    } catch (const std::logic_error &) {
        return true;
    }
    return false;
}

} // namespace

int RunEndpointOutputFlowLockSelfTest() {
    Checks checks;
    EndpointOutputFlowLock lock;
    const EndpointOutputFlowKey source_a{0, 3, 3, 0};
    const EndpointOutputFlowKey source_b{2, 3, 3, 0};
    const EndpointOutputFlowKey other_subflow{0, 3, 3, 1};

    checks.Check(!lock.Active() && lock.Residual() == 0,
                 "fresh lock is drained");
    lock.Acquire(source_a);
    checks.Check(lock.Active() && lock.Residual() == 1 &&
                     lock.OwnedBy(source_a),
                 "seq1 acquisition records the complete flow identity");
    checks.Check(!lock.OwnedBy(source_b) && !lock.OwnedBy(other_subflow),
                 "same destination/tag cannot alias source or subflow");
    checks.Check(Throws([&] { lock.Acquire(source_b); }),
                 "foreign seq1 cannot acquire an active output");
    checks.Check(Throws([&] { lock.Release(source_b); }),
                 "foreign tail cannot release the owner");
    lock.Release(source_a);
    checks.Check(!lock.Active() && lock.Residual() == 0,
                 "owner tail fully drains lock state");

    // N=4 AllGather, L=68 bytes: ceil(68/16)=5 endpoint fragments per
    // source. Model the Router's round-robin input scan with source A and B
    // simultaneously targeting rank 3 under the same transport tag.  Once A
    // seq1 wins, all five A fragments must leave contiguously before B seq1.
    struct Fragment {
        EndpointOutputFlowKey key;
        int sequence;
        bool tail;
    };
    std::array<std::vector<Fragment>, 2> inputs;
    for (int seq = 1; seq <= 5; ++seq) {
        inputs[0].push_back({source_a, seq, seq == 5});
        inputs[1].push_back({source_b, seq, seq == 5});
    }
    std::array<std::size_t, 2> next{};
    std::vector<Fragment> output;
    for (int sweep = 0; sweep < 20 && output.size() != 10; ++sweep) {
        for (std::size_t input = 0; input < inputs.size(); ++input) {
            if (next[input] == inputs[input].size())
                continue;
            const Fragment &fragment = inputs[input][next[input]];
            if (lock.Active() && !lock.OwnedBy(fragment.key))
                continue;
            if (!lock.Active()) {
                if (fragment.sequence != 1)
                    continue;
                lock.Acquire(fragment.key);
            }
            output.push_back(fragment);
            ++next[input];
            if (fragment.tail)
                lock.Release(fragment.key);
        }
    }
    bool contiguous = output.size() == 10;
    for (int i = 0; contiguous && i < 5; ++i)
        contiguous = output[static_cast<std::size_t>(i)].key == source_a &&
            output[static_cast<std::size_t>(i)].sequence == i + 1;
    for (int i = 0; contiguous && i < 5; ++i)
        contiguous = output[static_cast<std::size_t>(i + 5)].key == source_b &&
            output[static_cast<std::size_t>(i + 5)].sequence == i + 1;
    checks.Check(contiguous,
                 "N4 AllGather L68 same-tag sources never interleave");
    checks.Check(lock.Residual() == 0,
                 "N4 AllGather L68 finishes with zero lock residual");

    // Exercise the exact production pulse gate as two Router hops.  Four
    // consecutive control identities model REQUEST, completion ACK, EVENT,
    // and endpoint admission ACK.  A depth-one middle queue and a periodically
    // busy destination force backpressure without permitting a duplicate.
    ControlOutputPulseGate source_gate;
    ControlOutputPulseGate middle_gate;
    std::deque<int> source_queue{101, 102, 103, 104};
    std::deque<int> middle_queue;
    std::vector<int> delivered;
    bool source_level = false;
    bool middle_level = false;
    int source_wire = 0;
    int middle_wire = 0;
    for (int cycle = 0; cycle < 40; ++cycle) {
        if (middle_level) delivered.push_back(middle_wire);
        if (source_level) {
            if (middle_queue.size() == 1)
                throw std::logic_error(
                    "control pulse test overflowed depth-one Router queue");
            middle_queue.push_back(source_wire);
        }

        source_level = false;
        middle_level = false;
        const bool source_may_send = source_gate.BeginCycle();
        const bool middle_may_send = middle_gate.BeginCycle();
        const bool destination_busy = cycle % 3 == 1;
        if (middle_may_send && !destination_busy && !middle_queue.empty()) {
            middle_wire = middle_queue.front();
            middle_queue.pop_front();
            middle_level = true;
            middle_gate.MarkSent();
        }
        if (source_may_send && middle_queue.empty() &&
            !source_queue.empty()) {
            source_wire = source_queue.front();
            source_queue.pop_front();
            source_level = true;
            source_gate.MarkSent();
        }
    }
    checks.Check(delivered == std::vector<int>({101, 102, 103, 104}),
                 "two-hop REQUEST ACK EVENT control pulses are exactly once");
    checks.Check(source_queue.empty() && middle_queue.empty() &&
                     !source_level && !middle_level,
                 "depth-one control backpressure drains both Router queues");
    checks.Check(source_gate.Residual() == 0 &&
                     middle_gate.Residual() == 0,
                 "all Router control pulse cooldown state drains to zero");

    if (checks.failures() == 0)
        std::cout << "Endpoint output flow lock selftest passed ("
                  << checks.checks() << " checks)\n";
    return checks.failures();
}

#ifdef ENDPOINT_OUTPUT_FLOW_LOCK_SELFTEST_MAIN
int main() { return RunEndpointOutputFlowLockSelfTest(); }
#endif
