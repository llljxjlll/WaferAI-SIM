#include "workercore/serialized_wire_queue_selftest.h"

#include "workercore/serialized_wire_queue.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct TestState {
    int checks = 0;
    int failures = 0;

    void Check(bool condition, const std::string &description) {
        ++checks;
        if (condition) return;
        ++failures;
        std::cerr << "[serialized-wire-queue] FAIL: " << description << '\n';
    }
};

sc_bv<256> Wire(uint8_t producer, uint16_t sequence) {
    sc_bv<256> wire;
    wire = 0;
    wire.range(7, 0) = producer;
    wire.range(23, 8) = sequence;
    wire.range(255, 240) = static_cast<uint16_t>(producer ^ sequence);
    return wire;
}

template <class Function>
bool Throws(Function function) {
    try {
        function();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

} // namespace

int RunSerializedWireQueueSelfTest() {
    TestState test;

    // Producer 1 models normal legacy P2P DATA; producer 2 models a long DCA
    // stream. Alternating ownership plus a stopped consumer is the failure
    // shape that used to overwrite the shared send_buffer.
    SerializedWireQueue queue(64);
    std::vector<uint64_t> normal_tickets;
    std::vector<uint64_t> dca_tickets;
    for (uint16_t sequence = 0; sequence < 32; ++sequence) {
        normal_tickets.push_back(queue.Enqueue(Wire(1, sequence), false));
        dca_tickets.push_back(queue.Enqueue(Wire(2, sequence), false));
    }
    test.Check(queue.Full(), "mixed queue reaches its exact capacity");
    test.Check(Throws([&] { (void)queue.Enqueue(Wire(3, 0), true); }),
               "full queue rejects without overwriting an owned wire");
    test.Check(queue.Size() == 64,
               "failed enqueue leaves all mixed producer wires intact");

    for (int stalled_cycle = 0; stalled_cycle < 17; ++stalled_cycle)
        test.Check(queue.Size() == 64,
                   "data-channel backpressure preserves queued values");

    uint64_t previous_ticket = 0;
    int normal_completed = 0;
    int dca_completed = 0;
    for (uint16_t position = 0; position < 64; ++position) {
        const uint8_t expected_producer = position % 2 == 0 ? 1 : 2;
        const uint16_t expected_sequence = position / 2;
        const SerializedWireItem front = queue.Front();
        test.Check(front.wire == Wire(expected_producer, expected_sequence),
                   "mixed DATA/DCA wire preserves value and FIFO order");
        test.Check(!front.control,
                   "mixed DATA/DCA traffic remains on the data channel");
        const SerializedWireItem completed = queue.CompleteFront();
        test.Check(completed.ticket == previous_ticket + 1,
                   "each physical send completes one monotonic ticket");
        previous_ticket = completed.ticket;
        if (expected_producer == 1)
            ++normal_completed;
        else
            ++dca_completed;
        test.Check(normal_completed - dca_completed >= 0 &&
                       normal_completed - dca_completed <= 1,
                   "both continuously pending producers make bounded progress");
    }

    test.Check(normal_completed == 32 && dca_completed == 32,
               "normal DATA and DCA wires each complete exactly once");
    test.Check(queue.Empty() && queue.Size() == 0,
               "mixed traffic drains without residual ownership");
    test.Check(queue.Completed(normal_tickets.back()) &&
                   queue.Completed(dca_tickets.back()),
               "both producers observe their terminal completion ticket");
    test.Check(Throws([&] { (void)queue.CompleteFront(); }),
               "empty queue cannot fabricate a completion");

    if (test.failures == 0)
        std::cout << "serialized wire queue self-test passed (" << test.checks
                  << " checks)\n";
    return test.failures;
}

#ifdef SERIALIZED_WIRE_QUEUE_SELFTEST_MAIN
int sc_main(int, char **) { return RunSerializedWireQueueSelfTest(); }
#endif
