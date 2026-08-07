#include "dte/coll_stream_engine.h"

#include <iostream>
#include <string>
#include <vector>

namespace {
using namespace coll_refactor;

int failures = 0;
int checks = 0;

void Check(bool ok, const std::string &name) {
    ++checks;
    if (!ok) ++failures;
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name
              << std::endl;
}

ReduceStreamWireHeader Header(uint16_t tree, uint16_t source,
                              uint64_t elements,
                              uint64_t vector_bits = 512) {
    ReduceStreamWireHeader header;
    header.stream.tree_id = tree;
    header.stream.key = {{31, 37, 41}, 0, 1};
    header.stream.dtype = CollDType::UINT8;
    header.stream.op = CollReduceOp::SUM;
    header.stream.total_elements = elements;
    header.stream.physical_data_flits = CollCeilDiv(elements * 8, 128);
    const VectorWork work = ComputeVectorWork(
        elements, 1, vector_bits, CollDType::UINT8);
    header.stream.vector_beats = work.vector_beats;
    header.source_id = source;
    header.tail_valid_lanes = work.tail_valid_lanes;
    return header;
}

std::vector<ReduceStreamDataFlit> ConstantStream(
    const ReduceStreamWireHeader &header, uint64_t value,
    uint64_t vector_bits = 512) {
    const VectorWork work = ComputeVectorWork(
        header.stream.total_elements, 1, vector_bits, header.stream.dtype);
    std::vector<ReduceStreamDataFlit> data;
    for (uint64_t beat_id = 0; beat_id < work.vector_beats; ++beat_id) {
        const bool final = beat_id + 1 == work.vector_beats;
        VectorBeat geometry{header.stream.dtype, vector_bits,
                            {work.lanes,
                             final ? work.tail_valid_lanes : work.lanes}};
        std::vector<uint64_t> values(work.lanes, value);
        auto beat = PackReduceVectorValues(
            {header.stream.key, header.reduce_stage_id, beat_id},
            geometry, values);
        auto split = SplitReduceVectorBeat(
            header, beat, 128, vector_bits);
        data.insert(data.end(), split.begin(), split.end());
    }
    return data;
}

struct RunResult {
    std::vector<sc_bv<256>> wires;
    RouterReduceStreamStats stream;
    DcaComputePoolStats dca;
    size_t residual = 0;
};

RunResult RunNode(size_t inputs, uint64_t elements,
                  NocCollDcaConfig config = {}) {
    ResetCollectiveReduceFabric();
    const uint16_t tree = 91;
    const std::vector<Directions> directions{WEST, EAST, NORTH, SOUTH, CENTER};
    uint8_t bitmap = 0;
    for (size_t i = 0; i < inputs; ++i)
        bitmap |= 1u << directions[i];
    ProgramCollectiveReduceNode(tree, 0, {bitmap, CENTER});
    RouterReduceStreamEngine engine(0, config);
    std::vector<ReduceStreamWireHeader> headers;
    std::vector<std::vector<ReduceStreamDataFlit>> streams;
    for (size_t i = 0; i < inputs; ++i) {
        headers.push_back(Header(tree, static_cast<uint16_t>(i + 1),
                                 elements, config.vector_bits));
        streams.push_back(ConstantStream(headers.back(), i + 1,
                                         config.vector_bits));
        if (!engine.TryAcceptHeader(directions[i], headers.back()))
            throw std::runtime_error("test header unexpectedly backpressured");
    }
    RunResult result;
    auto collect = [&] {
        while (const auto *wire = engine.FrontEgress()) {
            result.wires.push_back(wire->wire);
            engine.PopEgress();
        }
    };
    collect();
    uint64_t cycle = 0;
    for (size_t seq = 0; seq < streams.front().size(); ++seq) {
        for (size_t input = 0; input < inputs; ++input) {
            const auto status = engine.TryAcceptData(
                directions[input], streams[input][seq]);
            if (status == ReduceStreamAcceptStatus::BACKPRESSURE)
                throw std::runtime_error(
                    "test failed to retry assembler backpressure");
        }
        engine.Tick(cycle++);
        collect();
    }
    for (uint64_t watchdog = 0;
         !engine.Drained() && watchdog < 10000; ++watchdog) {
        engine.Tick(cycle++);
        collect();
    }
    result.stream = engine.Stats();
    result.dca = engine.DcaStats();
    result.residual = engine.Residual();
    return result;
}

bool VerifyResult(const RunResult &run, uint64_t elements,
                  uint64_t expected, uint64_t vector_bits = 512) {
    if (run.wires.empty() ||
        !IsReduceStreamHeaderWire(run.wires.front()))
        return false;
    const auto header = DeserializeReduceStreamHeader(
        run.wires.front(), vector_bits);
    ReduceStreamAssembler assembler(header, 8, 128, vector_bits);
    size_t values_seen = 0;
    for (size_t i = 1; i < run.wires.size(); ++i) {
        if (!IsReduceStreamDataWire(run.wires[i])) return false;
        const auto status = assembler.Accept(
            DeserializeReduceStreamData(run.wires[i]));
        if (status == ReduceStreamAcceptStatus::BACKPRESSURE) return false;
        while (auto beat = assembler.PopBeat()) {
            const auto values = UnpackReduceVectorValues(*beat);
            for (uint64_t lane = 0;
                 lane < beat->geometry.lane_mask.valid_lanes; ++lane) {
                if (values[lane] != expected) return false;
                ++values_seen;
            }
        }
    }
    assembler.Finish();
    return values_seen == elements && assembler.Residual() == 0;
}

} // namespace

int RunCollR4SelfTest() {
    failures = checks = 0;
    std::cout << "==== NoC collective refactor R4 stream engine self-test ===="
              << std::endl;

    const auto two = RunNode(2, 70);
    Check(VerifyResult(two, 70, 3),
          "two-input stream produces bit-exact vector SUM with tail");
    Check(two.dca.dca_issued == 2 && two.dca.completions == 2,
          "two inputs issue one DCA request per vector beat");
    Check(two.stream.headers_in == 2 && two.stream.headers_out == 1 &&
              two.stream.data_in == 10 && two.stream.data_out == 5,
          "stream framing reduces two input streams to one output stream");
    Check(two.residual == 0,
          "two-input stream drains header/operand/issue/result/egress state");

    NocCollDcaConfig pipelined;
    pipelined.timing[0][0] = {4, 1};
    const auto three = RunNode(3, 129, pipelined);
    Check(VerifyResult(three, 129, 6),
          "three-input left fold preserves deterministic exact value");
    Check(three.dca.dca_issued == 6 && three.dca.completions == 6,
          "three inputs and three beats issue exactly 2B requests");
    Check(three.dca.inflight_peak > 1,
          "L/II stream engine permits multiple tagged inflight requests");
    Check(three.residual == 0,
          "three-input local-feedback stages fully drain");

    const auto one = RunNode(1, 70);
    Check(VerifyResult(one, 70, 1) && one.dca.dca_issued == 0,
          "single-input tree node bypasses DCA without changing values");
    Check(one.stream.bypass_beats == 2 && one.residual == 0,
          "bypass path accounts vector beats and drains");

    NocCollDcaConfig pressured;
    pressured.header_fifo_depth = 5;
    pressured.operand_fifo_depth = 5;
    pressured.result_fifo_depth = 1;
    pressured.timing[0][0] = {4, 1};
    const auto large = RunNode(2, 1024, pressured);
    Check(VerifyResult(large, 1024, 3),
          "64-flit stream survives finite result/egress backpressure");
    Check(large.dca.dca_issued == 16 &&
              large.dca.completions == 16 && large.residual == 0,
          "64-flit stream uses 16 vector issues and fully drains");

    std::cout << "R4 self-test: " << (checks - failures) << "/" << checks
              << " checks passed" << std::endl;
    return failures;
}
