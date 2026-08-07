#include "dte/coll_reduce_stream.h"

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

template <class E = std::exception, class F> bool Throws(F f) {
    try {
        f();
    } catch (const E &) {
        return true;
    } catch (...) {
    }
    return false;
}

ReduceStreamWireHeader Header(
    uint64_t elements = 70, uint32_t stream_id = 9,
    uint16_t source = 3, uint16_t stage = 1,
    CollDType dtype = CollDType::UINT8,
    CollReduceOp op = CollReduceOp::SUM) {
    ReduceStreamWireHeader header;
    header.stream.tree_id = 17;
    header.stream.key = {{5, 7, 11}, 2, stream_id};
    header.stream.dtype = dtype;
    header.stream.op = op;
    header.stream.total_elements = elements;
    const uint64_t bits = elements * CollDTypeBits(dtype);
    header.stream.physical_data_flits = CollCeilDiv(bits, 128);
    const VectorWork work = ComputeVectorWork(elements, 1, 512, dtype);
    header.stream.vector_beats = work.vector_beats;
    header.reduce_stage_id = stage;
    header.source_id = source;
    header.tail_valid_lanes = work.tail_valid_lanes;
    return header;
}

ReduceStreamDataFlit Data(const ReduceStreamWireHeader &header,
                          uint32_t seq) {
    ReduceStreamDataFlit data;
    data.route = header.Route();
    data.seq_id = seq;
    data.is_tail = seq + 1 == header.stream.physical_data_flits;
    const uint64_t total_bits =
        header.stream.total_elements * CollDTypeBits(header.stream.dtype);
    const uint64_t tail = total_bits % 128;
    data.length_bits = data.is_tail ? (tail == 0 ? 128 : tail) : 128;
    data.payload.range(31, 0) = seq + 0x10;
    return data;
}

ReduceVectorBeat Beat(const ReduceStreamWireHeader &header,
                      uint64_t beat_id = 0) {
    const VectorWork work = ComputeVectorWork(
        header.stream.total_elements, 1, 512, header.stream.dtype);
    const bool final = beat_id + 1 == header.stream.vector_beats;
    ReduceVectorBeat beat;
    beat.key = {header.stream.key, header.reduce_stage_id, beat_id};
    beat.geometry = {header.stream.dtype, 512,
                     {work.lanes,
                      final ? work.tail_valid_lanes : work.lanes}};
    beat.slices.resize(4);
    for (size_t i = 0; i < beat.slices.size(); ++i)
        beat.slices[i].range(31, 0) = beat_id * 4 + i + 0x10;
    return beat;
}

bool SameHeader(const ReduceStreamWireHeader &lhs,
                const ReduceStreamWireHeader &rhs) {
    return lhs.stream.wire_version == rhs.stream.wire_version &&
           lhs.stream.tree_id == rhs.stream.tree_id &&
           lhs.stream.key == rhs.stream.key &&
           lhs.stream.dtype == rhs.stream.dtype &&
           lhs.stream.op == rhs.stream.op &&
           lhs.stream.total_elements == rhs.stream.total_elements &&
           lhs.stream.physical_data_flits ==
               rhs.stream.physical_data_flits &&
           lhs.stream.vector_beats == rhs.stream.vector_beats &&
           lhs.reduce_stage_id == rhs.reduce_stage_id &&
           lhs.source_id == rhs.source_id &&
           lhs.tail_valid_lanes == rhs.tail_valid_lanes;
}

void DrainHeader(ReduceStreamFiniteState &state,
                 const ReduceStreamWireHeader &header) {
    for (uint32_t seq = 0; seq < header.stream.physical_data_flits; ++seq) {
        auto data = Data(header, seq);
        while (state.AcceptData(data) ==
               ReduceStreamAcceptStatus::BACKPRESSURE)
            (void)state.PopAssembled(header.Route());
    }
    while (state.PopAssembled(header.Route())) {}
    if (!state.TryCloseHeader(header.Route()))
        throw std::logic_error("test failed to drain stream header");
}

} // namespace

int RunCollR3SelfTest() {
    failures = checks = 0;
    std::cout << "==== NoC collective refactor R3 stream/state self-test ===="
              << std::endl;

    const auto header = Header();
    const auto header_wire = SerializeReduceStreamHeader(header);
    Check(IsReduceStreamHeaderWire(header_wire) &&
              SameHeader(header, DeserializeReduceStreamHeader(header_wire)),
          "stream header wire round-trips all identity and geometry fields");

    const auto data = Data(header, 4);
    const auto data_wire = SerializeReduceStreamData(data);
    const auto decoded_data = DeserializeReduceStreamData(data_wire);
    Check(IsReduceStreamDataWire(data_wire) &&
              decoded_data.route == data.route &&
              decoded_data.seq_id == data.seq_id &&
              decoded_data.length_bits == 48 && decoded_data.is_tail &&
              decoded_data.payload == data.payload,
          "stream data wire round-trips compact route, seq, tail, and payload");
    Check(!IsReduceStreamDataWire(header_wire) &&
              !IsReduceStreamHeaderWire(data_wire) &&
              !IsReduceStreamHeaderWire(sc_bv<256>(0)),
          "header/data detection is segmented and rejects ordinary zero wire");

    {
        auto bad = header_wire;
        bad[240] = true;
        auto bad_enum = header_wire;
        bad_enum.range(170, 168) = 7;
        auto bad_count = header_wire;
        bad_count.range(196, 173) = 71;
        Check(Throws([&] { DeserializeReduceStreamHeader(bad); }) &&
                  Throws([&] { DeserializeReduceStreamHeader(bad_enum); }) &&
                  Throws([&] { DeserializeReduceStreamHeader(bad_count); }),
              "header decoder rejects reserved bits, enums, and count mismatch");
    }
    {
        auto bad = data_wire;
        bad[250] = true;
        auto bad_length = data_wire;
        bad_length.range(199, 192) = 0;
        Check(Throws([&] { DeserializeReduceStreamData(bad); }) &&
                  Throws([&] { DeserializeReduceStreamData(bad_length); }),
              "data decoder rejects reserved bits and illegal length");
    }
    {
        auto bad = header;
        bad.stream.key.collective.group_id = 1u << 24;
        auto bad_tail = header;
        ++bad_tail.tail_valid_lanes;
        Check(Throws([&] { SerializeReduceStreamHeader(bad); }) &&
                  Throws([&] { SerializeReduceStreamHeader(bad_tail); }),
              "header encoder rejects range overflow and wrong lane tail");
    }
    Check(!Throws([] {
              ValidateReduceWireCompatibility(
                  ReduceWireVersion::STREAM_V2,
                  ReduceWireVersion::STREAM_V2);
          }) &&
              Throws([] {
                  ValidateReduceWireCompatibility(
                      ReduceWireVersion::LEGACY_TWO_SEGMENT,
                      ReduceWireVersion::STREAM_V2);
              }),
          "one collective accepts one wire version and rejects legacy mixing");

    const auto s1 = BuildBinaryReduceSchedule(1);
    const auto s2 = BuildBinaryReduceSchedule(2);
    const auto s3 = BuildBinaryReduceSchedule(3);
    const auto s5 = BuildBinaryReduceSchedule(5);
    Check(s1.bypass && s1.stages.empty() && s1.IssueCount(9) == 0,
          "single-member reduce bypasses DCA with zero issues");
    Check(s2.stages.size() == 1 && s3.stages.size() == 2 &&
              s5.stages.size() == 4,
          "1/2/3/5 inputs expand to 0/1/2/4 binary stages");
    Check(s5.stages[0].inputs[0] ==
              ReduceStageInputRef{ReduceStageInputKind::NETWORK_INPUT, 0} &&
              s5.stages[0].inputs[1] ==
              ReduceStageInputRef{ReduceStageInputKind::NETWORK_INPUT, 1} &&
              s5.stages[3].inputs[0] ==
              ReduceStageInputRef{ReduceStageInputKind::LOCAL_FEEDBACK, 2} &&
              s5.stages[3].inputs[1] ==
              ReduceStageInputRef{ReduceStageInputKind::NETWORK_INPUT, 4} &&
              s5.stages[3].final_output,
          "arbitrary fan-in uses deterministic left-fold feedback stages");
    Check(s5.IssueCount(7) == 28 && Throws([] {
              (void)BuildBinaryReduceSchedule(0);
          }),
          "stage issue count is stages times vector beats and rejects zero input");

    {
        ReduceStreamAssembler assembler(header, 2);
        std::vector<ReduceVectorBeat> beats;
        for (uint32_t seq = 0; seq < 5; ++seq) {
            const auto status = assembler.Accept(Data(header, seq));
            if (status == ReduceStreamAcceptStatus::BEAT_READY ||
                status == ReduceStreamAcceptStatus::STREAM_COMPLETE)
                if (auto beat = assembler.PopBeat())
                    beats.push_back(std::move(*beat));
        }
        assembler.Finish();
        Check(beats.size() == 2 && beats[0].slices.size() == 4 &&
                  beats[0].geometry.lane_mask.valid_lanes == 64 &&
                  beats[1].geometry.lane_mask.valid_lanes == 6,
              "128-bit flits assemble 4:1 into full and masked 512-bit beats");
        Check(beats[1].slices[1] == sc_bv<128>(0) &&
                  beats[1].slices[2] == sc_bv<128>(0) &&
                  beats[1].slices[3] == sc_bv<128>(0),
              "partial final vector beat is deterministically zero padded");
        std::vector<ReduceStreamDataFlit> split;
        for (const auto &beat : beats) {
            auto part = SplitReduceVectorBeat(header, beat);
            split.insert(split.end(), part.begin(), part.end());
        }
        Check(split.size() == 5 && split[0].seq_id == 0 &&
                  split[4].seq_id == 4 && split[4].is_tail &&
                  split[4].length_bits == 48,
              "splitter reverses vector beats to exact F flits and tail bits");
    }
    {
        ReduceStreamAssembler assembler(Header(80), 1);
        for (uint32_t seq = 0; seq < 4; ++seq)
            assembler.Accept(Data(Header(80), seq));
        Check(assembler.Accept(Data(Header(80), 4)) ==
                  ReduceStreamAcceptStatus::BACKPRESSURE,
              "full assembler ready queue backpressures without consuming seq");
        Check(assembler.PopBeat().has_value() &&
                  assembler.Accept(Data(Header(80), 4)) ==
                      ReduceStreamAcceptStatus::STREAM_COMPLETE &&
                  assembler.PopBeat().has_value(),
              "assembler retry resumes after ready queue drains");
        assembler.Finish();
        Check(assembler.Residual() == 0,
              "completed assembler has zero fragments and ready beats");
    }
    Check(Throws([&] {
              ReduceStreamAssembler a(header, 1);
              a.Accept(Data(header, 1));
          }) && Throws([&] {
              ReduceStreamAssembler a(header, 1);
              auto wrong = Data(header, 0);
              ++wrong.route.source_id;
              a.Accept(wrong);
          }),
          "assembler rejects out-of-order sequence and wrong route");
    Check(Throws([&] {
              ReduceStreamAssembler a(header, 1);
              auto wrong = Data(header, 0);
              wrong.is_tail = true;
              a.Accept(wrong);
          }) && Throws([&] {
              ReduceStreamAssembler a(header, 1);
              a.Accept(Data(header, 0));
              a.Finish();
          }),
          "assembler rejects wrong tail and truncated stream");
    Check(Throws([&] {
              ReduceStreamAssembler a(Header(16), 1);
              a.Accept(Data(Header(16), 0));
              a.Accept(Data(Header(16), 0));
          }),
          "assembler rejects extra or duplicate data after completion");
    Check(Throws([&] {
              auto wrong = Beat(header);
              ++wrong.key.stream.stream_id;
              (void)SplitReduceVectorBeat(header, wrong);
          }) && Throws([&] {
              auto wrong = Beat(header);
              wrong.geometry.dtype = CollDType::INT32;
              (void)SplitReduceVectorBeat(header, wrong);
          }),
          "splitter rejects key and dtype geometry mismatch");

    {
        ReduceStreamFiniteState state({1, 1, 2, 1, 1, 1, 1, 1});
        auto second = Header(70, 10, 4);
        Check(state.TryOpenHeader(header) && !state.TryOpenHeader(second),
              "finite header table backpressures at capacity");
        DrainHeader(state, header);
        Check(state.TryOpenHeader(second),
              "header capacity recovers after completed stream closes");
        DrainHeader(state, second);
        Check(state.Drained(),
              "header/assembler lifecycle drains every residual class");
    }
    {
        ReduceStreamFiniteState queues({1, 1, 4, 1, 1, 1, 2, 2});
        auto a = Beat(header, 0);
        auto b = Beat(header, 1);
        Check(queues.TryPushNetworkOperand({a, 0}) &&
                  queues.TryPushFeedbackOperand({b, 1}),
              "network and local-feedback inputs occupy independent FIFOs");
        auto first = queues.PopArbitratedInput();
        queues.TryPushNetworkOperand({a, 1});
        auto second = queues.PopArbitratedInput();
        Check(first && second && first->beat.key == a.key &&
                  second->beat.key == b.key,
              "network/feedback arbiter gives both continuously active sides progress");
        while (queues.PopArbitratedInput()) {}
        Check(queues.Drained(), "input arbitration queues drain to zero");
    }
    {
        ReduceStreamFiniteState state({1, 1, 3, 1, 1, 1, 1, 1});
        auto a = Beat(header, 0);
        auto b = Beat(header, 1);
        auto c = a;
        c.key.vector_beat_id = 7;
        Check(state.AcceptMatchedOperand({a, 0}) ==
                  ReduceStateStatus::ACCEPTED &&
                  state.AcceptMatchedOperand({b, 0}) ==
                  ReduceStateStatus::ACCEPTED &&
                  state.AcceptMatchedOperand({c, 0}) ==
                  ReduceStateStatus::BACKPRESSURE,
              "operand FIFO reserves its final slot and backpressures new keys");
        Check(state.AcceptMatchedOperand({a, 1}) ==
                  ReduceStateStatus::READY && state.TryQueueMatched(a.key),
              "reserved operand slot completes a pair and restores progress");
        auto wrong = b;
        wrong.geometry.lane_mask.valid_lanes = 1;
        Check(Throws([&] { state.AcceptMatchedOperand({wrong, 1}); }),
              "operand matcher rejects geometry mismatch");
        Check(state.AcceptMatchedOperand({b, 1}) ==
                  ReduceStateStatus::READY &&
                  !state.TryQueueMatched(b.key),
              "full issue queue backpressures a complete pair without loss");
        Check(state.TryIssue(1) && state.TryQueueMatched(b.key) &&
                  !state.TryIssue(2),
              "full inflight table backpressures issue and accepts retry later");
        Check(Throws([&] { state.TryIssue(1); }),
              "live DCA tags cannot be reused");
        Check(state.TryComplete(1, a) && state.TryIssue(2) &&
                  !state.TryComplete(2, b),
              "full result FIFO backpressures completion without losing inflight");
        Check(state.PopFinalResult().has_value() &&
                  state.TryComplete(2, b),
              "result completion retry succeeds after consumer drains FIFO");
        Check(state.TryPushFeedbackOperand({c, 0}) &&
                  !state.TryRouteResultToFeedback(c.key, 1),
              "full local-feedback FIFO atomically backpressures result routing");
        (void)state.PopArbitratedInput();
        Check(state.TryRouteResultToFeedback(c.key, 1) &&
                  state.PopArbitratedInput().has_value(),
              "local-feedback routing retries and preserves the result");
        Check(state.Drained(),
              "operand/issue/inflight/result/feedback pipeline drains to zero");
    }

    Check(Throws([] {
              ReduceStreamFiniteState bad({1, 1, 1, 1, 1, 1, 1, 1});
          }),
          "finite-state configuration rejects operand capacity below one pair");

    std::cout << "R3 self-test: " << (checks - failures) << "/" << checks
              << " checks passed" << std::endl;
    return failures;
}
