#include "dte/dte_unit.h"

#include "common/config.h"
#include "dte/dte_streaming.h"
#include "defs/spec.h"
#include "macros/macros.h"
#include "trace/Event_engine.h"
#include "utils/config_utils.h"
#include "utils/msg_utils.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
int g_fail = 0;
int g_total = 0;

void check(bool condition, const std::string &name) {
    ++g_total;
    if (condition) {
        std::cout << "  [ ok ] " << name << std::endl;
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << name << std::endl;
    }
}

long long cycleOf(const sc_time &time) {
    return static_cast<long long>(time.value() /
                                  sc_time(CYCLE, SC_NS).value());
}

struct IssueSpec {
    int issue_cycle;
    uint64_t payload_bits;
    DteDir dir = DteDir::SPM_TO_REMOTE;
};

struct DTEProbe : sc_module {
    std::unique_ptr<DTEUnit> dte;
    std::vector<IssueSpec> script;
    std::vector<DteTransferContext *> contexts;
    bool early_release_result = true;

    SC_HAS_PROCESS(DTEProbe);
    DTEProbe(sc_module_name name, const DTEConfig &config,
             Event_engine *event_engine = nullptr)
        : sc_module(name) {
        dte = std::make_unique<DTEUnit>("dte", config, -1, event_engine);
        SC_THREAD(drive);
    }

    void drive() {
        for (const IssueSpec &spec : script) {
            const sc_time target(spec.issue_cycle * CYCLE, SC_NS);
            if (sc_time_stamp() < target)
                wait(target - sc_time_stamp());
            DteTransferContext &ctx = dte->Issue(spec.payload_bits, spec.dir);
            contexts.push_back(&ctx);
            if (contexts.size() == 1)
                early_release_result = dte->Release(ctx.xfer_id);
        }
    }
};

struct DTEEventProbe : sc_module {
    std::unique_ptr<DTEUnit> dte;
    DteTransferContext *first = nullptr;
    DteTransferContext *second = nullptr;
    sc_event issued;
    bool first_woke = false;
    bool second_woke = false;
    long long first_wake_cycle = -1;
    long long second_wake_cycle = -1;

    SC_HAS_PROCESS(DTEEventProbe);
    DTEEventProbe(sc_module_name name, const DTEConfig &config)
        : sc_module(name) {
        dte = std::make_unique<DTEUnit>("dte", config);
        SC_THREAD(issueBoth);
        SC_THREAD(waitFirst);
        SC_THREAD(waitSecond);
    }

    void issueBoth() {
        first = &dte->Issue(128, DteDir::SPM_TO_REMOTE);
        second = &dte->Issue(128, DteDir::REMOTE_TO_SPM);
        issued.notify(SC_ZERO_TIME);
    }
    void waitFirst() {
        wait(issued);
        wait(first->done);
        first_woke = true;
        first_wake_cycle = cycleOf(sc_time_stamp());
    }
    void waitSecond() {
        wait(issued);
        wait(second->done);
        second_woke = true;
        second_wake_cycle = cycleOf(sc_time_stamp());
    }
};

bool allCompleted(const DTEProbe &probe) {
    if (probe.contexts.size() != probe.script.size())
        return false;
    for (const DteTransferContext *ctx : probe.contexts)
        if (ctx->state != DteTransferState::COMPLETED)
            return false;
    return true;
}

std::vector<long long> completionCycles(const DTEProbe &probe) {
    std::vector<long long> result;
    for (const DteTransferContext *ctx : probe.contexts)
        result.push_back(cycleOf(ctx->completion_time));
    return result;
}

bool busIntervalsDoNotOverlap(const DTEProbe &probe) {
    std::vector<std::pair<long long, long long>> intervals;
    for (const DteTransferContext *ctx : probe.contexts)
        intervals.push_back({cycleOf(ctx->transmit_start_time),
                             cycleOf(ctx->completion_time)});
    std::sort(intervals.begin(), intervals.end());
    for (size_t i = 1; i < intervals.size(); ++i)
        if (intervals[i - 1].second > intervals[i].first)
            return false;
    return true;
}

template <typename Exception, typename Function>
bool throwsExpected(Function function) {
    try {
        function();
    } catch (const Exception &) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}
} // namespace

int RunDTEV0SelfTest() {
    g_fail = 0;
    g_total = 0;
    std::cout << "==== DTE V0 self-test ====" << std::endl;

    // V0a: bit 契约和纯函数。
    {
        Send_prim prim(SEND_DATA);
        prim.max_packet = 1;
        prim.end_length = 1;
        check(ComputeSendPayloadBits(prim) == 1, "payload: 1 packet x 1 bit");
        prim.end_length = M_D_DATA;
        check(ComputeSendPayloadBits(prim) == 128,
              "payload: 1 full 128-bit packet");
        prim.max_packet = 2;
        prim.end_length = 1;
        check(ComputeSendPayloadBits(prim) == 129,
              "payload: full packet plus 1-bit tail");
        prim.end_length = M_D_DATA;
        check(ComputeSendPayloadBits(prim) == 256,
              "payload: 2 full packets");
        prim.stripe_count = 4;
        check(ComputeSendPayloadBits(prim) == 256,
              "payload independent of stripe_count");

        const int old_payload_per_cycle = HW_NOC_PAYLOAD_PER_CYCLE;
        HW_NOC_PAYLOAD_PER_CYCLE = 20;
        Send_prim grouped_full(SEND_DATA);
        CalculatePacketNum(1600, 1, 1, grouped_full.max_packet,
                           grouped_full.end_length, grouped_full.packet_scale,
                           grouped_full.packets_in_last_group);
        check(grouped_full.max_packet == 5 &&
                  grouped_full.packet_scale == 20 &&
                  grouped_full.packets_in_last_group == 20 &&
                  grouped_full.end_length == M_D_DATA,
              "packet calculation preserves 100 raw packets at scale 20");
        check(ComputeSendPayloadBits(grouped_full) == 12800,
              "DTE restores full payload from grouped packet metadata");

        Send_prim grouped_tail(SEND_DATA);
        CalculatePacketNum(1442, 1, 1, grouped_tail.max_packet,
                           grouped_tail.end_length, grouped_tail.packet_scale,
                           grouped_tail.packets_in_last_group);
        check(grouped_tail.max_packet == 5 &&
                  grouped_tail.packets_in_last_group == 11 &&
                  grouped_tail.end_length == 16 &&
                  ComputeSendPayloadBits(grouped_tail) == 11536,
              "DTE restores a partial final group and short tail exactly");
        grouped_tail.output_label = "dte_v0_payload";
        Send_prim decoded(SEND_DATA);
        decoded.deserialize(grouped_tail.serialize());
        check(decoded.packet_scale == 20 &&
                  decoded.packets_in_last_group == 11 &&
                  ComputeSendPayloadBits(decoded) == 11536,
              "Send_prim wire format preserves exact payload metadata");

        Send_prim request(SEND_REQ);
        request.max_packet = grouped_tail.max_packet;
        request.end_length = grouped_tail.end_length;
        request.packet_scale = grouped_tail.packet_scale;
        request.packets_in_last_group =
            grouped_tail.packets_in_last_group;
        Send_prim decoded_request(SEND_REQ);
        decoded_request.deserialize(request.serialize());
        check(ComputeSendPayloadBits(decoded_request) == 11536,
              "SEND_REQ wire carries its paired DATA payload metadata");

        Msg request_msg(MSG_TYPE::REQUEST, 1, 7, 0);
        request_msg.flow_packets_ = grouped_tail.max_packet;
        request_msg.dte_payload_bits_ =
            ComputeSendPayloadBits(grouped_tail);
        Msg decoded_msg = DeserializeMsg(SerializeMsg(request_msg));
        check(decoded_msg.flow_packets_ == grouped_tail.max_packet &&
                  decoded_msg.dte_payload_bits_ == 11536,
              "REQUEST wire carries exact DTE flow payload bits");
        Send_prim degenerate_request(SEND_REQ);
        degenerate_request.max_packet = 0;
        degenerate_request.end_length = 1;
        Msg disabled_request(MSG_TYPE::REQUEST, 1, 7, 0);
        bool disabled_path_unchanged = true;
        try {
            AttachRequestDtePayload(disabled_request, degenerate_request,
                                    false);
        } catch (...) {
            disabled_path_unchanged = false;
        }
        check(disabled_path_unchanged &&
                  disabled_request.dte_payload_bits_ == 0,
              "DTE-off REQUEST does not validate or attach payload metadata");
        check(throwsExpected<std::invalid_argument>([&]() {
                  AttachRequestDtePayload(disabled_request,
                                          degenerate_request, true);
              }),
              "DTE-on REQUEST applies strict payload validation");
        HW_NOC_PAYLOAD_PER_CYCLE = old_payload_per_cycle;

        prim.max_packet = std::numeric_limits<int>::max();
        prim.end_length = M_D_DATA;
        check(ComputeSendPayloadBits(prim) ==
                  uint64_t(std::numeric_limits<int>::max()) * M_D_DATA,
              "payload handles maximum encoded packet count");

        prim.max_packet = 0;
        prim.end_length = 0;
        check(ComputeSendPayloadBits(prim) == 0, "empty logical payload");
        prim.end_length = 1;
        check(throwsExpected<std::invalid_argument>(
                  [&]() { (void)ComputeSendPayloadBits(prim); }),
              "empty payload rejects non-zero tail");
        prim.max_packet = 1;
        prim.end_length = 129;
        check(throwsExpected<std::invalid_argument>(
                  [&]() { (void)ComputeSendPayloadBits(prim); }),
              "payload rejects tail wider than M_D_DATA");

        check(CeilDivU64(1, 128) == 1, "ceil_div 1/128");
        check(CeilDivU64(128, 128) == 1, "ceil_div 128/128");
        check(CeilDivU64(129, 128) == 2, "ceil_div 129/128");
        check(throwsExpected<std::invalid_argument>(
                  []() { (void)CeilDivU64(1, 0); }),
              "ceil_div rejects zero denominator");
        check(NanosecondsToDteCycles(5) == 3,
              "nanoseconds convert to DTE cycles with ceil rounding");
        check(ProjectDteSourceTailToDestinationNs(405, 899, 415) ==
                  909 &&
                  CombineDteStreamingTailsNs(909, 913, 429, 2) == 915,
              "DTE V2b closed-form tail helpers match the three-stage oracle");

        DTEConfig converted = MakeDTEConfig(2, 1024, 5, 3);
        check(converted.gamma_cycles == 3 && converted.tau_launch_cycles == 2,
              "DTE config converts nanoseconds to cycles");
        check(throwsExpected<std::invalid_argument>(
                  []() { (void)MakeDTEConfig(2, 1024, -1, 3); }),
              "DTE config rejects negative timing");

        Msg physical(true, DATA, 1, 1, 0, 7, 17, sc_bv<128>(0));
        check(ComputeMsgLogicalPayloadBits(physical, false) == 17,
              "physical Msg contributes its exact bit length");
        Msg representative = physical;
        representative.roofline_packets_ = 4;
        check(ComputeMsgLogicalPayloadBits(representative, true) == 401,
              "behavioral representative restores full logical payload bits");
        representative.roofline_packets_ = 0;
        check(throwsExpected<std::invalid_argument>(
                  [&]() {
                      (void)ComputeMsgLogicalPayloadBits(representative, true);
                  }),
              "behavioral representative requires a positive packet count");

        Msg streaming = physical;
        streaming.dte_stream_source_first_ns_ = 0x123456789abcULL;
        streaming.dte_stream_source_done_ns_ = 0x23456789abcdULL;
        streaming.dte_stream_network_tail_cycles_ = 0x89abcdefU;
        Msg streaming_wire = DeserializeMsg(SerializeMsg(streaming));
        check(streaming_wire.dte_stream_source_first_ns_ ==
                  streaming.dte_stream_source_first_ns_ &&
                  streaming_wire.dte_stream_source_done_ns_ ==
                  streaming.dte_stream_source_done_ns_ &&
                  streaming_wire.dte_stream_network_tail_cycles_ ==
                  streaming.dte_stream_network_tail_cycles_,
              "DTE V2b DATA streaming metadata survives wire round trip");

        Msg legacy = physical;
        legacy.data_ = sc_bv<128>(1);
        Msg legacy_wire = DeserializeMsg(SerializeMsg(legacy));
        check(legacy_wire.data_ == legacy.data_,
              "DTE V2b leaves legacy DATA wire payload unchanged");
        check(legacy_wire.dte_stream_source_first_ns_ == 1 &&
                  legacy_wire.dte_stream_source_done_ns_ == 0 &&
                  legacy_wire.dte_stream_network_tail_cycles_ == 0,
              "DTE V2b legacy DATA retains raw decode that consumers must gate");
    }

    // 配置负例不需要构造 SystemC module。
    {
        DTEConfig bad;
        bad.channel_count = 0;
        check(throwsExpected<std::invalid_argument>(
                  [&]() { DTEUnit::ValidateConfig(bad); }),
              "config rejects channel_count=0");
        bad = DTEConfig{};
        bad.bit_width_bits = 0;
        check(throwsExpected<std::invalid_argument>(
                  [&]() { DTEUnit::ValidateConfig(bad); }),
              "config rejects bit_width_bits=0");
        bad = DTEConfig{};
        bad.gamma_cycles = std::numeric_limits<uint64_t>::max();
        bad.tau_launch_cycles = 1;
        check(throwsExpected<std::overflow_error>(
                  [&]() { DTEUnit::ValidateConfig(bad); }),
              "config rejects launch cycle overflow");
    }

    // JSON 配置字段与仿真开关。
    {
        nlohmann::json valid = {
            {"id", 7}, {"dte_channel_count", 4}, {"dte_bit_width", 1024}};
        CoreHWConfig parsed = valid;
        check(parsed.dte_channel_count == 4 && parsed.dte_bit_width == 1024,
              "CoreHWConfig parses per-core DTE parameters");

        nlohmann::json defaults = {{"id", 8}};
        CoreHWConfig parsed_defaults = defaults;
        check(parsed_defaults.dte_channel_count == 2 &&
                  parsed_defaults.dte_bit_width == 2048,
              "CoreHWConfig supplies regression-safe DTE defaults");

        nlohmann::json bad_channel = {
            {"id", 9}, {"dte_channel_count", 0}};
        check(throwsExpected<std::invalid_argument>(
                  [&]() {
                      CoreHWConfig bad = bad_channel;
                      (void)bad;
                  }),
              "CoreHWConfig rejects invalid DTE channel count");

        const bool old_use_dte = SPEC_USE_BEHA_DTE;
        ParseSimulationConfig(
            nlohmann::json{{"dte", {{"use_beha_dte", true}}}});
        check(SPEC_USE_BEHA_DTE,
              "simulation config parses dte.use_beha_dte");

        const bool old_parallel = SPEC_SEND_RECV_PARALLEL;
        const bool old_streaming = SPEC_DTE_STREAMING;
        const SIM_MODE old_mode = SYSTEM_MODE;
        SPEC_USE_BEHA_DTE = false;
        SPEC_SEND_RECV_PARALLEL = false;
        SYSTEM_MODE = SIM_DATAFLOW;
        bool dataflow_parallel_accepted = true;
        try {
            ParseSimulationConfig(nlohmann::json{
                {"noc", {{"send_recv_parallel", true}}},
                {"dte", {{"use_beha_dte", true}}}});
        } catch (...) {
            dataflow_parallel_accepted = false;
        }
        check(dataflow_parallel_accepted && SPEC_USE_BEHA_DTE &&
                  SPEC_SEND_RECV_PARALLEL,
              "DTE V2a accepts parallel dataflow mode");

        SPEC_USE_BEHA_DTE = false;
        SPEC_SEND_RECV_PARALLEL = false;
        SYSTEM_MODE = SIM_PD;
        check(throwsExpected<std::invalid_argument>(
                  []() {
                      ParseSimulationConfig(nlohmann::json{
                          {"noc", {{"send_recv_parallel", true}}},
                          {"dte", {{"use_beha_dte", true}}}});
                  }),
              "DTE V2a rejects parallel non-dataflow mode");
        SPEC_USE_BEHA_DTE = false;
        SPEC_DTE_STREAMING = false;
        SPEC_SEND_RECV_PARALLEL = false;
        SYSTEM_MODE = SIM_DATAFLOW;
        bool streaming_accepted = true;
        try {
            ParseSimulationConfig(nlohmann::json{
                {"noc", {{"send_recv_parallel", false}}},
                {"dte", {{"use_beha_dte", true}, {"streaming", true}}}});
        } catch (...) {
            streaming_accepted = false;
        }
        check(streaming_accepted && SPEC_USE_BEHA_DTE &&
                  SPEC_DTE_STREAMING,
              "DTE V2b accepts sequential dataflow streaming mode");

        SPEC_USE_BEHA_DTE = false;
        SPEC_DTE_STREAMING = false;
        SPEC_SEND_RECV_PARALLEL = false;
        check(throwsExpected<std::invalid_argument>(
                  []() {
                      ParseSimulationConfig(nlohmann::json{
                          {"dte", {{"use_beha_dte", false},
                                   {"streaming", true}}}});
                  }),
              "DTE V2b rejects streaming while DTE is disabled");

        SPEC_USE_BEHA_DTE = false;
        SPEC_DTE_STREAMING = false;
        SPEC_SEND_RECV_PARALLEL = false;
        SYSTEM_MODE = SIM_PD;
        check(throwsExpected<std::invalid_argument>(
                  []() {
                      ParseSimulationConfig(nlohmann::json{
                          {"noc", {{"send_recv_parallel", false}}},
                          {"dte", {{"use_beha_dte", true},
                                   {"streaming", true}}}});
                  }),
              "DTE V2b rejects streaming in non-dataflow mode");

        SPEC_USE_BEHA_DTE = false;
        SPEC_DTE_STREAMING = false;
        SPEC_SEND_RECV_PARALLEL = false;
        SYSTEM_MODE = SIM_DATAFLOW;
        check(throwsExpected<std::invalid_argument>(
                  []() {
                      ParseSimulationConfig(nlohmann::json{
                          {"noc", {{"send_recv_parallel", true}}},
                          {"dte", {{"use_beha_dte", true},
                                   {"streaming", true}}}});
                  }),
              "DTE V2b rejects streaming in the parallel dispatcher");

        SYSTEM_MODE = old_mode;
        SPEC_SEND_RECV_PARALLEL = old_parallel;
        SPEC_DTE_STREAMING = old_streaming;
        SPEC_USE_BEHA_DTE = old_use_dte;
    }

    const DTEConfig c1{1, 128, 2, 1};
    const DTEConfig c2{2, 128, 2, 1};
    const DTEConfig c3{3, 128, 2, 1};
    const DTEConfig c4{4, 128, 2, 1};

    auto single1 = std::make_unique<DTEProbe>("dte_single_c1", c1);
    single1->script = {{0, 129}};
    auto single4 = std::make_unique<DTEProbe>("dte_single_c4", c4);
    single4->script = {{0, 129}};

    auto burst1 = std::make_unique<DTEProbe>("dte_burst_c1", c1);
    burst1->script = {{0, 129}, {0, 129}, {0, 129}, {0, 129}};
    auto burst2 = std::make_unique<DTEProbe>("dte_burst_c2", c2);
    burst2->script = burst1->script;
    auto burst4 = std::make_unique<DTEProbe>("dte_burst_c4", c4);
    burst4->script = burst1->script;

    auto rr = std::make_unique<DTEProbe>("dte_rr_c3", c3);
    rr->script = {{0, 128}, {0, 128}, {0, 128}};

    auto event_probe = std::make_unique<DTEEventProbe>("dte_event_probe", c2);

    auto trace_engine =
        std::make_unique<Event_engine>("dte_v0_trace_engine", 1000);
    auto trace_probe = std::make_unique<DTEProbe>(
        "dte_trace_probe", c1, trace_engine.get());
    trace_probe->script = {{0, 128}};

    auto stagger = std::make_unique<DTEProbe>("dte_stagger_c2", c2);
    stagger->script = {{0, 129, DteDir::SPM_TO_REMOTE},
                       {2, 128, DteDir::REMOTE_TO_SPM},
                       {9, 1, DteDir::SPM_TO_REMOTE}};

    auto zero = std::make_unique<DTEProbe>("dte_zero_reject", c1);
    zero->script = {};
    check(throwsExpected<std::invalid_argument>(
              [&]() { (void)zero->dte->Issue(0, DteDir::SPM_TO_REMOTE); }),
          "Issue rejects zero payload");

    sc_start(80 * CYCLE, SC_NS);

    check(allCompleted(*single1) && allCompleted(*single4),
          "single transfers complete");
    check(cycleOf(single1->contexts[0]->scheduled_completion_time) == 5,
          "transmit start records the scheduled completion time for V2b");
    check(completionCycles(*single1) == std::vector<long long>{5},
          "single completion = launch(3)+transmit(2) cycles");
    check(completionCycles(*single4) == std::vector<long long>{5},
          "single latency independent of channel_count");
    check(!single1->early_release_result,
          "context cannot be released before completion");
    const uint64_t release_id = single1->contexts[0]->xfer_id;
    check(single1->dte->Release(release_id),
          "completed context can be released");
    check(!single1->dte->Release(release_id),
          "released context cannot be released twice");

    check(allCompleted(*burst1) &&
              completionCycles(*burst1) ==
                  std::vector<long long>({5, 10, 15, 20}),
          "channel=1 admission serializes four transfers");
    check(allCompleted(*burst2) &&
              completionCycles(*burst2) ==
                  std::vector<long long>({5, 7, 10, 12}),
          "channel=2 overlaps launch and bounds active transfers");
    check(allCompleted(*burst4) &&
              completionCycles(*burst4) ==
                  std::vector<long long>({5, 7, 9, 11}),
          "channel=4 admits all while shared bus remains serial");
    check(burst1->dte->MaxActiveCount() == 1 &&
              burst2->dte->MaxActiveCount() == 2 &&
              burst4->dte->MaxActiveCount() == 4,
          "max active count obeys channel_count");
    check(busIntervalsDoNotOverlap(*burst1) &&
              busIntervalsDoNotOverlap(*burst2) &&
              busIntervalsDoNotOverlap(*burst4),
          "shared bus transmit intervals never overlap");

    check(allCompleted(*rr), "RR probe completes");
    check(rr->contexts.size() == 3 && rr->contexts[0]->channel_id == 0 &&
              rr->contexts[1]->channel_id == 1 &&
              rr->contexts[2]->channel_id == 2,
          "simultaneous descriptors admitted to channels 0,1,2");
    check(completionCycles(*rr) == std::vector<long long>({4, 5, 6}),
          "shared bus serves ready channels in RR order");
    check(allCompleted(*stagger) && busIntervalsDoNotOverlap(*stagger),
          "staggered issues and both directions complete without overlap");

    size_t dte_trace_events = 0;
    for (const Trace_event &event : trace_engine->traced_event_list)
        if (event.module_name == "dte_trace_probe.dte")
            ++dte_trace_events;
    check(allCompleted(*trace_probe) && dte_trace_events == 8,
          "trace records B/E for pending, launch, bus_wait and transmit");

    check(event_probe->first_woke && event_probe->second_woke &&
              event_probe->first_wake_cycle == 4 &&
              event_probe->second_wake_cycle == 5,
          "independent waiters receive their own completion events");
    check(burst1->dte->PendingCount() == 0 &&
              burst1->dte->ActiveCount() == 0 && !burst1->dte->BusBusy(),
          "completed simulation drains pending, active and bus state");

    std::cout << "DTE V0 self-test: "
              << (g_fail == 0 ? "PASS" : "FAILURES=" + std::to_string(g_fail))
              << " (" << g_total << " checks)" << std::endl;
    return g_fail;
}
