#include "memory/hbm_r4_selftest.h"

#include "defs/global.h"
#include "defs/spec.h"
#include "die/port.h"
#include "die/d2d_link.h"
#include "dte/coll_multicast.h"
#include "memory/core_mem_adapter.h"
#include "memory/hbm_address_map.h"
#include "memory/hbm_network.h"
#include "memory/hbm_runtime.h"
#include "router/router.h"
#include "trace/Event_engine.h"
#include "utils/router_utils.h"
#include "utils/msg_utils.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <string>
#include <iostream>
#include <memory>
#include <vector>

using namespace sc_core;

namespace {
int failures = 0, checks = 0;

void Check(bool condition, const char *name) {
    ++checks;
    if (!condition) { ++failures; std::cout << "  [FAIL] " << name << '\n'; }
    else std::cout << "  [ ok ] " << name << '\n';
}

void ConfigureR4() {
    GRID_X = GRID_Y = 3;
    GRID_SIZE = CORES_PER_DIE = 9;
    DIE_X = DIE_Y = 3;
    DIE_COUNT = 9;
    TOTAL_CORES = 81;
    HOST_ENDPOINT_ID = TOTAL_CORES;

    g_die_ports = D2DPortTable{};
    g_die_ports.active = true;
    g_die_ports.ports = {
        {0, 3, WEST, ROLE_C2C, WEST, 1, 2, 16},
        {1, 5, EAST, ROLE_C2C, EAST, 1, 2, 16},
        {2, 7, NORTH, ROLE_C2C, NORTH, 1, 2, 16},
        {3, 1, SOUTH, ROLE_C2C, SOUTH, 1, 2, 16},
    };
    g_die_ports.port_for_host.assign(CORES_PER_DIE, -1);
    g_die_ports.port_for.assign(
        CORES_PER_DIE, std::vector<int>(DIRECTIONS, -1));
    for (int tile = 0; tile < CORES_PER_DIE; ++tile) {
        g_die_ports.port_for[tile][WEST] = 0;
        g_die_ports.port_for[tile][EAST] = 1;
        g_die_ports.port_for[tile][NORTH] = 2;
        g_die_ports.port_for[tile][SOUTH] = 3;
    }
    BuildD2DLinks();
    g_d2d_cfg = D2DLinkConfig{};
    g_d2d_cfg.select_policy = SELECT_NEAREST;
    g_d2d_cfg.select_policy_explicit = true;

    g_memory_system_active = true;
    g_memory_topology = MemoryTopology::kDistributedHbm;
    g_memory_cache_policy = MemoryCachePolicy::kNone;
    ResetCollectiveFabric();
    HBMProfile profile;
    profile.generation = "HBM2";
    profile.channels_per_stack = 1;
    profile.pseudo_channels_per_channel = 2;
    profile.data_rate_gbps_per_pin = 2.0;
    profile.stack_bus_width_bits = 64;
    g_hbm_profiles = {{"r4", profile}};
    HBMStackConfig s0, s8;
    s0.stack_id = 0; s0.compute_die_id = 0; s0.profile = "r4";
    s0.side = NORTH; s0.start_idx = 1; s0.phy_span_tiles = 1;
    s0.capacity_bytes = 1ULL << 20;
    s0.backend_kind = HBMBackendKind::kBehavioral;
    s0.backend_granularity = HBMBackendGranularity::kChannel;
    s0.channel_dram_config = "../DRAMSys/configs/hbm2-example.json";
    s0.behavioral_base_latency_ns = 20;
    s8 = s0; s8.stack_id = 8; s8.compute_die_id = 8;
    g_hbm_stacks = {s0, s8};
    g_hbm_channels = {{0, 0, 4}, {8, 0, 4}};
    g_address_policy = AddressPolicyConfig{};
    g_address_policy.active = true;
    g_address_policy.mode = AddressPolicyMode::kNumaLocalInterleave;
    g_address_policy.home_ranges = {
        {0, 0, 1ULL << 20}, {8, 1ULL << 20, 1ULL << 20}};
    g_address_policy.stack_interleave_bytes = 256;
    g_address_policy.channel_interleave_bytes = 256;
    g_address_policy.pseudo_channel_interleave_bytes = 256;
    g_die_router_pkts.assign(DIE_COUNT, 0);
    g_die_mesh_pkts.assign(DIE_COUNT, 0);
    g_die_noc_sends.assign(DIE_COUNT, 0);
    g_die_noc_stalls.assign(DIE_COUNT, 0);
}

template <typename T>
using Signals = std::vector<std::unique_ptr<sc_signal<T>>>;

class R4RouterFabric {
public:
    struct LinkBundle {
        std::unique_ptr<sc_signal<bool>> in_avail, in_ctrl_avail;
        std::unique_ptr<sc_signal<sc_bv<256>>> out_data, out_ctrl;
        std::unique_ptr<sc_signal<bool>> out_sent, out_ctrl_sent;
        std::unique_ptr<sc_signal<bool>> data_credit, ctrl_credit;
        std::unique_ptr<D2DLinkUnit> unit;
        LinkBundle(int latency, int index)
            : in_avail(std::make_unique<sc_signal<bool>>(sc_gen_unique_name("r4_la"))),
              in_ctrl_avail(std::make_unique<sc_signal<bool>>(sc_gen_unique_name("r4_lca"))),
              out_data(std::make_unique<sc_signal<sc_bv<256>>>(sc_gen_unique_name("r4_ld"))),
              out_ctrl(std::make_unique<sc_signal<sc_bv<256>>>(sc_gen_unique_name("r4_lc"))),
              out_sent(std::make_unique<sc_signal<bool>>(sc_gen_unique_name("r4_ls"))),
              out_ctrl_sent(std::make_unique<sc_signal<bool>>(sc_gen_unique_name("r4_lcs"))),
              data_credit(std::make_unique<sc_signal<bool>>(sc_gen_unique_name("r4_lcr"))),
              ctrl_credit(std::make_unique<sc_signal<bool>>(sc_gen_unique_name("r4_lccr"))),
              unit(std::make_unique<D2DLinkUnit>(
                  sc_gen_unique_name("r4_d2d_link"), latency, index)) {}
    };
    RouterMonitor &monitor;
    Signals<sc_bv<256>> data[DIRECTIONS], ctrl[DIRECTIONS];
    Signals<bool> sent[DIRECTIONS], csent[DIRECTIONS];
    Signals<bool> avail[DIRECTIONS], cavail[DIRECTIONS];
    Signals<sc_bv<256>> center_in, center_ctrl_in;
    Signals<bool> center_sent, center_csent, core_busy, ctrl_busy;
    sc_signal<sc_bv<256>> term_data, term_ctrl;
    sc_signal<bool> term_false;
    std::vector<std::unique_ptr<LinkBundle>> links;

    explicit R4RouterFabric(RouterMonitor &m) : monitor(m) {
        for (int d = 0; d < DIRECTIONS; ++d)
            for (int r = 0; r < TOTAL_CORES; ++r) {
                data[d].push_back(std::make_unique<sc_signal<sc_bv<256>>>(sc_gen_unique_name("r4_d")));
                ctrl[d].push_back(std::make_unique<sc_signal<sc_bv<256>>>(sc_gen_unique_name("r4_c")));
                sent[d].push_back(std::make_unique<sc_signal<bool>>(sc_gen_unique_name("r4_s")));
                csent[d].push_back(std::make_unique<sc_signal<bool>>(sc_gen_unique_name("r4_cs")));
                avail[d].push_back(std::make_unique<sc_signal<bool>>(sc_gen_unique_name("r4_a")));
                cavail[d].push_back(std::make_unique<sc_signal<bool>>(sc_gen_unique_name("r4_ca")));
            }
        for (int r = 0; r < TOTAL_CORES; ++r) {
            center_in.push_back(std::make_unique<sc_signal<sc_bv<256>>>(sc_gen_unique_name("r4_ci")));
            center_ctrl_in.push_back(std::make_unique<sc_signal<sc_bv<256>>>(sc_gen_unique_name("r4_cci")));
            center_sent.push_back(std::make_unique<sc_signal<bool>>(sc_gen_unique_name("r4_si")));
            center_csent.push_back(std::make_unique<sc_signal<bool>>(sc_gen_unique_name("r4_csi")));
            core_busy.push_back(std::make_unique<sc_signal<bool>>(sc_gen_unique_name("r4_b")));
            ctrl_busy.push_back(std::make_unique<sc_signal<bool>>(sc_gen_unique_name("r4_cb")));
            center_sent.back()->write(false); center_csent.back()->write(false);
            core_busy.back()->write(false); ctrl_busy.back()->write(false);
        }
        term_false.write(false);
        Bind();
    }

private:
    const D2DPort *PortAt(int local, Directions side) const {
        for (const auto &p : g_die_ports.ports)
            if (p.tile == local && p.side == side) return &p;
        return nullptr;
    }
    const D2DLink *Outgoing(int die, int port) const {
        for (const auto &l : g_d2d_links)
            if (l.local_die == die && l.local_port == port) return &l;
        return nullptr;
    }
    const D2DLink *Incoming(int die, int port) const {
        for (const auto &l : g_d2d_links)
            if (l.remote_die == die && l.remote_port == port) return &l;
        return nullptr;
    }
    void Bind() {
        for (int r = 0; r < TOTAL_CORES; ++r) {
            RouterUnit *ru = monitor.routers[r];
            ru->core_busy_i(*core_busy[r]); ru->ctrl_core_busy_i(*ctrl_busy[r]);
            ru->channel_o[CENTER](*data[CENTER][r]);
            ru->data_sent_o[CENTER](*sent[CENTER][r]);
            ru->channel_avail_o[CENTER](*avail[CENTER][r]);
            ru->channel_i[CENTER](*center_in[r]);
            ru->data_sent_i[CENTER](*center_sent[r]);
            ru->ctrl_channel_o[CENTER](*ctrl[CENTER][r]);
            ru->ctrl_sent_o[CENTER](*csent[CENTER][r]);
            ru->ctrl_channel_avail_o[CENTER](*cavail[CENTER][r]);
            ru->ctrl_channel_i[CENTER](*center_ctrl_in[r]);
            ru->ctrl_sent_i[CENTER](*center_csent[r]);
            const int die = DieOfGlobal(r), local = LocalOfGlobal(r);
            for (int d = 0; d < DIRECTIONS - 1; ++d) {
                ru->channel_o[d](*data[d][r]); ru->data_sent_o[d](*sent[d][r]);
                ru->channel_avail_o[d](*avail[d][r]);
                ru->ctrl_channel_o[d](*ctrl[d][r]); ru->ctrl_sent_o[d](*csent[d][r]);
                ru->ctrl_channel_avail_o[d](*cavail[d][r]);
                int nb = OpenMeshNeighbor(r, static_cast<Directions>(d));
                if (nb >= 0) {
                    Directions od = GetOpposeDirection(static_cast<Directions>(d));
                    ru->channel_i[d](*data[od][nb]); ru->data_sent_i[d](*sent[od][nb]);
                    ru->ctrl_channel_i[d](*ctrl[od][nb]); ru->ctrl_sent_i[d](*csent[od][nb]);
                    ru->channel_avail_i[d](*avail[od][nb]);
                    ru->ctrl_channel_avail_i[d](*cavail[od][nb]);
                } else {
                    const D2DPort *p = PortAt(local, static_cast<Directions>(d));
                    const D2DLink *in = p ? Incoming(die, p->port_id) : nullptr;
                    const D2DLink *out = p ? Outgoing(die, p->port_id) : nullptr;
                    if (!in) {
                        ru->channel_i[d](term_data); ru->data_sent_i[d](term_false);
                        ru->ctrl_channel_i[d](term_ctrl); ru->ctrl_sent_i[d](term_false);
                    }
                    if (!out) {
                        ru->channel_avail_i[d](term_false);
                        ru->ctrl_channel_avail_i[d](term_false);
                        ru->d2d_data_credit_i[d](term_false);
                        ru->d2d_ctrl_credit_i[d](term_false);
                    }
                }
                if (nb >= 0) {
                    ru->d2d_data_credit_i[d](term_false);
                    ru->d2d_ctrl_credit_i[d](term_false);
                }
            }
        }
        int index = 0;
        for (const auto &link : g_d2d_links) {
            const auto &sp = g_die_ports.ports[link.local_port];
            const auto &dp = g_die_ports.ports[link.remote_port];
            int sr = GlobalId(link.local_die, sp.tile);
            int dr = GlobalId(link.remote_die, dp.tile);
            auto bundle = std::make_unique<LinkBundle>(link.latency, index++);
            auto &u = *bundle->unit;
            u.in_channel(*data[sp.side][sr]); u.in_sent(*sent[sp.side][sr]);
            u.in_avail(*bundle->in_avail);
            u.in_ctrl_channel(*ctrl[sp.side][sr]); u.in_ctrl_sent(*csent[sp.side][sr]);
            u.in_ctrl_avail(*bundle->in_ctrl_avail);
            u.out_channel(*bundle->out_data); u.out_sent(*bundle->out_sent);
            u.out_avail(*avail[dp.side][dr]);
            u.out_ctrl_channel(*bundle->out_ctrl); u.out_ctrl_sent(*bundle->out_ctrl_sent);
            u.out_ctrl_avail(*cavail[dp.side][dr]);
            u.data_credit_return(*bundle->data_credit);
            u.ctrl_credit_return(*bundle->ctrl_credit);
            RouterUnit *source = monitor.routers[sr];
            RouterUnit *dest = monitor.routers[dr];
            source->channel_avail_i[sp.side](*bundle->in_avail);
            source->ctrl_channel_avail_i[sp.side](*bundle->in_ctrl_avail);
            source->d2d_data_credit_i[sp.side](*bundle->data_credit);
            source->d2d_ctrl_credit_i[sp.side](*bundle->ctrl_credit);
            dest->channel_i[dp.side](*bundle->out_data);
            dest->data_sent_i[dp.side](*bundle->out_sent);
            dest->ctrl_channel_i[dp.side](*bundle->out_ctrl);
            dest->ctrl_sent_i[dp.side](*bundle->out_ctrl_sent);
            links.push_back(std::move(bundle));
        }
    }
};

class R4Driver : public sc_module {
public:
    SC_HAS_PROCESS(R4Driver);
    CoreMemAdapter local, remote, far, contender_a, contender_b;
    HBMNetwork &network;
    RouterMonitor &routers;
    sc_event go, done_a, done_b;

    R4Driver(const sc_module_name &name, HBMNetwork &n, RouterMonitor &r)
        : sc_module(name), local("r4_local", 4), remote("r4_remote", 4 * 9 + 4),
          far("r4_far", 8 * 9 + 4), contender_a("r4_ca", 0),
          contender_b("r4_cb", 3 * 9), network(n), routers(r) {
        local.BindTransport(&n); remote.BindTransport(&n); far.BindTransport(&n);
        contender_a.BindTransport(&n); contender_b.BindTransport(&n);
        SC_THREAD(Main); SC_THREAD(ContendA); SC_THREAD(ContendB);
    }

    void Main() {
        wait(1, SC_NS);
        std::vector<uint8_t> pattern(96);
        for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = uint8_t(i ^ 0x5a);
        local.Access(MemCommand::kWrite, 4096, pattern.size(), pattern);
        sc_time local_start = sc_time_stamp();
        MemMsg lr = local.Access(MemCommand::kRead, 4096, pattern.size());
        sc_time local_latency = sc_time_stamp() - local_start;
        sc_time remote_start = sc_time_stamp();
        MemMsg rr = remote.Access(MemCommand::kRead, 4096, pattern.size());
        sc_time remote_latency = sc_time_stamp() - remote_start;
        Check(lr.payload == pattern && rr.payload == pattern,
              "local and no-local-HBM cores observe the same backing data");
        Check(remote_latency > local_latency,
              "remote access includes additional NoC/C2C latency");

        std::vector<uint8_t> far_pattern(64, 0xa5);
        local.Access(MemCommand::kWrite, (1ULL << 20) + 8192,
                     far_pattern.size(), far_pattern);
        Check(far.Access(MemCommand::kRead, (1ULL << 20) + 8192,
                         far_pattern.size()).payload == far_pattern,
              "multi-hop remote write is visible at the home die");

        sc_time contention_start = sc_time_stamp();
        go.notify(SC_ZERO_TIME);
        wait(done_a & done_b);
        Check(sc_time_stamp() > contention_start,
              "concurrent HBM requests complete through shared finite queues");
        const auto &s = network.Stats();
        Check(s.c2c_request_hops > 0 && s.c2c_response_hops > 0,
              "both request and response directions consume C2C hops");
        Check(s.noc_hops > s.c2c_request_hops + s.c2c_response_hops,
              "traffic also traverses the per-die router mesh");
        Check(g_die_mesh_pkts[1] > 0 || g_die_mesh_pkts[4] > 0,
              "a true internal die observes in-die mesh traffic");
        Check(s.shared_noc_contention > 0,
              "HBM and collective traffic contend in a shared Router output queue");
        wait(20, SC_NS);
        Check(network.Residual() == 0,
              "all response waiters and endpoint assemblies drain");
        sc_stop();
    }
    void ContendA() {
        wait(go);
        Msg ordinary;
        ordinary.msg_type_ = DATA;
        ordinary.source_ = 0;
        ordinary.des_ = 8 * 9 + 4;
        ordinary.tag_id_ = ordinary.des_;
        ordinary.seq_id_ = 1;
        ordinary.is_end_ = true;
        ordinary.length_ = 128;
        ordinary.roofline_packets_ = 1;
        ordinary.exit_port_ = CrossDieSelectExit(
            ordinary.source_, ordinary.des_, ordinary.source_,
            ordinary.tag_id_, ordinary.subflow_);
        const Directions shared_out = DataMsgNextHop(ordinary, 0);
        const int next_router = OpenMeshNeighbor(0, shared_out);
        if (next_router < 0)
            throw std::runtime_error(
                "R4 collective contention path unexpectedly starts at C2C edge");
        const Directions next_ingress = GetOpposeDirection(shared_out);
        constexpr uint16_t tree_id = 401;
        ProgramCollectiveTreeEntry({tree_id, 0, WEST}, 1u << shared_out);
        ProgramCollectiveTreeEntry(
            {tree_id, static_cast<uint16_t>(next_router),
             static_cast<uint8_t>(next_ingress)},
            1u << CENTER);
        CollDataHeader collective;
        collective.tree_id = tree_id;
        collective.packet.collective = {4, 4, 0};
        collective.packet.phase_id = 1;
        collective.packet.chunk_id = 1;
        collective.packet.src_rank = 0;
        collective.packet.dst_rank = 1;
        collective.length_bits = 128;
        for (uint32_t seq = 1; seq <= 16; ++seq) {
            collective.seq_id = seq;
            collective.is_end = seq == 16;
            routers.routers[0]->buffer_i[WEST].push(
                SerializeCollData(collective));
        }
        routers.routers[0]->need_next_trigger.notify(CYCLE, SC_NS);
        contender_a.Access(MemCommand::kRead, (1ULL << 20) + 16384, 512);
        done_a.notify(SC_ZERO_TIME);
    }
    void ContendB() {
        wait(go);
        contender_b.Access(MemCommand::kRead, (1ULL << 20) + 32768, 512);
        done_b.notify(SC_ZERO_TIME);
    }
};


struct HbmExperimentPort {
    Directions side = NORTH;
    int index = 0;
    std::string label;
};

HbmExperimentPort ParseExperimentPort(const std::string &text) {
    if (text.size() < 2)
        throw std::runtime_error("HBM experiment port must be N0/S0/E0/W0 form");
    HbmExperimentPort port;
    switch (text[0]) {
    case 'N': port.side = NORTH; break;
    case 'S': port.side = SOUTH; break;
    case 'E': port.side = EAST; break;
    case 'W': port.side = WEST; break;
    default:
        throw std::runtime_error("HBM experiment port side must be N/S/E/W");
    }
    try {
        port.index = std::stoi(text.substr(1));
    } catch (...) {
        throw std::runtime_error("HBM experiment port index is not an integer");
    }
    if (port.index < 0 || port.index >= 4)
        throw std::runtime_error("HBM experiment port index must be in [0,3]");
    port.label = text;
    return port;
}

void ConfigureHbmContentionExperiment(const HbmExperimentPort &port) {
    GRID_X = GRID_Y = 4;
    GRID_SIZE = CORES_PER_DIE = 16;
    DIE_X = DIE_Y = DIE_COUNT = 1;
    TOTAL_CORES = 16;
    HOST_ENDPOINT_ID = TOTAL_CORES;

    g_die_ports = D2DPortTable{};
    g_d2d_links.clear();
    g_d2d_cfg = D2DLinkConfig{};
    g_d2d_cfg.select_policy = SELECT_NEAREST;
    g_d2d_cfg.select_policy_explicit = true;

    g_memory_system_active = true;
    g_memory_topology = MemoryTopology::kDistributedHbm;
    g_memory_cache_policy = MemoryCachePolicy::kNone;
    ResetCollectiveFabric();

    HBMProfile profile;
    profile.generation = "HBM2";
    profile.channels_per_stack = 1;
    profile.pseudo_channels_per_channel = 2;
    profile.data_rate_gbps_per_pin = 2.0;
    profile.stack_bus_width_bits = 64;
    g_hbm_profiles = {{"contention_hbm2", profile}};

    HBMStackConfig stack;
    stack.stack_id = 0;
    stack.compute_die_id = 0;
    stack.profile = "contention_hbm2";
    stack.side = port.side;
    stack.start_idx = port.index;
    stack.phy_span_tiles = 1;
    stack.capacity_bytes = 64ULL << 20;
    stack.backend_kind = HBMBackendKind::kBehavioral;
    stack.backend_granularity = HBMBackendGranularity::kChannel;
    stack.channel_dram_config = "../DRAMSys/configs/hbm2-example.json";
    stack.behavioral_base_latency_ns = 20;
    stack.behavioral_read_to_write_ns = 4;
    stack.behavioral_write_to_read_ns = 4;
    g_hbm_stacks = {stack};
    BuildMemAttach();
    ValidateMemAttach();

    g_address_policy = AddressPolicyConfig{};
    g_address_policy.active = true;
    g_address_policy.mode = AddressPolicyMode::kNumaLocalInterleave;
    g_address_policy.home_ranges = {{0, 0, stack.capacity_bytes}};
    g_address_policy.stack_interleave_bytes = 256;
    g_address_policy.channel_interleave_bytes = 256;
    g_address_policy.pseudo_channel_interleave_bytes = 256;

    g_die_router_pkts.assign(DIE_COUNT, 0);
    g_die_mesh_pkts.assign(DIE_COUNT, 0);
    g_die_noc_sends.assign(DIE_COUNT, 0);
    g_die_noc_stalls.assign(DIE_COUNT, 0);
}

class HbmContentionDriver : public sc_module {
public:
    SC_HAS_PROCESS(HbmContentionDriver);
    HBMNetwork &network;
    int core_count;
    int pairs_per_core;
    int bytes_per_access;
    std::vector<std::unique_ptr<CoreMemAdapter>> adapters;
    std::vector<double> latencies_ns;
    sc_event go;
    sc_event all_done;
    int completed_workers = 0;
    int data_errors = 0;
    sc_time started = SC_ZERO_TIME;
    sc_time ended = SC_ZERO_TIME;

    HbmContentionDriver(const sc_module_name &name, HBMNetwork &n,
                        int cores, int pairs, int bytes)
        : sc_module(name), network(n), core_count(cores),
          pairs_per_core(pairs), bytes_per_access(bytes) {
        static const int sources[8] = {0, 4, 8, 1, 5, 9, 2, 6};
        for (int i = 0; i < core_count; ++i) {
            auto adapter = std::make_unique<CoreMemAdapter>(
                sc_gen_unique_name("hbm_exp_adapter"), sources[i]);
            adapter->BindTransport(&network);
            adapters.push_back(std::move(adapter));
            sc_spawn(sc_bind(&HbmContentionDriver::Worker, this, i),
                     sc_gen_unique_name("hbm_exp_worker"));
        }
        SC_THREAD(Coordinator);
    }

private:
    void Coordinator() {
        wait(1, SC_NS);
        started = sc_time_stamp();
        go.notify(SC_ZERO_TIME);
        wait(all_done);
        wait(20, SC_NS);
        sc_stop();
    }

    void Worker(int worker) {
        wait(go);
        CoreMemAdapter &adapter = *adapters[worker];
        const uint64_t region_base =
            0x10000ULL + static_cast<uint64_t>(worker) * 0x100000ULL;
        std::vector<uint8_t> payload(bytes_per_access);
        for (int pair = 0; pair < pairs_per_core; ++pair) {
            for (int b = 0; b < bytes_per_access; ++b)
                payload[b] = static_cast<uint8_t>(
                    (adapter.core_id * 17 + pair * 13 + b) & 0xff);
            const uint64_t address =
                region_base + static_cast<uint64_t>(pair) * bytes_per_access;

            sc_time begin = sc_time_stamp();
            adapter.Access(MemCommand::kWrite, address, bytes_per_access,
                           payload);
            latencies_ns.push_back(
                (sc_time_stamp() - begin).to_seconds() * 1e9);

            begin = sc_time_stamp();
            MemMsg read =
                adapter.Access(MemCommand::kRead, address, bytes_per_access);
            latencies_ns.push_back(
                (sc_time_stamp() - begin).to_seconds() * 1e9);
            if (read.payload != payload)
                ++data_errors;
        }
        ++completed_workers;
        if (completed_workers == core_count) {
            ended = sc_time_stamp();
            all_done.notify(SC_ZERO_TIME);
        }
    }
};
} // namespace

int RunHbmR4SelfTest() {
    failures = checks = 0;
    std::cout << "==== distributed HBM R4 Router/NUMA self-test ====\n";
    ConfigureR4();
    Event_engine event_engine("r4_event_engine", 2);
    RouterMonitor routers("r4_routers", &event_engine);
    R4RouterFabric fabric(routers);
    auto runtime = BuildHBMBackends();
    HBMNetwork network("r4_hbm_network", routers, *runtime);
    R4Driver driver("r4_driver", network, routers);
    sc_start(100000, SC_NS);
    if (checks == 0) {
        std::cout << "  [diag] time=" << sc_time_stamp()
                  << " network_residual=" << network.Residual()
                  << " source_router_residual=" << routers.routers[4]->residual()
                  << " mem_router_residual=" << routers.routers[4]->residual()
                  << " requests=" << network.Stats().logical_requests
                  << " req_flits=" << network.Stats().request_flits
                  << " rsp_flits=" << network.Stats().response_flits << '\n';
        Check(false, "R4 SystemC driver executed");
    }
    std::cout << "distributed HBM R4 self-test: " << checks - failures
              << "/" << checks << " checks passed\n";
    return failures;
}


int RunHbmContentionExperiment(const std::string &port_text, int core_count,
                               int pairs_per_core, int bytes_per_access) {
    if (core_count < 1 || core_count > 8)
        throw std::runtime_error("HBM experiment core_count must be in [1,8]");
    if (pairs_per_core < 1 || pairs_per_core > 64)
        throw std::runtime_error(
            "HBM experiment pairs_per_core must be in [1,64]");
    if (bytes_per_access < 1 || bytes_per_access > 16384)
        throw std::runtime_error(
            "HBM experiment bytes_per_access must be in [1,16384]");

    const HbmExperimentPort port = ParseExperimentPort(port_text);
    ConfigureHbmContentionExperiment(port);
    const int mem_tile = g_hbm_channels.at(0).mem_tile;

    Event_engine event_engine("hbm_exp_event_engine", 2);
    RouterMonitor routers("hbm_exp_routers", &event_engine);
    R4RouterFabric fabric(routers);
    auto runtime = BuildHBMBackends();
    HBMNetwork network("hbm_exp_network", routers, *runtime);
    HbmContentionDriver driver("hbm_exp_driver", network, core_count,
                               pairs_per_core, bytes_per_access);

    sc_start(10, SC_MS);

    const HBMNetworkStats &net = network.Stats();
    const HBMRuntimeInstance &instance = runtime->Instances().at(0);
    const MemEndpointStats &endpoint = instance.endpoint->Stats();
    const HBMBackendStats &backend = instance.backend->Stats();

    std::vector<double> sorted = driver.latencies_ns;
    std::sort(sorted.begin(), sorted.end());
    double mean_latency_ns = 0.0;
    for (double latency : sorted)
        mean_latency_ns += latency;
    if (!sorted.empty())
        mean_latency_ns /= sorted.size();
    const size_t p95_index = sorted.empty()
        ? 0
        : std::min(sorted.size() - 1,
                   static_cast<size_t>(std::ceil(sorted.size() * 0.95)) - 1);
    const double p95_latency_ns = sorted.empty() ? 0.0 : sorted[p95_index];
    const double max_latency_ns = sorted.empty() ? 0.0 : sorted.back();
    const double elapsed_ns =
        (driver.ended - driver.started).to_seconds() * 1e9;
    const double useful_GBps =
        elapsed_ns > 0.0 ? static_cast<double>(net.request_bytes) / elapsed_ns
                         : 0.0;

    size_t router_residual = 0;
    for (int rid = 0; rid < TOTAL_CORES; ++rid)
        router_residual += routers.routers[rid]->residual();

    std::cout << std::fixed << std::setprecision(3)
              << "HBM_EXPERIMENT_CSV,"
              << port.label << ',' << mem_tile << ',' << core_count << ','
              << pairs_per_core << ',' << bytes_per_access << ','
              << net.logical_requests << ',' << net.reads << ','
              << net.writes << ',' << elapsed_ns << ',' << useful_GBps << ','
              << mean_latency_ns << ',' << p95_latency_ns << ','
              << max_latency_ns << ',' << net.noc_hops << ','
              << net.request_flits << ',' << net.response_flits << ','
              << net.injection_stalls << ','
              << net.mem_mem_noc_contention << ','
              << endpoint.queue_stalls << ','
              << endpoint.queue_wait.to_seconds() * 1e9 << ','
              << backend.service_time.to_seconds() * 1e9 << ','
              << network.Residual() << ',' << router_residual << ','
              << driver.data_errors << '\n';

    const uint64_t expected_each =
        static_cast<uint64_t>(core_count) * pairs_per_core;
    const bool complete =
        driver.completed_workers == core_count &&
        net.logical_requests == expected_each * 2 &&
        net.reads == expected_each && net.writes == expected_each &&
        endpoint.completed == expected_each * 2 &&
        backend.completed == expected_each * 2 &&
        network.Residual() == 0 && router_residual == 0 &&
        driver.data_errors == 0;
    return complete ? 0 : 1;
}
