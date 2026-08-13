#include "isa/collective_graph_v1.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace {

using Record = IsaV1NormalizedCollectiveRecord;

void Require(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}

bool IsZeroKey(const CollectiveKey &key) {
    return key.group_id == 0 && key.collective_id == 0 && key.epoch == 0;
}

bool IsReduction(CollOp op) {
    return op == CollOp::REDUCE || op == CollOp::REDUCESCATTER ||
           op == CollOp::ALLREDUCE;
}

bool IsSymmetric(CollOp op) {
    return op == CollOp::ALLTOALL || op == CollOp::ALLGATHER ||
           op == CollOp::REDUCESCATTER || op == CollOp::ALLREDUCE;
}

bool IsRootTransmit(CollOp op) {
    return op == CollOp::SCATTER || op == CollOp::BROADCAST;
}

bool IsRootReceive(CollOp op) {
    return op == CollOp::GATHER || op == CollOp::REDUCE;
}

bool ExpectedSend(CollOp op, size_t rank, size_t root, size_t p2p_source) {
    if (op == CollOp::P2P) return rank == p2p_source;
    if (IsRootTransmit(op)) return rank == root;
    return IsRootReceive(op) || IsSymmetric(op);
}

bool ExpectedReceive(CollOp op, size_t rank, size_t root,
                     size_t p2p_destination) {
    if (op == CollOp::P2P) return rank == p2p_destination;
    if (IsRootTransmit(op)) return true;
    if (IsRootReceive(op)) return rank == root;
    return IsSymmetric(op);
}

bool ExpectedCompute(CollOp op, size_t rank, size_t root) {
    if (op == CollOp::REDUCE) return rank == root;
    return op == CollOp::REDUCESCATTER || op == CollOp::ALLREDUCE;
}

struct Registry {
    uint32_t total_cores = 0;
    uint32_t cores_per_die = 0;
    std::set<uint16_t> active;
    std::map<uint32_t, std::vector<uint16_t>> groups;
};

Registry ValidateRegistry(const IsaV1CollectiveGroupRegistryView &view) {
    Require(view.total_cores != 0 && view.cores_per_die != 0 &&
                view.total_cores % view.cores_per_die == 0 &&
                view.total_cores <=
                    uint32_t(std::numeric_limits<uint16_t>::max()) + 1,
            "ISA-v1 collective graph topology is invalid");
    Require(!view.active_cores.empty(),
            "ISA-v1 collective graph active core set is empty");
    Require(std::is_sorted(view.active_cores.begin(), view.active_cores.end()),
            "ISA-v1 collective graph active cores must be sorted");

    Registry registry;
    registry.total_cores = view.total_cores;
    registry.cores_per_die = view.cores_per_die;
    for (uint16_t core : view.active_cores) {
        Require(core < view.total_cores,
                "ISA-v1 active core is outside topology");
        Require(registry.active.insert(core).second,
                "ISA-v1 active core is duplicated");
    }

    for (const auto &definition : view.groups) {
        Require(definition.group_id != 0,
                "ISA-v1 collective group ID zero is reserved");
        Require(!definition.members.empty(),
                "ISA-v1 collective group is empty");
        Require(definition.members.size() <=
                    std::numeric_limits<uint16_t>::max(),
                "ISA-v1 collective group exceeds u16 capacity");
        Require(std::is_sorted(definition.members.begin(),
                               definition.members.end()),
                "ISA-v1 collective group members must be sorted");
        Require(registry.groups.count(definition.group_id) == 0,
                "ISA-v1 collective group ID is duplicated");
        uint16_t previous = 0;
        bool first = true;
        for (uint16_t core : definition.members) {
            Require(core < view.total_cores,
                    "ISA-v1 collective group member is outside topology");
            Require(registry.active.count(core) != 0,
                    "ISA-v1 collective group member is not active");
            Require(first || previous != core,
                    "ISA-v1 collective group member is duplicated");
            previous = core;
            first = false;
        }
        registry.groups.emplace(definition.group_id, definition.members);
    }
    return registry;
}

uint64_t GraphDTypeBytes(CollDType dtype) {
    switch (dtype) {
    case CollDType::UINT8: return 1;
    case CollDType::INT32: return 4;
    case CollDType::INT64: return 8;
    case CollDType::FP32:
    case CollDType::FP16:
    case CollDType::FP8: break;
    }
    throw std::invalid_argument("ISA-v1 graph dtype is unsupported");
}

void ValidateEnumAndCanonicalFields(const Record &record) {
    switch (record.role) {
    case IsaV1CollectiveRecordRole::SEND:
        switch (record.tx_mode) {
        case CollTxKind::UNICAST:
        case CollTxKind::SCATTER:
        case CollTxKind::BROADCAST: break;
        default:
            throw std::invalid_argument("ISA-v1 SEND mode is invalid");
        }
        if (IsZeroKey(record.key))
            Require(record.asynchronous ? record.token != 0 : record.token == 0,
                    "ISA-v1 standalone SEND completion/token is not canonical");
        else
            Require(record.asynchronous && record.token != 0,
                    "ISA-v1 collective SEND must be ASYNC with a non-zero token");
        Require(record.logical_fsm_id_base != 0 && record.length_bytes != 0,
                "ISA-v1 SEND requires a non-zero fsm ID and length");
        Require(record.result_address_bytes == 0,
                "ISA-v1 SEND carries a result address");
        Require(record.expected_sources == 0,
                "ISA-v1 SEND carries expected_sources");
        Require(record.element_count == 0,
                "ISA-v1 SEND carries compute element_count");
        Require(record.root_rank == 0 && record.self_rank == 0,
                "ISA-v1 SEND carries compute ranks");
        break;
    case IsaV1CollectiveRecordRole::RECEIVE:
        switch (record.rx_mode) {
        case CollRxKind::UNICAST:
        case CollRxKind::GATHER:
        case CollRxKind::REDUCE: break;
        default:
            throw std::invalid_argument("ISA-v1 RECEIVE mode is invalid");
        }
        if (IsZeroKey(record.key))
            Require(record.asynchronous ? record.token != 0 : record.token == 0,
                    "ISA-v1 standalone RECEIVE completion/token is not canonical");
        else
            Require(record.asynchronous && record.token != 0,
                    "ISA-v1 collective RECEIVE must be ASYNC with a non-zero token");
        Require(record.logical_fsm_id_base != 0 && record.length_bytes != 0,
                "ISA-v1 RECEIVE requires a non-zero fsm ID and length");
        Require(record.result_address_bytes == 0,
                "ISA-v1 RECEIVE carries a result address");
        Require(record.element_count == 0,
                "ISA-v1 RECEIVE carries compute element_count");
        Require(record.root_rank == 0 && record.self_rank == 0,
                "ISA-v1 RECEIVE carries compute ranks");
        break;
    case IsaV1CollectiveRecordRole::REDUCE_COMPUTE:
        Require(!record.asynchronous && record.token == 0,
                "ISA-v1 REDUCE_COMPUTE cannot carry completion state");
        Require(record.logical_fsm_id_base == 0 && record.length_bytes == 0,
                "ISA-v1 REDUCE_COMPUTE fsm/length must be synthesized from RECEIVE");
        Require(record.expected_sources == 0 && record.peer_core == 0,
                "ISA-v1 REDUCE_COMPUTE carries endpoint routing fields");
        Require(record.element_count != 0,
                "ISA-v1 REDUCE_COMPUTE element_count must be positive");
        (void)GraphDTypeBytes(record.dtype);
        break;
    default:
        throw std::invalid_argument("ISA-v1 collective record role is invalid");
    }
    Require(record.tree_id == 0,
            "ISA-v1 collective graph supports tree_id=0 only");
}

struct FsmInterval {
    uint32_t first = 0;
    uint32_t last = 0;
    std::string context;
};

void ValidateIntervals(std::vector<FsmInterval> intervals) {
    std::sort(intervals.begin(), intervals.end(),
              [](const FsmInterval &a, const FsmInterval &b) {
                  return std::tie(a.first, a.last, a.context) <
                         std::tie(b.first, b.last, b.context);
              });
    for (size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].first <= intervals[i - 1].last)
            throw std::invalid_argument(
                intervals[i].context + ": logical fsm range collides with " +
                intervals[i - 1].context);
    }
}

std::string RecordContext(const Record &record) {
    std::ostringstream out;
    out << "collective key(" << record.key.group_id << ","
        << record.key.collective_id << "," << record.key.epoch << ") core "
        << record.core_id << " record " << record.record_index;
    return out.str();
}

struct StandaloneP2pKey {
    uint16_t source_core = 0;
    uint16_t destination_core = 0;
    uint32_t fsm_id = 0;

    bool operator<(const StandaloneP2pKey &other) const noexcept {
        return std::tie(source_core, destination_core, fsm_id) <
               std::tie(other.source_core, other.destination_core,
                        other.fsm_id);
    }
};

IsaV1CollectivePlan BuildOne(
    const CollectiveKey &key, const std::vector<const Record *> &records,
    const std::vector<uint16_t> &group, uint32_t cores_per_die,
    const IsaV1PlannerCapacity &capacity) {
    const size_t n = group.size();
    std::map<uint16_t, size_t> rank_of;
    for (size_t rank = 0; rank < n; ++rank)
        rank_of.emplace(group[rank], rank);

    using Slots = std::array<const Record *, 3>;
    std::vector<Slots> slots(n, Slots{{nullptr, nullptr, nullptr}});
    const auto first_endpoint_it = std::find_if(
        records.begin(), records.end(), [](const Record *record) {
            return record->role !=
                   IsaV1CollectiveRecordRole::REDUCE_COMPUTE;
        });
    Require(first_endpoint_it != records.end(),
            "ISA-v1 collective key has no endpoint records");
    const Record *first = *first_endpoint_it;
    bool have_tx = false;
    bool have_rx = false;
    CollTxKind tx = CollTxKind::UNICAST;
    CollRxKind rx = CollRxKind::UNICAST;

    for (const Record *record : records) {
        Require(record->key == key,
                "ISA-v1 internal collective key mismatch");
        if (record->role !=
            IsaV1CollectiveRecordRole::REDUCE_COMPUTE) {
            Require(record->logical_fsm_id_base ==
                        first->logical_fsm_id_base &&
                        record->length_bytes == first->length_bytes &&
                        record->tree_id == first->tree_id,
                    "ISA-v1 endpoint records with one key disagree on common fields");
        }
        const auto found_rank = rank_of.find(record->core_id);
        Require(found_rank != rank_of.end(),
                "ISA-v1 collective record core is not a group member");
        const size_t role = static_cast<size_t>(record->role);
        Require(role < slots[found_rank->second].size(),
                "ISA-v1 collective record role is invalid");
        Require(slots[found_rank->second][role] == nullptr,
                "ISA-v1 rank has a duplicate collective role");
        slots[found_rank->second][role] = record;

        if (record->role == IsaV1CollectiveRecordRole::SEND) {
            if (!have_tx) {
                tx = record->tx_mode;
                have_tx = true;
            } else {
                Require(record->tx_mode == tx,
                        "ISA-v1 SEND modes disagree within one key");
            }
        } else if (record->role == IsaV1CollectiveRecordRole::RECEIVE) {
            if (!have_rx) {
                rx = record->rx_mode;
                have_rx = true;
            } else {
                Require(record->rx_mode == rx,
                        "ISA-v1 RECEIVE modes disagree within one key");
            }
        }
    }
    Require(have_tx && have_rx,
            "ISA-v1 collective key is missing SEND or RECEIVE records");
    const CollOp op = IsaV1CollectiveOp(tx, rx);
    if (op != CollOp::P2P) {
        const uint32_t die = group.front() / cores_per_die;
        for (uint16_t core : group)
            Require(core / cores_per_die == die,
                    "ISA-v1 non-P2P collective group crosses dies");
    }

    std::vector<size_t> send_ranks;
    std::vector<size_t> receive_ranks;
    for (size_t rank = 0; rank < n; ++rank) {
        if (slots[rank][static_cast<size_t>(
                IsaV1CollectiveRecordRole::SEND)] != nullptr)
            send_ranks.push_back(rank);
        if (slots[rank][static_cast<size_t>(
                IsaV1CollectiveRecordRole::RECEIVE)] != nullptr)
            receive_ranks.push_back(rank);
    }

    size_t root = 0;
    size_t p2p_source = 0;
    size_t p2p_destination = 0;
    if (op == CollOp::P2P) {
        Require(send_ranks.size() == 1 && receive_ranks.size() == 1,
                "ISA-v1 keyed P2P requires one SEND and one RECEIVE root");
        p2p_source = send_ranks.front();
        p2p_destination = receive_ranks.front();
        root = p2p_source;
    } else if (IsRootTransmit(op)) {
        Require(send_ranks.size() == 1,
                "ISA-v1 scatter/broadcast requires one root SEND");
        root = send_ranks.front();
    } else if (IsRootReceive(op)) {
        Require(receive_ranks.size() == 1,
                "ISA-v1 gather/reduce requires one root RECEIVE");
        root = receive_ranks.front();
    }

    const uint16_t expected_sources =
        (rx == CollRxKind::GATHER || rx == CollRxKind::REDUCE)
            ? static_cast<uint16_t>(n - 1)
            : 0;
    CollDType plan_dtype = CollDType::UINT8;
    CollReduceOp plan_reduce = CollReduceOp::NONE;

    if (IsReduction(op)) {
        const Record *reduce_receive = nullptr;
        for (size_t rank = 0; rank < n && reduce_receive == nullptr; ++rank)
            reduce_receive = slots[rank][static_cast<size_t>(
                IsaV1CollectiveRecordRole::RECEIVE)];
        Require(reduce_receive != nullptr,
                "ISA-v1 reduction is missing a RECEIVE contract");
        plan_dtype = reduce_receive->dtype;
        plan_reduce = reduce_receive->reduce_op;
    }

    IsaV1CollectiveSpec spec;
    spec.tx_kind = tx;
    spec.rx_kind = rx;
    spec.key = key;
    spec.group = group;
    spec.root_rank = static_cast<uint16_t>(root);
    spec.p2p_source_rank = static_cast<uint16_t>(p2p_source);
    spec.p2p_destination_rank = static_cast<uint16_t>(p2p_destination);
    spec.length_bytes = first->length_bytes;
    spec.dtype = plan_dtype;
    spec.reduce_op = plan_reduce;
    spec.tree_id = 0;
    spec.expected_sources = expected_sources;
    spec.logical_fsm_id_base = first->logical_fsm_id_base;
    spec.rank_records.resize(n);

    for (size_t rank = 0; rank < n; ++rank) {
        const Record *send = slots[rank][static_cast<size_t>(
            IsaV1CollectiveRecordRole::SEND)];
        const Record *receive = slots[rank][static_cast<size_t>(
            IsaV1CollectiveRecordRole::RECEIVE)];
        const Record *compute = slots[rank][static_cast<size_t>(
            IsaV1CollectiveRecordRole::REDUCE_COMPUTE)];
        const bool want_send = ExpectedSend(op, rank, root, p2p_source);
        const bool want_receive =
            ExpectedReceive(op, rank, root, p2p_destination);
        const bool want_compute = ExpectedCompute(op, rank, root);
        Require((send != nullptr) == want_send,
                "ISA-v1 collective SEND membership is incomplete");
        Require((receive != nullptr) == want_receive,
                "ISA-v1 collective RECEIVE membership is incomplete");
        Require((compute != nullptr) == want_compute,
                "ISA-v1 reduction target is missing or has an extra compute record");

        if (send != nullptr) {
            Require(send->dtype == CollDType::UINT8 &&
                        send->reduce_op == CollReduceOp::NONE,
                    "ISA-v1 SEND must remain byte-oriented and non-reducing");
            auto &endpoint = spec.rank_records[rank].send;
            endpoint.present = true;
            endpoint.asynchronous = true;
            endpoint.token = send->token;
            endpoint.base_address_bytes = send->base_address_bytes;
        }
        if (receive != nullptr) {
            Require(receive->expected_sources == expected_sources,
                    "ISA-v1 RECEIVE expected_sources does not match N-1/zero");
            if (IsReduction(op)) {
                Require(receive->dtype == plan_dtype &&
                            receive->reduce_op == plan_reduce,
                        "ISA-v1 reduction RECEIVE dtype/op disagree");
            } else {
                Require(receive->dtype == CollDType::UINT8 &&
                            receive->reduce_op == CollReduceOp::NONE,
                        "ISA-v1 non-reduction RECEIVE is not UINT8/NONE");
            }
            auto &endpoint = spec.rank_records[rank].receive;
            endpoint.present = true;
            endpoint.asynchronous = true;
            endpoint.token = receive->token;
            endpoint.base_address_bytes = receive->base_address_bytes;
        }
        if (compute != nullptr) {
            Require(compute->dtype == plan_dtype &&
                        compute->reduce_op == plan_reduce,
                    "ISA-v1 REDUCE_COMPUTE dtype/op disagree with RECEIVE");
            Require(receive != nullptr &&
                        compute->base_address_bytes ==
                            receive->base_address_bytes,
                    "ISA-v1 REDUCE_COMPUTE staging does not match RECEIVE");
            Require(compute->self_rank == rank,
                    "ISA-v1 REDUCE_COMPUTE self_rank does not match its core rank");
            const uint16_t expected_root =
                op == CollOp::REDUCE ? static_cast<uint16_t>(root) : 0;
            Require(compute->root_rank == expected_root,
                    "ISA-v1 REDUCE_COMPUTE root_rank is not canonical for the operation");
            const uint64_t bytes = GraphDTypeBytes(compute->dtype);
            if (compute->element_count >
                std::numeric_limits<uint64_t>::max() / bytes)
                throw std::overflow_error(
                    "ISA-v1 REDUCE_COMPUTE element_count bytes overflow");
            Require(compute->element_count * bytes == first->length_bytes,
                    "ISA-v1 REDUCE_COMPUTE element_count*dtype_bytes does not equal RECEIVE L");
            spec.rank_records[rank].result_address_bytes =
                compute->result_address_bytes;
        }
    }
    return PlanIsaV1Collective(spec, capacity);
}

std::string InstanceContext(
    const CollectiveKey &key, const std::vector<const Record *> &records) {
    std::ostringstream out;
    out << "collective key(" << key.group_id << "," << key.collective_id
        << "," << key.epoch << ")";
    if (!records.empty())
        out << " core " << records.front()->core_id << " record "
            << records.front()->record_index;
    return out.str();
}

} // namespace

std::vector<IsaV1CollectivePlan> BuildIsaV1CollectiveGraph(
    const std::vector<IsaV1NormalizedCollectiveRecord> &records,
    const IsaV1CollectiveGroupRegistryView &registry_view,
    const IsaV1PlannerCapacity &capacity) {
    const Registry registry = ValidateRegistry(registry_view);
    std::set<std::pair<uint16_t, uint32_t>> provenance;
    std::set<std::pair<uint16_t, uint32_t>> aggregate_tokens;
    std::map<CollectiveKey, std::vector<const Record *>> instances;
    std::map<StandaloneP2pKey, const Record *> standalone_sends;
    std::map<StandaloneP2pKey, const Record *> standalone_receives;
    std::vector<FsmInterval> fsm_intervals;

    for (const auto &record : records) {
        const std::string context = RecordContext(record);
        try {
            ValidateEnumAndCanonicalFields(record);
            Require(record.core_id < registry.total_cores &&
                        registry.active.count(record.core_id) != 0,
                    "ISA-v1 collective record core is not active");
            Require(provenance.emplace(record.core_id, record.record_index).second,
                    "ISA-v1 normalized record provenance is duplicated");
            if (record.role != IsaV1CollectiveRecordRole::REDUCE_COMPUTE &&
                record.token != 0)
                Require(aggregate_tokens.emplace(record.core_id, record.token).second,
                        "ISA-v1 aggregate token collides on one core");

            if (IsZeroKey(record.key)) {
                Require(record.role !=
                            IsaV1CollectiveRecordRole::REDUCE_COMPUTE &&
                            ((record.role == IsaV1CollectiveRecordRole::SEND &&
                              record.tx_mode == CollTxKind::UNICAST) ||
                             (record.role ==
                                  IsaV1CollectiveRecordRole::RECEIVE &&
                              record.rx_mode == CollRxKind::UNICAST)),
                        "ISA-v1 zero key is reserved for standalone P2P");
                Require(record.tree_id == 0 && record.expected_sources == 0 &&
                            record.dtype == CollDType::UINT8 &&
                            record.reduce_op == CollReduceOp::NONE,
                        "ISA-v1 standalone P2P fields are not canonical");
                Require(record.peer_core < registry.total_cores &&
                            registry.active.count(record.peer_core) != 0,
                        "ISA-v1 standalone P2P peer is not active");
                Require(record.peer_core != record.core_id,
                        "ISA-v1 standalone P2P self peer is forbidden");
                StandaloneP2pKey key;
                key.fsm_id = record.logical_fsm_id_base;
                if (record.role == IsaV1CollectiveRecordRole::SEND) {
                    key.source_core = record.core_id;
                    key.destination_core = record.peer_core;
                    Require(standalone_sends.emplace(key, &record).second,
                            "ISA-v1 standalone P2P has a duplicate SEND endpoint");
                } else {
                    key.source_core = record.peer_core;
                    key.destination_core = record.core_id;
                    Require(standalone_receives.emplace(key, &record).second,
                            "ISA-v1 standalone P2P has a duplicate RECEIVE endpoint");
                }
                continue;
            }
            Require(record.key.group_id != 0,
                    "ISA-v1 collective key has a zero group ID");
            const auto group = registry.groups.find(record.key.group_id);
            Require(group != registry.groups.end(),
                    "ISA-v1 collective record references an unknown group");
            instances[record.key].push_back(&record);
        } catch (const std::overflow_error &error) {
            throw std::overflow_error(context + ": " + error.what());
        } catch (const std::invalid_argument &error) {
            throw std::invalid_argument(context + ": " + error.what());
        }
    }

    std::set<StandaloneP2pKey> standalone_keys;
    for (const auto &entry : standalone_sends) standalone_keys.insert(entry.first);
    for (const auto &entry : standalone_receives)
        standalone_keys.insert(entry.first);
    for (const StandaloneP2pKey &key : standalone_keys) {
        const auto send = standalone_sends.find(key);
        const auto receive = standalone_receives.find(key);
        const Record *anchor = send != standalone_sends.end()
                                   ? send->second
                                   : receive->second;
        const std::string context = RecordContext(*anchor);
        Require(send != standalone_sends.end(),
                (context + ": standalone P2P session has no SEND endpoint").c_str());
        Require(receive != standalone_receives.end(),
                (context + ": standalone P2P session has no RECEIVE endpoint").c_str());
        Require(send->second->length_bytes == receive->second->length_bytes &&
                    send->second->dtype == receive->second->dtype,
                (context + ": standalone P2P endpoint metadata mismatch").c_str());
        fsm_intervals.push_back(
            {key.fsm_id, key.fsm_id, context + " standalone P2P session"});
    }

    std::vector<IsaV1CollectivePlan> plans;
    plans.reserve(instances.size());
    for (const auto &instance : instances) {
        const auto group = registry.groups.find(instance.first.group_id);
        IsaV1CollectivePlan plan;
        const std::string context =
            InstanceContext(instance.first, instance.second);
        try {
            plan = BuildOne(instance.first, instance.second, group->second,
                            registry.cores_per_die, capacity);
        } catch (const std::overflow_error &error) {
            throw std::overflow_error(context + ": " + error.what());
        } catch (const std::invalid_argument &error) {
            throw std::invalid_argument(context + ": " + error.what());
        }
        if (!plan.child_flows.empty()) {
            fsm_intervals.push_back(
                {plan.logical_fsm_id_base, plan.child_flows.back().fsm_id,
                 context});
        }
        plans.push_back(std::move(plan));
    }
    ValidateIntervals(std::move(fsm_intervals));
    return plans;
}
