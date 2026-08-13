#include "prims/collective_launch_v1_prim.h"
#include "prims/collective_launch_v1_prim_selftest.h"

#include "isa/prim_manifest.h"
#include "utils/prim_utils.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Wire = std::vector<sc_bv<128>>;

struct Suite {
    int checks = 0;
    int failures = 0;

    void Check(bool condition, const std::string &name) {
        ++checks;
        if (condition) return;
        ++failures;
        std::cerr << "[COLLECTIVE LAUNCH V1 PRIM] FAIL: "
                  << name << '\n';
    }

    template <class Exception = std::exception, class Function>
    void Rejects(Function &&function, const std::string &name) {
        bool rejected = false;
        try {
            function();
        } catch (const Exception &) {
            rejected = true;
        } catch (...) {
        }
        Check(rejected, name);
    }

    template <class Exception = std::exception, class Function>
    void RejectsExact(Function &&function, const std::string &message,
                      const std::string &name) {
        bool matched = false;
        try {
            function();
        } catch (const Exception &error) {
            matched = error.what() == message;
        } catch (...) {
        }
        Check(matched, name);
    }
};

class LegacyModeGuard {
public:
    explicit LegacyModeGuard(bool enabled)
        : previous_(prim_wire::LegacyCompatibilityEnabled()) {
        prim_wire::SetLegacyCompatibility(enabled);
    }
    ~LegacyModeGuard() {
        prim_wire::SetLegacyCompatibility(previous_);
    }

private:
    bool previous_;
};

Collective_launch_v1_prim MakePrim(CollectiveLaunchV1Role role) {
    Collective_launch_v1_prim prim;
    prim.role = role;
    prim.image_generation = UINT64_C(0x1122334455667788);
    prim.plan_index = UINT32_C(0x99aabbcc);
    prim.external_record_index = UINT32_C(0xddeeff00);
    prim.expected_core = UINT16_C(0xfffe);
    prim.key = {UINT32_C(0x10203040), UINT32_C(0x50607080),
                UINT32_C(0x90a0b0c0)};
    prim.public_token =
        role == CollectiveLaunchV1Role::DECLARE_REDUCE_COMPUTE
            ? 0
            : UINT32_C(0x0badf00d);
    return prim;
}

bool SamePrim(const Collective_launch_v1_prim &left,
              const Collective_launch_v1_prim &right) {
    return left.role == right.role &&
           left.image_generation == right.image_generation &&
           left.plan_index == right.plan_index &&
           left.external_record_index == right.external_record_index &&
           left.expected_core == right.expected_core &&
           left.key == right.key && left.public_token == right.public_token;
}

bool SameWire(const Wire &left, const Wire &right) {
    if (left.size() != right.size()) return false;
    for (size_t index = 0; index < left.size(); ++index)
        if (left[index] != right[index]) return false;
    return true;
}

Wire ExpectedGolden(const Collective_launch_v1_prim &prim) {
    Wire expected(kCollectiveLaunchV1PrimWireSegments);
    for (uint8_t index = 0; index < expected.size(); ++index) {
        expected[index] = 0;
        expected[index].range(7, 0) =
            sc_bv<8>(PrimIdValue(PrimId::COLLECTIVE_LAUNCH_V1));
        expected[index].range(15, 8) = sc_bv<8>(index);
    }
    expected[0].range(23, 16) =
        sc_bv<8>(kCollectiveLaunchV1PrimWireVersion);
    expected[0].range(31, 24) =
        sc_bv<8>(kCollectiveLaunchV1PrimWireSegments);
    expected[0].range(33, 32) =
        sc_bv<2>(static_cast<uint8_t>(prim.role));
    expected[1].range(79, 16) = sc_bv<64>(prim.image_generation);
    expected[1].range(111, 80) = sc_bv<32>(prim.plan_index);
    expected[2].range(47, 16) =
        sc_bv<32>(prim.external_record_index);
    expected[2].range(63, 48) = sc_bv<16>(prim.expected_core);
    expected[2].range(95, 64) = sc_bv<32>(prim.public_token);
    expected[3].range(47, 16) = sc_bv<32>(prim.key.group_id);
    expected[3].range(79, 48) = sc_bv<32>(prim.key.collective_id);
    expected[3].range(111, 80) = sc_bv<32>(prim.key.epoch);
    return expected;
}

void TestGoldenAndRoundTrip(Suite &suite) {
    for (CollectiveLaunchV1Role role :
         {CollectiveLaunchV1Role::ISSUE_SEND,
          CollectiveLaunchV1Role::ISSUE_RECEIVE,
          CollectiveLaunchV1Role::DECLARE_REDUCE_COMPUTE}) {
        Collective_launch_v1_prim prim = MakePrim(role);
        const Wire wire = prim.serialize();
        Collective_launch_v1_prim decoded;
        decoded.deserialize(wire);
        suite.Check(SamePrim(prim, decoded) &&
                        SameWire(wire, decoded.serialize()),
                    "each launch role roundtrips deterministically");
    }

    Collective_launch_v1_prim prim =
        MakePrim(CollectiveLaunchV1Role::ISSUE_SEND);
    const Wire wire = prim.serialize();
    suite.Check(SameWire(wire, ExpectedGolden(prim)),
                "strict four-segment wire matches the fixed golden layout");
    bool headers = wire.size() == kCollectiveLaunchV1PrimWireSegments;
    for (size_t index = 0; index < wire.size(); ++index) {
        headers = headers &&
            wire[index].range(7, 0).to_uint64() ==
                PrimIdValue(PrimId::COLLECTIVE_LAUNCH_V1) &&
            wire[index].range(15, 8).to_uint64() == index;
    }
    suite.Check(headers &&
                    wire[0].range(23, 16).to_uint64() ==
                        kCollectiveLaunchV1PrimWireVersion &&
                    wire[0].range(31, 24).to_uint64() ==
                        kCollectiveLaunchV1PrimWireSegments &&
                    !wire[0].range(127, 34).or_reduce() &&
                    !wire[1].range(127, 112).or_reduce() &&
                    !wire[2].range(127, 96).or_reduce() &&
                    !wire[3].range(127, 112).or_reduce(),
                "every segment has ID60/ordinal and all reserved bits zero");
}

void TestWireRejectionsAndAtomicity(Suite &suite) {
    Collective_launch_v1_prim prim =
        MakePrim(CollectiveLaunchV1Role::ISSUE_RECEIVE);
    const Wire wire = prim.serialize();
    for (size_t segment = 0; segment < wire.size(); ++segment) {
        suite.Rejects<std::invalid_argument>([&] {
            Wire bad = wire;
            bad[segment].range(7, 0) =
                PrimIdValue(PrimId::COLLECTIVE_PHASE_BARRIER_V1);
            Collective_launch_v1_prim decoded;
            decoded.deserialize(std::move(bad));
        }, "each segment rejects a non-ID60 header");
        suite.Rejects<std::invalid_argument>([&] {
            Wire bad = wire;
            bad[segment].range(15, 8) =
                static_cast<uint8_t>((segment + 1) % wire.size());
            Collective_launch_v1_prim decoded;
            decoded.deserialize(std::move(bad));
        }, "each segment rejects a non-canonical ordinal");
    }
    suite.Rejects<std::invalid_argument>([&] {
        Wire bad = wire;
        bad.pop_back();
        Collective_launch_v1_prim decoded;
        decoded.deserialize(std::move(bad));
    }, "wrong vector segment count is rejected");
    suite.Rejects<std::invalid_argument>([&] {
        Wire bad = wire;
        bad[0].range(23, 16) = 2;
        Collective_launch_v1_prim decoded;
        decoded.deserialize(std::move(bad));
    }, "unsupported version is rejected");
    suite.Rejects<std::invalid_argument>([&] {
        Wire bad = wire;
        bad[0].range(31, 24) = 3;
        Collective_launch_v1_prim decoded;
        decoded.deserialize(std::move(bad));
    }, "inconsistent count field is rejected");
    const std::vector<std::pair<size_t, int>> reserved = {
        {0, 34}, {1, 112}, {2, 96}, {3, 112}};
    for (const auto &bit : reserved) {
        suite.Rejects<std::invalid_argument>([&] {
            Wire bad = wire;
            bad[bit.first][bit.second] = sc_dt::SC_LOGIC_1;
            Collective_launch_v1_prim decoded;
            decoded.deserialize(std::move(bad));
        }, "each segment rejects non-zero reserved bits");
    }
    suite.Rejects<std::invalid_argument>([&] {
        Wire bad = wire;
        bad[0].range(33, 32) = 3;
        Collective_launch_v1_prim decoded;
        decoded.deserialize(std::move(bad));
    }, "wire rejects unknown role enum");

    Collective_launch_v1_prim unchanged =
        MakePrim(CollectiveLaunchV1Role::DECLARE_REDUCE_COMPUTE);
    const Collective_launch_v1_prim before = unchanged;
    Wire bad = wire;
    bad[1].range(79, 16) = 0;
    suite.Rejects<std::invalid_argument>(
        [&] { unchanged.deserialize(bad); },
        "invalid generation rejects before decode commit");
    suite.Check(SamePrim(unchanged, before),
                "failed decode leaves the destination object unchanged");
}

void TestFieldContracts(Suite &suite) {
    Collective_launch_v1_prim prim =
        MakePrim(CollectiveLaunchV1Role::ISSUE_SEND);
    prim.image_generation = 0;
    suite.Rejects<std::invalid_argument>([&] { prim.Validate(); },
                                         "zero image generation is rejected");

    prim = MakePrim(CollectiveLaunchV1Role::ISSUE_SEND);
    prim.role = static_cast<CollectiveLaunchV1Role>(3);
    suite.Rejects<std::invalid_argument>([&] { prim.Validate(); },
                                         "unknown role enum is rejected");
    prim = MakePrim(CollectiveLaunchV1Role::ISSUE_SEND);
    prim.key.group_id = 0;
    suite.Rejects<std::invalid_argument>([&] { prim.Validate(); },
                                         "zero group key is rejected");
    prim = MakePrim(CollectiveLaunchV1Role::ISSUE_SEND);
    prim.key.collective_id = std::numeric_limits<uint32_t>::max();
    suite.Rejects<std::invalid_argument>([&] { prim.Validate(); },
                                         "GROUP_SYNC-reserved key is rejected");
    prim = MakePrim(CollectiveLaunchV1Role::ISSUE_SEND);
    prim.public_token = 0;
    suite.Rejects<std::invalid_argument>([&] { prim.Validate(); },
                                         "SEND requires a public token");
    prim = MakePrim(CollectiveLaunchV1Role::ISSUE_RECEIVE);
    prim.public_token = 0;
    suite.Rejects<std::invalid_argument>([&] { prim.Validate(); },
                                         "RECEIVE requires a public token");
    prim = MakePrim(CollectiveLaunchV1Role::DECLARE_REDUCE_COMPUTE);
    prim.public_token = 1;
    suite.Rejects<std::invalid_argument>([&] { prim.Validate(); },
                                         "REDUCE declaration forbids a token");

    prim = MakePrim(CollectiveLaunchV1Role::ISSUE_SEND);
    prim.expected_core = std::numeric_limits<uint16_t>::max();
    Collective_launch_v1_prim decoded;
    decoded.deserialize(prim.serialize());
    suite.Check(decoded.expected_core ==
                    std::numeric_limits<uint16_t>::max(),
                "expected_core preserves the full frozen u16 field");
}

void TestStrictFactoryAndDispatch(Suite &suite) {
    Collective_launch_v1_prim prim =
        MakePrim(CollectiveLaunchV1Role::ISSUE_SEND);
    const Wire wire = prim.serialize();
    suite.Rejects<std::invalid_argument>([&] {
        (void)prim_wire::LegacyTransportSegments(wire, prim.name);
    }, "legacy transport wrapper rejects ID60");
    {
        LegacyModeGuard legacy(true);
        suite.Rejects<std::invalid_argument>([&] {
            (void)prim.serialize();
        }, "legacy-mode serialize rejects ID60");
        suite.Rejects<std::invalid_argument>([&] {
            Collective_launch_v1_prim decoded;
            decoded.deserialize(wire);
        }, "legacy-mode deserialize rejects ID60");
    }
    suite.Check(!prim_wire::LegacyCompatibilityEnabled(),
                "strict-wire test restores transport mode");

    PrimFactory &factory = PrimFactory::getInstance();
    const PrimManifestEntry *manifest =
        LookupPrim(PrimId::COLLECTIVE_LAUNCH_V1);
    PrimBase *created = factory.createPrim(
        static_cast<int>(PrimIdValue(PrimId::COLLECTIVE_LAUNCH_V1)),
        false, false);
    suite.Check(manifest != nullptr &&
                    manifest->factory_name == "Collective_launch_v1_prim" &&
                    manifest->visibility == PrimVisibility::INTERNAL &&
                    manifest->primary_category ==
                        PrimCategory::COMMUNICATION &&
                    factory.getPrimId("Collective_launch_v1_prim") == 60 &&
                    dynamic_cast<Collective_launch_v1_prim *>(created) !=
                        nullptr &&
                    HasExactlyOnePrimMainCategory(created->prim_type) &&
                    PrimMainCategoryBits(created->prim_type) == COMM_PRIM,
                "ID60 manifest and untracked factory creator are canonical");
    delete created;

    suite.Check(LookupPrim(uint16_t{61}) == nullptr,
                "ID61 is the first unassigned PrimId");
    const size_t registered_before = factory.registeredCount();
    suite.RejectsExact<std::invalid_argument>([&] {
        factory.registerPrim(
            "__collective_launch_v1_id61__", static_cast<PrimId>(61),
            []() -> PrimBase * { return new Collective_launch_v1_prim(); });
    }, "unassigned PrimId cannot be registered: 61",
       "factory rejects first-unassigned ID61 exactly");
    suite.Check(factory.registeredCount() == registered_before,
                "failed ID61 registration does not mutate the factory");

    static int legacy_sram_address = 0;
    TaskCoreContext context(
        nullptr, nullptr, nullptr, nullptr, &legacy_sram_address,
        nullptr, nullptr, nullptr, nullptr, uint64_t{0}, unsigned{0});
    suite.RejectsExact<std::logic_error>([&] {
        (void)prim.taskCoreDefault(context);
    }, "Collective_launch_v1_prim requires Worker special dispatch",
       "default execution fails fast with the Worker dispatch gate");
}

} // namespace

int RunCollectiveLaunchV1PrimSelfTest() {
    Suite suite;
    TestGoldenAndRoundTrip(suite);
    TestWireRejectionsAndAtomicity(suite);
    TestFieldContracts(suite);
    TestStrictFactoryAndDispatch(suite);
    if (suite.failures == 0) {
        std::cout << "[COLLECTIVE LAUNCH V1 PRIM] PASS ("
                  << suite.checks << " checks)\n";
    } else {
        std::cerr << "[COLLECTIVE LAUNCH V1 PRIM] FAIL ("
                  << suite.failures << "/" << suite.checks
                  << " checks failed)\n";
    }
    return suite.failures;
}

#ifdef COLLECTIVE_LAUNCH_V1_PRIM_SELFTEST_MAIN
int sc_main(int, char **) { return RunCollectiveLaunchV1PrimSelfTest(); }
#endif
