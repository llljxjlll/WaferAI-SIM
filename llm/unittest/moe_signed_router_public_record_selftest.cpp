#include "isa/record_codec.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
SramAddressOperand Absolute(uint64_t address) {
    return {SramAddressKind::ABSOLUTE, address, 0, 0};
}

bool Reject(const std::function<void()> &action) {
    try {
        action();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}
} // namespace

int main() {
    try {
        MoeScoreWeightedForwardOperands forward;
        forward.route = Absolute(0);
        forward.score = Absolute(128);
        forward.returns = Absolute(256);
        forward.combined = Absolute(384);
        forward.rank_rows = 4;
        forward.hidden_size = 4;
        forward.expert_count = 2;
        forward.route_bytes = 80;
        const ExternalRecord f{Opcode::MOE_SCORE_WEIGHTED_FORWARD, forward};
        const auto fbytes = EncodeExternalRecord(f);
        if (fbytes.size() != kExternalRecordHeaderSize + 116 ||
            std::get<MoeScoreWeightedForwardOperands>(
                DecodeExternalRecordExact(fbytes).operands).route_bytes != 80)
            throw std::logic_error("0x27 fixed encoded80B route and full four addresses were lost");
        forward.route_bytes = 64;
        if (!Reject([&] { EncodeExternalRecord(
                {Opcode::MOE_SCORE_WEIGHTED_FORWARD, forward}); }))
            throw std::logic_error("0x27 truncated route must fail");
        forward.route_bytes = 80;
        forward.score = Absolute(64); // overlaps full80B route
        if (!Reject([&] { EncodeExternalRecord(
                {Opcode::MOE_SCORE_WEIGHTED_FORWARD, forward}); }))
            throw std::logic_error("0x27 route/score overlap must fail");

        MoeScoreWeightBackwardOperands backward;
        backward.route = Absolute(0);
        backward.score = Absolute(128);
        backward.returns = Absolute(256);
        backward.dcombined = Absolute(384);
        backward.dscore = Absolute(512);
        backward.dexpert = Absolute(640);
        backward.rank_rows = 4;
        backward.hidden_size = 4;
        backward.expert_count = 2;
        backward.route_bytes = 80;
        const ExternalRecord b{Opcode::MOE_SCORE_WEIGHT_BACKWARD, backward};
        const auto bbytes = EncodeExternalRecord(b);
        if (bbytes.size() != kExternalRecordHeaderSize + 168 ||
            std::get<MoeScoreWeightBackwardOperands>(
                DecodeExternalRecordExact(bbytes).operands).dexpert.absolute_address_bytes != 640)
            throw std::logic_error("0x28 distinct dScore/dExpert sixth address lost");
        backward.dexpert = backward.dscore;
        if (!Reject([&] { EncodeExternalRecord(
                {Opcode::MOE_SCORE_WEIGHT_BACKWARD, backward}); }))
            throw std::logic_error("0x28 shared dScore/dExpert alias must fail");
        backward.dexpert = Absolute(640);
        backward.route_datatype = ExternalDataType::FP16;
        if (!Reject([&] { EncodeExternalRecord(
                {Opcode::MOE_SCORE_WEIGHT_BACKWARD, backward}); }))
            throw std::logic_error("0x28 route dtype FP16 must fail");
        std::cout << "MOE_ROUTER_PUBLIC_CODEC 0x27/28 fixed route80B six-address PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
