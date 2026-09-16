#pragma once

#include "dte/coll_types.h"
#include "prims/base.h"

#include <cstdint>

inline constexpr uint8_t kCollectiveDataV1PrimWireVersion = 1;
inline constexpr uint8_t kCollectiveDataV2CastWireVersion = 2;
inline constexpr uint8_t kCollectiveDataV3StrideWireVersion = 3;
inline constexpr uint8_t kCollectiveDataV1PrimWireSegments = 6;

enum class CollectiveDataV1PrimMode : uint8_t {
    LOCAL_COPY = 0,
    REDUCE = 1,
};

// Strict-only real-byte collective data operation.  Addresses are relocated
// absolute SRAM byte addresses. REDUCE has N logical L-byte inputs; V1/V2
// stores them tightly, while V3 stores each at source + rank*input_stride_bytes.
// Destination is one L-byte result (or 2*L for the existing V2 cast).
class Collective_data_v1_prim final : public PrimBase {
public:
    CollectiveDataV1PrimMode mode = CollectiveDataV1PrimMode::LOCAL_COPY;
    CollectiveKey key;
    uint16_t phase_id = 0;
    uint64_t source_address_bytes = 0;
    uint64_t destination_address_bytes = 0;
    uint64_t length_bytes = 1;
    // Zero means the legacy tight rank-major layout (one L-byte input per rank).
    uint64_t input_stride_bytes = 0;
    uint16_t input_count = 1;
    CollDType dtype = CollDType::UINT8;
    // Equal to dtype on V1 wire. V2 is reserved for one-input FP16->FP32
    // local reduction/cast and writes twice the source byte count.
    CollDType output_dtype = CollDType::UINT8;
    CollReduceOp reduce_op = CollReduceOp::NONE;

    void Validate() const;
    int taskCoreDefault(TaskCoreContext &context) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>> wire) override;
    void printSelf() override;

    Collective_data_v1_prim() {
        name = "Collective_data_v1_prim";
        setPrimMainCategory(COMM_PRIM);
    }
};

// Byte-level arithmetic witness used by the ISA selftest for the V2 cast
// and variable-size FP32 gate gradient local sum.
bool CollectiveDataV2ArithmeticSelfTest();
