#pragma once

#include "dte/coll_types.h"
#include "prims/base.h"

#include <cstdint>

inline constexpr uint8_t kCollectiveDataV1PrimWireVersion = 1;
inline constexpr uint8_t kCollectiveDataV1PrimWireSegments = 6;

enum class CollectiveDataV1PrimMode : uint8_t {
    LOCAL_COPY = 0,
    REDUCE = 1,
};

// Strict-only real-byte collective data operation.  Addresses are relocated
// absolute SRAM byte addresses.  REDUCE source is rank-major N*L; destination
// is one L-byte result.
class Collective_data_v1_prim final : public PrimBase {
public:
    CollectiveDataV1PrimMode mode = CollectiveDataV1PrimMode::LOCAL_COPY;
    CollectiveKey key;
    uint16_t phase_id = 0;
    uint64_t source_address_bytes = 0;
    uint64_t destination_address_bytes = 0;
    uint64_t length_bytes = 1;
    uint16_t input_count = 1;
    CollDType dtype = CollDType::UINT8;
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
