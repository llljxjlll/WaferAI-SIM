#pragma once

#include "memory/sram/sram_types.h"
#include "prims/base.h"

#include <cstdint>
#include <string>

enum class SramLifecycleOp : uint8_t {
    ALLOC = 0,
    FREE = 1,
    RESIZE = 2,
    RENAME = 3,
    CLEAR_TARGETED = 4,
};

// Strict internal form shared by the public SRAM lifecycle instructions.
class Sram_lifecycle : public PrimBase {
public:
    SramLifecycleOp op = SramLifecycleOp::ALLOC;
    std::string region_name;
    std::string label;
    std::string new_label;
    uint64_t size_bytes = 0;
    uint64_t alignment_bytes = 0;
    sram::AllocationLifetime lifetime = sram::AllocationLifetime::kTask;
    bool spillable = false;

    int taskCoreDefault(TaskCoreContext &context);
    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> segments);
    void printSelf();

    Sram_lifecycle() {
        name = "Sram_lifecycle";
        setPrimMainCategory(MEM_PRIM);
    }
};
