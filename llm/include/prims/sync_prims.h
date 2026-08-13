#pragma once

#include "prims/base.h"

#include <cstdint>

class Group_sync_prim : public PrimBase {
public:
    uint32_t group_id = 0;
    uint32_t sync_seq = 0;

    int taskCoreDefault(TaskCoreContext &context) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>> buffer) override;
    void printSelf() override;

    Group_sync_prim() {
        name = "Group_sync_prim";
        setPrimMainCategory(SYNC_PRIM);
    }
};

enum class EventControlOp : uint8_t {
    SET = 0,
    WAIT = 1,
};

class Event_control_prim : public PrimBase {
public:
    EventControlOp op = EventControlOp::SET;
    uint16_t source_core = 0;
    uint16_t destination_core = 0;
    uint32_t tag = 0;
    uint32_t count = 1;

    int taskCoreDefault(TaskCoreContext &context) override;
    vector<sc_bv<128>> serialize() override;
    void deserialize(vector<sc_bv<128>> buffer) override;
    void printSelf() override;

    Event_control_prim() {
        name = "Event_control_prim";
        setPrimMainCategory(SYNC_PRIM);
    }
};
