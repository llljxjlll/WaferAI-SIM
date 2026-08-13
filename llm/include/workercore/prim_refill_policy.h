#pragma once

#include "defs/enums.h"
#include "dte/coll_types.h"

inline bool IsCollectiveReservedEndpointTag(int tag) noexcept {
    return tag >= static_cast<int>(COLL_TAG_BASE) &&
           tag <= static_cast<int>(COLL_TAG_MAX);
}

inline bool SendPrimMayRefill(SEND_TYPE type, int tag) noexcept {
    if (type == SEND_DONE) return false;
    if ((type == SEND_REQ || type == SEND_DATA) &&
        IsCollectiveReservedEndpointTag(tag))
        return false;
    return true;
}

inline bool RecvPrimMayRefill(RECV_TYPE type, int tag) noexcept {
    if (type == RECV_CONF || type == RECV_WEIGHT) return false;
    if ((type == RECV_ACK || type == RECV_DATA) &&
        IsCollectiveReservedEndpointTag(tag))
        return false;
    return true;
}
