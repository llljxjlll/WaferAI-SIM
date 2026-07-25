#pragma once

#include "dte/dte_types.h"

#include <cstdint>
#include <limits>


constexpr uint64_t DTE_ASYNC_INVALID_XFER_ID =
    std::numeric_limits<uint64_t>::max();
constexpr uint32_t DTE_ASYNC_INVALID_REMOTE_PEER =
    std::numeric_limits<uint32_t>::max();

enum class DteAsyncOp : uint8_t {
    ISSUE = 0,
    WAIT = 1,
    POLL = 2,
    FENCE = 3,
    CANCEL = 4,
};

enum class DteAsyncAccess : uint8_t {
    NONE = 0,
    READ = 1,
    WRITE = 2,
    READ_WRITE = 3,
};

inline const char *DteAsyncOpName(DteAsyncOp op) {
    switch (op) {
    case DteAsyncOp::ISSUE: return "issue";
    case DteAsyncOp::WAIT: return "wait";
    case DteAsyncOp::POLL: return "poll";
    case DteAsyncOp::FENCE: return "fence";
    case DteAsyncOp::CANCEL: return "cancel";
    }
    return "unknown";
}
