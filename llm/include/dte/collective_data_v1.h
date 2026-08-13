#pragma once

#include "dte/coll_types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Bounded, arrival-order-independent staging used by ISA-v1 GATHER and
// endpoint REDUCE.  Rank r owns the tight byte interval [r*L, (r+1)*L).
// Accept() is transactional: a malformed or overlapping chunk changes no
// payload, bitmap, or residual counters.
struct IsaV1CollectiveDataResidual {
    uint64_t expected_bytes = 0;
    uint64_t received_bytes = 0;
    uint64_t accepted_chunks = 0;

    bool Drained() const noexcept {
        return expected_bytes == 0 && received_bytes == 0 &&
               accepted_chunks == 0;
    }
};

class IsaV1CollectiveDataBuffer {
public:
    IsaV1CollectiveDataBuffer(uint16_t rank_count,
                              uint64_t bytes_per_rank,
                              uint64_t max_staging_bytes);

    void Accept(uint16_t source_rank, uint64_t offset_bytes,
                const std::vector<uint8_t> &bytes);

    bool Complete() const noexcept;
    const std::vector<uint8_t> &Staging() const noexcept { return staging_; }

    // GATHER result in canonical rank-major order.  Requires Complete().
    std::vector<uint8_t> TakeGathered();

    // Endpoint REDUCE over all rank-major operands.  Requires Complete().
    // Integer encoding is little-endian.  SUM wraps modulo the dtype width;
    // INT32/INT64 MAX uses signed two's-complement ordering.
    std::vector<uint8_t> TakeReduced(CollDType dtype,
                                     CollReduceOp reduce_op);

    void Abort() noexcept;
    IsaV1CollectiveDataResidual Residual() const noexcept;

    uint16_t rank_count() const noexcept { return rank_count_; }
    uint64_t bytes_per_rank() const noexcept { return bytes_per_rank_; }

private:
    void RequireComplete() const;
    void Reset() noexcept;

    uint16_t rank_count_ = 0;
    uint64_t bytes_per_rank_ = 0;
    uint64_t expected_bytes_ = 0;
    uint64_t received_bytes_ = 0;
    uint64_t accepted_chunks_ = 0;
    std::vector<uint8_t> staging_;
    std::vector<uint8_t> received_;
};

// Exact number of element-wise binary combines performed by endpoint reduce.
uint64_t IsaV1ReduceOperationCount(uint16_t rank_count,
                                   uint64_t bytes_per_rank,
                                   CollDType dtype);
