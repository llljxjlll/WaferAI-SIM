#include "dte/collective_data_v1.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {

uint64_t CheckedMultiply(uint64_t a, uint64_t b, const char *message) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
        throw std::overflow_error(message);
    return a * b;
}

size_t CheckedSize(uint64_t value, const char *message) {
    if (value > std::numeric_limits<size_t>::max())
        throw std::overflow_error(message);
    return static_cast<size_t>(value);
}

uint64_t DTypeBytes(CollDType dtype) {
    switch (dtype) {
    case CollDType::UINT8: return 1;
    case CollDType::INT32: return 4;
    case CollDType::INT64: return 8;
    case CollDType::FP32:
    case CollDType::FP16:
    case CollDType::FP8:
        break;
    }
    throw std::invalid_argument(
        "ISA-v1 endpoint reduce supports UINT8, INT32, and INT64 only");
}

uint64_t DecodeLittleEndian(const uint8_t *bytes, uint64_t width) {
    uint64_t value = 0;
    for (uint64_t i = 0; i < width; ++i)
        value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    return value;
}

void EncodeLittleEndian(uint64_t value, uint64_t width, uint8_t *bytes) {
    for (uint64_t i = 0; i < width; ++i)
        bytes[i] = static_cast<uint8_t>(value >> (8 * i));
}

uint64_t WidthMask(uint64_t width) {
    return width == 8 ? std::numeric_limits<uint64_t>::max()
                      : (UINT64_C(1) << (8 * width)) - 1;
}

bool SignedLess(uint64_t lhs, uint64_t rhs, uint64_t width) {
    const uint64_t sign = UINT64_C(1) << (8 * width - 1);
    return (lhs ^ sign) < (rhs ^ sign);
}

} // namespace

IsaV1CollectiveDataBuffer::IsaV1CollectiveDataBuffer(
    uint16_t rank_count, uint64_t bytes_per_rank,
    uint64_t max_staging_bytes)
    : rank_count_(rank_count), bytes_per_rank_(bytes_per_rank) {
    if (rank_count_ == 0)
        throw std::invalid_argument(
            "ISA-v1 collective data rank count must be positive");
    if (bytes_per_rank_ == 0)
        throw std::invalid_argument(
            "ISA-v1 collective bytes per rank must be positive");
    if (max_staging_bytes == 0)
        throw std::invalid_argument(
            "ISA-v1 collective staging capacity must be positive");
    expected_bytes_ = CheckedMultiply(
        rank_count_, bytes_per_rank_,
        "ISA-v1 collective rank_count*bytes_per_rank overflows");
    if (expected_bytes_ > max_staging_bytes)
        throw std::length_error(
            "ISA-v1 collective staging capacity is exhausted");
    const size_t size = CheckedSize(
        expected_bytes_, "ISA-v1 collective staging exceeds host size_t");
    staging_.assign(size, 0);
    received_.assign(size, 0);
}

void IsaV1CollectiveDataBuffer::Accept(
    uint16_t source_rank, uint64_t offset_bytes,
    const std::vector<uint8_t> &bytes) {
    if (source_rank >= rank_count_)
        throw std::invalid_argument(
            "ISA-v1 collective source rank is outside the group");
    if (bytes.empty())
        throw std::invalid_argument(
            "ISA-v1 collective chunk must contain at least one byte");
    if (offset_bytes > bytes_per_rank_ ||
        bytes.size() > bytes_per_rank_ - offset_bytes)
        throw std::out_of_range(
            "ISA-v1 collective chunk exceeds its rank slice");

    const uint64_t rank_base = CheckedMultiply(
        source_rank, bytes_per_rank_,
        "ISA-v1 collective rank staging offset overflows");
    const uint64_t begin_u64 = rank_base + offset_bytes;
    const size_t begin = CheckedSize(
        begin_u64, "ISA-v1 collective chunk offset exceeds host size_t");
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (received_[begin + i] != 0)
            throw std::invalid_argument(
                "ISA-v1 collective chunk overlaps previously received data");
    }

    std::copy(bytes.begin(), bytes.end(), staging_.begin() + begin);
    std::fill(received_.begin() + begin,
              received_.begin() + begin + bytes.size(), 1);
    received_bytes_ += bytes.size();
    ++accepted_chunks_;
}

bool IsaV1CollectiveDataBuffer::Complete() const noexcept {
    return expected_bytes_ != 0 && received_bytes_ == expected_bytes_;
}

void IsaV1CollectiveDataBuffer::RequireComplete() const {
    if (!Complete())
        throw std::logic_error(
            "ISA-v1 collective data is incomplete");
}

std::vector<uint8_t> IsaV1CollectiveDataBuffer::TakeGathered() {
    RequireComplete();
    std::vector<uint8_t> result = staging_;
    Reset();
    return result;
}

std::vector<uint8_t> IsaV1CollectiveDataBuffer::TakeReduced(
    CollDType dtype, CollReduceOp reduce_op) {
    RequireComplete();
    if (reduce_op != CollReduceOp::SUM && reduce_op != CollReduceOp::MAX)
        throw std::invalid_argument(
            "ISA-v1 endpoint reduce requires SUM or MAX");
    const uint64_t width = DTypeBytes(dtype);
    if (bytes_per_rank_ % width != 0)
        throw std::invalid_argument(
            "ISA-v1 endpoint reduce bytes are not dtype aligned");

    std::vector<uint8_t> result(
        CheckedSize(bytes_per_rank_,
                    "ISA-v1 reduce result exceeds host size_t"));
    const uint64_t mask = WidthMask(width);
    for (uint64_t offset = 0; offset < bytes_per_rank_; offset += width) {
        uint64_t accumulator = DecodeLittleEndian(
            &staging_[CheckedSize(offset, "ISA-v1 reduce offset overflow")],
            width);
        for (uint16_t rank = 1; rank < rank_count_; ++rank) {
            const uint64_t index = CheckedMultiply(
                rank, bytes_per_rank_,
                "ISA-v1 reduce rank offset overflows") + offset;
            const uint64_t operand = DecodeLittleEndian(
                &staging_[CheckedSize(index,
                                      "ISA-v1 reduce index overflow")],
                width);
            if (reduce_op == CollReduceOp::SUM) {
                accumulator = (accumulator + operand) & mask;
            } else if (dtype == CollDType::UINT8) {
                accumulator = std::max(accumulator, operand);
            } else if (SignedLess(accumulator, operand, width)) {
                accumulator = operand;
            }
        }
        EncodeLittleEndian(
            accumulator, width,
            &result[CheckedSize(offset, "ISA-v1 reduce output overflow")]);
    }
    Reset();
    return result;
}

void IsaV1CollectiveDataBuffer::Reset() noexcept {
    rank_count_ = 0;
    bytes_per_rank_ = 0;
    expected_bytes_ = 0;
    received_bytes_ = 0;
    accepted_chunks_ = 0;
    staging_.clear();
    received_.clear();
}

void IsaV1CollectiveDataBuffer::Abort() noexcept { Reset(); }

IsaV1CollectiveDataResidual
IsaV1CollectiveDataBuffer::Residual() const noexcept {
    return {expected_bytes_, received_bytes_, accepted_chunks_};
}

uint64_t IsaV1ReduceOperationCount(uint16_t rank_count,
                                   uint64_t bytes_per_rank,
                                   CollDType dtype) {
    if (rank_count == 0)
        throw std::invalid_argument(
            "ISA-v1 reduce rank count must be positive");
    const uint64_t width = DTypeBytes(dtype);
    if (bytes_per_rank == 0 || bytes_per_rank % width != 0)
        throw std::invalid_argument(
            "ISA-v1 reduce bytes are not dtype aligned");
    return CheckedMultiply(
        bytes_per_rank / width, static_cast<uint64_t>(rank_count - 1),
        "ISA-v1 reduce operation count overflows");
}
