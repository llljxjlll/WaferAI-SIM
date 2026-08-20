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
    case CollDType::FP32: return 4;
    case CollDType::FP16: return 2;
    case CollDType::FP8:
        break;
    }
    throw std::invalid_argument(
        "ISA-v1 endpoint reduce supports UINT8, INT32, INT64, FP16, and FP32 only");
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

uint32_t DecodeFp16Bits(uint16_t half) {
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 0x1fu;
    uint32_t fraction = half & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            int unbiased = -14;
            while ((fraction & 0x0400u) == 0) {
                fraction <<= 1;
                --unbiased;
            }
            fraction &= 0x03ffu;
            bits = sign |
                   (static_cast<uint32_t>(unbiased + 127) << 23) |
                   (fraction << 13);
        }
    } else if (exponent == 0x1fu) {
        // Every half NaN enters FP32 arithmetic in one canonical form.
        bits = fraction == 0 ? sign | 0x7f800000u : 0x7fc00000u;
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) |
               (fraction << 13);
    }
    return bits;
}

uint64_t ShiftRightJam(uint64_t value, unsigned distance) {
    if (distance == 0)
        return value;
    if (distance >= 64)
        return value != 0;
    const uint64_t discarded = value &
        ((uint64_t{1} << distance) - 1);
    return (value >> distance) | (discarded != 0);
}

// Deterministic IEEE-754 binary32 add, round-to-nearest ties-to-even. Three
// explicit GRS bits make host floating-point flags and -Ofast irrelevant.
uint32_t AddBinary32Rne(uint32_t left, uint32_t right) {
    const uint32_t left_abs = left & 0x7fffffffu;
    const uint32_t right_abs = right & 0x7fffffffu;
    const uint32_t left_exp = left_abs >> 23;
    const uint32_t right_exp = right_abs >> 23;
    const bool left_nan = left_exp == 0xffu &&
                          (left_abs & 0x7fffffu) != 0;
    const bool right_nan = right_exp == 0xffu &&
                           (right_abs & 0x7fffffu) != 0;
    if (left_nan || right_nan)
        return 0x7fc00000u;
    if (left_exp == 0xffu || right_exp == 0xffu) {
        if (left_exp == 0xffu && right_exp == 0xffu &&
            ((left ^ right) & 0x80000000u) != 0)
            return 0x7fc00000u;
        return left_exp == 0xffu ? left : right;
    }
    if (left_abs == 0 && right_abs == 0)
        return (left & right) & 0x80000000u;
    if (left_abs == 0)
        return right;
    if (right_abs == 0)
        return left;

    uint32_t a = left;
    uint32_t b = right;
    if ((a & 0x7fffffffu) < (b & 0x7fffffffu))
        std::swap(a, b);
    const uint32_t a_raw_exp = (a >> 23) & 0xffu;
    const uint32_t b_raw_exp = (b >> 23) & 0xffu;
    int exponent = static_cast<int>(a_raw_exp == 0 ? 1 : a_raw_exp);
    const int b_exponent =
        static_cast<int>(b_raw_exp == 0 ? 1 : b_raw_exp);
    uint64_t a_sig =
        (uint64_t{a & 0x7fffffu} |
         (a_raw_exp == 0 ? uint64_t{0} : uint64_t{0x800000u})) << 3;
    uint64_t b_sig =
        (uint64_t{b & 0x7fffffu} |
         (b_raw_exp == 0 ? uint64_t{0} : uint64_t{0x800000u})) << 3;
    b_sig = ShiftRightJam(
        b_sig, static_cast<unsigned>(exponent - b_exponent));

    const uint32_t sign = a & 0x80000000u;
    uint64_t result_sig = 0;
    if (((a ^ b) & 0x80000000u) == 0) {
        result_sig = a_sig + b_sig;
        if ((result_sig & (uint64_t{1} << 27)) != 0) {
            result_sig = ShiftRightJam(result_sig, 1);
            ++exponent;
        }
    } else {
        result_sig = a_sig - b_sig;
        if (result_sig == 0)
            return 0;
        while ((result_sig & (uint64_t{1} << 26)) == 0 && exponent > 1) {
            result_sig <<= 1;
            --exponent;
        }
    }

    uint32_t significand = static_cast<uint32_t>(result_sig >> 3);
    const uint32_t round_bits = static_cast<uint32_t>(result_sig & 7u);
    if (round_bits > 4 ||
        (round_bits == 4 && (significand & 1u) != 0)) {
        ++significand;
        if (significand == 0x1000000u) {
            significand >>= 1;
            ++exponent;
        }
    }
    if (exponent >= 0xff)
        return sign | 0x7f800000u;
    const uint32_t encoded_exponent =
        exponent == 1 && significand < 0x800000u
            ? 0
            : static_cast<uint32_t>(exponent);
    return sign | (encoded_exponent << 23) |
           (significand & 0x7fffffu);
}

uint16_t EncodeFp16Rne(uint32_t bits) {
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 0xffu)
        return fraction == 0 ? static_cast<uint16_t>(sign | 0x7c00u)
                             : uint16_t{0x7e00u};
    if (exponent == 0)
        return sign;

    const int unbiased = static_cast<int>(exponent) - 127;
    if (unbiased > 15)
        return static_cast<uint16_t>(sign | 0x7c00u);
    if (unbiased >= -14) {
        uint32_t half_exponent = static_cast<uint32_t>(unbiased + 15);
        uint32_t half_fraction = fraction >> 13;
        const uint32_t remainder = fraction & 0x1fffu;
        if (remainder > 0x1000u ||
            (remainder == 0x1000u && (half_fraction & 1u) != 0)) {
            ++half_fraction;
            if (half_fraction == 0x0400u) {
                half_fraction = 0;
                ++half_exponent;
                if (half_exponent == 0x1fu)
                    return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
        return static_cast<uint16_t>(sign | (half_exponent << 10) |
                                     half_fraction);
    }
    if (unbiased < -25)
        return sign;

    const uint32_t significand = 0x800000u | fraction;
    const unsigned shift = static_cast<unsigned>(13 + (-14 - unbiased));
    uint32_t half_fraction = significand >> shift;
    const uint32_t remainder_mask = (uint32_t{1} << shift) - 1;
    const uint32_t remainder = significand & remainder_mask;
    const uint32_t halfway = uint32_t{1} << (shift - 1);
    if (remainder > halfway ||
        (remainder == halfway && (half_fraction & 1u) != 0))
        ++half_fraction;
    // A rounded subnormal may become the minimum normal (0x0400).
    return static_cast<uint16_t>(sign | half_fraction);
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
    if ((dtype == CollDType::FP16 || dtype == CollDType::FP32) &&
        reduce_op != CollReduceOp::SUM)
        throw std::invalid_argument(
            "ISA-v1 floating-point reduce supports SUM only");
    if (bytes_per_rank_ % width != 0)
        throw std::invalid_argument(
            "ISA-v1 endpoint reduce bytes are not dtype aligned");

    std::vector<uint8_t> result(
        CheckedSize(bytes_per_rank_,
                    "ISA-v1 reduce result exceeds host size_t"));
    const uint64_t mask = WidthMask(width);
    for (uint64_t offset = 0; offset < bytes_per_rank_; offset += width) {
        if (dtype == CollDType::FP16 || dtype == CollDType::FP32) {
            const bool fp32 = dtype == CollDType::FP32;
            uint32_t accumulator = DecodeFp16Bits(static_cast<uint16_t>(
                fp32 ? 0 : DecodeLittleEndian(
                    &staging_[CheckedSize(
                        offset, "ISA-v1 FP16 reduce offset overflow")],
                    width)));
            if (fp32)
                accumulator = static_cast<uint32_t>(DecodeLittleEndian(
                    &staging_[CheckedSize(
                        offset, "ISA-v1 FP32 reduce offset overflow")],
                    width));
            for (uint16_t rank = 1; rank < rank_count_; ++rank) {
                const uint64_t index = CheckedMultiply(
                    rank, bytes_per_rank_,
                    "ISA-v1 FP16 reduce rank offset overflows") + offset;
                uint32_t operand = DecodeFp16Bits(static_cast<uint16_t>(
                    fp32 ? 0 : DecodeLittleEndian(
                        &staging_[CheckedSize(
                            index, "ISA-v1 FP16 reduce index overflow")],
                        width)));
                if (fp32)
                    operand = static_cast<uint32_t>(DecodeLittleEndian(
                        &staging_[CheckedSize(
                            index, "ISA-v1 FP32 reduce index overflow")],
                        width));
                accumulator = AddBinary32Rne(accumulator, operand);
            }
            EncodeLittleEndian(
                fp32 ? accumulator : EncodeFp16Rne(accumulator), width,
                &result[CheckedSize(
                    offset, "ISA-v1 FP16 reduce output overflow")]);
            continue;
        }
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
