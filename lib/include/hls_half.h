// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
/*
   __VIVADO_HLS_COPYRIGHT-INFO__
*/
#ifndef __HLS_HALF_H__
#define __HLS_HALF_H__

#ifdef __SYNTHESIS__

#ifndef _HLS_HALF_DEFINED_
typedef __fp16 half;
#endif
#else // AESL_SYN

#ifndef __cplusplus
#error C++ is required to include this header file
#endif

#ifndef HLS_NO_XIL_FPO_LIB
#include "hls_half_fpo.h"
#elif defined(__x86_64__)
#include "hls_half_sim.h"
#elif defined(__arm__) || defined(__aarch64__)
#include "hls_half_sim.h"
#endif

#include <limits>
#include <climits>
/// Extensions to the C++ standard library.
//  It is used in hlsmath lib.
namespace std {
/// Numeric limits for half-precision floats.
/// Because of the underlying single-precision implementation of many operations, it inherits some properties from
/// `std::numeric_limits<float>`.
template <>
struct numeric_limits<half> : public numeric_limits<float> {
   public:
    /// Supports signed values.
    static const bool is_signed = true;

    /// Is not exact.
    static const bool is_exact = false;

    /// Doesn't provide modulo arithmetic.
    static const bool is_modulo = false;

    /// IEEE conformant.
    static const bool is_iec559 = true;

    /// Supports infinity.
    static const bool has_infinity = true;

    /// Supports quiet NaNs.
    static const bool has_quiet_NaN = true;

    /// Supports subnormal values.
    static const float_denorm_style has_denorm = denorm_present;

    /// Significant digits.
    static const int digits = 11;

    /// Significant decimal digits.
    static const int digits10 = 3;

    /// Required decimal digits to represent all possible values.
    static const int max_digits10 = 5;

    /// Number base.
    static const int radix = 2;

    /// One more than smallest exponent.
    static const int min_exponent = -13;

    /// Smallest normalized representable power of 10.
    static const int min_exponent10 = -4;

    /// One more than largest exponent
    static const int max_exponent = 16;

    /// Largest finitely representable power of 10.
    static const int max_exponent10 = 4;

    /// Smallest positive normal value.
    static half min() throw() {
        half t;
        t.set_bits(0x0400);
        return t;
    }

    /// Smallest finite value.
    static half lowest() throw() {
        half t;
        t.set_bits(0xFBFF);
        return t;
    }
    /// Largest finite value.
    static half max() throw() {
        half t;
        t.set_bits(0x7BFF);
        return t;
    }

    /// Difference between one and next representable value.
    static half epsilon() throw() {
        half t;
        t.set_bits(0x1400);
        return t;
    }

    /// Maximum rounding error.
    static half round_error() throw() {
        half t;
        t.set_bits(0x3800);
        return t;
    }

    /// Positive infinity.
    static half infinity() throw() {
        half t;
        t.set_bits(0x7C00);
        return t;
    }

    /// Quiet NaN.
    static half quiet_NaN() throw() {
        half t;
        t.set_bits(0x7FFF);
        return t;
    }

    /// Signalling NaN.
    static half signaling_NaN() throw() {
        half t;
        t.set_bits(0x7DFF);
        return t;
    }

    /// Smallest positive subnormal value.
    static half denorm_min() throw() {
        half t;
        t.set_bits(0x0001);
        return t;
    }
}; // numeric_limits<half>
} // namespace std

#include "hls_half_x_utils.h"

#endif // __SYNTHESIS__

#endif // __HLS_HALF__
