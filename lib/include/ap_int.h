// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
/*
   __VIVADO_HLS_COPYRIGHT-INFO__
*/

#ifndef __AP_INT_H__
#define __AP_INT_H__

#include <ap_common.h>
#include <ap_int_base.h>
#include <ap_int_ref.h>

//---------------------------------------------------------------

/// Sign Arbitrary Precision Type.
template <int _AP_W>
struct ap_int : ap_int_base<_AP_W, true> {
  typedef ap_int_base<_AP_W, true> Base;
  // Constructor
  INLINE ap_int() : Base() {}
  template <int _AP_W2>
  INLINE ap_int(const ap_int<_AP_W2>& op) {
    Base::V = op.V;
  }

  template <int _AP_W2>
  INLINE ap_int(const volatile ap_int<_AP_W2>& op) {
    Base::V = op.V;
  }

  template <int _AP_W2>
  INLINE ap_int(const ap_uint<_AP_W2>& op) {
    Base::V = op.V;
  }

  template <int _AP_W2>
  INLINE ap_int(const volatile ap_uint<_AP_W2>& op) {
    Base::V = op.V;
  }

#ifdef __SYNTHESIS__
#if ((__clang_major__ != 3) || (__clang_minor__ != 1))
  /// ctor from raw val
  INLINE ap_int(unsigned V __attribute__((bitwidth(_AP_W))), bool raw) {
    Base::V = V;
  }
#endif
#endif

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_int(const ap_range_ref<_AP_W2, _AP_S2>& ref) : Base(ref) {}

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_int(const ap_bit_ref<_AP_W2, _AP_S2>& ref) : Base(ref) {}

  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_int(const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& ref)
      : Base(ref) {}

  template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
            int _AP_N2>
  INLINE ap_int(const ap_fixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base((ap_fixed_base<_AP_W2, _AP_I2, true, _AP_Q2, _AP_O2, _AP_N2>)op) {}

  template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
            int _AP_N2>
  INLINE ap_int(const ap_ufixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base((ap_fixed_base<_AP_W2, _AP_I2, false, _AP_Q2, _AP_O2, _AP_N2>)op) {
  }

  template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
            int _AP_N2>
  INLINE ap_int(
      const volatile ap_fixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base((ap_fixed_base<_AP_W2, _AP_I2, true, _AP_Q2, _AP_O2, _AP_N2>)op) {}

  template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
            int _AP_N2>
  INLINE ap_int(
      const volatile ap_ufixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base((ap_fixed_base<_AP_W2, _AP_I2, false, _AP_Q2, _AP_O2, _AP_N2>)op) {
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_int(const ap_int_base<_AP_W2, _AP_S2>& op) {
    Base::V = op.V;
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_int(
      const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base(op) {}

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_int(
      const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base(op) {}

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_int(
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base(op) {}

#define CTOR(TYPE) \
  INLINE ap_int(TYPE val) { Base::V = val; }
  CTOR(bool)
  CTOR(char)
  CTOR(signed char)
  CTOR(unsigned char)
  CTOR(short)
  CTOR(unsigned short)
  CTOR(int)
  CTOR(unsigned int)
  CTOR(long)
  CTOR(unsigned long)
  CTOR(ap_slong)
  CTOR(ap_ulong)
#undef CTOR
  ap_int(double val) : Base(val) {}
  ap_int(float val) : Base(val) {}
  ap_int(half val) : Base(val) {}

  // ap_int_base will guess radix if radix is not provided.
  INLINE ap_int(const char* s) : Base(s) {}

  INLINE ap_int(const char* s, signed char rd) : Base(s, rd) {}

  // Assignment
  /* ctor will be used when right is not of proper type. */

  INLINE ap_int& operator=(const ap_int<_AP_W>& op2) {
    Base::V = op2.V;
    return *this;
  }

  /* cannot bind volatile reference to non-volatile type. */
  INLINE ap_int& operator=(const volatile ap_int<_AP_W>& op2) {
    Base::V = op2.V;
    return *this;
  }

  /* cannot return volatile *this. */
  INLINE void operator=(const ap_int<_AP_W>& op2) volatile { Base::V = op2.V; }

  INLINE void operator=(const volatile ap_int<_AP_W>& op2) volatile {
    Base::V = op2.V;
  }

}; // struct ap_int.

//---------------------------------------------------------------

/// Unsigned Arbitrary Precision Type.
template <int _AP_W>
struct ap_uint : ap_int_base<_AP_W, false> {
  typedef ap_int_base<_AP_W, false> Base;
  // Constructor
  INLINE ap_uint() : Base() {}
  template <int _AP_W2>
  INLINE ap_uint(const ap_uint<_AP_W2>& op) {
    Base::V = op.V;
  }

  template <int _AP_W2>
  INLINE ap_uint(const ap_int<_AP_W2>& op) {
    Base::V = op.V;
  }

  template <int _AP_W2>
  INLINE ap_uint(const volatile ap_uint<_AP_W2>& op) {
    Base::V = op.V;
  }

  template <int _AP_W2>
  INLINE ap_uint(const volatile ap_int<_AP_W2>& op) {
    Base::V = op.V;
  }

#ifdef __SYNTHESIS__
#if ((__clang_major__ != 3) || (__clang_minor__ != 1))
  /// ctor from raw val
  INLINE ap_uint(unsigned V __attribute__((bitwidth(_AP_W))), bool raw) {
    Base::V = V;
  }
#endif
#endif

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_uint(const ap_range_ref<_AP_W2, _AP_S2>& ref) : Base(ref) {}

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_uint(const ap_bit_ref<_AP_W2, _AP_S2>& ref) : Base(ref) {}

  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_uint(const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& ref)
      : Base(ref) {}

  template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
            int _AP_N2>
  INLINE ap_uint(const ap_fixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base((ap_fixed_base<_AP_W2, _AP_I2, true, _AP_Q2, _AP_O2, _AP_N2>)op) {}

  template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
            int _AP_N2>
  INLINE ap_uint(const ap_ufixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base((ap_fixed_base<_AP_W2, _AP_I2, false, _AP_Q2, _AP_O2, _AP_N2>)op) {
  }

  template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
            int _AP_N2>
  INLINE ap_uint(
      const volatile ap_fixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base((ap_fixed_base<_AP_W2, _AP_I2, true, _AP_Q2, _AP_O2, _AP_N2>)op) {}

  template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
            int _AP_N2>
  INLINE ap_uint(
      const volatile ap_ufixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base((ap_fixed_base<_AP_W2, _AP_I2, false, _AP_Q2, _AP_O2, _AP_N2>)op) {
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_uint(const ap_int_base<_AP_W2, _AP_S2>& op) {
    Base::V = op.V;
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_uint(
      const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base(op) {}

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_uint(
      const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base(op) {}

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_uint(
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base(op) {}

#define CTOR(TYPE) \
  INLINE ap_uint(TYPE val) { Base::V = val; }
  CTOR(bool)
  CTOR(char)
  CTOR(signed char)
  CTOR(unsigned char)
  CTOR(short)
  CTOR(unsigned short)
  CTOR(int)
  CTOR(unsigned int)
  CTOR(long)
  CTOR(unsigned long)
  CTOR(ap_slong)
  CTOR(ap_ulong)
#undef CTOR
  ap_uint(double val) : Base(val) {}
  ap_uint(float val) : Base(val) {}
  ap_uint(half val) : Base(val) {}

  // ap_int_base will guess radix if radix is not provided.
  INLINE ap_uint(const char* s) : Base(s) {}

  INLINE ap_uint(const char* s, signed char rd) : Base(s, rd) {}

  // Assignment
  /* XXX ctor will be used when right is not of proper type. */

  INLINE ap_uint& operator=(const ap_uint<_AP_W>& op2) {
    Base::V = op2.V;
    return *this;
  }

  /* cannot bind volatile reference to non-volatile type. */
  INLINE ap_uint& operator=(const volatile ap_uint<_AP_W>& op2) {
    Base::V = op2.V;
    return *this;
  }

  /* cannot return volatile *this. */
  INLINE void operator=(const ap_uint<_AP_W>& op2) volatile { Base::V = op2.V; }

  INLINE void operator=(const volatile ap_uint<_AP_W>& op2) volatile {
    Base::V = op2.V;
  }

}; // struct ap_uint.

#define ap_bigint ap_int
#define ap_biguint ap_uint

#if !defined(__SYNTHESIS__) && (defined(SYSTEMC_H) || defined(SYSTEMC_INCLUDED))
// XXX sc_trace overload for ap_fixed is already included in
// "ap_sysc/ap_sc_extras.h", so do not define in synthesis.
template <int _AP_W>
INLINE void sc_trace(sc_core::sc_trace_file* tf, const ap_int<_AP_W>& op,
                     const std::string& name) {
  if (tf) tf->trace(sc_dt::sc_lv<_AP_W>(op.to_string(2).c_str()), name);
}

template <int _AP_W>
INLINE void sc_trace(sc_core::sc_trace_file* tf, const ap_uint<_AP_W>& op,
                     const std::string& name) {
  if (tf) tf->trace(sc_dt::sc_lv<_AP_W>(op.to_string(2).c_str()), name);
}
#endif // System C sim

#include <ap_int_special.h>

#endif // ifndef __AP_INT_H__ else

// FIXME user should include ap_fixed.h when using ap_fixed.
// to avoid circular inclusion, must check whether this is required by
// ap_fixed.h
#ifndef __AP_FIXED_H__
#include <ap_fixed.h>
#endif

// -*- cpp -*-
