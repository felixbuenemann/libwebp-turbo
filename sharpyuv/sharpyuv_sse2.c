// Copyright 2022 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Speed-critical functions for Sharp YUV.
//
// Author: Skal (pascal.massimino@gmail.com)

#include "./sharpyuv_dsp.h"

#if defined(WEBP_USE_SSE2)
#include <emmintrin.h>
#include <stdlib.h>

#include "src/dsp/cpu.h"
#include "webp/types.h"

static uint16_t clip_SSE2(int v, int max) {
  return (v < 0) ? 0 : (v > max) ? max : (uint16_t)v;
}

static uint64_t SharpYuvUpdateY_SSE2(const uint16_t* ref, const uint16_t* src,
                                     uint16_t* dst, int len, int bit_depth) {
  const int max_y = (1 << bit_depth) - 1;
  uint64_t diff = 0;
  uint32_t tmp[4];
  int i;
  const __m128i zero = _mm_setzero_si128();
  const __m128i max = _mm_set1_epi16(max_y);
  const __m128i one = _mm_set1_epi16(1);
  __m128i sum = zero;

  for (i = 0; i + 8 <= len; i += 8) {
    const __m128i A = _mm_loadu_si128((const __m128i*)(ref + i));
    const __m128i B = _mm_loadu_si128((const __m128i*)(src + i));
    const __m128i C = _mm_loadu_si128((const __m128i*)(dst + i));
    const __m128i D = _mm_sub_epi16(A, B);       // diff_y
    const __m128i E = _mm_cmpgt_epi16(zero, D);  // sign (-1 or 0)
    const __m128i F = _mm_add_epi16(C, D);       // new_y
    const __m128i G = _mm_or_si128(E, one);      // -1 or 1
    const __m128i H = _mm_max_epi16(_mm_min_epi16(F, max), zero);
    const __m128i I = _mm_madd_epi16(D, G);  // sum(abs(...))
    _mm_storeu_si128((__m128i*)(dst + i), H);
    sum = _mm_add_epi32(sum, I);
  }
  _mm_storeu_si128((__m128i*)tmp, sum);
  diff = tmp[3] + tmp[2] + tmp[1] + tmp[0];
  for (; i < len; ++i) {
    const int diff_y = ref[i] - src[i];
    const int new_y = (int)dst[i] + diff_y;
    dst[i] = clip_SSE2(new_y, max_y);
    diff += (uint64_t)abs(diff_y);
  }
  return diff;
}

static void SharpYuvUpdateRGB_SSE2(const int16_t* ref, const int16_t* src,
                                   int16_t* dst, int len) {
  int i = 0;
  for (i = 0; i + 8 <= len; i += 8) {
    const __m128i A = _mm_loadu_si128((const __m128i*)(ref + i));
    const __m128i B = _mm_loadu_si128((const __m128i*)(src + i));
    const __m128i C = _mm_loadu_si128((const __m128i*)(dst + i));
    const __m128i D = _mm_sub_epi16(A, B);  // diff_uv
    const __m128i E = _mm_add_epi16(C, D);  // new_uv
    _mm_storeu_si128((__m128i*)(dst + i), E);
  }
  for (; i < len; ++i) {
    const int diff_uv = ref[i] - src[i];
    dst[i] += diff_uv;
  }
}

static void SharpYuvFilterRow16_SSE2(const int16_t* A, const int16_t* B,
                                     int len, const uint16_t* best_y,
                                     uint16_t* out, int bit_depth) {
  const int max_y = (1 << bit_depth) - 1;
  int i;
  const __m128i kCst8 = _mm_set1_epi16(8);
  const __m128i max = _mm_set1_epi16(max_y);
  const __m128i zero = _mm_setzero_si128();
  for (i = 0; i + 8 <= len; i += 8) {
    const __m128i a0 = _mm_loadu_si128((const __m128i*)(A + i + 0));
    const __m128i a1 = _mm_loadu_si128((const __m128i*)(A + i + 1));
    const __m128i b0 = _mm_loadu_si128((const __m128i*)(B + i + 0));
    const __m128i b1 = _mm_loadu_si128((const __m128i*)(B + i + 1));
    const __m128i a0b1 = _mm_add_epi16(a0, b1);
    const __m128i a1b0 = _mm_add_epi16(a1, b0);
    const __m128i a0a1b0b1 = _mm_add_epi16(a0b1, a1b0);  // A0+A1+B0+B1
    const __m128i a0a1b0b1_8 = _mm_add_epi16(a0a1b0b1, kCst8);
    const __m128i a0b1_2 = _mm_add_epi16(a0b1, a0b1);  // 2*(A0+B1)
    const __m128i a1b0_2 = _mm_add_epi16(a1b0, a1b0);  // 2*(A1+B0)
    const __m128i c0 = _mm_srai_epi16(_mm_add_epi16(a0b1_2, a0a1b0b1_8), 3);
    const __m128i c1 = _mm_srai_epi16(_mm_add_epi16(a1b0_2, a0a1b0b1_8), 3);
    const __m128i d0 = _mm_add_epi16(c1, a0);
    const __m128i d1 = _mm_add_epi16(c0, a1);
    const __m128i e0 = _mm_srai_epi16(d0, 1);
    const __m128i e1 = _mm_srai_epi16(d1, 1);
    const __m128i f0 = _mm_unpacklo_epi16(e0, e1);
    const __m128i f1 = _mm_unpackhi_epi16(e0, e1);
    const __m128i g0 = _mm_loadu_si128((const __m128i*)(best_y + 2 * i + 0));
    const __m128i g1 = _mm_loadu_si128((const __m128i*)(best_y + 2 * i + 8));
    const __m128i h0 = _mm_add_epi16(g0, f0);
    const __m128i h1 = _mm_add_epi16(g1, f1);
    const __m128i i0 = _mm_max_epi16(_mm_min_epi16(h0, max), zero);
    const __m128i i1 = _mm_max_epi16(_mm_min_epi16(h1, max), zero);
    _mm_storeu_si128((__m128i*)(out + 2 * i + 0), i0);
    _mm_storeu_si128((__m128i*)(out + 2 * i + 8), i1);
  }
  for (; i < len; ++i) {
    //   (9 * A0 + 3 * A1 + 3 * B0 + B1 + 8) >> 4 =
    // = (8 * A0 + 2 * (A1 + B0) + (A0 + A1 + B0 + B1 + 8)) >> 4
    // We reuse the common sub-expressions.
    const int a0b1 = A[i + 0] + B[i + 1];
    const int a1b0 = A[i + 1] + B[i + 0];
    const int a0a1b0b1 = a0b1 + a1b0 + 8;
    const int v0 = (8 * A[i + 0] + 2 * a1b0 + a0a1b0b1) >> 4;
    const int v1 = (8 * A[i + 1] + 2 * a0b1 + a0a1b0b1) >> 4;
    out[2 * i + 0] = clip_SSE2(best_y[2 * i + 0] + v0, max_y);
    out[2 * i + 1] = clip_SSE2(best_y[2 * i + 1] + v1, max_y);
  }
}

static WEBP_INLINE __m128i s16_to_s32(__m128i in) {
  return _mm_srai_epi32(_mm_unpacklo_epi16(in, in), 16);
}

static void SharpYuvFilterRow32_SSE2(const int16_t* A, const int16_t* B,
                                     int len, const uint16_t* best_y,
                                     uint16_t* out, int bit_depth) {
  const int max_y = (1 << bit_depth) - 1;
  int i;
  const __m128i kCst8 = _mm_set1_epi32(8);
  const __m128i max = _mm_set1_epi16(max_y);
  const __m128i zero = _mm_setzero_si128();
  for (i = 0; i + 4 <= len; i += 4) {
    const __m128i a0 = s16_to_s32(_mm_loadl_epi64((const __m128i*)(A + i + 0)));
    const __m128i a1 = s16_to_s32(_mm_loadl_epi64((const __m128i*)(A + i + 1)));
    const __m128i b0 = s16_to_s32(_mm_loadl_epi64((const __m128i*)(B + i + 0)));
    const __m128i b1 = s16_to_s32(_mm_loadl_epi64((const __m128i*)(B + i + 1)));
    const __m128i a0b1 = _mm_add_epi32(a0, b1);
    const __m128i a1b0 = _mm_add_epi32(a1, b0);
    const __m128i a0a1b0b1 = _mm_add_epi32(a0b1, a1b0);  // A0+A1+B0+B1
    const __m128i a0a1b0b1_8 = _mm_add_epi32(a0a1b0b1, kCst8);
    const __m128i a0b1_2 = _mm_add_epi32(a0b1, a0b1);  // 2*(A0+B1)
    const __m128i a1b0_2 = _mm_add_epi32(a1b0, a1b0);  // 2*(A1+B0)
    const __m128i c0 = _mm_srai_epi32(_mm_add_epi32(a0b1_2, a0a1b0b1_8), 3);
    const __m128i c1 = _mm_srai_epi32(_mm_add_epi32(a1b0_2, a0a1b0b1_8), 3);
    const __m128i d0 = _mm_add_epi32(c1, a0);
    const __m128i d1 = _mm_add_epi32(c0, a1);
    const __m128i e0 = _mm_srai_epi32(d0, 1);
    const __m128i e1 = _mm_srai_epi32(d1, 1);
    const __m128i f0 = _mm_unpacklo_epi32(e0, e1);
    const __m128i f1 = _mm_unpackhi_epi32(e0, e1);
    const __m128i g = _mm_loadu_si128((const __m128i*)(best_y + 2 * i + 0));
    const __m128i h_16 = _mm_add_epi16(g, _mm_packs_epi32(f0, f1));
    const __m128i final = _mm_max_epi16(_mm_min_epi16(h_16, max), zero);
    _mm_storeu_si128((__m128i*)(out + 2 * i + 0), final);
  }
  for (; i < len; ++i) {
    //   (9 * A0 + 3 * A1 + 3 * B0 + B1 + 8) >> 4 =
    // = (8 * A0 + 2 * (A1 + B0) + (A0 + A1 + B0 + B1 + 8)) >> 4
    // We reuse the common sub-expressions.
    const int a0b1 = A[i + 0] + B[i + 1];
    const int a1b0 = A[i + 1] + B[i + 0];
    const int a0a1b0b1 = a0b1 + a1b0 + 8;
    const int v0 = (8 * A[i + 0] + 2 * a1b0 + a0a1b0b1) >> 4;
    const int v1 = (8 * A[i + 1] + 2 * a0b1 + a0a1b0b1) >> 4;
    out[2 * i + 0] = clip_SSE2(best_y[2 * i + 0] + v0, max_y);
    out[2 * i + 1] = clip_SSE2(best_y[2 * i + 1] + v1, max_y);
  }
}

static void SharpYuvFilterRow_SSE2(const int16_t* A, const int16_t* B, int len,
                                   const uint16_t* best_y, uint16_t* out,
                                   int bit_depth) {
  if (bit_depth <= 10) {
    SharpYuvFilterRow16_SSE2(A, B, len, best_y, out, bit_depth);
  } else {
    SharpYuvFilterRow32_SSE2(A, B, len, best_y, out, bit_depth);
  }
}

//------------------------------------------------------------------------------
// Gamma-space gray computation, going through linear space for averages. The
// sRGB transfer function and its inverse are evaluated with polynomial
// approximations whose error is below the quantization error of the
// table-based scalar versions in sharpyuv_gamma.c.

#include "./sharpyuv_gamma.h"

// thresh = 0.018053968510807, a = 0.09929682680944 (as in sharpyuv_gamma.c).
#define SHARPYUV_GAMMA_THRESH 0.018053968510807f
#define SHARPYUV_GAMMA_A 0.09929682680944f

// res = (mask) ? a : b
static WEBP_INLINE __m128 Select_SSE2(const __m128 mask, const __m128 a,
                                      const __m128 b) {
  return _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b));
}

// Degree-6 least-squares fit of ((g + a) / (1 + a)) ^ (1 / 0.45) on
// [4.5 * thresh, 1]. Max error 4.4e-6, below the 16-bit quantization of the
// table used by SharpYuvGammaToLinear().
static WEBP_INLINE __m128 GammaToLinear_SSE2(const __m128 g) {
  const __m128 lo = _mm_mul_ps(g, _mm_set1_ps(1.0f / 4.5f));
  __m128 p = _mm_set1_ps(-2.035152666e-02f);
  p = _mm_add_ps(_mm_mul_ps(p, g), _mm_set1_ps(9.086542551e-02f));
  p = _mm_add_ps(_mm_mul_ps(p, g), _mm_set1_ps(-1.905903016e-01f));
  p = _mm_add_ps(_mm_mul_ps(p, g), _mm_set1_ps(3.256689955e-01f));
  p = _mm_add_ps(_mm_mul_ps(p, g), _mm_set1_ps(6.849562000e-01f));
  p = _mm_add_ps(_mm_mul_ps(p, g), _mm_set1_ps(1.045739367e-01f));
  p = _mm_add_ps(_mm_mul_ps(p, g), _mm_set1_ps(4.874765193e-03f));
  return Select_SSE2(
      _mm_cmple_ps(g, _mm_set1_ps(SHARPYUV_GAMMA_THRESH * 4.5f)), lo, p);
}

// (1 + a) * l ^ 0.45 - a, with l ^ 0.45 evaluated as a degree-7 fit in
// u = sqrt(sqrt(l)) on [thresh, 1]. Max error 1e-7, about 500x below the
// quantization of the 512-entry interpolated table used by
// SharpYuvLinearToGamma().
static WEBP_INLINE __m128 LinearToGamma_SSE2(const __m128 l) {
  const __m128 lo = _mm_mul_ps(l, _mm_set1_ps(4.5f));
  const __m128 u = _mm_sqrt_ps(_mm_sqrt_ps(l));
  __m128 p = _mm_set1_ps(-1.622143889e-02f);
  p = _mm_add_ps(_mm_mul_ps(p, u), _mm_set1_ps(9.720616133e-02f));
  p = _mm_add_ps(_mm_mul_ps(p, u), _mm_set1_ps(-2.609642418e-01f));
  p = _mm_add_ps(_mm_mul_ps(p, u), _mm_set1_ps(4.250514527e-01f));
  p = _mm_add_ps(_mm_mul_ps(p, u), _mm_set1_ps(-5.155822037e-01f));
  p = _mm_add_ps(_mm_mul_ps(p, u), _mm_set1_ps(1.208584132e+00f));
  p = _mm_add_ps(_mm_mul_ps(p, u), _mm_set1_ps(6.450501113e-02f));
  p = _mm_add_ps(_mm_mul_ps(p, u), _mm_set1_ps(-2.578887858e-03f));
  p = _mm_sub_ps(_mm_mul_ps(p, _mm_set1_ps(1.0f + SHARPYUV_GAMMA_A)),
                 _mm_set1_ps(SHARPYUV_GAMMA_A));
  return Select_SSE2(_mm_cmple_ps(l, _mm_set1_ps(SHARPYUV_GAMMA_THRESH)), lo,
                     p);
}

// Loads 4 16-bit samples and normalizes them to [0, 1].
static WEBP_INLINE __m128 LoadNormalized_SSE2(const uint16_t* src,
                                              const __m128 norm) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i s = _mm_loadl_epi64((const __m128i*)src);
  return _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(s, zero)), norm);
}

static int RGBToGray_SSE2(int64_t r, int64_t g, int64_t b) {
  const int64_t luma = 13933 * r + 46871 * g + 4732 * b + (1 << 15);
  return (int)(luma >> 16);
}

static void SharpYuvUpdateW_SSE2(const uint16_t* src, uint16_t* dst, int w,
                                 int bit_depth,
                                 SharpYuvTransferFunctionType transfer_type) {
  int i = 0;
  if (transfer_type != kSharpYuvTransferFunctionSrgb) {
    SharpYuvUpdateW_C(src, dst, w, bit_depth, transfer_type);
    return;
  }
  {
    const __m128 norm = _mm_set1_ps(1.0f / (1 << bit_depth));
    const __m128 scale = _mm_set1_ps((float)(1 << bit_depth));
    // RGBToGray() coefficients, normalized back from their 16 bit fixed-point.
    const __m128 coeff_r = _mm_set1_ps(13933.0f / 65536.0f);
    const __m128 coeff_g = _mm_set1_ps(46871.0f / 65536.0f);
    const __m128 coeff_b = _mm_set1_ps(4732.0f / 65536.0f);
    for (; i + 4 <= w; i += 4) {
      const __m128 r =
          GammaToLinear_SSE2(LoadNormalized_SSE2(src + 0 * w + i, norm));
      const __m128 g =
          GammaToLinear_SSE2(LoadNormalized_SSE2(src + 1 * w + i, norm));
      const __m128 b =
          GammaToLinear_SSE2(LoadNormalized_SSE2(src + 2 * w + i, norm));
      const __m128 gray =
          _mm_add_ps(_mm_add_ps(_mm_mul_ps(r, coeff_r), _mm_mul_ps(g, coeff_g)),
                     _mm_mul_ps(b, coeff_b));
      const __m128 y = _mm_mul_ps(LinearToGamma_SSE2(gray), scale);
      const __m128i out = _mm_cvtps_epi32(y);  // values fit in 14 bits
      _mm_storel_epi64((__m128i*)(dst + i), _mm_packs_epi32(out, out));
    }
  }
  for (; i < w; ++i) {  // left-over
    const uint32_t R =
        SharpYuvGammaToLinear(src[0 * w + i], bit_depth, transfer_type);
    const uint32_t G =
        SharpYuvGammaToLinear(src[1 * w + i], bit_depth, transfer_type);
    const uint32_t B =
        SharpYuvGammaToLinear(src[2 * w + i], bit_depth, transfer_type);
    const uint32_t Y = RGBToGray_SSE2(R, G, B);
    dst[i] = (uint16_t)SharpYuvLinearToGamma(Y, bit_depth, transfer_type);
  }
}

// Loads 8 gamma-space samples, converts them to linear space and adds them
// pairwise.
static WEBP_INLINE __m128 LoadLinearPairs_SSE2(const uint16_t* src,
                                               const __m128 norm) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i s = _mm_loadu_si128((const __m128i*)src);
  const __m128 lo = GammaToLinear_SSE2(
      _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(s, zero)), norm));
  const __m128 hi = GammaToLinear_SSE2(
      _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(s, zero)), norm));
  const __m128 evens = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));
  const __m128 odds = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 1, 3, 1));
  return _mm_add_ps(evens, odds);
}

static void SharpYuvUpdateChroma_SSE2(
    const uint16_t* src1, const uint16_t* src2, int16_t* dst, int uv_w,
    int bit_depth, SharpYuvTransferFunctionType transfer_type) {
  int i = 0;
  if (transfer_type != kSharpYuvTransferFunctionSrgb) {
    SharpYuvUpdateChroma_C(src1, src2, dst, uv_w, bit_depth, transfer_type);
    return;
  }
  {
    const __m128 norm = _mm_set1_ps(1.0f / (1 << bit_depth));
    const __m128 scale = _mm_set1_ps((float)(1 << bit_depth));
    // RGBToGray() coefficients, normalized back from their 16 bit fixed-point.
    const __m128 coeff_r = _mm_set1_ps(13933.0f / 65536.0f);
    const __m128 coeff_g = _mm_set1_ps(46871.0f / 65536.0f);
    const __m128 coeff_b = _mm_set1_ps(4732.0f / 65536.0f);
    for (; i + 4 <= uv_w; i += 4) {
      __m128 v[3];
      __m128i vi[3], gray;
      int p;
      for (p = 0; p < 3; ++p) {
        // Average 2x2 blocks of gamma-space samples in linear space.
        const __m128 sum =
            _mm_add_ps(LoadLinearPairs_SSE2(src1 + 2 * p * uv_w + 2 * i, norm),
                       LoadLinearPairs_SSE2(src2 + 2 * p * uv_w + 2 * i, norm));
        const __m128 avg = _mm_mul_ps(sum, _mm_set1_ps(0.25f));
        vi[p] = _mm_cvtps_epi32(_mm_mul_ps(LinearToGamma_SSE2(avg), scale));
        v[p] = _mm_cvtepi32_ps(vi[p]);
      }
      // Like RGBToGray(), computed in float as SSE2 has no 32-bit integer
      // multiplication (the rounding very rarely differs by one).
      gray = _mm_cvtps_epi32(_mm_add_ps(
          _mm_add_ps(_mm_mul_ps(v[0], coeff_r), _mm_mul_ps(v[1], coeff_g)),
          _mm_mul_ps(v[2], coeff_b)));
      for (p = 0; p < 3; ++p) {
        const __m128i diff = _mm_sub_epi32(vi[p], gray);
        _mm_storel_epi64((__m128i*)(dst + p * uv_w + i),
                         _mm_packs_epi32(diff, diff));
      }
    }
  }
  for (; i < uv_w; ++i) {  // left-over
    int v[3], W, p;
    for (p = 0; p < 3; ++p) {
      const int off = 2 * p * uv_w + 2 * i;
      const uint32_t A =
          SharpYuvGammaToLinear(src1[off + 0], bit_depth, transfer_type);
      const uint32_t B =
          SharpYuvGammaToLinear(src1[off + 1], bit_depth, transfer_type);
      const uint32_t C =
          SharpYuvGammaToLinear(src2[off + 0], bit_depth, transfer_type);
      const uint32_t D =
          SharpYuvGammaToLinear(src2[off + 1], bit_depth, transfer_type);
      v[p] = SharpYuvLinearToGamma((A + B + C + D + 2) >> 2, bit_depth,
                                   transfer_type);
    }
    W = RGBToGray_SSE2(v[0], v[1], v[2]);
    for (p = 0; p < 3; ++p) {
      dst[p * uv_w + i] = (int16_t)(v[p] - W);
    }
  }
}

#undef SHARPYUV_GAMMA_THRESH
#undef SHARPYUV_GAMMA_A

//------------------------------------------------------------------------------

extern void InitSharpYuvSSE2(void);

WEBP_TSAN_IGNORE_FUNCTION void InitSharpYuvSSE2(void) {
  SharpYuvUpdateY = SharpYuvUpdateY_SSE2;
  SharpYuvUpdateRGB = SharpYuvUpdateRGB_SSE2;
  SharpYuvFilterRow = SharpYuvFilterRow_SSE2;
  SharpYuvUpdateW = SharpYuvUpdateW_SSE2;
  SharpYuvUpdateChroma = SharpYuvUpdateChroma_SSE2;
}
#else  // !WEBP_USE_SSE2

extern void InitSharpYuvSSE2(void);

void InitSharpYuvSSE2(void) {}

#endif  // WEBP_USE_SSE2
