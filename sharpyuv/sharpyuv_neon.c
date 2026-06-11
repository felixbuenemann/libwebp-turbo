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

#include "sharpyuv/sharpyuv_dsp.h"

#if defined(WEBP_USE_NEON)
#include <arm_neon.h>
#include <assert.h>
#include <stdlib.h>

static uint16_t clip_NEON(int v, int max) {
  return (v < 0) ? 0 : (v > max) ? max : (uint16_t)v;
}

static uint64_t SharpYuvUpdateY_NEON(const uint16_t* ref, const uint16_t* src,
                                     uint16_t* dst, int len, int bit_depth) {
  const int max_y = (1 << bit_depth) - 1;
  int i;
  const int16x8_t zero = vdupq_n_s16(0);
  const int16x8_t max = vdupq_n_s16(max_y);
  uint64x2_t sum = vdupq_n_u64(0);
  uint64_t diff;

  for (i = 0; i + 8 <= len; i += 8) {
    const int16x8_t A = vreinterpretq_s16_u16(vld1q_u16(ref + i));
    const int16x8_t B = vreinterpretq_s16_u16(vld1q_u16(src + i));
    const int16x8_t C = vreinterpretq_s16_u16(vld1q_u16(dst + i));
    const int16x8_t D = vsubq_s16(A, B);  // diff_y
    const int16x8_t F = vaddq_s16(C, D);  // new_y
    const uint16x8_t H =
        vreinterpretq_u16_s16(vmaxq_s16(vminq_s16(F, max), zero));
    const int16x8_t I = vabsq_s16(D);  // abs(diff_y)
    vst1q_u16(dst + i, H);
    sum = vpadalq_u32(sum, vpaddlq_u16(vreinterpretq_u16_s16(I)));
  }
  diff = vgetq_lane_u64(sum, 0) + vgetq_lane_u64(sum, 1);
  for (; i < len; ++i) {
    const int diff_y = ref[i] - src[i];
    const int new_y = (int)(dst[i]) + diff_y;
    dst[i] = clip_NEON(new_y, max_y);
    diff += (uint64_t)(abs(diff_y));
  }
  return diff;
}

static void SharpYuvUpdateRGB_NEON(const int16_t* ref, const int16_t* src,
                                   int16_t* dst, int len) {
  int i;
  for (i = 0; i + 8 <= len; i += 8) {
    const int16x8_t A = vld1q_s16(ref + i);
    const int16x8_t B = vld1q_s16(src + i);
    const int16x8_t C = vld1q_s16(dst + i);
    const int16x8_t D = vsubq_s16(A, B);  // diff_uv
    const int16x8_t E = vaddq_s16(C, D);  // new_uv
    vst1q_s16(dst + i, E);
  }
  for (; i < len; ++i) {
    const int diff_uv = ref[i] - src[i];
    dst[i] += diff_uv;
  }
}

static void SharpYuvFilterRow16_NEON(const int16_t* A, const int16_t* B,
                                     int len, const uint16_t* best_y,
                                     uint16_t* out, int bit_depth) {
  const int max_y = (1 << bit_depth) - 1;
  int i;
  const int16x8_t max = vdupq_n_s16(max_y);
  const int16x8_t zero = vdupq_n_s16(0);
  for (i = 0; i + 8 <= len; i += 8) {
    const int16x8_t a0 = vld1q_s16(A + i + 0);
    const int16x8_t a1 = vld1q_s16(A + i + 1);
    const int16x8_t b0 = vld1q_s16(B + i + 0);
    const int16x8_t b1 = vld1q_s16(B + i + 1);
    const int16x8_t a0b1 = vaddq_s16(a0, b1);
    const int16x8_t a1b0 = vaddq_s16(a1, b0);
    const int16x8_t a0a1b0b1 = vaddq_s16(a0b1, a1b0);  // A0+A1+B0+B1
    const int16x8_t a0b1_2 = vaddq_s16(a0b1, a0b1);    // 2*(A0+B1)
    const int16x8_t a1b0_2 = vaddq_s16(a1b0, a1b0);    // 2*(A1+B0)
    const int16x8_t c0 = vshrq_n_s16(vaddq_s16(a0b1_2, a0a1b0b1), 3);
    const int16x8_t c1 = vshrq_n_s16(vaddq_s16(a1b0_2, a0a1b0b1), 3);
    const int16x8_t e0 = vrhaddq_s16(c1, a0);
    const int16x8_t e1 = vrhaddq_s16(c0, a1);
    const int16x8x2_t f = vzipq_s16(e0, e1);
    const int16x8_t g0 = vreinterpretq_s16_u16(vld1q_u16(best_y + 2 * i + 0));
    const int16x8_t g1 = vreinterpretq_s16_u16(vld1q_u16(best_y + 2 * i + 8));
    const int16x8_t h0 = vaddq_s16(g0, f.val[0]);
    const int16x8_t h1 = vaddq_s16(g1, f.val[1]);
    const int16x8_t i0 = vmaxq_s16(vminq_s16(h0, max), zero);
    const int16x8_t i1 = vmaxq_s16(vminq_s16(h1, max), zero);
    vst1q_u16(out + 2 * i + 0, vreinterpretq_u16_s16(i0));
    vst1q_u16(out + 2 * i + 8, vreinterpretq_u16_s16(i1));
  }
  for (; i < len; ++i) {
    const int a0b1 = A[i + 0] + B[i + 1];
    const int a1b0 = A[i + 1] + B[i + 0];
    const int a0a1b0b1 = a0b1 + a1b0 + 8;
    const int v0 = (8 * A[i + 0] + 2 * a1b0 + a0a1b0b1) >> 4;
    const int v1 = (8 * A[i + 1] + 2 * a0b1 + a0a1b0b1) >> 4;
    out[2 * i + 0] = clip_NEON(best_y[2 * i + 0] + v0, max_y);
    out[2 * i + 1] = clip_NEON(best_y[2 * i + 1] + v1, max_y);
  }
}

static void SharpYuvFilterRow32_NEON(const int16_t* A, const int16_t* B,
                                     int len, const uint16_t* best_y,
                                     uint16_t* out, int bit_depth) {
  const int max_y = (1 << bit_depth) - 1;
  int i;
  const uint16x8_t max = vdupq_n_u16(max_y);
  for (i = 0; i + 4 <= len; i += 4) {
    const int16x4_t a0 = vld1_s16(A + i + 0);
    const int16x4_t a1 = vld1_s16(A + i + 1);
    const int16x4_t b0 = vld1_s16(B + i + 0);
    const int16x4_t b1 = vld1_s16(B + i + 1);
    const int32x4_t a0b1 = vaddl_s16(a0, b1);
    const int32x4_t a1b0 = vaddl_s16(a1, b0);
    const int32x4_t a0a1b0b1 = vaddq_s32(a0b1, a1b0);  // A0+A1+B0+B1
    const int32x4_t a0b1_2 = vaddq_s32(a0b1, a0b1);    // 2*(A0+B1)
    const int32x4_t a1b0_2 = vaddq_s32(a1b0, a1b0);    // 2*(A1+B0)
    const int32x4_t c0 = vshrq_n_s32(vaddq_s32(a0b1_2, a0a1b0b1), 3);
    const int32x4_t c1 = vshrq_n_s32(vaddq_s32(a1b0_2, a0a1b0b1), 3);
    const int32x4_t e0 = vrhaddq_s32(c1, vmovl_s16(a0));
    const int32x4_t e1 = vrhaddq_s32(c0, vmovl_s16(a1));
    const int32x4x2_t f = vzipq_s32(e0, e1);

    const int16x8_t g = vreinterpretq_s16_u16(vld1q_u16(best_y + 2 * i));
    const int32x4_t h0 = vaddw_s16(f.val[0], vget_low_s16(g));
    const int32x4_t h1 = vaddw_s16(f.val[1], vget_high_s16(g));
    const uint16x8_t i_16 = vcombine_u16(vqmovun_s32(h0), vqmovun_s32(h1));
    const uint16x8_t i_clamped = vminq_u16(i_16, max);
    vst1q_u16(out + 2 * i + 0, i_clamped);
  }
  for (; i < len; ++i) {
    const int a0b1 = A[i + 0] + B[i + 1];
    const int a1b0 = A[i + 1] + B[i + 0];
    const int a0a1b0b1 = a0b1 + a1b0 + 8;
    const int v0 = (8 * A[i + 0] + 2 * a1b0 + a0a1b0b1) >> 4;
    const int v1 = (8 * A[i + 1] + 2 * a0b1 + a0a1b0b1) >> 4;
    out[2 * i + 0] = clip_NEON(best_y[2 * i + 0] + v0, max_y);
    out[2 * i + 1] = clip_NEON(best_y[2 * i + 1] + v1, max_y);
  }
}

static void SharpYuvFilterRow_NEON(const int16_t* A, const int16_t* B, int len,
                                   const uint16_t* best_y, uint16_t* out,
                                   int bit_depth) {
  if (bit_depth <= 10) {
    SharpYuvFilterRow16_NEON(A, B, len, best_y, out, bit_depth);
  } else {
    SharpYuvFilterRow32_NEON(A, B, len, best_y, out, bit_depth);
  }
}

//------------------------------------------------------------------------------
// Gamma-space gray computation, going through linear space for averages. The
// sRGB transfer function and its inverse are evaluated with polynomial
// approximations whose error is below the quantization error of the
// table-based scalar versions in sharpyuv_gamma.c. Requires AArch64 for
// vsqrtq_f32/vcvtnq_u32_f32/vpaddq_f32.

#if WEBP_AARCH64

#include "sharpyuv/sharpyuv_gamma.h"

// thresh = 0.018053968510807, a = 0.09929682680944 (as in sharpyuv_gamma.c).
#define SHARPYUV_GAMMA_THRESH 0.018053968510807f
#define SHARPYUV_GAMMA_A 0.09929682680944f

// Degree-6 least-squares fit of ((g + a) / (1 + a)) ^ (1 / 0.45) on
// [4.5 * thresh, 1]. Max error 4.4e-6, below the 16-bit quantization of the
// table used by SharpYuvGammaToLinear().
static WEBP_INLINE float32x4_t GammaToLinear_NEON(const float32x4_t g) {
  const float32x4_t lo = vmulq_n_f32(g, 1.0f / 4.5f);
  float32x4_t p = vdupq_n_f32(-2.035152666e-02f);
  p = vfmaq_f32(vdupq_n_f32(9.086542551e-02f), p, g);
  p = vfmaq_f32(vdupq_n_f32(-1.905903016e-01f), p, g);
  p = vfmaq_f32(vdupq_n_f32(3.256689955e-01f), p, g);
  p = vfmaq_f32(vdupq_n_f32(6.849562000e-01f), p, g);
  p = vfmaq_f32(vdupq_n_f32(1.045739367e-01f), p, g);
  p = vfmaq_f32(vdupq_n_f32(4.874765193e-03f), p, g);
  return vbslq_f32(vcleq_f32(g, vdupq_n_f32(SHARPYUV_GAMMA_THRESH * 4.5f)), lo,
                   p);
}

// (1 + a) * l ^ 0.45 - a, with l ^ 0.45 evaluated as a degree-7 fit in
// u = sqrt(sqrt(l)) on [thresh, 1]. Max error 1e-7, about 500x below the
// quantization of the 512-entry interpolated table used by
// SharpYuvLinearToGamma().
static WEBP_INLINE float32x4_t LinearToGamma_NEON(const float32x4_t l) {
  const float32x4_t lo = vmulq_n_f32(l, 4.5f);
  const float32x4_t u = vsqrtq_f32(vsqrtq_f32(l));
  float32x4_t p = vdupq_n_f32(-1.622143889e-02f);
  p = vfmaq_f32(vdupq_n_f32(9.720616133e-02f), p, u);
  p = vfmaq_f32(vdupq_n_f32(-2.609642418e-01f), p, u);
  p = vfmaq_f32(vdupq_n_f32(4.250514527e-01f), p, u);
  p = vfmaq_f32(vdupq_n_f32(-5.155822037e-01f), p, u);
  p = vfmaq_f32(vdupq_n_f32(1.208584132e+00f), p, u);
  p = vfmaq_f32(vdupq_n_f32(6.450501113e-02f), p, u);
  p = vfmaq_f32(vdupq_n_f32(-2.578887858e-03f), p, u);
  p = vfmaq_f32(vdupq_n_f32(-SHARPYUV_GAMMA_A), p,
                vdupq_n_f32(1.0f + SHARPYUV_GAMMA_A));
  return vbslq_f32(vcleq_f32(l, vdupq_n_f32(SHARPYUV_GAMMA_THRESH)), lo, p);
}

static WEBP_INLINE float32x4_t LoadNormalized_NEON(const uint16_t* src,
                                                   const float32x4_t norm) {
  return vmulq_f32(vcvtq_f32_u32(vmovl_u16(vld1_u16(src))), norm);
}

static int RGBToGray_NEON(int64_t r, int64_t g, int64_t b) {
  const int64_t luma = 13933 * r + 46871 * g + 4732 * b + (1 << 15);
  return (int)(luma >> 16);
}

static void SharpYuvUpdateW_NEON(const uint16_t* src, uint16_t* dst, int w,
                                 int bit_depth,
                                 SharpYuvTransferFunctionType transfer_type) {
  int i = 0;
  if (transfer_type != kSharpYuvTransferFunctionSrgb) {
    SharpYuvUpdateW_C(src, dst, w, bit_depth, transfer_type);
    return;
  }
  {
    const float32x4_t norm = vdupq_n_f32(1.0f / (1 << bit_depth));
    const float32x4_t scale = vdupq_n_f32((float)(1 << bit_depth));
    // RGBToGray() coefficients, normalized back from their 16 bit fixed-point.
    const float32x4_t coeff_g = vdupq_n_f32(46871.0f / 65536.0f);
    const float32x4_t coeff_b = vdupq_n_f32(4732.0f / 65536.0f);
    for (; i + 4 <= w; i += 4) {
      const float32x4_t r =
          GammaToLinear_NEON(LoadNormalized_NEON(src + 0 * w + i, norm));
      const float32x4_t g =
          GammaToLinear_NEON(LoadNormalized_NEON(src + 1 * w + i, norm));
      const float32x4_t b =
          GammaToLinear_NEON(LoadNormalized_NEON(src + 2 * w + i, norm));
      const float32x4_t gray = vfmaq_f32(
          vfmaq_f32(vmulq_n_f32(r, 13933.0f / 65536.0f), g, coeff_g), b,
          coeff_b);
      const float32x4_t y = vmulq_f32(LinearToGamma_NEON(gray), scale);
      vst1_u16(dst + i, vmovn_u32(vcvtnq_u32_f32(y)));
    }
  }
  for (; i < w; ++i) {  // left-over
    const uint32_t R =
        SharpYuvGammaToLinear(src[0 * w + i], bit_depth, transfer_type);
    const uint32_t G =
        SharpYuvGammaToLinear(src[1 * w + i], bit_depth, transfer_type);
    const uint32_t B =
        SharpYuvGammaToLinear(src[2 * w + i], bit_depth, transfer_type);
    const uint32_t Y = RGBToGray_NEON(R, G, B);
    dst[i] = (uint16_t)SharpYuvLinearToGamma(Y, bit_depth, transfer_type);
  }
}

// Loads 8 gamma-space samples, converts them to linear space and adds them
// pairwise.
static WEBP_INLINE float32x4_t LoadLinearPairs_NEON(const uint16_t* src,
                                                    const float32x4_t norm) {
  const uint16x8_t s = vld1q_u16(src);
  const float32x4_t lo = GammaToLinear_NEON(
      vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(s))), norm));
  const float32x4_t hi = GammaToLinear_NEON(
      vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(s))), norm));
  return vpaddq_f32(lo, hi);
}

static void SharpYuvUpdateChroma_NEON(
    const uint16_t* src1, const uint16_t* src2, int16_t* dst, int uv_w,
    int bit_depth, SharpYuvTransferFunctionType transfer_type) {
  int i = 0;
  if (transfer_type != kSharpYuvTransferFunctionSrgb) {
    SharpYuvUpdateChroma_C(src1, src2, dst, uv_w, bit_depth, transfer_type);
    return;
  }
  {
    const float32x4_t norm = vdupq_n_f32(1.0f / (1 << bit_depth));
    const float32x4_t scale = vdupq_n_f32((float)(1 << bit_depth));
    for (; i + 4 <= uv_w; i += 4) {
      int32x4_t v[3];
      int32x4_t gray;
      int p;
      for (p = 0; p < 3; ++p) {
        // Average 2x2 blocks of gamma-space samples in linear space.
        const float32x4_t sum =
            vaddq_f32(LoadLinearPairs_NEON(src1 + 2 * p * uv_w + 2 * i, norm),
                      LoadLinearPairs_NEON(src2 + 2 * p * uv_w + 2 * i, norm));
        const float32x4_t avg = vmulq_n_f32(sum, 0.25f);
        v[p] = vcvtnq_s32_f32(vmulq_f32(LinearToGamma_NEON(avg), scale));
      }
      // Same as RGBToGray(): exact, as the products fit in int32 (the samples
      // have at most 14 bits).
      gray = vmulq_s32(v[0], vdupq_n_s32(13933));
      gray = vmlaq_s32(gray, v[1], vdupq_n_s32(46871));
      gray = vmlaq_s32(gray, v[2], vdupq_n_s32(4732));
      gray = vshrq_n_s32(vaddq_s32(gray, vdupq_n_s32(1 << 15)), 16);
      for (p = 0; p < 3; ++p) {
        vst1_s16(dst + p * uv_w + i, vmovn_s32(vsubq_s32(v[p], gray)));
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
    W = RGBToGray_NEON(v[0], v[1], v[2]);
    for (p = 0; p < 3; ++p) {
      dst[p * uv_w + i] = (int16_t)(v[p] - W);
    }
  }
}

#undef SHARPYUV_GAMMA_THRESH
#undef SHARPYUV_GAMMA_A

#endif  // WEBP_AARCH64

//------------------------------------------------------------------------------

extern void InitSharpYuvNEON(void);

WEBP_TSAN_IGNORE_FUNCTION void InitSharpYuvNEON(void) {
  SharpYuvUpdateY = SharpYuvUpdateY_NEON;
  SharpYuvUpdateRGB = SharpYuvUpdateRGB_NEON;
  SharpYuvFilterRow = SharpYuvFilterRow_NEON;
#if WEBP_AARCH64
  SharpYuvUpdateW = SharpYuvUpdateW_NEON;
  SharpYuvUpdateChroma = SharpYuvUpdateChroma_NEON;
#endif
}

#else  // !WEBP_USE_NEON

extern void InitSharpYuvNEON(void);

void InitSharpYuvNEON(void) {}

#endif  // WEBP_USE_NEON
