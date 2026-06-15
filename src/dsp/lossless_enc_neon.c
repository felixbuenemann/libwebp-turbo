// Copyright 2015 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// NEON variant of methods for lossless encoder
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_NEON)

#include <arm_neon.h>

#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"
#include "src/dsp/neon.h"
#include "src/utils/utils.h"
#include "src/webp/format_constants.h"

//------------------------------------------------------------------------------
// Subtract-Green Transform

// vtbl?_u8 are marked unavailable for iOS arm64 with Xcode < 6.3, use
// non-standard versions there.
#if defined(__APPLE__) && WEBP_AARCH64 && defined(__apple_build_version__) && \
    (__apple_build_version__ < 6020037)
#define USE_VTBLQ
#endif

#ifdef USE_VTBLQ
// 255 = byte will be zeroed
static const uint8_t kGreenShuffle[16] = {1, 255, 1, 255, 5,  255, 5,  255,
                                          9, 255, 9, 255, 13, 255, 13, 255};

static WEBP_INLINE uint8x16_t DoGreenShuffle_NEON(const uint8x16_t argb,
                                                  const uint8x16_t shuffle) {
  return vcombine_u8(vtbl1q_u8(argb, vget_low_u8(shuffle)),
                     vtbl1q_u8(argb, vget_high_u8(shuffle)));
}
#else   // !USE_VTBLQ
// 255 = byte will be zeroed
static const uint8_t kGreenShuffle[8] = {1, 255, 1, 255, 5, 255, 5, 255};

static WEBP_INLINE uint8x16_t DoGreenShuffle_NEON(const uint8x16_t argb,
                                                  const uint8x8_t shuffle) {
  return vcombine_u8(vtbl1_u8(vget_low_u8(argb), shuffle),
                     vtbl1_u8(vget_high_u8(argb), shuffle));
}
#endif  // USE_VTBLQ

static void SubtractGreenFromBlueAndRed_NEON(uint32_t* argb_data,
                                             int num_pixels) {
  const uint32_t* const end = argb_data + (num_pixels & ~3);
#ifdef USE_VTBLQ
  const uint8x16_t shuffle = vld1q_u8(kGreenShuffle);
#else
  const uint8x8_t shuffle = vld1_u8(kGreenShuffle);
#endif
  for (; argb_data < end; argb_data += 4) {
    const uint8x16_t argb = vld1q_u8((uint8_t*)argb_data);
    const uint8x16_t greens = DoGreenShuffle_NEON(argb, shuffle);
    vst1q_u8((uint8_t*)argb_data, vsubq_u8(argb, greens));
  }
  // fallthrough and finish off with plain-C
  VP8LSubtractGreenFromBlueAndRed_C(argb_data, num_pixels & 3);
}

//------------------------------------------------------------------------------
// Color Transform

static void TransformColor_NEON(const VP8LMultipliers* WEBP_RESTRICT const m,
                                uint32_t* WEBP_RESTRICT argb_data,
                                int num_pixels) {
  // sign-extended multiplying constants, pre-shifted by 6.
#define CST(X) (((int16_t)(m->X << 8)) >> 6)
  const int16_t rb[8] = {CST(green_to_blue), CST(green_to_red),
                         CST(green_to_blue), CST(green_to_red),
                         CST(green_to_blue), CST(green_to_red),
                         CST(green_to_blue), CST(green_to_red)};
  const int16x8_t mults_rb = vld1q_s16(rb);
  const int16_t b2[8] = {
      0, CST(red_to_blue), 0, CST(red_to_blue),
      0, CST(red_to_blue), 0, CST(red_to_blue),
  };
  const int16x8_t mults_b2 = vld1q_s16(b2);
#undef CST
#ifdef USE_VTBLQ
  static const uint8_t kg0g0[16] = {255, 1, 255, 1, 255, 5,  255, 5,
                                    255, 9, 255, 9, 255, 13, 255, 13};
  const uint8x16_t shuffle = vld1q_u8(kg0g0);
#else
  static const uint8_t k0g0g[8] = {255, 1, 255, 1, 255, 5, 255, 5};
  const uint8x8_t shuffle = vld1_u8(k0g0g);
#endif
  const uint32x4_t mask_rb = vdupq_n_u32(0x00ff00ffu);  // red-blue masks
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const uint8x16_t in = vld1q_u8((uint8_t*)(argb_data + i));
    // 0 g 0 g
    const uint8x16_t greens = DoGreenShuffle_NEON(in, shuffle);
    // x dr  x db1
    const int16x8_t A = vqdmulhq_s16(vreinterpretq_s16_u8(greens), mults_rb);
    // r 0   b   0
    const int16x8_t B = vshlq_n_s16(vreinterpretq_s16_u8(in), 8);
    // x db2 0   0
    const int16x8_t C = vqdmulhq_s16(B, mults_b2);
    // 0 0   x db2
    const uint32x4_t D = vshrq_n_u32(vreinterpretq_u32_s16(C), 16);
    // x dr  x  db
    const int8x16_t E =
        vaddq_s8(vreinterpretq_s8_u32(D), vreinterpretq_s8_s16(A));
    // 0 dr  0  db
    const uint32x4_t F = vandq_u32(vreinterpretq_u32_s8(E), mask_rb);
    const int8x16_t out =
        vsubq_s8(vreinterpretq_s8_u8(in), vreinterpretq_s8_u32(F));
    vst1q_s8((int8_t*)(argb_data + i), out);
  }
  // fallthrough and finish off with plain-C
  VP8LTransformColor_C(m, argb_data + i, num_pixels - i);
}

#undef USE_VTBLQ

//------------------------------------------------------------------------------
// Color-space conversion: collecting histograms

// sign-extended multiplying constant, pre-shifted by 6 so that the doubling
// multiply (vqdmulhq_s16) computes ColorTransformDelta, i.e. (mult * c) >> 5.
#define CST_5b(X) (((int16_t)((X) << 8)) >> 6)
#define SPAN 8

// Adds the 8 values (one per byte of 'v') to 'histo'. Batches of identical
// values are added in one go; this is faster on uniform areas and, more
// importantly, it breaks the load-add-store dependency chain on a single
// histogram entry.
static WEBP_INLINE void AccumulateHisto_NEON(uint64_t v, uint32_t histo[]) {
  if (v == ((v >> 8) | (v << 56))) {  // all bytes equal
    histo[v & 0xff] += SPAN;
  } else {
    int k;
    for (k = 0; k < SPAN; ++k) ++histo[(v >> (8 * k)) & 0xff];
  }
}

static void CollectColorBlueTransforms_NEON(const uint32_t* WEBP_RESTRICT argb,
                                            int stride, int tile_width,
                                            int tile_height, int green_to_blue,
                                            int red_to_blue, uint32_t histo[]) {
  const int16x8_t mults_g = vdupq_n_s16(CST_5b(green_to_blue));
  const int16x8_t mults_r = vdupq_n_s16(CST_5b(red_to_blue));
  const uint32x4_t mask_g = vdupq_n_u32(0x0000ff00u);  // green mask
  int y;
  for (y = 0; y < tile_height; ++y) {
    const uint32_t* const src = argb + y * stride;
    int x;
    for (x = 0; x + SPAN <= tile_width; x += SPAN) {
      const uint32x4_t in0 = vld1q_u32(src + x + 0);
      const uint32x4_t in1 = vld1q_u32(src + x + SPAN / 2);
      // 0 g 0 0 (green at the top of the low 16-bit lane)
      const int16x8_t g0 = vreinterpretq_s16_u32(vandq_u32(in0, mask_g));
      const int16x8_t g1 = vreinterpretq_s16_u32(vandq_u32(in1, mask_g));
      // x x | x dbg  ((green * green_to_blue) >> 5 in the low 16-bit lane)
      const int16x8_t A0 = vqdmulhq_s16(g0, mults_g);
      const int16x8_t A1 = vqdmulhq_s16(g1, mults_g);
      // r 0 | b 0
      const int16x8_t B0 = vshlq_n_s16(vreinterpretq_s16_u32(in0), 8);
      const int16x8_t B1 = vshlq_n_s16(vreinterpretq_s16_u32(in1), 8);
      // x dbr | x x  ((red * red_to_blue) >> 5 in the high 16-bit lane)
      const int16x8_t C0 = vqdmulhq_s16(B0, mults_r);
      const int16x8_t C1 = vqdmulhq_s16(B1, mults_r);
      // 0 0 | x dbr
      const uint32x4_t D0 = vshrq_n_u32(vreinterpretq_u32_s16(C0), 16);
      const uint32x4_t D1 = vshrq_n_u32(vreinterpretq_u32_s16(C1), 16);
      // x x | x db  (total delta in the low byte)
      const uint8x16_t E0 =
          vaddq_u8(vreinterpretq_u8_s16(A0), vreinterpretq_u8_u32(D0));
      const uint8x16_t E1 =
          vaddq_u8(vreinterpretq_u8_s16(A1), vreinterpretq_u8_u32(D1));
      // x x | x b'  (new blue in the low byte)
      const uint32x4_t F0 =
          vreinterpretq_u32_u8(vsubq_u8(vreinterpretq_u8_u32(in0), E0));
      const uint32x4_t F1 =
          vreinterpretq_u32_u8(vsubq_u8(vreinterpretq_u8_u32(in1), E1));
      const uint8x8_t b8 =
          vmovn_u16(vcombine_u16(vmovn_u32(F0), vmovn_u32(F1)));
      AccumulateHisto_NEON(vget_lane_u64(vreinterpret_u64_u8(b8), 0), histo);
    }
  }
  {
    const int left_over = tile_width & (SPAN - 1);
    if (left_over > 0) {
      VP8LCollectColorBlueTransforms_C(argb + tile_width - left_over, stride,
                                       left_over, tile_height, green_to_blue,
                                       red_to_blue, histo);
    }
  }
}

static void CollectColorRedTransforms_NEON(const uint32_t* WEBP_RESTRICT argb,
                                           int stride, int tile_width,
                                           int tile_height, int green_to_red,
                                           uint32_t histo[]) {
  const int16x8_t mults_g = vdupq_n_s16(CST_5b(green_to_red));
  const uint32x4_t mask_g = vdupq_n_u32(0xff000000u);
  int y;
  for (y = 0; y < tile_height; ++y) {
    const uint32_t* const src = argb + y * stride;
    int x;
    for (x = 0; x + SPAN <= tile_width; x += SPAN) {
      const uint32x4_t in0 = vld1q_u32(src + x + 0);
      const uint32x4_t in1 = vld1q_u32(src + x + SPAN / 2);
      // g 0 | 0 0  (green at the top of the high 16-bit lane)
      const int16x8_t g0 = vreinterpretq_s16_u32(
          vandq_u32(vshlq_n_u32(in0, 16), mask_g));
      const int16x8_t g1 = vreinterpretq_s16_u32(
          vandq_u32(vshlq_n_u32(in1, 16), mask_g));
      // x drg | x x  ((green * green_to_red) >> 5 in the high 16-bit lane)
      const int16x8_t A0 = vqdmulhq_s16(g0, mults_g);
      const int16x8_t A1 = vqdmulhq_s16(g1, mults_g);
      // x r' | x x  (new red in byte 2)
      const uint32x4_t E0 = vreinterpretq_u32_u8(
          vsubq_u8(vreinterpretq_u8_u32(in0), vreinterpretq_u8_s16(A0)));
      const uint32x4_t E1 = vreinterpretq_u32_u8(
          vsubq_u8(vreinterpretq_u8_u32(in1), vreinterpretq_u8_s16(A1)));
      const uint8x8_t r8 = vmovn_u16(vcombine_u16(
          vmovn_u32(vshrq_n_u32(E0, 16)), vmovn_u32(vshrq_n_u32(E1, 16))));
      AccumulateHisto_NEON(vget_lane_u64(vreinterpret_u64_u8(r8), 0), histo);
    }
  }
  {
    const int left_over = tile_width & (SPAN - 1);
    if (left_over > 0) {
      VP8LCollectColorRedTransforms_C(argb + tile_width - left_over, stride,
                                      left_over, tile_height, green_to_red,
                                      histo);
    }
  }
}

static void CollectArgbHistos_NEON(const uint32_t* WEBP_RESTRICT argb,
                                   int num_pixels, uint32_t histo[4 * 256]) {
  int i;
  for (i = 0; i + SPAN <= num_pixels; i += SPAN) {
    // De-interleave the channels of 8 pixels: val[0] = blue, ..., val[3] =
    // alpha (little-endian).
    const uint8x8x4_t v = vld4_u8((const uint8_t*)(argb + i));
    AccumulateHisto_NEON(vget_lane_u64(vreinterpret_u64_u8(v.val[3]), 0),
                         histo + 0 * 256);
    AccumulateHisto_NEON(vget_lane_u64(vreinterpret_u64_u8(v.val[2]), 0),
                         histo + 1 * 256);
    AccumulateHisto_NEON(vget_lane_u64(vreinterpret_u64_u8(v.val[1]), 0),
                         histo + 2 * 256);
    AccumulateHisto_NEON(vget_lane_u64(vreinterpret_u64_u8(v.val[0]), 0),
                         histo + 3 * 256);
  }
  for (; i < num_pixels; ++i) {
    const uint32_t pix = argb[i];
    ++histo[0 * 256 + (pix >> 24)];
    ++histo[1 * 256 + ((pix >> 16) & 0xff)];
    ++histo[2 * 256 + ((pix >> 8) & 0xff)];
    ++histo[3 * 256 + (pix & 0xff)];
  }
}

#undef SPAN
#undef CST_5b

//------------------------------------------------------------------------------
// Predictor Transform

// Average of two packed pixels, per byte: floor((a + b) / 2).
static WEBP_INLINE uint8x16_t Average2_NEON(const uint8x16_t a,
                                            const uint8x16_t b) {
  return vhaddq_u8(a, b);
}

// Predictor0: ARGB_BLACK.
static void PredictorSub0_NEON(const uint32_t* in, const uint32_t* upper,
                               int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  const uint8x16_t black = vreinterpretq_u8_u32(vdupq_n_u32(ARGB_BLACK));
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const uint8x16_t src = vreinterpretq_u8_u32(vld1q_u32(&in[i]));
    vst1q_u32(&out[i], vreinterpretq_u32_u8(vsubq_u8(src, black)));
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_C[0](in + i, NULL, num_pixels - i, out + i);
  }
  (void)upper;
}

#define GENERATE_PREDICTOR_1(X, IN)                                          \
  static void PredictorSub##X##_NEON(                                        \
      const uint32_t* const in, const uint32_t* const upper, int num_pixels, \
      uint32_t* WEBP_RESTRICT const out) {                                   \
    int i;                                                                   \
    for (i = 0; i + 4 <= num_pixels; i += 4) {                               \
      const uint8x16_t src = vreinterpretq_u8_u32(vld1q_u32(&in[i]));        \
      const uint8x16_t pred = vreinterpretq_u8_u32(vld1q_u32(&(IN)));        \
      vst1q_u32(&out[i], vreinterpretq_u32_u8(vsubq_u8(src, pred)));         \
    }                                                                        \
    if (i != num_pixels) {                                                   \
      VP8LPredictorsSub_C[(X)](in + i, WEBP_OFFSET_PTR(upper, i),            \
                               num_pixels - i, out + i);                     \
    }                                                                        \
  }

GENERATE_PREDICTOR_1(1, in[i - 1])     // Predictor1: L
GENERATE_PREDICTOR_1(2, upper[i])      // Predictor2: T
GENERATE_PREDICTOR_1(3, upper[i + 1])  // Predictor3: TR
GENERATE_PREDICTOR_1(4, upper[i - 1])  // Predictor4: TL
#undef GENERATE_PREDICTOR_1

// Predictor5: avg2(avg2(L, TR), T)
static void PredictorSub5_NEON(const uint32_t* in, const uint32_t* upper,
                               int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const uint8x16_t L = vreinterpretq_u8_u32(vld1q_u32(&in[i - 1]));
    const uint8x16_t T = vreinterpretq_u8_u32(vld1q_u32(&upper[i]));
    const uint8x16_t TR = vreinterpretq_u8_u32(vld1q_u32(&upper[i + 1]));
    const uint8x16_t src = vreinterpretq_u8_u32(vld1q_u32(&in[i]));
    const uint8x16_t pred = Average2_NEON(Average2_NEON(L, TR), T);
    vst1q_u32(&out[i], vreinterpretq_u32_u8(vsubq_u8(src, pred)));
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_C[5](in + i, upper + i, num_pixels - i, out + i);
  }
}

#define GENERATE_PREDICTOR_2(X, A, B)                                       \
  static void PredictorSub##X##_NEON(const uint32_t* in,                    \
                                     const uint32_t* upper, int num_pixels, \
                                     uint32_t* WEBP_RESTRICT out) {         \
    int i;                                                                  \
    for (i = 0; i + 4 <= num_pixels; i += 4) {                              \
      const uint8x16_t tA = vreinterpretq_u8_u32(vld1q_u32(&(A)));          \
      const uint8x16_t tB = vreinterpretq_u8_u32(vld1q_u32(&(B)));          \
      const uint8x16_t src = vreinterpretq_u8_u32(vld1q_u32(&in[i]));       \
      const uint8x16_t pred = Average2_NEON(tA, tB);                        \
      vst1q_u32(&out[i], vreinterpretq_u32_u8(vsubq_u8(src, pred)));        \
    }                                                                       \
    if (i != num_pixels) {                                                  \
      VP8LPredictorsSub_C[(X)](in + i, upper + i, num_pixels - i, out + i); \
    }                                                                       \
  }

GENERATE_PREDICTOR_2(6, in[i - 1], upper[i - 1])  // Predictor6: avg(L, TL)
GENERATE_PREDICTOR_2(7, in[i - 1], upper[i])      // Predictor7: avg(L, T)
GENERATE_PREDICTOR_2(8, upper[i - 1], upper[i])   // Predictor8: avg(TL, T)
GENERATE_PREDICTOR_2(9, upper[i], upper[i + 1])   // Predictor9: average(T, TR)
#undef GENERATE_PREDICTOR_2

// Predictor10: avg(avg(L, TL), avg(T, TR)).
static void PredictorSub10_NEON(const uint32_t* in, const uint32_t* upper,
                                int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const uint8x16_t L = vreinterpretq_u8_u32(vld1q_u32(&in[i - 1]));
    const uint8x16_t TL = vreinterpretq_u8_u32(vld1q_u32(&upper[i - 1]));
    const uint8x16_t T = vreinterpretq_u8_u32(vld1q_u32(&upper[i]));
    const uint8x16_t TR = vreinterpretq_u8_u32(vld1q_u32(&upper[i + 1]));
    const uint8x16_t src = vreinterpretq_u8_u32(vld1q_u32(&in[i]));
    const uint8x16_t pred =
        Average2_NEON(Average2_NEON(L, TL), Average2_NEON(T, TR));
    vst1q_u32(&out[i], vreinterpretq_u32_u8(vsubq_u8(src, pred)));
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_C[10](in + i, upper + i, num_pixels - i, out + i);
  }
}

// Predictor11: select.
static void PredictorSub11_NEON(const uint32_t* in, const uint32_t* upper,
                                int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const uint8x16_t L = vreinterpretq_u8_u32(vld1q_u32(&in[i - 1]));
    const uint8x16_t TL = vreinterpretq_u8_u32(vld1q_u32(&upper[i - 1]));
    const uint8x16_t T = vreinterpretq_u8_u32(vld1q_u32(&upper[i]));
    const uint8x16_t src = vreinterpretq_u8_u32(vld1q_u32(&in[i]));
    // Per-pixel sums of absolute byte differences.
    const uint32x4_t pa = vpaddlq_u16(vpaddlq_u8(vabdq_u8(T, TL)));
    const uint32x4_t pb = vpaddlq_u16(vpaddlq_u8(vabdq_u8(L, TL)));
    const uint32x4_t mask = vcgtq_u32(pb, pa);  // pb > pa ? L : T
    const uint8x16_t pred = vreinterpretq_u8_u32(vbslq_u32(
        mask, vreinterpretq_u32_u8(L), vreinterpretq_u32_u8(T)));
    vst1q_u32(&out[i], vreinterpretq_u32_u8(vsubq_u8(src, pred)));
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_C[11](in + i, upper + i, num_pixels - i, out + i);
  }
}

// Predictor12: ClampedAddSubtractFull.
static void PredictorSub12_NEON(const uint32_t* in, const uint32_t* upper,
                                int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const uint8x16_t L = vreinterpretq_u8_u32(vld1q_u32(&in[i - 1]));
    const uint8x16_t TL = vreinterpretq_u8_u32(vld1q_u32(&upper[i - 1]));
    const uint8x16_t T = vreinterpretq_u8_u32(vld1q_u32(&upper[i]));
    const uint8x16_t src = vreinterpretq_u8_u32(vld1q_u32(&in[i]));
    // pred = clamp_to_u8(L + T - TL), computed in 16-bit.
    const int16x8_t L_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(L)));
    const int16x8_t L_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(L)));
    const int16x8_t T_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(T)));
    const int16x8_t T_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(T)));
    const int16x8_t TL_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(TL)));
    const int16x8_t TL_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(TL)));
    const int16x8_t pred_lo = vaddq_s16(L_lo, vsubq_s16(T_lo, TL_lo));
    const int16x8_t pred_hi = vaddq_s16(L_hi, vsubq_s16(T_hi, TL_hi));
    const uint8x16_t pred =
        vcombine_u8(vqmovun_s16(pred_lo), vqmovun_s16(pred_hi));
    vst1q_u32(&out[i], vreinterpretq_u32_u8(vsubq_u8(src, pred)));
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_C[12](in + i, upper + i, num_pixels - i, out + i);
  }
}

// Predictor13: ClampedAddSubtractHalf.
static WEBP_INLINE int16x8_t ClampedAddSubtractHalf_NEON(const int16x8_t avg,
                                                         const int16x8_t TL) {
  // pred = clamp_to_u8(avg + (avg - TL) / 2), with the same rounding towards
  // zero as the C code: (avg - TL + (TL > avg)) >> 1.
  const int16x8_t A1 = vsubq_s16(avg, TL);
  const int16x8_t bit_fix = vreinterpretq_s16_u16(vcgtq_s16(TL, avg));
  const int16x8_t A2 = vsubq_s16(A1, bit_fix);
  const int16x8_t A3 = vshrq_n_s16(A2, 1);
  return vaddq_s16(avg, A3);
}

static void PredictorSub13_NEON(const uint32_t* in, const uint32_t* upper,
                                int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const uint8x16_t L = vreinterpretq_u8_u32(vld1q_u32(&in[i - 1]));
    const uint8x16_t TL = vreinterpretq_u8_u32(vld1q_u32(&upper[i - 1]));
    const uint8x16_t T = vreinterpretq_u8_u32(vld1q_u32(&upper[i]));
    const uint8x16_t src = vreinterpretq_u8_u32(vld1q_u32(&in[i]));
    const uint16x8_t sum_lo = vaddl_u8(vget_low_u8(L), vget_low_u8(T));
    const uint16x8_t sum_hi = vaddl_u8(vget_high_u8(L), vget_high_u8(T));
    const int16x8_t avg_lo = vreinterpretq_s16_u16(vshrq_n_u16(sum_lo, 1));
    const int16x8_t avg_hi = vreinterpretq_s16_u16(vshrq_n_u16(sum_hi, 1));
    const int16x8_t TL_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(TL)));
    const int16x8_t TL_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(TL)));
    const int16x8_t A4_lo = ClampedAddSubtractHalf_NEON(avg_lo, TL_lo);
    const int16x8_t A4_hi = ClampedAddSubtractHalf_NEON(avg_hi, TL_hi);
    const uint8x16_t pred = vcombine_u8(vqmovun_s16(A4_lo), vqmovun_s16(A4_hi));
    vst1q_u32(&out[i], vreinterpretq_u32_u8(vsubq_u8(src, pred)));
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_C[13](in + i, upper + i, num_pixels - i, out + i);
  }
}

//------------------------------------------------------------------------------
// Matching length (LZ77 / hash-chain)

// Returns the number of leading uint32_t elements that compare equal between
// array1 and array2 (capped at |length|). Bit-identical to the scalar C
// reference (it computes the same match length).
//
// NEON has no movemask, so the per-iteration NEON->GPR "all lanes equal" test
// is the costly part. Two design choices keep this a win at every match length
// on both wide (Apple) and narrow (Graviton N1) cores:
//   1. amortize the cross-domain extract over 16 lanes in the bulk loop;
//   2. resolve the common SHORT match with a lean scalar head, and keep the
//      NEON bulk in a separate noinline function so its code never bloats the
//      hot short-match path. Short matches then run at scalar speed; only an
//      established long match (HEAD elems equal) pays into the vector path.
// Measured (perf): never slower than scalar for any match length on either uarch.

// All-ones test of a uint32x4_t equality mask: AND the two 64-bit halves and
// extract once (one cross-domain move).
static WEBP_INLINE int AllEqual_NEON(const uint32x4_t cmp) {
  const uint64x2_t c64 = vreinterpretq_u64_u32(cmp);
  const uint64x1_t r = vand_u64(vget_low_u64(c64), vget_high_u64(c64));
  return vget_lane_u64(r, 0) == ~(uint64_t)0;
}

#define VECTOR_MISMATCH_HEAD 16  // scalar-head length; covers N1's NEON-loses region

// Cold path: NEON bulk for an established long match (match_len already matched).
// noinline so it does not bloat the hot short-match path in VectorMismatch_NEON.
#if defined(__GNUC__)
__attribute__((noinline))
#endif
static int VectorMismatchBulk_NEON(const uint32_t* const array1,
                                   const uint32_t* const array2, int length,
                                   int match_len) {
  while (match_len + 16 <= length) {
    const uint32x4_t c0 = vceqq_u32(vld1q_u32(array1 + match_len),
                                    vld1q_u32(array2 + match_len));
    const uint32x4_t c1 = vceqq_u32(vld1q_u32(array1 + match_len + 4),
                                    vld1q_u32(array2 + match_len + 4));
    const uint32x4_t c2 = vceqq_u32(vld1q_u32(array1 + match_len + 8),
                                    vld1q_u32(array2 + match_len + 8));
    const uint32x4_t c3 = vceqq_u32(vld1q_u32(array1 + match_len + 12),
                                    vld1q_u32(array2 + match_len + 12));
    if (!AllEqual_NEON(vandq_u32(vandq_u32(c0, c1), vandq_u32(c2, c3)))) break;
    match_len += 16;
  }
  while (match_len + 4 <= length) {
    const uint32x4_t c = vceqq_u32(vld1q_u32(array1 + match_len),
                                   vld1q_u32(array2 + match_len));
    if (!AllEqual_NEON(c)) break;
    match_len += 4;
  }
  while (match_len < length && array1[match_len] == array2[match_len]) {
    ++match_len;
  }
  return match_len;
}

// Hot path: lean scalar head; short matches resolve here at scalar speed.
static int VectorMismatch_NEON(const uint32_t* const array1,
                               const uint32_t* const array2, int length) {
  int match_len = 0;
  const int head = length < VECTOR_MISMATCH_HEAD ? length : VECTOR_MISMATCH_HEAD;
  while (match_len < head && array1[match_len] == array2[match_len]) ++match_len;
  if (match_len == VECTOR_MISMATCH_HEAD) {
    return VectorMismatchBulk_NEON(array1, array2, length, match_len);
  }
  return match_len;
}

//------------------------------------------------------------------------------
// Combined Shannon entropy (sparse-histogram cost)

// Returns non-zero iff any of the 16 bins (8 lanes across two vectors per side)
// is set, i.e. the 16-bin block has at least one non-zero X or Y entry.
static WEBP_INLINE uint32_t BlockNonZero_NEON(uint32x4_t a, uint32x4_t b,
                                              uint32x4_t c, uint32x4_t d) {
  const uint32x4_t any = vorrq_u32(vorrq_u32(a, b), vorrq_u32(c, d));
  const uint32x2_t r = vorr_u32(vget_low_u32(any), vget_high_u32(any));
  return vget_lane_u32(r, 0) | vget_lane_u32(r, 1);
}

// Same result as CombinedShannonEntropy_C. Histograms are sparse, so cheaply
// skip whole 16-bin blocks that are entirely zero (one OR-reduce, no per-bin
// work); for non-empty blocks fall back to the exact scalar inner loop. This
// avoids the set-bit-iteration cost that makes a movemask approach lose on the
// dense histograms where there is nothing to skip (NEON has no cheap movemask),
// so it is never slower than scalar and faster on the sparse common case.
static uint64_t CombinedShannonEntropy_NEON(const uint32_t X[256],
                                            const uint32_t Y[256]) {
  int i;
  uint64_t retval = 0;
  uint32_t sumX = 0, sumXY = 0;
  for (i = 0; i < 256; i += 16) {
    const uint32x4_t x0 = vld1q_u32(X + i + 0), x1 = vld1q_u32(X + i + 4);
    const uint32x4_t x2 = vld1q_u32(X + i + 8), x3 = vld1q_u32(X + i + 12);
    if (!BlockNonZero_NEON(x0, x1, x2, x3) &&
        !BlockNonZero_NEON(vld1q_u32(Y + i + 0), vld1q_u32(Y + i + 4),
                           vld1q_u32(Y + i + 8), vld1q_u32(Y + i + 12))) {
      continue;  // all 16 X and Y bins zero -> nothing to add
    }
    {
      int j;
      for (j = i; j < i + 16; ++j) {
        const uint32_t x = X[j];
        if (x != 0) {
          const uint32_t xy = x + Y[j];
          sumX += x;
          retval += VP8LFastSLog2(x);
          sumXY += xy;
          retval += VP8LFastSLog2(xy);
        } else if (Y[j] != 0) {
          sumXY += Y[j];
          retval += VP8LFastSLog2(Y[j]);
        }
      }
    }
  }
  retval = VP8LFastSLog2(sumX) + VP8LFastSLog2(sumXY) - retval;
  return retval;
}

//------------------------------------------------------------------------------
// Combined entropy of two streams (run-length cost)

// Local copy of the scalar helper (it is static in lossless_enc.c). Folds in the
// contribution of a run of constant (X+Y) ending at index i.
static WEBP_INLINE void EntropyUnrefinedHelper_NEON(
    uint32_t val, int i, uint32_t* WEBP_RESTRICT const val_prev,
    int* WEBP_RESTRICT const i_prev, VP8LBitEntropy* WEBP_RESTRICT const e,
    VP8LStreaks* WEBP_RESTRICT const stats) {
  const int streak = i - *i_prev;
  if (*val_prev != 0) {
    e->sum += (*val_prev) * streak;
    e->nonzeros += streak;
    e->nonzero_code = *i_prev;
    e->entropy += VP8LFastSLog2(*val_prev) * streak;
    if (e->max_val < *val_prev) e->max_val = *val_prev;
  }
  stats->counts[*val_prev != 0] += (streak > 3);
  stats->streaks[*val_prev != 0][(streak > 3)] += streak;
  *val_prev = val;
  *i_prev = i;
}

// Same result as GetCombinedEntropyUnrefined_C. The cost is the scan that finds
// where xy = X + Y changes; screen-content runs are long, so skip 4-at-a-time
// through any block that stays equal to the current run value, dropping to the
// exact scalar per-element path only on blocks that contain a change.
static void GetCombinedEntropyUnrefined_NEON(
    const uint32_t X[], const uint32_t Y[], int length,
    VP8LBitEntropy* WEBP_RESTRICT const bit_entropy,
    VP8LStreaks* WEBP_RESTRICT const stats) {
  int i = 1, i_prev = 0;
  uint32_t xy_prev = X[0] + Y[0];
  memset(stats, 0, sizeof(*stats));
  VP8LBitEntropyInit(bit_entropy);
  while (i + 4 <= length) {
    const uint32x4_t xy = vaddq_u32(vld1q_u32(X + i), vld1q_u32(Y + i));
    if (AllEqual_NEON(vceqq_u32(xy, vdupq_n_u32(xy_prev)))) {
      i += 4;  // whole block continues the current run
    } else {
      int k;
      for (k = 0; k < 4; ++k, ++i) {
        const uint32_t xyk = X[i] + Y[i];
        if (xyk != xy_prev) {
          EntropyUnrefinedHelper_NEON(xyk, i, &xy_prev, &i_prev, bit_entropy,
                                      stats);
        }
      }
    }
  }
  for (; i < length; ++i) {
    const uint32_t xy = X[i] + Y[i];
    if (xy != xy_prev) {
      EntropyUnrefinedHelper_NEON(xy, i, &xy_prev, &i_prev, bit_entropy, stats);
    }
  }
  EntropyUnrefinedHelper_NEON(0, i, &xy_prev, &i_prev, bit_entropy, stats);
  bit_entropy->entropy = VP8LFastSLog2(bit_entropy->sum) - bit_entropy->entropy;
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8LEncDspInitNEON(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LEncDspInitNEON(void) {
  VP8LSubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed_NEON;
  VP8LTransformColor = TransformColor_NEON;
  VP8LCollectColorBlueTransforms = CollectColorBlueTransforms_NEON;
  VP8LCollectColorRedTransforms = CollectColorRedTransforms_NEON;
  VP8LCollectArgbHistos = CollectArgbHistos_NEON;
  VP8LVectorMismatch = VectorMismatch_NEON;
  VP8LCombinedShannonEntropy = CombinedShannonEntropy_NEON;
  VP8LGetCombinedEntropyUnrefined = GetCombinedEntropyUnrefined_NEON;

  VP8LPredictorsSub[0] = PredictorSub0_NEON;
  VP8LPredictorsSub[1] = PredictorSub1_NEON;
  VP8LPredictorsSub[2] = PredictorSub2_NEON;
  VP8LPredictorsSub[3] = PredictorSub3_NEON;
  VP8LPredictorsSub[4] = PredictorSub4_NEON;
  VP8LPredictorsSub[5] = PredictorSub5_NEON;
  VP8LPredictorsSub[6] = PredictorSub6_NEON;
  VP8LPredictorsSub[7] = PredictorSub7_NEON;
  VP8LPredictorsSub[8] = PredictorSub8_NEON;
  VP8LPredictorsSub[9] = PredictorSub9_NEON;
  VP8LPredictorsSub[10] = PredictorSub10_NEON;
  VP8LPredictorsSub[11] = PredictorSub11_NEON;
  VP8LPredictorsSub[12] = PredictorSub12_NEON;
  VP8LPredictorsSub[13] = PredictorSub13_NEON;
  VP8LPredictorsSub[14] = PredictorSub0_NEON;  // <= padding security sentinels
  VP8LPredictorsSub[15] = PredictorSub0_NEON;
}

#else  // !WEBP_USE_NEON

WEBP_DSP_INIT_STUB(VP8LEncDspInitNEON)

#endif  // WEBP_USE_NEON
