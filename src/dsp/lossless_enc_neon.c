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
#include "src/dsp/neon.h"
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
// Entry point

extern void VP8LEncDspInitNEON(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LEncDspInitNEON(void) {
  VP8LSubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed_NEON;
  VP8LTransformColor = TransformColor_NEON;
  VP8LCollectColorBlueTransforms = CollectColorBlueTransforms_NEON;
  VP8LCollectColorRedTransforms = CollectColorRedTransforms_NEON;
  VP8LCollectArgbHistos = CollectArgbHistos_NEON;

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
