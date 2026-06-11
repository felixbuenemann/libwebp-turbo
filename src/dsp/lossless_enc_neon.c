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

#undef SPAN
#undef CST_5b

//------------------------------------------------------------------------------
// Entry point

extern void VP8LEncDspInitNEON(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LEncDspInitNEON(void) {
  VP8LSubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed_NEON;
  VP8LTransformColor = TransformColor_NEON;
  VP8LCollectColorBlueTransforms = CollectColorBlueTransforms_NEON;
  VP8LCollectColorRedTransforms = CollectColorRedTransforms_NEON;
}

#else  // !WEBP_USE_NEON

WEBP_DSP_INIT_STUB(VP8LEncDspInitNEON)

#endif  // WEBP_USE_NEON
