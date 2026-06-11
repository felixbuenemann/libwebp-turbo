// Copyright 2022 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Sharp RGB to YUV conversion.
//
// Author: Skal (pascal.massimino@gmail.com)

#include "./sharpyuv.h"

#include <assert.h>
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "./sharpyuv_cpu.h"
#include "./sharpyuv_dsp.h"
#include "./sharpyuv_gamma.h"
#include "webp/types.h"

//------------------------------------------------------------------------------

int SharpYuvGetVersion(void) { return SHARPYUV_VERSION; }

//------------------------------------------------------------------------------
// Sharp RGB->YUV conversion

static const int kNumIterations = 4;

#define YUV_FIX 16  // fixed-point precision for RGB->YUV
static const int kYuvHalf = 1 << (YUV_FIX - 1);

// Max bit depth so that intermediate calculations fit in 16 bits.
static const int kMaxBitDepth = 14;

// Returns the precision shift to use based on the input rgb_bit_depth.
static int GetPrecisionShift(int rgb_bit_depth) {
  // Try to add 2 bits of precision if it fits in kMaxBitDepth. Otherwise remove
  // bits if needed.
  return ((rgb_bit_depth + 2) <= kMaxBitDepth) ? 2
                                               : (kMaxBitDepth - rgb_bit_depth);
}

typedef int16_t fixed_t;     // signed type with extra precision for UV
typedef uint16_t fixed_y_t;  // unsigned type with extra precision for W

//------------------------------------------------------------------------------

static uint8_t clip_8b(fixed_t v) {
  return (!(v & ~0xff)) ? (uint8_t)v : (v < 0) ? 0u : 255u;
}

static uint16_t clip(fixed_t v, int max) {
  return (v < 0) ? 0 : (v > max) ? max : (uint16_t)v;
}

static fixed_y_t clip_bit_depth(int y, int bit_depth) {
  const int max = (1 << bit_depth) - 1;
  return (!(y & ~max)) ? (fixed_y_t)y : (y < 0) ? 0 : max;
}

//------------------------------------------------------------------------------

static int RGBToGray(int64_t r, int64_t g, int64_t b) {
  const int64_t luma = 13933 * r + 46871 * g + 4732 * b + kYuvHalf;
  return (int)(luma >> YUV_FIX);
}

static void StoreGray(const fixed_y_t* rgb, fixed_y_t* y, int w) {
  int i = 0;
  assert(w > 0);
  do {
    y[i] = RGBToGray(rgb[0 * w + i], rgb[1 * w + i], rgb[2 * w + i]);
  } while (++i < w);
}

//------------------------------------------------------------------------------

static WEBP_INLINE fixed_y_t Filter2(int A, int B, int W0, int bit_depth) {
  const int v0 = (A * 3 + B + 2) >> 2;
  return clip_bit_depth(v0 + W0, bit_depth);
}

//------------------------------------------------------------------------------

static WEBP_INLINE int Shift(int v, int shift) {
  return (shift >= 0) ? (v << shift) : (v >> -shift);
}

static void ImportOneRow(const uint8_t* const r_ptr, const uint8_t* const g_ptr,
                         const uint8_t* const b_ptr, int rgb_step,
                         int rgb_bit_depth, int pic_width,
                         fixed_y_t* const dst) {
  // Convert the rgb_step from a number of bytes to a number of uint8_t or
  // uint16_t values depending the bit depth.
  const int step = (rgb_bit_depth > 8) ? rgb_step / 2 : rgb_step;
  const int w = (pic_width + 1) & ~1;
  const int shift = GetPrecisionShift(rgb_bit_depth);
  const int max_val = (1 << rgb_bit_depth) - 1;
  int i = 0;

  if (rgb_bit_depth == 8) {
    do {
      const int off = i * step;
      dst[i + 0 * w] = Shift(r_ptr[off], shift);
      dst[i + 1 * w] = Shift(g_ptr[off], shift);
      dst[i + 2 * w] = Shift(b_ptr[off], shift);
    } while (++i < pic_width);
  } else if (rgb_bit_depth < 16) {
    do {
      const int off = i * step;
      int r = ((const uint16_t*)r_ptr)[off];
      int g = ((const uint16_t*)g_ptr)[off];
      int b = ((const uint16_t*)b_ptr)[off];
      dst[i + 0 * w] = Shift(r > max_val ? max_val : r, shift);
      dst[i + 1 * w] = Shift(g > max_val ? max_val : g, shift);
      dst[i + 2 * w] = Shift(b > max_val ? max_val : b, shift);
    } while (++i < pic_width);
  } else {  // rgb_bit_depth == 16
    do {
      const int off = i * step;
      int r = ((const uint16_t*)r_ptr)[off];
      int g = ((const uint16_t*)g_ptr)[off];
      int b = ((const uint16_t*)b_ptr)[off];
      dst[i + 0 * w] = Shift(r, shift);
      dst[i + 1 * w] = Shift(g, shift);
      dst[i + 2 * w] = Shift(b, shift);
    } while (++i < pic_width);
  }

  if (pic_width & 1) {  // replicate rightmost pixel
    dst[pic_width + 0 * w] = dst[pic_width + 0 * w - 1];
    dst[pic_width + 1 * w] = dst[pic_width + 1 * w - 1];
    dst[pic_width + 2 * w] = dst[pic_width + 2 * w - 1];
  }
}

static void InterpolateTwoRows(const fixed_y_t* const best_y,
                               const fixed_t* prev_uv, const fixed_t* cur_uv,
                               const fixed_t* next_uv, int w, fixed_y_t* out1,
                               fixed_y_t* out2, int bit_depth) {
  const int uv_w = w >> 1;
  const int len = (w - 1) >> 1;  // length to filter
  int k = 3;
  while (k-- > 0) {  // process each R/G/B segments in turn
    // special boundary case for i==0
    out1[0] = Filter2(cur_uv[0], prev_uv[0], best_y[0], bit_depth);
    out2[0] = Filter2(cur_uv[0], next_uv[0], best_y[w], bit_depth);

    SharpYuvFilterRow(cur_uv, prev_uv, len, best_y + 0 + 1, out1 + 1,
                      bit_depth);
    SharpYuvFilterRow(cur_uv, next_uv, len, best_y + w + 1, out2 + 1,
                      bit_depth);

    // special boundary case for i == w - 1 when w is even
    if (!(w & 1)) {
      out1[w - 1] = Filter2(cur_uv[uv_w - 1], prev_uv[uv_w - 1],
                            best_y[w - 1 + 0], bit_depth);
      out2[w - 1] = Filter2(cur_uv[uv_w - 1], next_uv[uv_w - 1],
                            best_y[w - 1 + w], bit_depth);
    }
    out1 += w;
    out2 += w;
    prev_uv += uv_w;
    cur_uv += uv_w;
    next_uv += uv_w;
  }
}

static WEBP_INLINE int RGBToYUVComponent(int r, int g, int b,
                                         const int coeffs[4], int sfix) {
  const int64_t srounder = 1LL << (YUV_FIX + sfix - 1);
  const int64_t luma = (int64_t)coeffs[0] * r + (int64_t)coeffs[1] * g +
                       (int64_t)coeffs[2] * b + coeffs[3] + srounder;
  return (int)(luma >> (YUV_FIX + sfix));
}

//------------------------------------------------------------------------------
// Main function

static void* SafeMalloc(uint64_t nmemb, size_t size) {
  const uint64_t total_size = nmemb * (uint64_t)size;
  if (total_size != (size_t)total_size) return NULL;
  return malloc((size_t)total_size);
}

//------------------------------------------------------------------------------
// Band processing.
// The image is split into at most kMaxBands horizontal bands of row pairs;
// the bands only depend on the image size, never on the number of threads, so
// the output is the same whether the bands are processed sequentially or
// concurrently. Within an iteration, a band resolves its rows like the
// original code, except that its first row reads the (iteration-start)
// snapshot of the last chroma row of the band above instead of its updated
// value.

enum { kMaxBands = 8 };
enum { kMinBandUVRows = 32 };

typedef struct {
  const uint8_t *r_ptr, *g_ptr, *b_ptr;
  int rgb_step, rgb_stride, rgb_bit_depth;
  uint8_t *y_ptr, *u_ptr, *v_ptr;
  int y_stride, u_stride, v_stride;
  int yuv_bit_depth;
  int width, height;
  int w, h, uv_w, uv_h, y_bit_depth;
  const SharpYuvConversionMatrix* yuv_matrix;
  SharpYuvTransferFunctionType transfer_type;
  fixed_y_t *best_y_base, *target_y_base;
  fixed_t *best_uv_base, *target_uv_base;
} SharpYuvCtx;

typedef struct {
  SharpYuvCtx* ctx;
  int uv_row_start, uv_row_end;  // range of row pairs of the band
  fixed_y_t* tmp;                // 6 * w private scratch
  fixed_y_t* best_rgb_y;         // 2 * w
  fixed_t* best_rgb_uv;          // 3 * uv_w
  // Iteration-start snapshots of the chroma rows just above and just below
  // the band (the rows of the neighboring bands may be concurrently updated).
  fixed_t* prev_uv_snapshot;  // 3 * uv_w
  fixed_t* next_uv_snapshot;  // 3 * uv_w
  uint64_t diff_y_sum;
  int phase;  // 0: import, 1: iterate, 2: final conversion
} SharpYuvBand;

// Import RGB samples to W/RGB representation.
static void ImportBand(const SharpYuvBand* const band) {
  const SharpYuvCtx* const ctx = band->ctx;
  const int w = ctx->w, uv_w = ctx->uv_w;
  int p;
  for (p = band->uv_row_start; p < band->uv_row_end; ++p) {
    const int j = 2 * p;
    const int is_last_row = (j == ctx->height - 1);
    const uint8_t* const r_ptr = ctx->r_ptr + (ptrdiff_t)j * ctx->rgb_stride;
    const uint8_t* const g_ptr = ctx->g_ptr + (ptrdiff_t)j * ctx->rgb_stride;
    const uint8_t* const b_ptr = ctx->b_ptr + (ptrdiff_t)j * ctx->rgb_stride;
    fixed_y_t* const best_y = ctx->best_y_base + (ptrdiff_t)j * w;
    fixed_y_t* const target_y = ctx->target_y_base + (ptrdiff_t)j * w;
    fixed_t* const best_uv = ctx->best_uv_base + (ptrdiff_t)p * 3 * uv_w;
    fixed_t* const target_uv = ctx->target_uv_base + (ptrdiff_t)p * 3 * uv_w;
    fixed_y_t* const src1 = band->tmp + 0 * w;
    fixed_y_t* const src2 = band->tmp + 3 * w;

    // prepare two rows of input
    ImportOneRow(r_ptr, g_ptr, b_ptr, ctx->rgb_step, ctx->rgb_bit_depth,
                 ctx->width, src1);
    if (!is_last_row) {
      ImportOneRow(r_ptr + ctx->rgb_stride, g_ptr + ctx->rgb_stride,
                   b_ptr + ctx->rgb_stride, ctx->rgb_step, ctx->rgb_bit_depth,
                   ctx->width, src2);
    } else {
      memcpy(src2, src1, 3 * w * sizeof(*src2));
    }
    StoreGray(src1, best_y + 0, w);
    StoreGray(src2, best_y + w, w);

    SharpYuvUpdateW(src1, target_y, w, ctx->y_bit_depth, ctx->transfer_type);
    SharpYuvUpdateW(src2, target_y + w, w, ctx->y_bit_depth,
                    ctx->transfer_type);
    SharpYuvUpdateChroma(src1, src2, target_uv, uv_w, ctx->y_bit_depth,
                         ctx->transfer_type);
    memcpy(best_uv, target_uv, 3 * uv_w * sizeof(*best_uv));
  }
}

// One iteration of clipping conflict resolution over the band rows.
static void IterateBand(SharpYuvBand* const band) {
  const SharpYuvCtx* const ctx = band->ctx;
  const int w = ctx->w, h = ctx->h, uv_w = ctx->uv_w;
  const fixed_t* prev_uv =
      (band->uv_row_start == 0)
          ? ctx->best_uv_base
          : band->prev_uv_snapshot;
  int p;
  band->diff_y_sum = 0;
  for (p = band->uv_row_start; p < band->uv_row_end; ++p) {
    const int j = 2 * p;
    fixed_y_t* const best_y = ctx->best_y_base + (ptrdiff_t)j * w;
    fixed_y_t* const target_y = ctx->target_y_base + (ptrdiff_t)j * w;
    fixed_t* const best_uv = ctx->best_uv_base + (ptrdiff_t)p * 3 * uv_w;
    fixed_t* const target_uv = ctx->target_uv_base + (ptrdiff_t)p * 3 * uv_w;
    const fixed_t* const cur_uv = best_uv;
    const fixed_t* const next_uv =
        (j >= h - 2) ? cur_uv
        : (p == band->uv_row_end - 1) ? band->next_uv_snapshot
                                      : cur_uv + 3 * uv_w;
    fixed_y_t* const src1 = band->tmp + 0 * w;
    fixed_y_t* const src2 = band->tmp + 3 * w;

    InterpolateTwoRows(best_y, prev_uv, cur_uv, next_uv, w, src1, src2,
                       ctx->y_bit_depth);

    SharpYuvUpdateW(src1, band->best_rgb_y + 0 * w, w, ctx->y_bit_depth,
                    ctx->transfer_type);
    SharpYuvUpdateW(src2, band->best_rgb_y + 1 * w, w, ctx->y_bit_depth,
                    ctx->transfer_type);
    SharpYuvUpdateChroma(src1, src2, band->best_rgb_uv, uv_w, ctx->y_bit_depth,
                         ctx->transfer_type);

    // update two rows of Y and one row of RGB
    band->diff_y_sum += SharpYuvUpdateY(target_y, band->best_rgb_y, best_y,
                                        2 * w, ctx->y_bit_depth);
    SharpYuvUpdateRGB(target_uv, band->best_rgb_uv, best_uv, 3 * uv_w);
    prev_uv = cur_uv;  // the updated row (same as the original code)
  }
}

// Final conversion of the band rows to the destination YUV planes.
static void ConvertBand(const SharpYuvBand* const band) {
  const SharpYuvCtx* const ctx = band->ctx;
  const int w = ctx->w, uv_w = ctx->uv_w;
  const int sfix = GetPrecisionShift(ctx->rgb_bit_depth);
  const int yuv_max = (1 << ctx->yuv_bit_depth) - 1;
  const SharpYuvConversionMatrix* const yuv_matrix = ctx->yuv_matrix;
  int i, j, p;
  for (j = 2 * band->uv_row_start;
       j < 2 * band->uv_row_end && j < ctx->height; ++j) {
    const fixed_y_t* const best_y = ctx->best_y_base + (ptrdiff_t)j * w;
    const fixed_t* const best_uv =
        ctx->best_uv_base + (ptrdiff_t)(j >> 1) * 3 * uv_w;
    uint8_t* const y_ptr = ctx->y_ptr + (ptrdiff_t)j * ctx->y_stride;
    for (i = 0; i < ctx->width; ++i) {
      const int off = (i >> 1);
      const int W = best_y[i];
      const int r = best_uv[off + 0 * uv_w] + W;
      const int g = best_uv[off + 1 * uv_w] + W;
      const int b = best_uv[off + 2 * uv_w] + W;
      const int y = RGBToYUVComponent(r, g, b, yuv_matrix->rgb_to_y, sfix);
      if (ctx->yuv_bit_depth <= 8) {
        y_ptr[i] = clip_8b(y);
      } else {
        ((uint16_t*)y_ptr)[i] = clip(y, yuv_max);
      }
    }
  }
  for (p = band->uv_row_start;
       p < band->uv_row_end && p < (ctx->height + 1) / 2; ++p) {
    const fixed_t* const best_uv =
        ctx->best_uv_base + (ptrdiff_t)p * 3 * uv_w;
    uint8_t* const u_ptr = ctx->u_ptr + (ptrdiff_t)p * ctx->u_stride;
    uint8_t* const v_ptr = ctx->v_ptr + (ptrdiff_t)p * ctx->v_stride;
    for (i = 0; i < uv_w; ++i) {
      // Note r, g and b values here are off by W, but a constant offset on all
      // 3 components doesn't change the value of u and v with a YCbCr matrix.
      const int r = best_uv[i + 0 * uv_w];
      const int g = best_uv[i + 1 * uv_w];
      const int b = best_uv[i + 2 * uv_w];
      const int u = RGBToYUVComponent(r, g, b, yuv_matrix->rgb_to_u, sfix);
      const int v = RGBToYUVComponent(r, g, b, yuv_matrix->rgb_to_v, sfix);
      if (ctx->yuv_bit_depth <= 8) {
        u_ptr[i] = clip_8b(u);
        v_ptr[i] = clip_8b(v);
      } else {
        ((uint16_t*)u_ptr)[i] = clip(u, yuv_max);
        ((uint16_t*)v_ptr)[i] = clip(v, yuv_max);
      }
    }
  }
}

static void RunBandPhase(SharpYuvBand* const band) {
  switch (band->phase) {
    case 0: ImportBand(band); break;
    case 1: IterateBand(band); break;
    default: ConvertBand(band); break;
  }
}

#if defined(WEBP_USE_THREAD) && !defined(_WIN32)
#include <pthread.h>  // NOLINT

static void* SharpYuvBandThread(void* arg) {
  RunBandPhase((SharpYuvBand*)arg);
  return NULL;
}
#endif

// Runs 'phase' on all the bands, concurrently if allowed. The bands are
// independent within a phase, so the output does not depend on the schedule.
static void RunBands(SharpYuvBand bands[], int num_bands, int phase,
                     int use_threads) {
  int i;
  for (i = 0; i < num_bands; ++i) bands[i].phase = phase;
#if defined(WEBP_USE_THREAD) && !defined(_WIN32)
  if (use_threads && num_bands > 1) {
    pthread_t threads[kMaxBands];
    int started[kMaxBands] = {0};
    for (i = 1; i < num_bands; ++i) {
      started[i] =
          !pthread_create(&threads[i], NULL, SharpYuvBandThread, &bands[i]);
    }
    RunBandPhase(&bands[0]);
    for (i = 1; i < num_bands; ++i) {
      if (started[i]) {
        pthread_join(threads[i], NULL);
      } else {
        RunBandPhase(&bands[i]);  // thread could not start: run inline
      }
    }
    return;
  }
#else
  (void)use_threads;
#endif
  for (i = 0; i < num_bands; ++i) RunBandPhase(&bands[i]);
}

static int DoSharpArgbToYuv(const uint8_t* r_ptr, const uint8_t* g_ptr,
                            const uint8_t* b_ptr, int rgb_step, int rgb_stride,
                            int rgb_bit_depth, uint8_t* y_ptr, int y_stride,
                            uint8_t* u_ptr, int u_stride, uint8_t* v_ptr,
                            int v_stride, int yuv_bit_depth, int width,
                            int height,
                            const SharpYuvConversionMatrix* yuv_matrix,
                            SharpYuvTransferFunctionType transfer_type,
                            int use_threads) {
  // we expand the right/bottom border if needed
  const int w = (width + 1) & ~1;
  const int h = (height + 1) & ~1;
  const int uv_w = w >> 1;
  const int uv_h = h >> 1;
  const int y_bit_depth = rgb_bit_depth + GetPrecisionShift(rgb_bit_depth);
  uint64_t prev_diff_y_sum = ~0;
  int iter, i;
  // The number of bands only depends on the image size so that the output is
  // independent of the number of threads.
  const int num_bands =
      (uv_h / kMinBandUVRows < 1)         ? 1
      : (uv_h / kMinBandUVRows > kMaxBands) ? kMaxBands
                                            : uv_h / kMinBandUVRows;

  const uint64_t band_scratch_size =
      (uint64_t)num_bands * (6 * w + 2 * w + 3 * 3 * uv_w);
  const uint64_t best_y_base_size = (uint64_t)w * h;
  const uint64_t target_y_base_size = (uint64_t)w * h;
  const uint64_t best_uv_base_size = (uint64_t)uv_w * 3 * uv_h;
  const uint64_t target_uv_base_size = (uint64_t)uv_w * 3 * uv_h;
  fixed_y_t* const tmp_buffer = (fixed_y_t*)SafeMalloc(
      band_scratch_size + best_y_base_size + target_y_base_size +
          best_uv_base_size + target_uv_base_size,
      sizeof(*tmp_buffer));
  const uint64_t diff_y_threshold = (uint64_t)(3.0 * w * h);
  SharpYuvCtx ctx;
  SharpYuvBand bands[kMaxBands];
  fixed_y_t* mem;
  int ok;
  assert(w > 0);
  assert(h > 0);
  assert(sizeof(fixed_y_t) == sizeof(fixed_t));

  if (tmp_buffer == NULL) {
    ok = 0;
    goto End;
  }
  mem = tmp_buffer;
  ctx.r_ptr = r_ptr;
  ctx.g_ptr = g_ptr;
  ctx.b_ptr = b_ptr;
  ctx.rgb_step = rgb_step;
  ctx.rgb_stride = rgb_stride;
  ctx.rgb_bit_depth = rgb_bit_depth;
  ctx.y_ptr = y_ptr;
  ctx.u_ptr = u_ptr;
  ctx.v_ptr = v_ptr;
  ctx.y_stride = y_stride;
  ctx.u_stride = u_stride;
  ctx.v_stride = v_stride;
  ctx.yuv_bit_depth = yuv_bit_depth;
  ctx.width = width;
  ctx.height = height;
  ctx.w = w;
  ctx.h = h;
  ctx.uv_w = uv_w;
  ctx.uv_h = uv_h;
  ctx.y_bit_depth = y_bit_depth;
  ctx.yuv_matrix = yuv_matrix;
  ctx.transfer_type = transfer_type;
  ctx.best_y_base = mem;
  mem += best_y_base_size;
  ctx.target_y_base = mem;
  mem += target_y_base_size;
  ctx.best_uv_base = (fixed_t*)mem;
  mem += best_uv_base_size;
  ctx.target_uv_base = (fixed_t*)mem;
  mem += target_uv_base_size;
  {
    const int rows_per_band = uv_h / num_bands;
    const int remainder = uv_h % num_bands;
    int row = 0;
    for (i = 0; i < num_bands; ++i) {
      SharpYuvBand* const band = &bands[i];
      band->ctx = &ctx;
      band->uv_row_start = row;
      row += rows_per_band + (i < remainder);
      band->uv_row_end = row;
      band->tmp = mem;
      mem += 6 * w;
      band->best_rgb_y = mem;
      mem += 2 * w;
      band->best_rgb_uv = (fixed_t*)mem;
      mem += 3 * uv_w;
      band->prev_uv_snapshot = (fixed_t*)mem;
      mem += 3 * uv_w;
      band->next_uv_snapshot = (fixed_t*)mem;
      mem += 3 * uv_w;
      band->diff_y_sum = 0;
    }
    assert(row == uv_h);
  }

  // Import RGB samples to W/RGB representation.
  RunBands(bands, num_bands, /*phase=*/0, use_threads);

  // Iterate and resolve clipping conflicts.
  for (iter = 0; iter < kNumIterations; ++iter) {
    uint64_t diff_y_sum = 0;
    // Snapshot the chroma rows around each band before they are updated.
    for (i = 1; i < num_bands; ++i) {
      memcpy(bands[i].prev_uv_snapshot,
             ctx.best_uv_base +
                 (ptrdiff_t)(bands[i].uv_row_start - 1) * 3 * uv_w,
             3 * uv_w * sizeof(fixed_t));
      memcpy(bands[i - 1].next_uv_snapshot,
             ctx.best_uv_base + (ptrdiff_t)bands[i].uv_row_start * 3 * uv_w,
             3 * uv_w * sizeof(fixed_t));
    }
    RunBands(bands, num_bands, /*phase=*/1, use_threads);
    for (i = 0; i < num_bands; ++i) diff_y_sum += bands[i].diff_y_sum;
    // test exit condition
    if (iter > 0) {
      if (diff_y_sum < diff_y_threshold) break;
      if (diff_y_sum > prev_diff_y_sum) break;
    }
    prev_diff_y_sum = diff_y_sum;
  }

  // final reconstruction
  RunBands(bands, num_bands, /*phase=*/2, use_threads);
  ok = 1;

End:
  free(tmp_buffer);
  return ok;
}

#if defined(WEBP_USE_THREAD) && !defined(_WIN32)
#include <pthread.h>  // NOLINT

#define LOCK_ACCESS                                                 \
  static pthread_mutex_t sharpyuv_lock = PTHREAD_MUTEX_INITIALIZER; \
  if (pthread_mutex_lock(&sharpyuv_lock)) return
#define UNLOCK_ACCESS_AND_RETURN                \
  do {                                          \
    (void)pthread_mutex_unlock(&sharpyuv_lock); \
    return;                                     \
  } while (0)
#else  // !(defined(WEBP_USE_THREAD) && !defined(_WIN32))
#define LOCK_ACCESS \
  do {              \
  } while (0)
#define UNLOCK_ACCESS_AND_RETURN return
#endif  // defined(WEBP_USE_THREAD) && !defined(_WIN32)

// Hidden exported init function.
// By default SharpYuvConvert calls it with SharpYuvGetCPUInfo. If needed,
// users can declare it as extern and call it with an alternate VP8CPUInfo
// function.
extern VP8CPUInfo SharpYuvGetCPUInfo;
SHARPYUV_EXTERN void SharpYuvInit(VP8CPUInfo cpu_info_func);
void SharpYuvInit(VP8CPUInfo cpu_info_func) {
  static volatile VP8CPUInfo sharpyuv_last_cpuinfo_used =
      (VP8CPUInfo)&sharpyuv_last_cpuinfo_used;
  LOCK_ACCESS;
  // Only update SharpYuvGetCPUInfo when called from external code to avoid a
  // race on reading the value in SharpYuvConvert().
  if (cpu_info_func != (VP8CPUInfo)&SharpYuvGetCPUInfo) {
    SharpYuvGetCPUInfo = cpu_info_func;
  }
  if (sharpyuv_last_cpuinfo_used == SharpYuvGetCPUInfo) {
    UNLOCK_ACCESS_AND_RETURN;
  }

  SharpYuvInitDsp();
  SharpYuvInitGammaTables();

  sharpyuv_last_cpuinfo_used = SharpYuvGetCPUInfo;
  UNLOCK_ACCESS_AND_RETURN;
}

int SharpYuvConvert(const void* r_ptr, const void* g_ptr, const void* b_ptr,
                    int rgb_step, int rgb_stride, int rgb_bit_depth,
                    void* y_ptr, int y_stride, void* u_ptr, int u_stride,
                    void* v_ptr, int v_stride, int yuv_bit_depth, int width,
                    int height, const SharpYuvConversionMatrix* yuv_matrix) {
  SharpYuvOptions options;
  options.yuv_matrix = yuv_matrix;
  options.transfer_type = kSharpYuvTransferFunctionSrgb;
  options.use_threads = 0;
  return SharpYuvConvertWithOptions(
      r_ptr, g_ptr, b_ptr, rgb_step, rgb_stride, rgb_bit_depth, y_ptr, y_stride,
      u_ptr, u_stride, v_ptr, v_stride, yuv_bit_depth, width, height, &options);
}

int SharpYuvOptionsInitInternal(const SharpYuvConversionMatrix* yuv_matrix,
                                SharpYuvOptions* options, int version) {
  const int major = (version >> 24);
  const int minor = (version >> 16) & 0xff;
  if (options == NULL || yuv_matrix == NULL ||
      (major == SHARPYUV_VERSION_MAJOR && major == 0 &&
       minor != SHARPYUV_VERSION_MINOR) ||
      (major != SHARPYUV_VERSION_MAJOR)) {
    return 0;
  }
  options->yuv_matrix = yuv_matrix;
  options->transfer_type = kSharpYuvTransferFunctionSrgb;
  options->use_threads = 0;
  return 1;
}

int SharpYuvConvertWithOptions(const void* r_ptr, const void* g_ptr,
                               const void* b_ptr, int rgb_step, int rgb_stride,
                               int rgb_bit_depth, void* y_ptr, int y_stride,
                               void* u_ptr, int u_stride, void* v_ptr,
                               int v_stride, int yuv_bit_depth, int width,
                               int height, const SharpYuvOptions* options) {
  const SharpYuvConversionMatrix* yuv_matrix = options->yuv_matrix;
  SharpYuvTransferFunctionType transfer_type = options->transfer_type;
  SharpYuvConversionMatrix scaled_matrix;
  const int rgb_max = (1 << rgb_bit_depth) - 1;
  const int rgb_round = 1 << (rgb_bit_depth - 1);
  const int yuv_max = (1 << yuv_bit_depth) - 1;
  const int sfix = GetPrecisionShift(rgb_bit_depth);

  if (width < 1 || height < 1 || width == INT_MAX || height == INT_MAX ||
      r_ptr == NULL || g_ptr == NULL || b_ptr == NULL || y_ptr == NULL ||
      u_ptr == NULL || v_ptr == NULL) {
    return 0;
  }
  if (rgb_bit_depth != 8 && rgb_bit_depth != 10 && rgb_bit_depth != 12 &&
      rgb_bit_depth != 16) {
    return 0;
  }
  if (yuv_bit_depth != 8 && yuv_bit_depth != 10 && yuv_bit_depth != 12) {
    return 0;
  }
  if (rgb_bit_depth > 8 && (rgb_step % 2 != 0 || rgb_stride % 2 != 0)) {
    // Step/stride should be even for uint16_t buffers.
    return 0;
  }
  {
    const uint64_t yuv_bytes = (yuv_bit_depth > 8) ? 2 : 1;
    const uint64_t uv_width = (width + 1) / 2;
    const uint64_t abs_step =
        (uint64_t)((rgb_step < 0) ? -(int64_t)rgb_step : (int64_t)rgb_step);
    const uint64_t abs_stride =
        (uint64_t)((rgb_stride < 0) ? -(int64_t)rgb_stride
                                    : (int64_t)rgb_stride);
    const uint64_t total_rgb_size = (uint64_t)height * abs_stride;
    const uint64_t uv_height = (height + 1) / 2;
    const uint64_t total_y_size = (uint64_t)height * y_stride;
    const uint64_t total_u_size = uv_height * u_stride;
    const uint64_t total_v_size = uv_height * v_stride;

    if (y_stride < 0 || (uint64_t)y_stride < (uint64_t)width * yuv_bytes ||
        u_stride < 0 || (uint64_t)u_stride < uv_width * yuv_bytes ||
        v_stride < 0 || (uint64_t)v_stride < uv_width * yuv_bytes) {
      return 0;
    }
    if (abs_step == 0 || abs_stride < (uint64_t)width * abs_step) {
      return 0;
    }
    if (total_rgb_size != (size_t)total_rgb_size ||
        total_y_size != (size_t)total_y_size ||
        total_u_size != (size_t)total_u_size ||
        total_v_size != (size_t)total_v_size) {
      return 0;
    }
  }
  if (yuv_bit_depth > 8 &&
      (y_stride % 2 != 0 || u_stride % 2 != 0 || v_stride % 2 != 0)) {
    // Stride should be even for uint16_t buffers.
    return 0;
  }
  // The address of the function pointer is used to avoid a read race.
  SharpYuvInit((VP8CPUInfo)&SharpYuvGetCPUInfo);

  // Add scaling factor to go from rgb_bit_depth to yuv_bit_depth, to the
  // rgb->yuv conversion matrix.
  if (rgb_bit_depth == yuv_bit_depth) {
    memcpy(&scaled_matrix, yuv_matrix, sizeof(scaled_matrix));
  } else {
    int i;
    for (i = 0; i < 3; ++i) {
      scaled_matrix.rgb_to_y[i] =
          (yuv_matrix->rgb_to_y[i] * yuv_max + rgb_round) / rgb_max;
      scaled_matrix.rgb_to_u[i] =
          (yuv_matrix->rgb_to_u[i] * yuv_max + rgb_round) / rgb_max;
      scaled_matrix.rgb_to_v[i] =
          (yuv_matrix->rgb_to_v[i] * yuv_max + rgb_round) / rgb_max;
    }
  }
  // Also incorporate precision change scaling.
  scaled_matrix.rgb_to_y[3] = Shift(yuv_matrix->rgb_to_y[3], sfix);
  scaled_matrix.rgb_to_u[3] = Shift(yuv_matrix->rgb_to_u[3], sfix);
  scaled_matrix.rgb_to_v[3] = Shift(yuv_matrix->rgb_to_v[3], sfix);

  return DoSharpArgbToYuv(
      (const uint8_t*)r_ptr, (const uint8_t*)g_ptr, (const uint8_t*)b_ptr,
      rgb_step, rgb_stride, rgb_bit_depth, (uint8_t*)y_ptr, y_stride,
      (uint8_t*)u_ptr, u_stride, (uint8_t*)v_ptr, v_stride, yuv_bit_depth,
      width, height, &scaled_matrix, transfer_type, options->use_threads);
}

//------------------------------------------------------------------------------
