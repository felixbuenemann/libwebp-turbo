// Copyright 2016 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Image transform methods for lossless encoder.
//
// Authors: Vikas Arora (vikaas.arora@gmail.com)
//          Jyrki Alakuijala (jyrki@google.com)
//          Urvang Joshi (urvang@google.com)
//          Vincent Rabaud (vrabaud@google.com)

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"
#include "src/enc/vp8i_enc.h"
#include "src/enc/vp8li_enc.h"
#include "src/utils/thread_utils.h"
#include "src/utils/utils.h"
#include "src/webp/encode.h"
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

#define HISTO_SIZE (4 * 256)
static const int64_t kSpatialPredictorBias = 15ll << LOG_2_PRECISION_BITS;
static const int kPredLowEffort = 11;
static const uint32_t kMaskAlpha = 0xff000000;
static const int kNumPredModes = 14;

// Mostly used to reduce code size + readability
static WEBP_INLINE int GetMin(int a, int b) { return (a > b) ? b : a; }
static WEBP_INLINE int GetMax(int a, int b) { return (a < b) ? b : a; }

//------------------------------------------------------------------------------
// Methods to calculate Entropy (Shannon).

// Compute a bias for prediction entropy using a global heuristic to favor
// values closer to 0. Hence the final negative sign.
// 'exp_val' has a scaling factor of 1/100.
static int64_t PredictionCostBias(const uint32_t counts[256], uint64_t weight_0,
                                  uint64_t exp_val) {
  const int significant_symbols = 256 >> 4;
  const uint64_t exp_decay_factor = 6;  // has a scaling factor of 1/10
  uint64_t bits = (weight_0 * counts[0]) << LOG_2_PRECISION_BITS;
  int i;
  exp_val <<= LOG_2_PRECISION_BITS;
  for (i = 1; i < significant_symbols; ++i) {
    bits += DivRound(exp_val * (counts[i] + counts[256 - i]), 100);
    exp_val = DivRound(exp_decay_factor * exp_val, 10);
  }
  return -DivRound((int64_t)bits, 10);
}

static int64_t PredictionCostSpatialHistogram(
    const uint32_t accumulated[HISTO_SIZE], const uint32_t tile[HISTO_SIZE],
    int mode, int left_mode, int above_mode) {
  int i;
  int64_t retval = 0;
  for (i = 0; i < 4; ++i) {
    const uint64_t kExpValue = 94;
    retval += PredictionCostBias(&tile[i * 256], 1, kExpValue);
    // Compute the new cost if 'tile' is added to 'accumulate' but also add the
    // cost of the current histogram to guide the spatial predictor selection.
    // Basically, favor low entropy, locally and globally.
    retval += (int64_t)VP8LCombinedShannonEntropy(&tile[i * 256],
                                                  &accumulated[i * 256]);
  }
  // Favor keeping the areas locally similar.
  if (mode == left_mode) retval -= kSpatialPredictorBias;
  if (mode == above_mode) retval -= kSpatialPredictorBias;
  return retval;
}

//------------------------------------------------------------------------------
// Spatial transform functions.

static WEBP_INLINE void PredictBatch(int mode, int x_start, int y,
                                     int num_pixels, const uint32_t* current,
                                     const uint32_t* upper, uint32_t* out) {
  if (x_start == 0) {
    if (y == 0) {
      // ARGB_BLACK.
      VP8LPredictorsSub[0](current, NULL, 1, out);
    } else {
      // Top one.
      VP8LPredictorsSub[2](current, upper, 1, out);
    }
    ++x_start;
    ++out;
    --num_pixels;
  }
  if (y == 0) {
    // Left one.
    VP8LPredictorsSub[1](current + x_start, NULL, num_pixels, out);
  } else {
    VP8LPredictorsSub[mode](current + x_start, upper + x_start, num_pixels,
                            out);
  }
}

#if (WEBP_NEAR_LOSSLESS == 1)
static int MaxDiffBetweenPixels(uint32_t p1, uint32_t p2) {
  const int diff_a = abs((int)(p1 >> 24) - (int)(p2 >> 24));
  const int diff_r = abs((int)((p1 >> 16) & 0xff) - (int)((p2 >> 16) & 0xff));
  const int diff_g = abs((int)((p1 >> 8) & 0xff) - (int)((p2 >> 8) & 0xff));
  const int diff_b = abs((int)(p1 & 0xff) - (int)(p2 & 0xff));
  return GetMax(GetMax(diff_a, diff_r), GetMax(diff_g, diff_b));
}

static int MaxDiffAroundPixel(uint32_t current, uint32_t up, uint32_t down,
                              uint32_t left, uint32_t right) {
  const int diff_up = MaxDiffBetweenPixels(current, up);
  const int diff_down = MaxDiffBetweenPixels(current, down);
  const int diff_left = MaxDiffBetweenPixels(current, left);
  const int diff_right = MaxDiffBetweenPixels(current, right);
  return GetMax(GetMax(diff_up, diff_down), GetMax(diff_left, diff_right));
}

static uint32_t AddGreenToBlueAndRed(uint32_t argb) {
  const uint32_t green = (argb >> 8) & 0xff;
  uint32_t red_blue = argb & 0x00ff00ffu;
  red_blue += (green << 16) | green;
  red_blue &= 0x00ff00ffu;
  return (argb & 0xff00ff00u) | red_blue;
}

static void MaxDiffsForRow(int width, int stride, const uint32_t* const argb,
                           uint8_t* const max_diffs, int used_subtract_green) {
  uint32_t current, up, down, left, right;
  int x;
  if (width <= 2) return;
  current = argb[0];
  right = argb[1];
  if (used_subtract_green) {
    current = AddGreenToBlueAndRed(current);
    right = AddGreenToBlueAndRed(right);
  }
  // max_diffs[0] and max_diffs[width - 1] are never used.
  for (x = 1; x < width - 1; ++x) {
    up = argb[-stride + x];
    down = argb[stride + x];
    left = current;
    current = right;
    right = argb[x + 1];
    if (used_subtract_green) {
      up = AddGreenToBlueAndRed(up);
      down = AddGreenToBlueAndRed(down);
      right = AddGreenToBlueAndRed(right);
    }
    max_diffs[x] = MaxDiffAroundPixel(current, up, down, left, right);
  }
}

// Quantize the difference between the actual component value and its prediction
// to a multiple of quantization, working modulo 256, taking care not to cross
// a boundary (inclusive upper limit).
static uint8_t NearLosslessComponent(uint8_t value, uint8_t predict,
                                     uint8_t boundary, int quantization) {
  const int residual = (value - predict) & 0xff;
  const int boundary_residual = (boundary - predict) & 0xff;
  const int lower = residual & ~(quantization - 1);
  const int upper = lower + quantization;
  // Resolve ties towards a value closer to the prediction (i.e. towards lower
  // if value comes after prediction and towards upper otherwise).
  const int bias = ((boundary - value) & 0xff) < boundary_residual;
  if (residual - lower < upper - residual + bias) {
    // lower is closer to residual than upper.
    if (residual > boundary_residual && lower <= boundary_residual) {
      // Halve quantization step to avoid crossing boundary. This midpoint is
      // on the same side of boundary as residual because midpoint >= residual
      // (since lower is closer than upper) and residual is above the boundary.
      return lower + (quantization >> 1);
    }
    return lower;
  } else {
    // upper is closer to residual than lower.
    if (residual <= boundary_residual && upper > boundary_residual) {
      // Halve quantization step to avoid crossing boundary. This midpoint is
      // on the same side of boundary as residual because midpoint <= residual
      // (since upper is closer than lower) and residual is below the boundary.
      return lower + (quantization >> 1);
    }
    return upper & 0xff;
  }
}

static WEBP_INLINE uint8_t NearLosslessDiff(uint8_t a, uint8_t b) {
  return (uint8_t)((((int)(a) - (int)(b))) & 0xff);
}

// Quantize every component of the difference between the actual pixel value and
// its prediction to a multiple of a quantization (a power of 2, not larger than
// max_quantization which is a power of 2, smaller than max_diff). Take care if
// value and predict have undergone subtract green, which means that red and
// blue are represented as offsets from green.
static uint32_t NearLossless(uint32_t value, uint32_t predict,
                             int max_quantization, int max_diff,
                             int used_subtract_green) {
  int quantization;
  uint8_t new_green = 0;
  uint8_t green_diff = 0;
  uint8_t a, r, g, b;
  if (max_diff <= 2) {
    return VP8LSubPixels(value, predict);
  }
  quantization = max_quantization;
  while (quantization >= max_diff) {
    quantization >>= 1;
  }
  if ((value >> 24) == 0 || (value >> 24) == 0xff) {
    // Preserve transparency of fully transparent or fully opaque pixels.
    a = NearLosslessDiff((value >> 24) & 0xff, (predict >> 24) & 0xff);
  } else {
    a = NearLosslessComponent(value >> 24, predict >> 24, 0xff, quantization);
  }
  g = NearLosslessComponent((value >> 8) & 0xff, (predict >> 8) & 0xff, 0xff,
                            quantization);
  if (used_subtract_green) {
    // The green offset will be added to red and blue components during decoding
    // to obtain the actual red and blue values.
    new_green = ((predict >> 8) + g) & 0xff;
    // The amount by which green has been adjusted during quantization. It is
    // subtracted from red and blue for compensation, to avoid accumulating two
    // quantization errors in them.
    green_diff = NearLosslessDiff(new_green, (value >> 8) & 0xff);
  }
  r = NearLosslessComponent(NearLosslessDiff((value >> 16) & 0xff, green_diff),
                            (predict >> 16) & 0xff, 0xff - new_green,
                            quantization);
  b = NearLosslessComponent(NearLosslessDiff(value & 0xff, green_diff),
                            predict & 0xff, 0xff - new_green, quantization);
  return ((uint32_t)a << 24) | ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
}
#endif  // (WEBP_NEAR_LOSSLESS == 1)

// Stores the difference between the pixel and its prediction in "out".
// In case of a lossy encoding, updates the source image to avoid propagating
// the deviation further to pixels which depend on the current pixel for their
// predictions.
// Returns 1 if the pixels in the row segment [x_start, x_end) are all opaque.
static int RowSegmentIsOpaque(const uint32_t* const row, int x_start,
                              int x_end) {
  int x;
  for (x = x_start; x < x_end; ++x) {
    if ((row[x] & kMaskAlpha) == 0) return 0;
  }
  return 1;
}

// 'segment_is_opaque' must be true only if the pixels [x_start, x_end) of
// 'current_row' all have a non-zero alpha value.
static WEBP_INLINE void GetResidual(
    int width, int height, uint32_t* const upper_row,
    uint32_t* const current_row, const uint8_t* const max_diffs, int mode,
    int x_start, int x_end, int y, int max_quantization, int exact,
    int used_subtract_green, int segment_is_opaque, uint32_t* const out) {
  int use_batch = exact;
  if (!use_batch) {
    // On a fully opaque segment with no near-lossless quantization, the
    // pixel-by-pixel loop below neither quantizes nor cleans up any pixel
    // and computes the same residuals as the (vectorized) batch prediction.
#if (WEBP_NEAR_LOSSLESS == 1)
    use_batch = (max_quantization == 1) && segment_is_opaque;
#else
    use_batch = segment_is_opaque;
#endif
  }
  if (use_batch) {
    PredictBatch(mode, x_start, y, x_end - x_start, current_row, upper_row,
                 out);
  } else {
    const VP8LPredictorFunc pred_func = VP8LPredictors[mode];
    int x;
    for (x = x_start; x < x_end; ++x) {
      uint32_t predict;
      uint32_t residual;
      if (y == 0) {
        predict = (x == 0) ? ARGB_BLACK : current_row[x - 1];  // Left.
      } else if (x == 0) {
        predict = upper_row[x];  // Top.
      } else {
        predict = pred_func(&current_row[x - 1], upper_row + x);
      }
#if (WEBP_NEAR_LOSSLESS == 1)
      if (max_quantization == 1 || mode == 0 || y == 0 || y == height - 1 ||
          x == 0 || x == width - 1) {
        residual = VP8LSubPixels(current_row[x], predict);
      } else {
        residual = NearLossless(current_row[x], predict, max_quantization,
                                max_diffs[x], used_subtract_green);
        // Update the source image.
        current_row[x] = VP8LAddPixels(predict, residual);
        // x is never 0 here so we do not need to update upper_row like below.
      }
#else
      (void)max_diffs;
      (void)height;
      (void)max_quantization;
      (void)used_subtract_green;
      residual = VP8LSubPixels(current_row[x], predict);
#endif
      if ((current_row[x] & kMaskAlpha) == 0) {
        // If alpha is 0, cleanup RGB. We can choose the RGB values of the
        // residual for best compression. The prediction of alpha itself can be
        // non-zero and must be kept though. We choose RGB of the residual to be
        // 0.
        residual &= kMaskAlpha;
        // Update the source image.
        current_row[x] = predict & ~kMaskAlpha;
        // The prediction for the rightmost pixel in a row uses the leftmost
        // pixel
        // in that row as its top-right context pixel. Hence if we change the
        // leftmost pixel of current_row, the corresponding change must be
        // applied
        // to upper_row as well where top-right context is being read from.
        if (x == 0 && y != 0) upper_row[width] = current_row[0];
      }
      out[x - x_start] = residual;
    }
  }
}

// Accessors to residual histograms.
static WEBP_INLINE uint32_t* GetHistoArgb(uint32_t* const all_histos,
                                          int subsampling_index, int mode) {
  return &all_histos[(subsampling_index * kNumPredModes + mode) * HISTO_SIZE];
}

static WEBP_INLINE const uint32_t* GetHistoArgbConst(
    const uint32_t* const all_histos, int subsampling_index, int mode) {
  return &all_histos[subsampling_index * kNumPredModes * HISTO_SIZE +
                     mode * HISTO_SIZE];
}

// Accessors to accumulated residual histogram.
static WEBP_INLINE uint32_t* GetAccumulatedHisto(uint32_t* all_accumulated,
                                                 int subsampling_index) {
  return &all_accumulated[subsampling_index * HISTO_SIZE];
}

// Find and store the best predictor for a tile at subsampling
// 'subsampling_index'. 'tile_histos' are the kNumPredModes residual
// histograms of the tile (HISTO_SIZE each).
static void GetBestPredictorForTile(const uint32_t* const tile_histos,
                                    int subsampling_index, int tile_x,
                                    int tile_y, int tiles_per_row,
                                    uint32_t* all_accumulated_argb,
                                    uint32_t** const all_modes,
                                    uint32_t* const all_pred_histos) {
  uint32_t* const accumulated_argb =
      GetAccumulatedHisto(all_accumulated_argb, subsampling_index);
  uint32_t* const modes = all_modes[subsampling_index];
  uint32_t* const pred_histos =
      &all_pred_histos[subsampling_index * kNumPredModes];
  // Prediction modes of the left and above neighbor tiles.
  const int left_mode =
      (tile_x > 0) ? (modes[tile_y * tiles_per_row + tile_x - 1] >> 8) & 0xff
                   : 0xff;
  const int above_mode =
      (tile_y > 0) ? (modes[(tile_y - 1) * tiles_per_row + tile_x] >> 8) & 0xff
                   : 0xff;
  int mode;
  int64_t best_diff = WEBP_INT64_MAX;
  uint32_t best_mode = 0;
  const uint32_t* best_histo = tile_histos;
  for (mode = 0; mode < kNumPredModes; ++mode) {
    const uint32_t* const histo_argb = &tile_histos[mode * HISTO_SIZE];
    const int64_t cur_diff = PredictionCostSpatialHistogram(
        accumulated_argb, histo_argb, mode, left_mode, above_mode);

    if (cur_diff < best_diff) {
      best_histo = histo_argb;
      best_diff = cur_diff;
      best_mode = mode;
    }
  }
  // Update the accumulated histogram.
  VP8LAddVectorEq(best_histo, accumulated_argb, HISTO_SIZE);
  modes[tile_y * tiles_per_row + tile_x] = ARGB_BLACK | (best_mode << 8);
  ++pred_histos[best_mode];
}

// Computes the residuals for the different predictors.
// If max_quantization > 1, assumes that near lossless processing will be
// applied, quantizing residuals to multiples of quantization levels up to
// max_quantization (the actual quantization level depends on smoothness near
// the given pixel).
static void ComputeResidualsForTile(
    int width, int height, int tile_x, int tile_y, int min_bits,
    uint32_t update_up_to_index, uint32_t* const all_argb,
    uint32_t* const argb_scratch, const uint32_t* const argb,
    int max_quantization, int exact, int used_subtract_green) {
  const int start_x = tile_x << min_bits;
  const int start_y = tile_y << min_bits;
  const int tile_size = 1 << min_bits;
  const int max_y = GetMin(tile_size, height - start_y);
  const int max_x = GetMin(tile_size, width - start_x);
  // Whether there exist columns just outside the tile.
  const int have_left = (start_x > 0);
  // Position and size of the strip covering the tile and adjacent columns if
  // they exist.
  const int context_start_x = start_x - have_left;
#if (WEBP_NEAR_LOSSLESS == 1)
  const int context_width = max_x + have_left + (max_x < width - start_x);
#endif
  // The width of upper_row and current_row is one pixel larger than image width
  // to allow the top right pixel to point to the leftmost pixel of the next row
  // when at the right edge.
  uint32_t* upper_row = argb_scratch;
  uint32_t* current_row = upper_row + width + 1;
  uint8_t* const max_diffs = (uint8_t*)(current_row + width + 1);
  int mode;
  // Need pointers to be able to swap arrays.
  uint32_t residuals[1 << MAX_TRANSFORM_BITS];
  // Whether each row segment is fully opaque (the rows are re-read for each
  // mode and alpha values are never modified, so compute the flags once).
  uint8_t row_is_opaque[1 << MAX_TRANSFORM_BITS];
  int relative_y;
  assert(max_x <= (1 << MAX_TRANSFORM_BITS));
  assert(max_y <= (1 << MAX_TRANSFORM_BITS));
  for (relative_y = 0; relative_y < max_y; ++relative_y) {
    row_is_opaque[relative_y] = (uint8_t)RowSegmentIsOpaque(
        argb + (start_y + relative_y) * width, start_x, start_x + max_x);
  }
  for (mode = 0; mode < kNumPredModes; ++mode) {
    uint32_t* const histo_argb =
        GetHistoArgb(all_argb, /*subsampling_index=*/0, mode);
    if (start_y > 0) {
      // Read the row above the tile which will become the first upper_row.
      // Include a pixel to the left if it exists; include a pixel to the right
      // in all cases (wrapping to the leftmost pixel of the next row if it does
      // not exist).
      memcpy(current_row + context_start_x,
             argb + (start_y - 1) * width + context_start_x,
             sizeof(*argb) * (max_x + have_left + 1));
    }
    for (relative_y = 0; relative_y < max_y; ++relative_y) {
      const int y = start_y + relative_y;
      uint32_t* tmp = upper_row;
      upper_row = current_row;
      current_row = tmp;
      // Read current_row. Include a pixel to the left if it exists; include a
      // pixel to the right in all cases except at the bottom right corner of
      // the image (wrapping to the leftmost pixel of the next row if it does
      // not exist in the current row).
      memcpy(current_row + context_start_x, argb + y * width + context_start_x,
             sizeof(*argb) * (max_x + have_left + (y + 1 < height)));
#if (WEBP_NEAR_LOSSLESS == 1)
      if (max_quantization > 1 && y >= 1 && y + 1 < height) {
        MaxDiffsForRow(context_width, width, argb + y * width + context_start_x,
                       max_diffs + context_start_x, used_subtract_green);
      }
#endif

      GetResidual(width, height, upper_row, current_row, max_diffs, mode,
                  start_x, start_x + max_x, y, max_quantization, exact,
                  used_subtract_green, row_is_opaque[relative_y], residuals);
      VP8LCollectArgbHistos(residuals, max_x, histo_argb);
      if (update_up_to_index > 0) {
        uint32_t subsampling_index;
        for (subsampling_index = 1; subsampling_index <= update_up_to_index;
             ++subsampling_index) {
          uint32_t* const super_histo =
              GetHistoArgb(all_argb, subsampling_index, mode);
          VP8LCollectArgbHistos(residuals, max_x, super_histo);
        }
      }
    }
  }
}

// Converts pixels of the image to residuals with respect to predictions.
// If max_quantization > 1, applies near lossless processing, quantizing
// residuals to multiples of quantization levels up to max_quantization
// (the actual quantization level depends on smoothness near the given pixel).
static void CopyImageWithPrediction(int width, int height, int bits,
                                    const uint32_t* const modes,
                                    uint32_t* const argb_scratch,
                                    uint32_t* const argb, int low_effort,
                                    int max_quantization, int exact,
                                    int used_subtract_green) {
  const int tiles_per_row = VP8LSubSampleSize(width, bits);
  // The width of upper_row and current_row is one pixel larger than image width
  // to allow the top right pixel to point to the leftmost pixel of the next row
  // when at the right edge.
  uint32_t* upper_row = argb_scratch;
  uint32_t* current_row = upper_row + width + 1;
  uint8_t* current_max_diffs = (uint8_t*)(current_row + width + 1);
#if (WEBP_NEAR_LOSSLESS == 1)
  uint8_t* lower_max_diffs = current_max_diffs + width;
#endif
  int y;

  for (y = 0; y < height; ++y) {
    int x;
    uint32_t* const tmp32 = upper_row;
    upper_row = current_row;
    current_row = tmp32;
    memcpy(current_row, argb + y * width,
           sizeof(*argb) * (width + (y + 1 < height)));

    if (low_effort) {
      PredictBatch(kPredLowEffort, 0, y, width, current_row, upper_row,
                   argb + y * width);
    } else {
#if (WEBP_NEAR_LOSSLESS == 1)
      if (max_quantization > 1) {
        // Compute max_diffs for the lower row now, because that needs the
        // contents of argb for the current row, which we will overwrite with
        // residuals before proceeding with the next row.
        uint8_t* const tmp8 = current_max_diffs;
        current_max_diffs = lower_max_diffs;
        lower_max_diffs = tmp8;
        if (y + 2 < height) {
          MaxDiffsForRow(width, width, argb + (y + 1) * width, lower_max_diffs,
                         used_subtract_green);
        }
      }
#endif
      for (x = 0; x < width;) {
        const int mode =
            (modes[(y >> bits) * tiles_per_row + (x >> bits)] >> 8) & 0xff;
        int x_end = x + (1 << bits);
        if (x_end > width) x_end = width;
        GetResidual(width, height, upper_row, current_row, current_max_diffs,
                    mode, x, x_end, y, max_quantization, exact,
                    used_subtract_green,
                    RowSegmentIsOpaque(current_row, x, x_end),
                    argb + y * width + x);
        x = x_end;
      }
    }
  }
}

// Checks whether 'image' can be subsampled by finding the biggest power of 2
// squares (defined by 'best_bits') of uniform value it is made out of.
void VP8LOptimizeSampling(uint32_t* const image, int full_width,
                          int full_height, int bits, int max_bits,
                          int* best_bits_out) {
  int width = VP8LSubSampleSize(full_width, bits);
  int height = VP8LSubSampleSize(full_height, bits);
  int old_width, x, y, square_size;
  int best_bits = bits;
  *best_bits_out = bits;
  // Check rows first.
  while (best_bits < max_bits) {
    const int new_square_size = 1 << (best_bits + 1 - bits);
    int is_good = 1;
    square_size = 1 << (best_bits - bits);
    for (y = 0; y + square_size < height; y += new_square_size) {
      // Check the first lines of consecutive line groups.
      if (memcmp(&image[y * width], &image[(y + square_size) * width],
                 width * sizeof(*image)) != 0) {
        is_good = 0;
        break;
      }
    }
    if (is_good) {
      ++best_bits;
    } else {
      break;
    }
  }
  if (best_bits == bits) return;

  // Check columns.
  while (best_bits > bits) {
    int is_good = 1;
    square_size = 1 << (best_bits - bits);
    for (y = 0; is_good && y < height; ++y) {
      for (x = 0; is_good && x < width; x += square_size) {
        int i;
        for (i = x + 1; i < GetMin(x + square_size, width); ++i) {
          if (image[y * width + i] != image[y * width + x]) {
            is_good = 0;
            break;
          }
        }
      }
    }
    if (is_good) {
      break;
    }
    --best_bits;
  }
  if (best_bits == bits) return;

  // Subsample the image.
  old_width = width;
  square_size = 1 << (best_bits - bits);
  width = VP8LSubSampleSize(full_width, best_bits);
  height = VP8LSubSampleSize(full_height, best_bits);
  for (y = 0; y < height; ++y) {
    for (x = 0; x < width; ++x) {
      image[y * width + x] = image[square_size * (y * old_width + x)];
    }
  }
  *best_bits_out = best_bits;
}

// Computes the best predictor image.
// Finds the best predictors per tile. Once done, finds the best predictor image
// sampling.
// best_bits is set to 0 in case of error.
// The following requires some glossary:
// - a tile is a square of side 2^min_bits pixels.
// - a super-tile of a tile is a square of side 2^bits pixels with bits in
// [min_bits+1, max_bits].
// - the max-tile of a tile is the square of 2^max_bits pixels containing it.
//   If this max-tile crosses the border of an image, it is cropped.
// - tile, super-tiles and max_tile are aligned on powers of 2 in the original
//   image.
// - coordinates for tile, super-tile, max-tile are respectively named
//   tile_x, super_tile_x, max_tile_x at their bit scale.
// - in the max-tile, a tile has local coordinates (local_tile_x, local_tile_y).
// The tiles are processed in the following zigzag order to complete the
// super-tiles as soon as possible:
//   1  2|  5  6
//   3  4|  7  8
// --------------
//   9 10| 13 14
//  11 12| 15 16
// When computing the residuals for a tile, the histogram of the above
// super-tile is updated. If this super-tile is finished, its histogram is used
// to update the histogram of the next super-tile and so on up to the max-tile.
// State of the Z-order traversal of the smallest tiles used by
// GetBestPredictorsAndSubSampling(): within a max-tile (a square of
// (1 << max_subsampling_index) tiles), the tiles are visited in Z-order so
// that the super-tiles complete as early as possible.
typedef struct {
  uint32_t tiles_per_row, tiles_per_col;
  uint32_t max_subsampling_index;
  uint32_t max_tile_size;               // in smallest-tile units
  uint32_t tile_x, tile_y;              // current tile
  uint32_t local_tile_x, local_tile_y;  // coordinates within the max-tile
  uint32_t max_tile_x, max_tile_y;      // coordinates of the max-tile
  uint32_t num_completed;  // number of super-tiles the current tile completes
} TileTraversal;

// Number of subsampling levels for which the current tile is the last one of
// its super-tile (the smallest tile is at the end of a line/column of the
// image or of a super-tile of size (1 << index)). Purely geometric.
static uint32_t TileTraversalNumCompleted(const TileTraversal* const t) {
  uint32_t idx;
  for (idx = 1; idx <= t->max_subsampling_index; ++idx) {
    if (!((t->tile_x == t->tiles_per_row - 1 ||
           (t->local_tile_x + 1) % (1u << idx) == 0) &&
          (t->tile_y == t->tiles_per_col - 1 ||
           (t->local_tile_y + 1) % (1u << idx) == 0))) {
      break;
    }
  }
  return idx - 1;
}

static void TileTraversalInit(int width, int height, int min_bits,
                              uint32_t max_subsampling_index,
                              TileTraversal* const t) {
  t->tiles_per_row = VP8LSubSampleSize(width, min_bits);
  t->tiles_per_col = VP8LSubSampleSize(height, min_bits);
  t->max_subsampling_index = max_subsampling_index;
  t->max_tile_size = 1u << max_subsampling_index;
  t->tile_x = t->tile_y = 0;
  t->local_tile_x = t->local_tile_y = 0;
  t->max_tile_x = t->max_tile_y = 0;
  t->num_completed = TileTraversalNumCompleted(t);
}

static int TileTraversalDone(const TileTraversal* const t) {
  return (t->tile_y >= t->tiles_per_col);
}

static void TileTraversalNext(TileTraversal* const t) {
  if (t->num_completed == t->max_subsampling_index) {
    // A new max-tile is started.
    if (t->tile_x == t->tiles_per_row - 1) {
      t->max_tile_x = 0;
      ++t->max_tile_y;
    } else {
      ++t->max_tile_x;
    }
    t->local_tile_x = 0;
    t->local_tile_y = 0;
  } else {
    // Proceed with the Z traversal.
    uint32_t coord_x = t->local_tile_x >> t->num_completed;
    uint32_t coord_y = t->local_tile_y >> t->num_completed;
    if (t->tile_x == t->tiles_per_row - 1 && coord_x % 2 == 0) {
      ++coord_y;
    } else {
      if (coord_x % 2 == 0) {
        ++coord_x;
      } else {
        // Z traversal.
        ++coord_y;
        --coord_x;
      }
    }
    t->local_tile_x = coord_x << t->num_completed;
    t->local_tile_y = coord_y << t->num_completed;
  }
  t->tile_x = t->max_tile_x * t->max_tile_size + t->local_tile_x;
  t->tile_y = t->max_tile_y * t->max_tile_size + t->local_tile_y;
  if (!TileTraversalDone(t)) t->num_completed = TileTraversalNumCompleted(t);
}

#ifdef WEBP_USE_THREAD

// Multi-threaded analysis: the per-tile residual histograms (the expensive
// part) are computed by workers into a ring of slots following the traversal
// order; the (strictly ordered) predictor selections are made by the calling
// thread, consuming the slots in order. The histograms it accumulates are
// integer sums of the same per-tile histograms as in the single-threaded
// code, so the selections are identical.

#define PRED_ANALYSIS_MAX_THREADS 8
#define PRED_ANALYSIS_NUM_SLOTS 64   // ring slots (one per-tile histogram set)
#define PRED_ANALYSIS_MIN_TILES 128  // don't bother below that many tiles

typedef struct PredAnalysisCtx PredAnalysisCtx;

typedef struct {
  PredAnalysisCtx* ctx;
  uint32_t* scratch;  // private prediction scratch (2 argb scanlines + bytes)
} PredAnalysisWorker;

struct PredAnalysisCtx {
  int width, height, min_bits;
  uint32_t max_subsampling_index;
  const uint32_t* argb;
  int max_quantization, exact, used_subtract_green;
  uint32_t num_tiles;
  uint32_t* slots;  // PRED_ANALYSIS_NUM_SLOTS histogram sets
  void* monitor;
  // Shared state, protected by the monitor.
  uint32_t next_seq;  // next tile (in traversal order) to be claimed
  uint32_t consumed;  // number of tiles processed by the consumer
  uint32_t produced[PRED_ANALYSIS_NUM_SLOTS];  // tile index + 1, 0 if empty
  int abort;
  int num_workers;
  PredAnalysisWorker workers[PRED_ANALYSIS_MAX_THREADS];
  WebPWorker threads[PRED_ANALYSIS_MAX_THREADS];
};

static uint32_t* PredAnalysisSlot(const PredAnalysisCtx* const ctx,
                                  uint32_t seq) {
  return &ctx->slots[(seq % PRED_ANALYSIS_NUM_SLOTS) *
                     (uint32_t)(kNumPredModes * HISTO_SIZE)];
}

static int PredAnalysisWorkerHook(void* arg1, void* arg2) {
  PredAnalysisWorker* const w = (PredAnalysisWorker*)arg1;
  PredAnalysisCtx* const ctx = w->ctx;
  TileTraversal trav;
  uint32_t trav_seq = 0;
  (void)arg2;
  TileTraversalInit(ctx->width, ctx->height, ctx->min_bits,
                    ctx->max_subsampling_index, &trav);
  while (1) {
    uint32_t seq;
    WebPMonitorLock(ctx->monitor);
    while (!ctx->abort && ctx->next_seq < ctx->num_tiles &&
           ctx->next_seq >= ctx->consumed + PRED_ANALYSIS_NUM_SLOTS) {
      WebPMonitorWait(ctx->monitor);  // ring is full
    }
    if (ctx->abort || ctx->next_seq >= ctx->num_tiles) {
      WebPMonitorUnlock(ctx->monitor);
      break;
    }
    seq = ctx->next_seq++;
    WebPMonitorUnlock(ctx->monitor);

    while (trav_seq < seq) {  // advance the private traversal to 'seq'
      TileTraversalNext(&trav);
      ++trav_seq;
    }
    {
      uint32_t* const slot = PredAnalysisSlot(ctx, seq);
      memset(slot, 0, kNumPredModes * HISTO_SIZE * sizeof(*slot));
      ComputeResidualsForTile(ctx->width, ctx->height, trav.tile_x,
                              trav.tile_y, ctx->min_bits,
                              /*update_up_to_index=*/0, slot, w->scratch,
                              ctx->argb, ctx->max_quantization, ctx->exact,
                              ctx->used_subtract_green);
    }
    WebPMonitorLock(ctx->monitor);
    ctx->produced[seq % PRED_ANALYSIS_NUM_SLOTS] = seq + 1;
    WebPMonitorBroadcast(ctx->monitor);
    WebPMonitorUnlock(ctx->monitor);
  }
  return 1;
}

// Returns 1 on success, 0 if multi-threading could not be set up (the caller
// then uses the single-threaded code).
static int PredAnalysisStart(int width, int height, int min_bits,
                             uint32_t max_subsampling_index,
                             const uint32_t* const argb, int max_quantization,
                             int exact, int used_subtract_green,
                             uint32_t num_tiles, PredAnalysisCtx* const ctx) {
  const WebPWorkerInterface* const winterface = WebPGetWorkerInterface();
  const uint64_t scratch_size =
      (width + 1) * 2 +
      (width * 2 + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  int i;
  memset(ctx, 0, sizeof(*ctx));
  ctx->width = width;
  ctx->height = height;
  ctx->min_bits = min_bits;
  ctx->max_subsampling_index = max_subsampling_index;
  ctx->argb = argb;
  ctx->max_quantization = max_quantization;
  ctx->exact = exact;
  ctx->used_subtract_green = used_subtract_green;
  ctx->num_tiles = num_tiles;
  ctx->num_workers = PRED_ANALYSIS_MAX_THREADS;
  ctx->monitor = WebPMonitorNew();
  ctx->slots = (uint32_t*)WebPSafeMalloc(
      PRED_ANALYSIS_NUM_SLOTS, kNumPredModes * HISTO_SIZE * sizeof(uint32_t));
  if (ctx->monitor == NULL || ctx->slots == NULL) goto Error;
  for (i = 0; i < ctx->num_workers; ++i) {
    ctx->workers[i].ctx = ctx;
    ctx->workers[i].scratch =
        (uint32_t*)WebPSafeMalloc(scratch_size, sizeof(uint32_t));
    if (ctx->workers[i].scratch == NULL) goto Error;
  }
  for (i = 0; i < ctx->num_workers; ++i) {
    WebPWorker* const thread = &ctx->threads[i];
    winterface->Init(thread);
    thread->data1 = &ctx->workers[i];
    thread->data2 = NULL;
    thread->hook = PredAnalysisWorkerHook;
    if (!winterface->Reset(thread)) {
      ctx->num_workers = i;  // only End() the threads reset so far
      goto Error;
    }
    winterface->Launch(thread);
  }
  return 1;

 Error:
  ctx->abort = 1;
  return 0;
}

// Waits for the workers to finish and releases the resources.
static void PredAnalysisEnd(PredAnalysisCtx* const ctx) {
  const WebPWorkerInterface* const winterface = WebPGetWorkerInterface();
  int i;
  if (ctx->monitor != NULL) {
    WebPMonitorLock(ctx->monitor);
    ctx->consumed = ctx->num_tiles;  // unblock any waiting worker
    WebPMonitorBroadcast(ctx->monitor);
    WebPMonitorUnlock(ctx->monitor);
  }
  for (i = 0; i < ctx->num_workers; ++i) {
    winterface->Sync(&ctx->threads[i]);
    winterface->End(&ctx->threads[i]);
  }
  for (i = 0; i < PRED_ANALYSIS_MAX_THREADS; ++i) {
    WebPSafeFree(ctx->workers[i].scratch);
  }
  WebPSafeFree(ctx->slots);
  WebPMonitorDelete(ctx->monitor);
}

#endif  // WEBP_USE_THREAD

static void GetBestPredictorsAndSubSampling(
    int width, int height, const int min_bits, const int max_bits,
    uint32_t* const argb_scratch, const uint32_t* const argb,
    int max_quantization, int exact, int used_subtract_green, int use_threads,
    const WebPPicture* const pic, int percent_range, int* const percent,
    uint32_t** const all_modes, int* best_bits, uint32_t** best_mode) {
  const uint32_t tiles_per_row = VP8LSubSampleSize(width, min_bits);
  const uint32_t tiles_per_col = VP8LSubSampleSize(height, min_bits);
  int64_t best_cost;
  uint32_t subsampling_index;
  const uint32_t max_subsampling_index = max_bits - min_bits;
  // Compute the needed memory size for residual histograms, accumulated
  // residual histograms and predictor histograms.
  const int num_argb = (max_subsampling_index + 1) * kNumPredModes * HISTO_SIZE;
  const int num_accumulated_rgb = (max_subsampling_index + 1) * HISTO_SIZE;
  const int num_predictors = (max_subsampling_index + 1) * kNumPredModes;
  uint32_t* const raw_data = (uint32_t*)WebPSafeCalloc(
      num_argb + num_accumulated_rgb + num_predictors, sizeof(uint32_t));
  uint32_t* const all_argb = raw_data;
  uint32_t* const all_accumulated_argb = all_argb + num_argb;
  uint32_t* const all_pred_histos = all_accumulated_argb + num_accumulated_rgb;
  int percent_start = *percent;
  // When using the residuals of a tile for its super-tiles, you can either:
  // - use each residual to update the histogram of the super-tile, with a cost
  //   of 4 * (1<<n)^2 increment operations (4 for the number of channels, and
  //   (1<<n)^2 for the number of pixels in the tile)
  // - use the histogram of the tile to update the histogram of the super-tile,
  //   with a cost of HISTO_SIZE (1024)
  // The first method is therefore faster until n==4. 'update_up_to_index'
  // defines the maximum subsampling_index for which the residuals should be
  // individually added to the super-tile histogram.
  const uint32_t update_up_to_index =
      GetMax(GetMin(4, max_bits), min_bits) - min_bits;
  TileTraversal trav;
#ifdef WEBP_USE_THREAD
  PredAnalysisCtx* mt = NULL;
  uint32_t mt_seq = 0;
#endif

  *best_bits = 0;
  *best_mode = NULL;
  if (raw_data == NULL) return;

  TileTraversalInit(width, height, min_bits, max_subsampling_index, &trav);

#ifdef WEBP_USE_THREAD
  if (use_threads && tiles_per_row * tiles_per_col >= PRED_ANALYSIS_MIN_TILES) {
    mt = (PredAnalysisCtx*)WebPSafeMalloc(1ULL, sizeof(*mt));
    if (mt != NULL &&
        !PredAnalysisStart(width, height, min_bits, max_subsampling_index,
                           argb, max_quantization, exact, used_subtract_green,
                           tiles_per_row * tiles_per_col, mt)) {
      PredAnalysisEnd(mt);  // could not start -> use the single-threaded code
      WebPSafeFree(mt);
      mt = NULL;
    }
  }
#else
  (void)use_threads;
#endif

  while (!TileTraversalDone(&trav)) {
    const uint32_t tile_x = trav.tile_x;
    const uint32_t tile_y = trav.tile_y;
    const uint32_t num_completed = trav.num_completed;
    const uint32_t* tile_histos;

#ifdef WEBP_USE_THREAD
    if (mt != NULL) {
      // Wait for the tile histograms computed by the workers.
      uint32_t* const slot = PredAnalysisSlot(mt, mt_seq);
      WebPMonitorLock(mt->monitor);
      while (mt->produced[mt_seq % PRED_ANALYSIS_NUM_SLOTS] != mt_seq + 1) {
        WebPMonitorWait(mt->monitor);
      }
      WebPMonitorUnlock(mt->monitor);
      tile_histos = slot;
      // Update the partial super-tile histograms: the first level gets the
      // tile histograms, the next ones get the (then complete) histograms of
      // the level below. The resulting sums are the same as in the
      // single-threaded code.
      for (subsampling_index = 1; subsampling_index <= max_subsampling_index;
           ++subsampling_index) {
        const uint32_t* const src =
            (subsampling_index == 1)
                ? slot
                : GetHistoArgbConst(all_argb, subsampling_index - 1, 0);
        VP8LAddVectorEq(src, GetHistoArgb(all_argb, subsampling_index, 0),
                        HISTO_SIZE * kNumPredModes);
        if (subsampling_index > num_completed) break;
      }
    } else
#endif
    {
      ComputeResidualsForTile(width, height, tile_x, tile_y, min_bits,
                              update_up_to_index, all_argb, argb_scratch, argb,
                              max_quantization, exact, used_subtract_green);
      tile_histos = GetHistoArgbConst(all_argb, 0, 0);
      // Update the partial super-tile histograms that are not updated
      // residual-per-residual by ComputeResidualsForTile().
      for (subsampling_index = 1; subsampling_index <= max_subsampling_index;
           ++subsampling_index) {
        if (subsampling_index > update_up_to_index) {
          VP8LAddVectorEq(
              GetHistoArgbConst(all_argb, subsampling_index - 1, 0),
              GetHistoArgb(all_argb, subsampling_index, 0),
              HISTO_SIZE * kNumPredModes);
        }
        if (subsampling_index > num_completed) break;
      }
    }

    // Find the best predictors of the tile and of all the super-tiles it
    // completes.
    GetBestPredictorForTile(tile_histos, 0, tile_x, tile_y, tiles_per_row,
                            all_accumulated_argb, all_modes, all_pred_histos);
    for (subsampling_index = 1; subsampling_index <= num_completed;
         ++subsampling_index) {
      GetBestPredictorForTile(
          GetHistoArgbConst(all_argb, subsampling_index, 0), subsampling_index,
          tile_x >> subsampling_index, tile_y >> subsampling_index,
          VP8LSubSampleSize(width, min_bits + subsampling_index),
          all_accumulated_argb, all_modes, all_pred_histos);
    }

    // Reset all the histograms belonging to finished tiles.
#ifdef WEBP_USE_THREAD
    if (mt != NULL) {
      if (num_completed > 0) {
        memset(GetHistoArgb(all_argb, 1, 0), 0,
               HISTO_SIZE * kNumPredModes * num_completed *
                   sizeof(*all_argb));
      }
      WebPMonitorLock(mt->monitor);
      mt->produced[mt_seq % PRED_ANALYSIS_NUM_SLOTS] = 0;
      mt->consumed = ++mt_seq;
      WebPMonitorBroadcast(mt->monitor);
      WebPMonitorUnlock(mt->monitor);
    } else
#endif
    {
      memset(all_argb, 0,
             HISTO_SIZE * kNumPredModes * (num_completed + 1) *
                 sizeof(*all_argb));
    }

    TileTraversalNext(&trav);
    if (trav.tile_x == 0 &&
        !WebPReportProgress(
            pic, percent_start + percent_range * trav.tile_y / tiles_per_col,
            percent)) {
#ifdef WEBP_USE_THREAD
      if (mt != NULL) {
        WebPMonitorLock(mt->monitor);
        mt->abort = 1;
        WebPMonitorBroadcast(mt->monitor);
        WebPMonitorUnlock(mt->monitor);
        PredAnalysisEnd(mt);
        WebPSafeFree(mt);
      }
#endif
      WebPSafeFree(raw_data);
      return;
    }
  }

#ifdef WEBP_USE_THREAD
  if (mt != NULL) {
    PredAnalysisEnd(mt);
    WebPSafeFree(mt);
  }
#endif

  // Figure out the best sampling.
  best_cost = WEBP_INT64_MAX;
  for (subsampling_index = 0; subsampling_index <= max_subsampling_index;
       ++subsampling_index) {
    int plane;
    const uint32_t* const accumulated =
        GetAccumulatedHisto(all_accumulated_argb, subsampling_index);
    int64_t cost = VP8LShannonEntropy(
        &all_pred_histos[subsampling_index * kNumPredModes], kNumPredModes);
    for (plane = 0; plane < 4; ++plane) {
      cost += VP8LShannonEntropy(&accumulated[plane * 256], 256);
    }
    if (cost < best_cost) {
      best_cost = cost;
      *best_bits = min_bits + subsampling_index;
      *best_mode = all_modes[subsampling_index];
    }
  }

  WebPSafeFree(raw_data);

  VP8LOptimizeSampling(*best_mode, width, height, *best_bits,
                       MAX_TRANSFORM_BITS, best_bits);
}

// Finds the best predictor for each tile, and converts the image to residuals
// with respect to predictions. If near_lossless_quality < 100, applies
// near lossless processing, shaving off more bits of residuals for lower
// qualities.
int VP8LResidualImage(int width, int height, int min_bits, int max_bits,
                      int low_effort, uint32_t* const argb,
                      uint32_t* const argb_scratch, uint32_t* const image,
                      int near_lossless_quality, int exact,
                      int used_subtract_green, int use_threads,
                      const WebPPicture* const pic, int percent_range,
                      int* const percent, int* const best_bits) {
  int percent_start = *percent;
  const int max_quantization = 1 << VP8LNearLosslessBits(near_lossless_quality);
  if (low_effort) {
    const int tiles_per_row = VP8LSubSampleSize(width, max_bits);
    const int tiles_per_col = VP8LSubSampleSize(height, max_bits);
    int i;
    for (i = 0; i < tiles_per_row * tiles_per_col; ++i) {
      image[i] = ARGB_BLACK | (kPredLowEffort << 8);
    }
    *best_bits = max_bits;
  } else {
    // Allocate data to try all samplings from min_bits to max_bits.
    int bits;
    uint32_t sum_num_pixels = 0u;
    uint32_t *modes_raw, *best_mode;
    uint32_t* modes[MAX_TRANSFORM_BITS + 1];
    uint32_t num_pixels[MAX_TRANSFORM_BITS + 1];
    for (bits = min_bits; bits <= max_bits; ++bits) {
      const int tiles_per_row = VP8LSubSampleSize(width, bits);
      const int tiles_per_col = VP8LSubSampleSize(height, bits);
      num_pixels[bits] = tiles_per_row * tiles_per_col;
      sum_num_pixels += num_pixels[bits];
    }
    modes_raw = (uint32_t*)WebPSafeMalloc(sum_num_pixels, sizeof(*modes_raw));
    if (modes_raw == NULL) return 0;
    // Have modes point to the right global memory modes_raw.
    modes[min_bits] = modes_raw;
    for (bits = min_bits + 1; bits <= max_bits; ++bits) {
      modes[bits] = modes[bits - 1] + num_pixels[bits - 1];
    }
    // Find the best sampling.
    GetBestPredictorsAndSubSampling(
        width, height, min_bits, max_bits, argb_scratch, argb, max_quantization,
        exact, used_subtract_green, use_threads, pic, percent_range, percent,
        &modes[min_bits], best_bits, &best_mode);
    if (*best_bits == 0) {
      WebPSafeFree(modes_raw);
      return 0;
    }
    // Keep the best predictor image.
    memcpy(image, best_mode,
           VP8LSubSampleSize(width, *best_bits) *
               VP8LSubSampleSize(height, *best_bits) * sizeof(*image));
    WebPSafeFree(modes_raw);
  }

  CopyImageWithPrediction(width, height, *best_bits, image, argb_scratch, argb,
                          low_effort, max_quantization, exact,
                          used_subtract_green);
  return WebPReportProgress(pic, percent_start + percent_range, percent);
}

//------------------------------------------------------------------------------
// Color transform functions.

static WEBP_INLINE void MultipliersClear(VP8LMultipliers* const m) {
  m->green_to_red = 0;
  m->green_to_blue = 0;
  m->red_to_blue = 0;
}

static WEBP_INLINE void ColorCodeToMultipliers(uint32_t color_code,
                                               VP8LMultipliers* const m) {
  m->green_to_red = (color_code >> 0) & 0xff;
  m->green_to_blue = (color_code >> 8) & 0xff;
  m->red_to_blue = (color_code >> 16) & 0xff;
}

static WEBP_INLINE uint32_t
MultipliersToColorCode(const VP8LMultipliers* const m) {
  return 0xff000000u | ((uint32_t)(m->red_to_blue) << 16) |
         ((uint32_t)(m->green_to_blue) << 8) | m->green_to_red;
}

static int64_t PredictionCostCrossColor(const uint32_t accumulated[256],
                                        const uint32_t counts[256]) {
  // Favor low entropy, locally and globally.
  // Favor small absolute values for PredictionCostSpatial
  static const uint64_t kExpValue = 240;
  return (int64_t)VP8LCombinedShannonEntropy(counts, accumulated) +
         PredictionCostBias(counts, 3, kExpValue);
}

static int64_t GetPredictionCostCrossColorRed(
    const uint32_t* argb, int stride, int tile_width, int tile_height,
    VP8LMultipliers prev_x, VP8LMultipliers prev_y, int green_to_red,
    const uint32_t accumulated_red_histo[256]) {
  uint32_t histo[256] = {0};
  int64_t cur_diff;

  VP8LCollectColorRedTransforms(argb, stride, tile_width, tile_height,
                                green_to_red, histo);

  cur_diff = PredictionCostCrossColor(accumulated_red_histo, histo);
  if ((uint8_t)green_to_red == prev_x.green_to_red) {
    // favor keeping the areas locally similar
    cur_diff -= 3ll << LOG_2_PRECISION_BITS;
  }
  if ((uint8_t)green_to_red == prev_y.green_to_red) {
    // favor keeping the areas locally similar
    cur_diff -= 3ll << LOG_2_PRECISION_BITS;
  }
  if (green_to_red == 0) {
    cur_diff -= 3ll << LOG_2_PRECISION_BITS;
  }
  return cur_diff;
}

static void GetBestGreenToRed(const uint32_t* argb, int stride, int tile_width,
                              int tile_height, VP8LMultipliers prev_x,
                              VP8LMultipliers prev_y, int quality,
                              const uint32_t accumulated_red_histo[256],
                              VP8LMultipliers* const best_tx) {
  const int kMaxIters = 4 + ((7 * quality) >> 8);  // in range [4..6]
  int green_to_red_best = 0;
  int iter, offset;
  int64_t best_diff = GetPredictionCostCrossColorRed(
      argb, stride, tile_width, tile_height, prev_x, prev_y, green_to_red_best,
      accumulated_red_histo);
  for (iter = 0; iter < kMaxIters; ++iter) {
    // ColorTransformDelta is a 3.5 bit fixed point, so 32 is equal to
    // one in color computation. Having initial delta here as 1 is sufficient
    // to explore the range of (-2, 2).
    const int delta = 32 >> iter;
    // Try a negative and a positive delta from the best known value.
    for (offset = -delta; offset <= delta; offset += 2 * delta) {
      const int green_to_red_cur = offset + green_to_red_best;
      const int64_t cur_diff = GetPredictionCostCrossColorRed(
          argb, stride, tile_width, tile_height, prev_x, prev_y,
          green_to_red_cur, accumulated_red_histo);
      if (cur_diff < best_diff) {
        best_diff = cur_diff;
        green_to_red_best = green_to_red_cur;
      }
    }
  }
  best_tx->green_to_red = (green_to_red_best & 0xff);
}

static int64_t GetPredictionCostCrossColorBlue(
    const uint32_t* argb, int stride, int tile_width, int tile_height,
    VP8LMultipliers prev_x, VP8LMultipliers prev_y, int green_to_blue,
    int red_to_blue, const uint32_t accumulated_blue_histo[256]) {
  uint32_t histo[256] = {0};
  int64_t cur_diff;

  VP8LCollectColorBlueTransforms(argb, stride, tile_width, tile_height,
                                 green_to_blue, red_to_blue, histo);

  cur_diff = PredictionCostCrossColor(accumulated_blue_histo, histo);
  if ((uint8_t)green_to_blue == prev_x.green_to_blue) {
    // favor keeping the areas locally similar
    cur_diff -= 3ll << LOG_2_PRECISION_BITS;
  }
  if ((uint8_t)green_to_blue == prev_y.green_to_blue) {
    // favor keeping the areas locally similar
    cur_diff -= 3ll << LOG_2_PRECISION_BITS;
  }
  if ((uint8_t)red_to_blue == prev_x.red_to_blue) {
    // favor keeping the areas locally similar
    cur_diff -= 3ll << LOG_2_PRECISION_BITS;
  }
  if ((uint8_t)red_to_blue == prev_y.red_to_blue) {
    // favor keeping the areas locally similar
    cur_diff -= 3ll << LOG_2_PRECISION_BITS;
  }
  if (green_to_blue == 0) {
    cur_diff -= 3ll << LOG_2_PRECISION_BITS;
  }
  if (red_to_blue == 0) {
    cur_diff -= 3ll << LOG_2_PRECISION_BITS;
  }
  return cur_diff;
}

#define kGreenRedToBlueNumAxis 8
#define kGreenRedToBlueMaxIters 7
static void GetBestGreenRedToBlue(const uint32_t* argb, int stride,
                                  int tile_width, int tile_height,
                                  VP8LMultipliers prev_x,
                                  VP8LMultipliers prev_y, int quality,
                                  const uint32_t accumulated_blue_histo[256],
                                  VP8LMultipliers* const best_tx) {
  const int8_t offset[kGreenRedToBlueNumAxis][2] = {
      {0, -1}, {0, 1}, {-1, 0}, {1, 0}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
  const int8_t delta_lut[kGreenRedToBlueMaxIters] = {16, 16, 8, 4, 2, 2, 2};
  // Only axis aligned diffs for lower quality.
  const int iters = (quality < 25)   ? 1
                    : (quality > 50) ? kGreenRedToBlueMaxIters
                                     : 4;
  int green_to_blue_best = 0;
  int red_to_blue_best = 0;
  int iter;
  // Initial value at origin:
  int64_t best_diff = GetPredictionCostCrossColorBlue(
      argb, stride, tile_width, tile_height, prev_x, prev_y, green_to_blue_best,
      red_to_blue_best, accumulated_blue_histo);
  for (iter = 0; iter < iters; ++iter) {
    const int delta = delta_lut[iter];
    int axis;
    for (axis = 0; axis < kGreenRedToBlueNumAxis; ++axis) {
      const int green_to_blue_cur =
          offset[axis][0] * delta + green_to_blue_best;
      const int red_to_blue_cur = offset[axis][1] * delta + red_to_blue_best;
      const int64_t cur_diff = GetPredictionCostCrossColorBlue(
          argb, stride, tile_width, tile_height, prev_x, prev_y,
          green_to_blue_cur, red_to_blue_cur, accumulated_blue_histo);
      if (cur_diff < best_diff) {
        best_diff = cur_diff;
        green_to_blue_best = green_to_blue_cur;
        red_to_blue_best = red_to_blue_cur;
      }
    }
    if (delta == 2 && green_to_blue_best == 0 && red_to_blue_best == 0) {
      // Further iterations would not help.
      break;  // out of iter-loop.
    }
  }
  best_tx->green_to_blue = green_to_blue_best & 0xff;
  best_tx->red_to_blue = red_to_blue_best & 0xff;
}
#undef kGreenRedToBlueMaxIters
#undef kGreenRedToBlueNumAxis

static VP8LMultipliers GetBestColorTransformForTile(
    int tile_x, int tile_y, int bits, VP8LMultipliers prev_x,
    VP8LMultipliers prev_y, int quality, int xsize, int ysize,
    const uint32_t accumulated_red_histo[256],
    const uint32_t accumulated_blue_histo[256], const uint32_t* const argb) {
  const int max_tile_size = 1 << bits;
  const int tile_y_offset = tile_y * max_tile_size;
  const int tile_x_offset = tile_x * max_tile_size;
  const int all_x_max = GetMin(tile_x_offset + max_tile_size, xsize);
  const int all_y_max = GetMin(tile_y_offset + max_tile_size, ysize);
  const int tile_width = all_x_max - tile_x_offset;
  const int tile_height = all_y_max - tile_y_offset;
  const uint32_t* const tile_argb =
      argb + tile_y_offset * xsize + tile_x_offset;
  VP8LMultipliers best_tx;
  MultipliersClear(&best_tx);

  GetBestGreenToRed(tile_argb, xsize, tile_width, tile_height, prev_x, prev_y,
                    quality, accumulated_red_histo, &best_tx);
  GetBestGreenRedToBlue(tile_argb, xsize, tile_width, tile_height, prev_x,
                        prev_y, quality, accumulated_blue_histo, &best_tx);
  return best_tx;
}

static void CopyTileWithColorTransform(int xsize, int ysize, int tile_x,
                                       int tile_y, int max_tile_size,
                                       VP8LMultipliers color_transform,
                                       uint32_t* argb) {
  const int xscan = GetMin(max_tile_size, xsize - tile_x);
  int yscan = GetMin(max_tile_size, ysize - tile_y);
  argb += tile_y * xsize + tile_x;
  while (yscan-- > 0) {
    VP8LTransformColor(&color_transform, argb, xscan);
    argb += xsize;
  }
}

// The accumulated red/blue histograms guiding the multiplier search advance
// in phases of CROSS_COLOR_PHASE_ROWS tile rows: the tiles of a phase are
// evaluated against the histograms accumulated over the previous phases, and
// the contributions of the phase are merged at the phase boundary. The merge
// sums are integers, so they do not depend on the order in which the tiles
// were processed and the multi-threaded loop below is bit-exact with the
// single-threaded one.
#define CROSS_COLOR_PHASE_ROWS 4

typedef struct {
  int width, height, bits, quality;
  uint32_t* argb;
  uint32_t* image;
  int tile_xsize, tile_ysize;
  // Snapshot of the accumulated histograms (read-only within a phase).
  uint32_t accumulated_red_histo[256];
  uint32_t accumulated_blue_histo[256];
} CrossColorCtx;

// Processes one tile: finds the best multipliers (using the phase snapshot
// and the left/above neighbor multipliers), stores them, transforms the tile
// pixels and gathers the histogram contributions into 'delta_red'/'delta_blue'
// ('prev_x' is the left neighbor's multipliers, updated to the current ones).
static void CrossColorProcessTile(const CrossColorCtx* const ctx, int tile_x,
                                  int tile_y, VP8LMultipliers* const prev_x,
                                  uint32_t delta_red[256],
                                  uint32_t delta_blue[256]) {
  const int width = ctx->width;
  const int max_tile_size = 1 << ctx->bits;
  const int tile_x_offset = tile_x * max_tile_size;
  const int tile_y_offset = tile_y * max_tile_size;
  const int all_x_max = GetMin(tile_x_offset + max_tile_size, width);
  const int all_y_max = GetMin(tile_y_offset + max_tile_size, ctx->height);
  const int offset = tile_y * ctx->tile_xsize + tile_x;
  uint32_t* const argb = ctx->argb;
  VP8LMultipliers prev_y;
  int y;
  MultipliersClear(&prev_y);
  if (tile_y != 0) {
    ColorCodeToMultipliers(ctx->image[offset - ctx->tile_xsize], &prev_y);
  }
  *prev_x = GetBestColorTransformForTile(
      tile_x, tile_y, ctx->bits, *prev_x, prev_y, ctx->quality, width,
      ctx->height, ctx->accumulated_red_histo, ctx->accumulated_blue_histo,
      argb);
  ctx->image[offset] = MultipliersToColorCode(prev_x);
  CopyTileWithColorTransform(width, ctx->height, tile_x_offset, tile_y_offset,
                             max_tile_size, *prev_x, argb);

  // Gather the histogram contributions of the (transformed) tile.
  for (y = tile_y_offset; y < all_y_max; ++y) {
    int ix = y * width + tile_x_offset;
    const int ix_end = ix + all_x_max - tile_x_offset;
    for (; ix < ix_end; ++ix) {
      const uint32_t pix = argb[ix];
      if (ix >= 2 && pix == argb[ix - 2] && pix == argb[ix - 1]) {
        continue;  // repeated pixels are handled by backward references
      }
      if (ix >= width + 2 && argb[ix - 2] == argb[ix - width - 2] &&
          argb[ix - 1] == argb[ix - width - 1] && pix == argb[ix - width]) {
        continue;  // repeated pixels are handled by backward references
      }
      ++delta_red[(pix >> 16) & 0xff];
      ++delta_blue[(pix >> 0) & 0xff];
    }
  }
}

#ifdef WEBP_USE_THREAD

// Multi-threaded cross-color analysis: within a phase, the tile rows are
// processed by several workers in a wavefront pattern (a tile needs the
// multipliers of its left and above neighbors, and the transformed pixels of
// the row above one column ahead). Each worker accumulates its histogram
// contributions privately; they are merged at the phase boundary.

#define CROSS_COLOR_MAX_THREADS 8
#define CROSS_COLOR_SYNC_RANGE 8  // publish progress every that many tiles

typedef struct CrossColorMTCtx CrossColorMTCtx;

typedef struct {
  CrossColorMTCtx* ctx;
  uint32_t delta_red[256];
  uint32_t delta_blue[256];
} CrossColorWorker;

struct CrossColorMTCtx {
  CrossColorCtx* base;
  void* monitor;
  int* row_progress;  // per tile row, number of processed tiles
  int next_row;       // next row to be claimed
  int phase_end;      // rows in [0, phase_end) can be processed
  int rows_done;      // fully processed rows
  int abort;
  int num_workers;
  CrossColorWorker workers[CROSS_COLOR_MAX_THREADS];
  WebPWorker threads[CROSS_COLOR_MAX_THREADS - 1];
};

// Returns 0 if the processing was aborted.
static int CrossColorProcessRow(CrossColorWorker* const w, int tile_y) {
  CrossColorMTCtx* const ctx = w->ctx;
  const int tile_xsize = ctx->base->tile_xsize;
  VP8LMultipliers prev_x;
  int tile_x;
  MultipliersClear(&prev_x);
  for (tile_x = 0; tile_x < tile_xsize; ++tile_x) {
    if (tile_y > 0) {  // wait for the wavefront dependency
      const int needed = tile_x + 1;
      int aborted;
      WebPMonitorLock(ctx->monitor);
      while (!ctx->abort && ctx->row_progress[tile_y - 1] < needed) {
        WebPMonitorWait(ctx->monitor);
      }
      aborted = ctx->abort;
      WebPMonitorUnlock(ctx->monitor);
      if (aborted) return 0;
    }
    CrossColorProcessTile(ctx->base, tile_x, tile_y, &prev_x, w->delta_red,
                          w->delta_blue);
    if ((tile_x + 1) % CROSS_COLOR_SYNC_RANGE == 0 ||
        (tile_x + 1) == tile_xsize) {
      WebPMonitorLock(ctx->monitor);
      ctx->row_progress[tile_y] = tile_x + 1;
      if ((tile_x + 1) == tile_xsize) ++ctx->rows_done;
      WebPMonitorBroadcast(ctx->monitor);
      WebPMonitorUnlock(ctx->monitor);
    }
  }
  return 1;
}

static void CrossColorWorkLoop(CrossColorWorker* const w) {
  CrossColorMTCtx* const ctx = w->ctx;
  const int tile_ysize = ctx->base->tile_ysize;
  while (1) {
    int tile_y = -1;
    WebPMonitorLock(ctx->monitor);
    while (!ctx->abort && ctx->next_row >= ctx->phase_end &&
           ctx->phase_end < tile_ysize) {
      WebPMonitorWait(ctx->monitor);  // wait for the next phase
    }
    if (!ctx->abort && ctx->next_row < ctx->phase_end) {
      tile_y = ctx->next_row++;
    }
    WebPMonitorUnlock(ctx->monitor);
    if (tile_y < 0) break;  // aborted or no more rows
    if (!CrossColorProcessRow(w, tile_y)) break;
  }
  return;
}

static int CrossColorWorkerHook(void* arg1, void* arg2) {
  (void)arg2;
  CrossColorWorkLoop((CrossColorWorker*)arg1);
  return 1;
}

// Returns 1 on success, 0 on set-up failure (the caller then uses the
// single-threaded code).
static int CrossColorMT(CrossColorCtx* const base, const WebPPicture* const pic,
                        int percent_range, int* const percent,
                        int* const ok) {
  const WebPWorkerInterface* const winterface = WebPGetWorkerInterface();
  const int tile_ysize = base->tile_ysize;
  int percent_start = *percent;
  CrossColorMTCtx* ctx = NULL;
  int rows_replayed = 0;
  int i, s;

  *ok = 1;
  ctx = (CrossColorMTCtx*)WebPSafeCalloc(1ULL, sizeof(*ctx));
  if (ctx == NULL) return 0;
  ctx->base = base;
  ctx->monitor = WebPMonitorNew();
  ctx->row_progress = (int*)WebPSafeCalloc(tile_ysize, sizeof(int));
  ctx->num_workers = CROSS_COLOR_MAX_THREADS;
  if (ctx->num_workers > (tile_ysize + 1) / 2) {
    ctx->num_workers = (tile_ysize + 1) / 2;
  }
  if (ctx->monitor == NULL || ctx->row_progress == NULL) goto Error;
  for (i = 0; i < ctx->num_workers; ++i) {
    ctx->workers[i].ctx = ctx;
  }
  for (i = 1; i < ctx->num_workers; ++i) {
    WebPWorker* const thread = &ctx->threads[i - 1];
    winterface->Init(thread);
    thread->data1 = &ctx->workers[i];
    thread->data2 = NULL;
    thread->hook = CrossColorWorkerHook;
    if (!winterface->Reset(thread)) {
      ctx->num_workers = i;  // only End() the threads reset so far
      goto Error;
    }
  }

  ctx->phase_end = (CROSS_COLOR_PHASE_ROWS < tile_ysize)
                       ? CROSS_COLOR_PHASE_ROWS
                       : tile_ysize;
  for (i = 1; i < ctx->num_workers; ++i) {
    winterface->Launch(&ctx->threads[i - 1]);
  }
  while (1) {
    int tile_y = -1;
    WebPMonitorLock(ctx->monitor);
    while (!ctx->abort && ctx->next_row >= ctx->phase_end &&
           ctx->rows_done < ctx->phase_end) {
      WebPMonitorWait(ctx->monitor);  // wait for the phase to drain
    }
    if (!ctx->abort && ctx->next_row < ctx->phase_end) {
      tile_y = ctx->next_row++;
    }
    WebPMonitorUnlock(ctx->monitor);
    if (ctx->abort) break;
    if (tile_y >= 0) {  // the calling thread processes rows too
      CrossColorProcessRow(&ctx->workers[0], tile_y);
      continue;
    }
    // The phase is complete: merge the contributions into the snapshot.
    // The sums are integers: the merge order does not matter.
    for (i = 0; i < ctx->num_workers; ++i) {
      CrossColorWorker* const w = &ctx->workers[i];
      for (s = 0; s < 256; ++s) {
        base->accumulated_red_histo[s] += w->delta_red[s];
        base->accumulated_blue_histo[s] += w->delta_blue[s];
      }
      memset(w->delta_red, 0, sizeof(w->delta_red));
      memset(w->delta_blue, 0, sizeof(w->delta_blue));
    }
    rows_replayed = ctx->phase_end;
    if (ctx->phase_end == tile_ysize) break;  // done
    if (!WebPReportProgress(
            pic, percent_start + percent_range * rows_replayed / tile_ysize,
            percent)) {
      *ok = 0;  // user abort
      WebPMonitorLock(ctx->monitor);
      ctx->abort = 1;
      WebPMonitorBroadcast(ctx->monitor);
      WebPMonitorUnlock(ctx->monitor);
      break;
    }
    WebPMonitorLock(ctx->monitor);
    ctx->phase_end += CROSS_COLOR_PHASE_ROWS;
    if (ctx->phase_end > tile_ysize) ctx->phase_end = tile_ysize;
    WebPMonitorBroadcast(ctx->monitor);
    WebPMonitorUnlock(ctx->monitor);
  }

  for (i = 1; i < ctx->num_workers; ++i) {
    winterface->Sync(&ctx->threads[i - 1]);
    winterface->End(&ctx->threads[i - 1]);
  }
  WebPSafeFree(ctx->row_progress);
  WebPMonitorDelete(ctx->monitor);
  WebPSafeFree(ctx);
  return 1;

 Error:
  for (i = 1; i < ctx->num_workers; ++i) winterface->End(&ctx->threads[i - 1]);
  WebPSafeFree(ctx->row_progress);
  WebPMonitorDelete(ctx->monitor);
  WebPSafeFree(ctx);
  return 0;
}

#endif  // WEBP_USE_THREAD

int VP8LColorSpaceTransform(int width, int height, int bits, int quality,
                            uint32_t* const argb, uint32_t* image,
                            int use_threads, const WebPPicture* const pic,
                            int percent_range, int* const percent,
                            int* const best_bits) {
  const int tile_xsize = VP8LSubSampleSize(width, bits);
  const int tile_ysize = VP8LSubSampleSize(height, bits);
  int percent_start = *percent;
  CrossColorCtx ctx;
  int tile_x, tile_y;

  memset(&ctx, 0, sizeof(ctx));
  ctx.width = width;
  ctx.height = height;
  ctx.bits = bits;
  ctx.quality = quality;
  ctx.argb = argb;
  ctx.image = image;
  ctx.tile_xsize = tile_xsize;
  ctx.tile_ysize = tile_ysize;

#ifdef WEBP_USE_THREAD
  if (use_threads && tile_ysize >= 2 * CROSS_COLOR_PHASE_ROWS &&
      tile_xsize * tile_ysize >= 128) {
    int ok;
    if (CrossColorMT(&ctx, pic, percent_range, percent, &ok)) {
      if (!ok) return 0;
      VP8LOptimizeSampling(image, width, height, bits, MAX_TRANSFORM_BITS,
                           best_bits);
      return 1;
    }
    // Multi-threading could not be set up: use the single-threaded code.
  }
#else
  (void)use_threads;
#endif

  for (tile_y = 0; tile_y < tile_ysize; tile_y += CROSS_COLOR_PHASE_ROWS) {
    const int phase_end = GetMin(tile_y + CROSS_COLOR_PHASE_ROWS, tile_ysize);
    uint32_t delta_red[256] = {0};
    uint32_t delta_blue[256] = {0};
    int y;
    for (y = tile_y; y < phase_end; ++y) {
      VP8LMultipliers prev_x;
      MultipliersClear(&prev_x);
      for (tile_x = 0; tile_x < tile_xsize; ++tile_x) {
        CrossColorProcessTile(&ctx, tile_x, y, &prev_x, delta_red, delta_blue);
      }
    }
    {
      int s;
      for (s = 0; s < 256; ++s) {
        ctx.accumulated_red_histo[s] += delta_red[s];
        ctx.accumulated_blue_histo[s] += delta_blue[s];
      }
    }
    if (!WebPReportProgress(
            pic, percent_start + percent_range * phase_end / tile_ysize,
            percent)) {
      return 0;
    }
  }
  VP8LOptimizeSampling(image, width, height, bits, MAX_TRANSFORM_BITS,
                       best_bits);
  return 1;
}
