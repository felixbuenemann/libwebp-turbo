// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
//   frame coding and analysis
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "src/dec/common_dec.h"
#include "src/dsp/dsp.h"
#include "src/enc/cost_enc.h"
#include "src/enc/vp8i_enc.h"
#include "src/utils/bit_writer_utils.h"
#include "src/webp/encode.h"
#include "src/webp/format_constants.h"  // RIFF constants
#include "src/webp/types.h"

#define SEGMENT_VISU 0
#define DEBUG_SEARCH 0  // useful to track search convergence

//------------------------------------------------------------------------------
// multi-pass convergence

#define HEADER_SIZE_ESTIMATE \
  (RIFF_HEADER_SIZE + CHUNK_HEADER_SIZE + VP8_FRAME_HEADER_SIZE)
#define DQ_LIMIT 0.4  // convergence is considered reached if dq < DQ_LIMIT
// we allow 2k of extra head-room in PARTITION0 limit.
#define PARTITION0_SIZE_LIMIT ((VP8_MAX_PARTITION0_SIZE - 2048ULL) << 11)

static float Clamp(float v, float min, float max) {
  return (v < min) ? min : (v > max) ? max : v;
}

typedef struct {  // struct for organizing convergence in either size or PSNR
  int is_first;
  float dq;
  float q, last_q;
  float qmin, qmax;
  double value, last_value;  // PSNR or size
  double target;
  int do_size_search;
} PassStats;

static int InitPassStats(const VP8Encoder* const enc, PassStats* const s) {
  const uint64_t target_size = (uint64_t)enc->config->target_size;
  const int do_size_search = (target_size != 0);
  const float target_PSNR = enc->config->target_PSNR;

  s->is_first = 1;
  s->dq = 10.f;
  s->qmin = 1.f * enc->config->qmin;
  s->qmax = 1.f * enc->config->qmax;
  s->q = s->last_q = Clamp(enc->config->quality, s->qmin, s->qmax);
  s->target = do_size_search       ? (double)target_size
              : (target_PSNR > 0.) ? target_PSNR
                                   : 40.;  // default, just in case
  s->value = s->last_value = 0.;
  s->do_size_search = do_size_search;
  return do_size_search;
}

static float ComputeNextQ(PassStats* const s) {
  float dq;
  if (s->is_first) {
    dq = (s->value > s->target) ? -s->dq : s->dq;
    s->is_first = 0;
  } else if (s->value != s->last_value) {
    const double slope = (s->target - s->value) / (s->last_value - s->value);
    dq = (float)(slope * (s->last_q - s->q));
  } else {
    dq = 0.;  // we're done?!
  }
  // Limit variable to avoid large swings.
  s->dq = Clamp(dq, -30.f, 30.f);
  s->last_q = s->q;
  s->last_value = s->value;
  s->q = Clamp(s->q + s->dq, s->qmin, s->qmax);
  return s->q;
}

//------------------------------------------------------------------------------
// Tables for level coding

const uint8_t VP8Cat3[] = {173, 148, 140};
const uint8_t VP8Cat4[] = {176, 155, 140, 135};
const uint8_t VP8Cat5[] = {180, 157, 141, 134, 130};
const uint8_t VP8Cat6[] = {254, 254, 243, 230, 196, 177,
                           153, 140, 133, 130, 129};

//------------------------------------------------------------------------------
// Reset the statistics about: number of skips, token proba, level cost,...

static void ResetStats(VP8Encoder* const enc) {
  VP8EncProba* const proba = &enc->proba;
  VP8CalculateLevelCosts(proba);
  proba->nb_skip = 0;
}

//------------------------------------------------------------------------------
// Skip decision probability

#define SKIP_PROBA_THRESHOLD 250  // value below which using skip_proba is OK.

static int CalcSkipProba(uint64_t nb, uint64_t total) {
  return (int)(total ? (total - nb) * 255 / total : 255);
}

// Returns the bit-cost for coding the skip probability.
static int FinalizeSkipProba(VP8Encoder* const enc) {
  VP8EncProba* const proba = &enc->proba;
  const int nb_mbs = enc->mb_w * enc->mb_h;
  const int nb_events = proba->nb_skip;
  int size;
  proba->skip_proba = CalcSkipProba(nb_events, nb_mbs);
  proba->use_skip_proba = (proba->skip_proba < SKIP_PROBA_THRESHOLD);
  size = 256;  // 'use_skip_proba' bit
  if (proba->use_skip_proba) {
    size += nb_events * VP8BitCost(1, proba->skip_proba) +
            (nb_mbs - nb_events) * VP8BitCost(0, proba->skip_proba);
    size += 8 * 256;  // cost of signaling the 'skip_proba' itself.
  }
  return size;
}

// Collect statistics and deduce probabilities for next coding pass.
// Return the total bit-cost for coding the probability updates.
static int CalcTokenProba(int nb, int total) {
  assert(nb <= total);
  return nb ? (255 - nb * 255 / total) : 255;
}

// Cost of coding 'nb' 1's and 'total-nb' 0's using 'proba' probability.
static int BranchCost(int nb, int total, int proba) {
  return nb * VP8BitCost(1, proba) + (total - nb) * VP8BitCost(0, proba);
}

static void ResetTokenStats(VP8Encoder* const enc) {
  VP8EncProba* const proba = &enc->proba;
  memset(proba->stats, 0, sizeof(proba->stats));
}

static int FinalizeTokenProbas(VP8EncProba* const proba) {
  int has_changed = 0;
  int size = 0;
  int t, b, c, p;
  for (t = 0; t < NUM_TYPES; ++t) {
    for (b = 0; b < NUM_BANDS; ++b) {
      for (c = 0; c < NUM_CTX; ++c) {
        for (p = 0; p < NUM_PROBAS; ++p) {
          const proba_t stats = proba->stats[t][b][c][p];
          const int nb = (stats >> 0) & 0xffff;
          const int total = (stats >> 16) & 0xffff;
          const int update_proba = VP8CoeffsUpdateProba[t][b][c][p];
          const int old_p = VP8CoeffsProba0[t][b][c][p];
          const int new_p = CalcTokenProba(nb, total);
          const int old_cost =
              BranchCost(nb, total, old_p) + VP8BitCost(0, update_proba);
          const int new_cost = BranchCost(nb, total, new_p) +
                               VP8BitCost(1, update_proba) + 8 * 256;
          const int use_new_p = (old_cost > new_cost);
          size += VP8BitCost(use_new_p, update_proba);
          if (use_new_p) {  // only use proba that seem meaningful enough.
            proba->coeffs[t][b][c][p] = new_p;
            has_changed |= (new_p != old_p);
            size += 8 * 256;
          } else {
            proba->coeffs[t][b][c][p] = old_p;
          }
        }
      }
    }
  }
  proba->dirty = has_changed;
  return size;
}

//------------------------------------------------------------------------------
// Finalize Segment probability based on the coding tree

static int GetProba(int a, int b) {
  const int total = a + b;
  return (total == 0) ? 255  // that's the default probability.
                      : (255 * a + total / 2) / total;  // rounded proba
}

static void ResetSegments(VP8Encoder* const enc) {
  int n;
  for (n = 0; n < enc->mb_w * enc->mb_h; ++n) {
    enc->mb_info[n].segment = 0;
  }
}

static void SetSegmentProbas(VP8Encoder* const enc) {
  int p[NUM_MB_SEGMENTS] = {0};
  int n;

  for (n = 0; n < enc->mb_w * enc->mb_h; ++n) {
    const VP8MBInfo* const mb = &enc->mb_info[n];
    ++p[mb->segment];
  }
#if !defined(WEBP_DISABLE_STATS)
  if (enc->pic->stats != NULL) {
    for (n = 0; n < NUM_MB_SEGMENTS; ++n) {
      enc->pic->stats->segment_size[n] = p[n];
    }
  }
#endif
  if (enc->segment_hdr.num_segments > 1) {
    uint8_t* const probas = enc->proba.segments;
    probas[0] = GetProba(p[0] + p[1], p[2] + p[3]);
    probas[1] = GetProba(p[0], p[1]);
    probas[2] = GetProba(p[2], p[3]);

    enc->segment_hdr.update_map =
        (probas[0] != 255) || (probas[1] != 255) || (probas[2] != 255);
    if (!enc->segment_hdr.update_map) ResetSegments(enc);
    enc->segment_hdr.size =
        p[0] * (VP8BitCost(0, probas[0]) + VP8BitCost(0, probas[1])) +
        p[1] * (VP8BitCost(0, probas[0]) + VP8BitCost(1, probas[1])) +
        p[2] * (VP8BitCost(1, probas[0]) + VP8BitCost(0, probas[2])) +
        p[3] * (VP8BitCost(1, probas[0]) + VP8BitCost(1, probas[2]));
  } else {
    enc->segment_hdr.update_map = 0;
    enc->segment_hdr.size = 0;
  }
}

//------------------------------------------------------------------------------
// Coefficient coding

static int PutCoeffs(VP8BitWriter* const bw, int ctx, const VP8Residual* res) {
  int n = res->first;
  // should be prob[VP8EncBands[n]], but it's equivalent for n=0 or 1
  const uint8_t* p = res->prob[n][ctx];
  if (!VP8PutBit(bw, res->last >= 0, p[0])) {
    return 0;
  }

  while (n < 16) {
    const int c = res->coeffs[n++];
    const int sign = c < 0;
    int v = sign ? -c : c;
    if (!VP8PutBit(bw, v != 0, p[1])) {
      p = res->prob[VP8EncBands[n]][0];
      continue;
    }
    if (!VP8PutBit(bw, v > 1, p[2])) {
      p = res->prob[VP8EncBands[n]][1];
    } else {
      if (!VP8PutBit(bw, v > 4, p[3])) {
        if (VP8PutBit(bw, v != 2, p[4])) {
          VP8PutBit(bw, v == 4, p[5]);
        }
      } else if (!VP8PutBit(bw, v > 10, p[6])) {
        if (!VP8PutBit(bw, v > 6, p[7])) {
          VP8PutBit(bw, v == 6, 159);
        } else {
          VP8PutBit(bw, v >= 9, 165);
          VP8PutBit(bw, !(v & 1), 145);
        }
      } else {
        int mask;
        const uint8_t* tab;
        if (v < 3 + (8 << 1)) {  // VP8Cat3  (3b)
          VP8PutBit(bw, 0, p[8]);
          VP8PutBit(bw, 0, p[9]);
          v -= 3 + (8 << 0);
          mask = 1 << 2;
          tab = VP8Cat3;
        } else if (v < 3 + (8 << 2)) {  // VP8Cat4  (4b)
          VP8PutBit(bw, 0, p[8]);
          VP8PutBit(bw, 1, p[9]);
          v -= 3 + (8 << 1);
          mask = 1 << 3;
          tab = VP8Cat4;
        } else if (v < 3 + (8 << 3)) {  // VP8Cat5  (5b)
          VP8PutBit(bw, 1, p[8]);
          VP8PutBit(bw, 0, p[10]);
          v -= 3 + (8 << 2);
          mask = 1 << 4;
          tab = VP8Cat5;
        } else {  // VP8Cat6 (11b)
          VP8PutBit(bw, 1, p[8]);
          VP8PutBit(bw, 1, p[10]);
          v -= 3 + (8 << 3);
          mask = 1 << 10;
          tab = VP8Cat6;
        }
        while (mask) {
          VP8PutBit(bw, !!(v & mask), *tab++);
          mask >>= 1;
        }
      }
      p = res->prob[VP8EncBands[n]][2];
    }
    VP8PutBitUniform(bw, sign);
    if (n == 16 || !VP8PutBit(bw, n <= res->last, p[0])) {
      return 1;  // EOB
    }
  }
  return 1;
}

static void CodeResiduals(VP8BitWriter* const bw, VP8EncIterator* const it,
                          const VP8ModeScore* const rd) {
  int x, y, ch;
  VP8Residual res;
  uint64_t pos1, pos2, pos3;
  const int i16 = (it->mb->type == 1);
  const int segment = it->mb->segment;
  VP8Encoder* const enc = it->enc;

  VP8IteratorNzToBytes(it);

  pos1 = VP8BitWriterPos(bw);
  if (i16) {
    VP8InitResidual(0, 1, enc, &res);
    VP8SetResidualCoeffs(rd->y_dc_levels, &res);
    it->top_nz[8] = it->left_nz[8] =
        PutCoeffs(bw, it->top_nz[8] + it->left_nz[8], &res);
    VP8InitResidual(1, 0, enc, &res);
  } else {
    VP8InitResidual(0, 3, enc, &res);
  }

  // luma-AC
  for (y = 0; y < 4; ++y) {
    for (x = 0; x < 4; ++x) {
      const int ctx = it->top_nz[x] + it->left_nz[y];
      VP8SetResidualCoeffs(rd->y_ac_levels[x + y * 4], &res);
      it->top_nz[x] = it->left_nz[y] = PutCoeffs(bw, ctx, &res);
    }
  }
  pos2 = VP8BitWriterPos(bw);

  // U/V
  VP8InitResidual(0, 2, enc, &res);
  for (ch = 0; ch <= 2; ch += 2) {
    for (y = 0; y < 2; ++y) {
      for (x = 0; x < 2; ++x) {
        const int ctx = it->top_nz[4 + ch + x] + it->left_nz[4 + ch + y];
        VP8SetResidualCoeffs(rd->uv_levels[ch * 2 + x + y * 2], &res);
        it->top_nz[4 + ch + x] = it->left_nz[4 + ch + y] =
            PutCoeffs(bw, ctx, &res);
      }
    }
  }
  pos3 = VP8BitWriterPos(bw);
  it->luma_bits = pos2 - pos1;
  it->uv_bits = pos3 - pos2;
  it->bit_count[segment][i16] += it->luma_bits;
  it->bit_count[segment][2] += it->uv_bits;
  VP8IteratorBytesToNz(it);
}

// Same as CodeResiduals, but doesn't actually write anything.
// Instead, it just records the event distribution.
static void RecordResiduals(VP8EncIterator* const it,
                            const VP8ModeScore* const rd) {
  int x, y, ch;
  VP8Residual res;
  VP8Encoder* const enc = it->enc;

  VP8IteratorNzToBytes(it);

  if (it->mb->type == 1) {  // i16x16
    VP8InitResidual(0, 1, enc, &res);
    VP8SetResidualCoeffs(rd->y_dc_levels, &res);
    it->top_nz[8] = it->left_nz[8] =
        VP8RecordCoeffs(it->top_nz[8] + it->left_nz[8], &res);
    VP8InitResidual(1, 0, enc, &res);
  } else {
    VP8InitResidual(0, 3, enc, &res);
  }

  // luma-AC
  for (y = 0; y < 4; ++y) {
    for (x = 0; x < 4; ++x) {
      const int ctx = it->top_nz[x] + it->left_nz[y];
      VP8SetResidualCoeffs(rd->y_ac_levels[x + y * 4], &res);
      it->top_nz[x] = it->left_nz[y] = VP8RecordCoeffs(ctx, &res);
    }
  }

  // U/V
  VP8InitResidual(0, 2, enc, &res);
  for (ch = 0; ch <= 2; ch += 2) {
    for (y = 0; y < 2; ++y) {
      for (x = 0; x < 2; ++x) {
        const int ctx = it->top_nz[4 + ch + x] + it->left_nz[4 + ch + y];
        VP8SetResidualCoeffs(rd->uv_levels[ch * 2 + x + y * 2], &res);
        it->top_nz[4 + ch + x] = it->left_nz[4 + ch + y] =
            VP8RecordCoeffs(ctx, &res);
      }
    }
  }

  VP8IteratorBytesToNz(it);
}

//------------------------------------------------------------------------------
// Token buffer

#if !defined(DISABLE_TOKEN_BUFFER)

// 'stats' is where the token statistics get accumulated: enc->proba.stats
// for direct recording, or a private scratch when the canonical statistics
// are rebuilt later with VP8TokenReplayStats() (multi-threaded loop).
static int RecordTokens(VP8EncIterator* const it, const VP8ModeScore* const rd,
                        VP8TBuffer* const tokens,
                        StatsArray (*const stats)[NUM_BANDS]) {
  int x, y, ch;
  VP8Residual res;
  VP8Encoder* const enc = it->enc;

  VP8IteratorNzToBytes(it);
  if (it->mb->type == 1) {  // i16x16
    const int ctx = it->top_nz[8] + it->left_nz[8];
    VP8InitResidual(0, 1, enc, &res);
    res.stats = stats[1];
    VP8SetResidualCoeffs(rd->y_dc_levels, &res);
    it->top_nz[8] = it->left_nz[8] = VP8RecordCoeffTokens(ctx, &res, tokens);
    VP8InitResidual(1, 0, enc, &res);
    res.stats = stats[0];
  } else {
    VP8InitResidual(0, 3, enc, &res);
    res.stats = stats[3];
  }

  // luma-AC
  for (y = 0; y < 4; ++y) {
    for (x = 0; x < 4; ++x) {
      const int ctx = it->top_nz[x] + it->left_nz[y];
      VP8SetResidualCoeffs(rd->y_ac_levels[x + y * 4], &res);
      it->top_nz[x] = it->left_nz[y] = VP8RecordCoeffTokens(ctx, &res, tokens);
    }
  }

  // U/V
  VP8InitResidual(0, 2, enc, &res);
  res.stats = stats[2];
  for (ch = 0; ch <= 2; ch += 2) {
    for (y = 0; y < 2; ++y) {
      for (x = 0; x < 2; ++x) {
        const int ctx = it->top_nz[4 + ch + x] + it->left_nz[4 + ch + y];
        VP8SetResidualCoeffs(rd->uv_levels[ch * 2 + x + y * 2], &res);
        it->top_nz[4 + ch + x] = it->left_nz[4 + ch + y] =
            VP8RecordCoeffTokens(ctx, &res, tokens);
      }
    }
  }
  VP8IteratorBytesToNz(it);
  return !tokens->error;
}

#endif  // !DISABLE_TOKEN_BUFFER

//------------------------------------------------------------------------------
// ExtraInfo map / Debug function

#if !defined(WEBP_DISABLE_STATS)

#if SEGMENT_VISU
static void SetBlock(uint8_t* p, int value, int size) {
  int y;
  for (y = 0; y < size; ++y) {
    memset(p, value, size);
    p += BPS;
  }
}
#endif

static void ResetSSE(VP8Encoder* const enc) {
  enc->sse[0] = 0;
  enc->sse[1] = 0;
  enc->sse[2] = 0;
  // Note: enc->sse[3] is managed by alpha.c
  enc->sse_count = 0;
}

// Per-pass accumulators for the side statistics: they are summed separately
// (possibly per-worker) and applied to the encoder at the end of the last
// pass, so that the totals do not depend on the accumulation order.
typedef struct {
  uint64_t sse[3];
  uint64_t sse_count;
  int block_count[3];
} SideInfoSums;

static void StoreSSE(const VP8EncIterator* const it,
                     SideInfoSums* const sums) {
  const uint8_t* const in = it->yuv_in;
  const uint8_t* const out = it->yuv_out;
  // Note: not totally accurate at boundary. And doesn't include in-loop filter.
  sums->sse[0] += VP8SSE16x16(in + Y_OFF_ENC, out + Y_OFF_ENC);
  sums->sse[1] += VP8SSE8x8(in + U_OFF_ENC, out + U_OFF_ENC);
  sums->sse[2] += VP8SSE8x8(in + V_OFF_ENC, out + V_OFF_ENC);
  sums->sse_count += 16 * 16;
}

static void StoreSideInfo(const VP8EncIterator* const it,
                          SideInfoSums* const sums) {
  VP8Encoder* const enc = it->enc;
  const VP8MBInfo* const mb = it->mb;
  WebPPicture* const pic = enc->pic;

  if (pic->stats != NULL) {
    StoreSSE(it, sums);
    sums->block_count[0] += (mb->type == 0);
    sums->block_count[1] += (mb->type == 1);
    sums->block_count[2] += (mb->skip != 0);
  }

  if (pic->extra_info != NULL) {
    uint8_t* const info = &pic->extra_info[it->x + it->y * enc->mb_w];
    switch (pic->extra_info_type) {
      case 1:
        *info = mb->type;
        break;
      case 2:
        *info = mb->segment;
        break;
      case 3:
        *info = enc->dqm[mb->segment].quant;
        break;
      case 4:
        *info = (mb->type == 1) ? it->preds[0] : 0xff;
        break;
      case 5:
        *info = mb->uv_mode;
        break;
      case 6: {
        const int b = (int)((it->luma_bits + it->uv_bits + 7) >> 3);
        *info = (b > 255) ? 255 : b;
        break;
      }
      case 7:
        *info = mb->alpha;
        break;
      default:
        *info = 0;
        break;
    }
  }
#if SEGMENT_VISU  // visualize segments and prediction modes
  SetBlock(it->yuv_out + Y_OFF_ENC, mb->segment * 64, 16);
  SetBlock(it->yuv_out + U_OFF_ENC, it->preds[0] * 64, 8);
  SetBlock(it->yuv_out + V_OFF_ENC, mb->uv_mode * 64, 8);
#endif
}

#ifdef WEBP_USE_THREAD
static void MergeSideInfoSums(SideInfoSums* const dst,
                              const SideInfoSums* const src) {
  dst->sse[0] += src->sse[0];
  dst->sse[1] += src->sse[1];
  dst->sse[2] += src->sse[2];
  dst->sse_count += src->sse_count;
  dst->block_count[0] += src->block_count[0];
  dst->block_count[1] += src->block_count[1];
  dst->block_count[2] += src->block_count[2];
}
#endif  // WEBP_USE_THREAD

static void ApplySideInfoSums(VP8Encoder* const enc,
                              const SideInfoSums* const sums) {
  enc->sse[0] += sums->sse[0];
  enc->sse[1] += sums->sse[1];
  enc->sse[2] += sums->sse[2];
  enc->sse_count += sums->sse_count;
  enc->block_count[0] += sums->block_count[0];
  enc->block_count[1] += sums->block_count[1];
  enc->block_count[2] += sums->block_count[2];
}

static void ResetSideInfo(const VP8EncIterator* const it) {
  VP8Encoder* const enc = it->enc;
  WebPPicture* const pic = enc->pic;
  if (pic->stats != NULL) {
    memset(enc->block_count, 0, sizeof(enc->block_count));
  }
  ResetSSE(enc);
}
#else   // defined(WEBP_DISABLE_STATS)
typedef struct { int unused; } SideInfoSums;

static void ResetSSE(VP8Encoder* const enc) { (void)enc; }
static void StoreSideInfo(const VP8EncIterator* const it,
                          SideInfoSums* const sums) {
  VP8Encoder* const enc = it->enc;
  WebPPicture* const pic = enc->pic;
  (void)sums;
  if (pic->extra_info != NULL) {
    if (it->x == 0 && it->y == 0) {  // only do it once, at start
      memset(pic->extra_info, 0,
             enc->mb_w * enc->mb_h * sizeof(*pic->extra_info));
    }
  }
}

#ifdef WEBP_USE_THREAD
static void MergeSideInfoSums(SideInfoSums* const dst,
                              const SideInfoSums* const src) {
  (void)dst;
  (void)src;
}
#endif  // WEBP_USE_THREAD

static void ApplySideInfoSums(VP8Encoder* const enc,
                              const SideInfoSums* const sums) {
  (void)enc;
  (void)sums;
}

static void ResetSideInfo(const VP8EncIterator* const it) { (void)it; }
#endif  // !defined(WEBP_DISABLE_STATS)

static double GetPSNR(uint64_t mse, uint64_t size) {
  return (mse > 0 && size > 0) ? 10. * log10(255. * 255. * size / mse) : 99;
}

//------------------------------------------------------------------------------
//  StatLoop(): only collect statistics (number of skips, token usage, ...).
//  This is used for deciding optimal probabilities. It also modifies the
//  quantizer value if some target (size, PSNR) was specified.

static void SetLoopParams(VP8Encoder* const enc, float q) {
  // Make sure the quality parameter is inside valid bounds
  q = Clamp(q, 0.f, 100.f);

  VP8SetSegmentParams(enc, q);  // setup segment quantizations and filters
  SetSegmentProbas(enc);        // compute segment probabilities

  ResetStats(enc);
  ResetSSE(enc);
}

static uint64_t OneStatPass(VP8Encoder* const enc, VP8RDLevel rd_opt,
                            int nb_mbs, int percent_delta, PassStats* const s) {
  VP8EncIterator it;
  uint64_t size = 0;
  uint64_t size_p0 = 0;
  uint64_t distortion = 0;
  const uint64_t pixel_count = (uint64_t)nb_mbs * 384;

  VP8IteratorInit(enc, &it);
  SetLoopParams(enc, s->q);
  do {
    VP8ModeScore info;
    VP8IteratorImport(&it, NULL);
    if (VP8Decimate(&it, &info, rd_opt)) {
      // Just record the number of skips and act like skip_proba is not used.
      ++enc->proba.nb_skip;
    }
    RecordResiduals(&it, &info);
    size += info.R + info.H;
    size_p0 += info.H;
    distortion += info.D;
    if (percent_delta && !VP8IteratorProgress(&it, percent_delta)) {
      return 0;
    }
    VP8IteratorSaveBoundary(&it);
  } while (VP8IteratorNext(&it) && --nb_mbs > 0);
  VP8IteratorMergeMaxEdge(&it);

  size_p0 += enc->segment_hdr.size;
  if (s->do_size_search) {
    size += FinalizeSkipProba(enc);
    size += FinalizeTokenProbas(&enc->proba);
    size = ((size + size_p0 + 1024) >> 11) + HEADER_SIZE_ESTIMATE;
    s->value = (double)size;
  } else {
    s->value = GetPSNR(distortion, pixel_count);
  }
  return size_p0;
}

static int StatLoop(VP8Encoder* const enc) {
  const int method = enc->method;
  const int do_search = enc->do_search;
  const int fast_probe = ((method == 0 || method == 3) && !do_search);
  int num_pass_left = enc->config->pass;
  const int task_percent = 20;
  const int percent_per_pass =
      (task_percent + num_pass_left / 2) / num_pass_left;
  const int final_percent = enc->percent + task_percent;
  const VP8RDLevel rd_opt =
      (method >= 3 || do_search) ? RD_OPT_BASIC : RD_OPT_NONE;
  int nb_mbs = enc->mb_w * enc->mb_h;
  PassStats stats;

  InitPassStats(enc, &stats);
  ResetTokenStats(enc);

  // Fast mode: quick analysis pass over few mbs. Better than nothing.
  if (fast_probe) {
    if (method == 3) {  // we need more stats for method 3 to be reliable.
      nb_mbs = (nb_mbs > 200) ? nb_mbs >> 1 : 100;
    } else {
      nb_mbs = (nb_mbs > 200) ? nb_mbs >> 2 : 50;
    }
  }

  while (num_pass_left-- > 0) {
    const int is_last_pass = (fabs(stats.dq) <= DQ_LIMIT) ||
                             (num_pass_left == 0) ||
                             (enc->max_i4_header_bits == 0);
    const uint64_t size_p0 =
        OneStatPass(enc, rd_opt, nb_mbs, percent_per_pass, &stats);
    if (size_p0 == 0) return 0;
#if (DEBUG_SEARCH > 0)
    printf("#%d value:%.1lf -> %.1lf   q:%.2f -> %.2f\n", num_pass_left,
           stats.last_value, stats.value, stats.last_q, stats.q);
#endif
    if (enc->max_i4_header_bits > 0 && size_p0 > PARTITION0_SIZE_LIMIT) {
      ++num_pass_left;
      enc->max_i4_header_bits >>= 1;  // strengthen header bit limitation...
      continue;                       // ...and start over
    }
    if (is_last_pass) {
      break;
    }
    // If no target size: just do several pass without changing 'q'
    if (do_search) {
      ComputeNextQ(&stats);
      if (fabs(stats.dq) <= DQ_LIMIT) break;
    }
  }
  if (!do_search || !stats.do_size_search) {
    // Need to finalize probas now, since it wasn't done during the search.
    FinalizeSkipProba(enc);
    FinalizeTokenProbas(&enc->proba);
  }
  VP8CalculateLevelCosts(&enc->proba);  // finalize costs
  return WebPReportProgress(enc->pic, final_percent, &enc->percent);
}

//------------------------------------------------------------------------------
// Main loops
//

static const uint8_t kAverageBytesPerMB[8] = {50, 24, 16, 9, 7, 5, 3, 2};

static int PreLoopInitialize(VP8Encoder* const enc) {
  int p;
  int ok = 1;
  const int average_bytes_per_MB = kAverageBytesPerMB[enc->base_quant >> 4];
  const int bytes_per_parts =
      enc->mb_w * enc->mb_h * average_bytes_per_MB / enc->num_parts;
  // Initialize the bit-writers
  for (p = 0; ok && p < enc->num_parts; ++p) {
    ok = VP8BitWriterInit(enc->parts + p, bytes_per_parts);
  }
  if (!ok) {
    VP8EncFreeBitWriters(enc);  // malloc error occurred
    return WebPEncodingSetError(enc->pic, VP8_ENC_ERROR_OUT_OF_MEMORY);
  }
  return ok;
}

static int PostLoopFinalize(VP8EncIterator* const it, int ok) {
  VP8Encoder* const enc = it->enc;
  if (ok) {  // Finalize the partitions, check for extra errors.
    int p;
    for (p = 0; p < enc->num_parts; ++p) {
      VP8BitWriterFinish(enc->parts + p);
      ok &= !enc->parts[p].error;
    }
  }

  if (ok) {  // All good. Finish up.
#if !defined(WEBP_DISABLE_STATS)
    if (enc->pic->stats != NULL) {  // finalize byte counters...
      int i, s;
      for (i = 0; i <= 2; ++i) {
        for (s = 0; s < NUM_MB_SEGMENTS; ++s) {
          enc->residual_bytes[i][s] = (int)((it->bit_count[s][i] + 7) >> 3);
        }
      }
    }
#endif
    VP8AdjustFilterStrength(it);  // ...and store filter stats.
  } else {
    // Something bad happened -> need to do some memory cleanup.
    VP8EncFreeBitWriters(enc);
    return WebPEncodingSetError(enc->pic, VP8_ENC_ERROR_OUT_OF_MEMORY);
  }
  return ok;
}

//------------------------------------------------------------------------------
//  VP8EncLoop(): does the final bitstream coding.

static void ResetAfterSkip(VP8EncIterator* const it) {
  if (it->mb->type == 1) {
    *it->nz = 0;  // reset all predictors
    it->left_nz[8] = 0;
  } else {
    *it->nz &= (1 << 24);  // preserve the dc_nz bit
  }
}

int VP8EncLoop(VP8Encoder* const enc) {
  VP8EncIterator it;
  SideInfoSums side_sums;
  int ok = PreLoopInitialize(enc);
  if (!ok) return 0;

  StatLoop(enc);  // stats-collection loop

  VP8IteratorInit(enc, &it);
  VP8InitFilter(&it);
  memset(&side_sums, 0, sizeof(side_sums));
  do {
    VP8ModeScore info;
    const int dont_use_skip = !enc->proba.use_skip_proba;
    const VP8RDLevel rd_opt = enc->rd_opt_level;

    VP8IteratorImport(&it, NULL);
    // Warning! order is important: first call VP8Decimate() and
    // *then* decide how to code the skip decision if there's one.
    if (!VP8Decimate(&it, &info, rd_opt) || dont_use_skip) {
      CodeResiduals(it.bw, &it, &info);
      if (it.bw->error) {
        // enc->pic->error_code is set in PostLoopFinalize().
        ok = 0;
        break;
      }
    } else {  // reset predictors after a skip
      ResetAfterSkip(&it);
    }
    StoreSideInfo(&it, &side_sums);
    VP8StoreFilterStats(&it);
    VP8IteratorExport(&it);
    ok = VP8IteratorProgress(&it, 20);
    VP8IteratorSaveBoundary(&it);
  } while (ok && VP8IteratorNext(&it));
  VP8IteratorMergeMaxEdge(&it);

  if (ok) ApplySideInfoSums(enc, &side_sums);
  return PostLoopFinalize(&it, ok);
}

//------------------------------------------------------------------------------
// Single pass using Token Buffer.

#if !defined(DISABLE_TOKEN_BUFFER)

#define MIN_COUNT 96  // minimum number of macroblocks before updating stats

//------------------------------------------------------------------------------
// Multi-threaded token loop: the macroblock rows are processed by several
// workers in a wavefront pattern, a macroblock being processed as soon as the
// reconstruction of its top-right neighbor is available. The shared context
// strips (y_top/uv_top, nz, preds, top_derr) are naturally ordered by the
// wavefront. The probability refreshes happen at row-group boundaries where
// all workers gather; the canonical statistics are rebuilt there by replaying
// the recorded tokens in raster order. All inputs of a macroblock decision are
// therefore the same as in the single-threaded loop and the output bitstream
// is identical.

#ifdef WEBP_USE_THREAD

#if defined(_WIN32)
#include <windows.h>  // GetSystemInfo()
#else
#include <unistd.h>   // sysconf()
#endif

// Broadcast progress to the waiting workers at least every that many MBs (the
// actual per-MB value is published lock-free; see WfStoreRelease()).
#define WAVEFRONT_SYNC_RANGE 16
// Upper bound on worker threads. The effective count is min(this, detected
// processors or the WEBP_ENC_THREADS override, and half the macroblock rows).
#define WAVEFRONT_MAX_THREADS 64
#define WAVEFRONT_MIN_MB_ROWS 4  // don't bother below that many rows

// Size of a cache line (128 covers Apple Silicon; 64 elsewhere). The per-row
// progress counters are padded to this so a producer writing its own row does
// not invalidate the counters that other workers poll (false sharing).
#define WAVEFRONT_CACHE_LINE 128

// Lock-free access to the per-row progress counters. A worker publishes its
// row's progress with a release store; the worker on the row below observes it
// with an acquire load and only blocks on the monitor when the dependency is
// genuinely not satisfied yet. This removes the per-macroblock mutex round-trip
// that otherwise serializes all workers on a single global lock. Compilers
// without the GCC/Clang atomic builtins fall back to the original always-locked
// path (correct, just with the old contention).
#if defined(__GNUC__) || defined(__clang__)
#define WAVEFRONT_LOCKFREE 1
static WEBP_INLINE int WfLoadAcquire(const int* const p) {
  return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
static WEBP_INLINE void WfStoreRelease(int* const p, int v) {
  __atomic_store_n(p, v, __ATOMIC_RELEASE);
}
#else
#define WAVEFRONT_LOCKFREE 0
static WEBP_INLINE int WfLoadAcquire(const int* const p) { return *p; }
static WEBP_INLINE void WfStoreRelease(int* const p, int v) { *p = v; }
#endif

// One per macroblock row, padded to its own cache line (see above).
typedef union {
  int v;  // number of processed macroblocks on the row
  char padding[WAVEFRONT_CACHE_LINE];
} WavefrontProgress;

// Number of logical processors available, or 1 if it cannot be determined.
static int WavefrontDetectProcessors(void) {
#if defined(_WIN32)
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return (info.dwNumberOfProcessors > 0) ? (int)info.dwNumberOfProcessors : 1;
#elif defined(_SC_NPROCESSORS_ONLN)
  const long n = sysconf(_SC_NPROCESSORS_ONLN);
  return (n > 0) ? (int)n : 1;
#else
  return 1;
#endif
}

typedef struct VP8WavefrontCtx VP8WavefrontCtx;

typedef struct {
  VP8WavefrontCtx* ctx;
  // Scratch statistics, discarded: the canonical ones are rebuilt by
  // replaying the tokens. This keeps RecordTokens() off the shared state.
  StatsArray scratch_stats[NUM_TYPES][NUM_BANDS];
  uint64_t size_p0;
  uint64_t distortion;
  SideInfoSums side_sums;
  LFStats lf_stats;  // auto-filter statistics (fixed-point: order-free merge)
} VP8WavefrontWorker;

struct VP8WavefrontCtx {
  VP8Encoder* enc;
  void* monitor;
  VP8RDLevel rd_opt;
  int is_last_pass;
  int rows_per_refresh;
  VP8TBuffer* row_tokens;          // per-row recorded tokens
  WavefrontProgress* row_progress; // per-row processed-MB count (padded)
  int next_row;            // next row to be claimed by a worker
  int phase_end;           // rows in [0, phase_end) can be processed
  int rows_done;           // fully processed rows
  int abort;               // error or cancellation: stop as soon as possible
  int num_threads;         // total number of threads (including the caller)
  VP8WavefrontWorker* workers;  // one per thread
  WebPWorker* threads;          // auxiliary threads (num_threads - 1)
  // Pipelined token replay: a dedicated thread replays completed rows into
  // proba->stats in strict raster order, concurrent with the wavefront, so the
  // (serial, order-dependent) replay overlaps the token loop instead of
  // stalling at the refresh barrier. NULL -> serial replay at the barrier.
  WebPWorker* stats_thread;
  int replayed_upto;       // rows [0, replayed_upto) folded into proba->stats
};

// Returns true if macroblock 'x' of row 'y' can be processed ('y' > 0).
// The monitor must be locked.
static int WavefrontReady(const VP8WavefrontCtx* const ctx, int x, int y) {
  int needed = x + 2;  // top and top-right neighbors must be reconstructed
  if (needed > ctx->enc->mb_w) needed = ctx->enc->mb_w;
  return (WfLoadAcquire(&ctx->row_progress[y - 1].v) >= needed);
}

// Dedicated stats thread: replays the recorded tokens into proba->stats in
// strict raster order (row 0, 1, 2, ...), each row as soon as the wavefront has
// finished it, and waits at the refresh boundary until the caller has consumed
// the stats and opened the next phase. Because it applies exactly the same
// per-token sequence (and overflow halving) as the single-threaded encoder, the
// resulting probabilities are bit-identical -- the replay is merely overlapped
// with the token loop instead of run serially at the barrier.
static int WavefrontStatsHook(void* arg1, void* arg2) {
  VP8WavefrontCtx* const ctx = (VP8WavefrontCtx*)arg1;
  proba_t* const stats = (proba_t*)ctx->enc->proba.stats;
  const int mb_w = ctx->enc->mb_w;
  const int mb_h = ctx->enc->mb_h;
  int r = 0;
  (void)arg2;
  while (r < mb_h) {
    // Wait until row r is fully processed and the phase it belongs to is open
    // (so we never write past a boundary the caller is still reading).
    WebPMonitorLock(ctx->monitor);
    while (!ctx->abort && !(WfLoadAcquire(&ctx->row_progress[r].v) >= mb_w &&
                            r < ctx->phase_end)) {
      WebPMonitorWait(ctx->monitor);
    }
    if (ctx->abort) {
      WebPMonitorUnlock(ctx->monitor);
      break;
    }
    WebPMonitorUnlock(ctx->monitor);
    VP8TokenReplayStats(&ctx->row_tokens[r], stats);  // exact, in raster order
    ++r;
    WebPMonitorLock(ctx->monitor);
    ctx->replayed_upto = r;
    WebPMonitorBroadcast(ctx->monitor);  // wake the caller waiting at a barrier
    WebPMonitorUnlock(ctx->monitor);
  }
  return 1;
}

static int WavefrontProcessRow(VP8WavefrontWorker* const w,
                               VP8EncIterator* const it, int y) {
  VP8WavefrontCtx* const ctx = w->ctx;
  const int mb_w = ctx->enc->mb_w;
  VP8TBuffer* const tokens = &ctx->row_tokens[y];
  int x;
  int ok = 1;
  VP8IteratorSetRow(it, y);
  VP8IteratorSetCountDown(it, mb_w);
  for (x = 0; ok && x < mb_w; ++x) {
    VP8ModeScore info;
    if (y > 0) {  // wait for the wavefront dependency
      // Lock-free fast path: when the row above has already published enough
      // progress (the common case once the wavefront is established) we proceed
      // without touching the monitor at all. Only block when it has not. The
      // fallback (no atomic builtins) always takes the slow, locked path.
#if WAVEFRONT_LOCKFREE
      if (!WavefrontReady(ctx, x, y))
#endif
      {
        WebPMonitorLock(ctx->monitor);
        while (!ctx->abort && !WavefrontReady(ctx, x, y)) {
          WebPMonitorWait(ctx->monitor);
        }
        ok = !ctx->abort;
        WebPMonitorUnlock(ctx->monitor);
        if (!ok) break;
      }
    }
    VP8IteratorImport(it, NULL);
    VP8Decimate(it, &info, ctx->rd_opt);
    ok = RecordTokens(it, &info, tokens, w->scratch_stats);
    w->size_p0 += info.H;
    w->distortion += info.D;
    if (ctx->is_last_pass) {
      StoreSideInfo(it, &w->side_sums);
      VP8StoreFilterStats(it);
      VP8IteratorExport(it);
    }
    VP8IteratorSaveBoundary(it);
    (void)VP8IteratorNext(it);
#if WAVEFRONT_LOCKFREE
    // Publish progress every macroblock with a release store so the worker on
    // the row below advances with single-MB granularity and rarely has to
    // block. Only grab the monitor to wake blocked workers / the caller, which
    // we still do every WAVEFRONT_SYNC_RANGE MBs and at row end / on error.
    WfStoreRelease(&ctx->row_progress[y].v, x + 1);
    if (!ok || ((x + 1) % WAVEFRONT_SYNC_RANGE) == 0 || (x + 1) == mb_w) {
      WebPMonitorLock(ctx->monitor);
      if (!ok) {
        ctx->abort = 1;
      } else if ((x + 1) == mb_w) {
        ++ctx->rows_done;  // row complete; wakes the caller / stats thread
      }
      WebPMonitorBroadcast(ctx->monitor);
      WebPMonitorUnlock(ctx->monitor);
    }
#else
    if (!ok || ((x + 1) % WAVEFRONT_SYNC_RANGE) == 0 || (x + 1) == mb_w) {
      WebPMonitorLock(ctx->monitor);
      ctx->row_progress[y].v = x + 1;
      if (!ok) {
        ctx->abort = 1;
      } else if ((x + 1) == mb_w) {
        ++ctx->rows_done;
      }
      WebPMonitorBroadcast(ctx->monitor);
      WebPMonitorUnlock(ctx->monitor);
    }
#endif
  }
  return ok;
}

// Claims and processes rows until the pass is complete (or aborted).
static void WavefrontWorkLoop(VP8WavefrontWorker* const w,
                              VP8EncIterator* const it) {
  VP8WavefrontCtx* const ctx = w->ctx;
  const int mb_h = ctx->enc->mb_h;
  while (1) {
    int y = -1;
    WebPMonitorLock(ctx->monitor);
    while (!ctx->abort && ctx->next_row >= ctx->phase_end &&
           ctx->phase_end < mb_h) {
      WebPMonitorWait(ctx->monitor);  // wait for the next phase
    }
    if (!ctx->abort && ctx->next_row < ctx->phase_end) {
      y = ctx->next_row++;
    }
    WebPMonitorUnlock(ctx->monitor);
    if (y < 0) break;  // aborted or no more rows
    if (!WavefrontProcessRow(w, it, y)) break;
  }
}

static int WavefrontWorkerHook(void* arg1, void* arg2) {
  VP8WavefrontWorker* const w = (VP8WavefrontWorker*)arg1;
  VP8EncIterator it;
  (void)arg2;
  // Note: VP8IteratorInit() would reset the shared top boundary conditions,
  // racing with the rows being processed by the other threads.
  VP8IteratorInitWorker(w->ctx->enc, &it);
  if (it.lf_stats != NULL) {
    it.lf_stats = &w->lf_stats;  // accumulate the auto-filter stats privately
  }
  WavefrontWorkLoop(w, &it);
  // 'max_edge' is a maximum: merging it under the lock makes the result
  // independent of the workers' ordering.
  WebPMonitorLock(w->ctx->monitor);
  VP8IteratorMergeMaxEdge(&it);
  WebPMonitorUnlock(w->ctx->monitor);
  return 1;
}

// Runs one full pass over the image with the wavefront workers, accumulating
// into 'size_p0', 'distortion' and 'side_sums'. Returns 0 on error.
static int WavefrontOnePass(VP8WavefrontCtx* const ctx,
                            VP8EncIterator* const it, int is_last_pass,
                            VP8RDLevel rd_opt, int pass_progress,
                            uint64_t* const size_p0,
                            uint64_t* const distortion,
                            SideInfoSums* const side_sums) {
  const WebPWorkerInterface* const winterface = WebPGetWorkerInterface();
  VP8Encoder* const enc = ctx->enc;
  VP8EncProba* const proba = &enc->proba;
  const int mb_h = enc->mb_h;
  int rows_replayed = 0;
  int ok = 1;
  int y, i;

  ctx->rd_opt = rd_opt;
  ctx->is_last_pass = is_last_pass;
  ctx->next_row = 0;
  ctx->rows_done = 0;
  ctx->replayed_upto = 0;
  ctx->abort = 0;
  ctx->phase_end =
      (ctx->rows_per_refresh < mb_h) ? ctx->rows_per_refresh : mb_h;
  for (y = 0; y < mb_h; ++y) {
    ctx->row_progress[y].v = 0;  // single-threaded here (before Launch)
    VP8TBufferClear(&ctx->row_tokens[y]);
  }
  for (i = 0; i < ctx->num_threads; ++i) {
    VP8WavefrontWorker* const w = &ctx->workers[i];
    w->size_p0 = 0;
    w->distortion = 0;
    memset(&w->side_sums, 0, sizeof(w->side_sums));
    memset(&w->lf_stats, 0, sizeof(w->lf_stats));
  }
  for (i = 1; i < ctx->num_threads; ++i) {
    winterface->Launch(&ctx->threads[i - 1]);
  }
  if (ctx->stats_thread != NULL) winterface->Launch(ctx->stats_thread);

  while (1) {
    int y_claim = -1;
    WebPMonitorLock(ctx->monitor);
    // Wait until the phase is processed -- and, when the stats thread is in
    // use, replayed (replayed_upto reaching phase_end implies processed).
    while (!ctx->abort && ctx->next_row >= ctx->phase_end &&
           (ctx->stats_thread != NULL ? ctx->replayed_upto < ctx->phase_end
                                      : ctx->rows_done < ctx->phase_end)) {
      WebPMonitorWait(ctx->monitor);
    }
    if (ctx->abort) {
      WebPMonitorUnlock(ctx->monitor);
      ok = 0;
      break;
    }
    if (ctx->next_row < ctx->phase_end) {
      y_claim = ctx->next_row++;
    }
    WebPMonitorUnlock(ctx->monitor);
    if (y_claim >= 0) {  // the calling thread processes rows too
      if (!WavefrontProcessRow(&ctx->workers[0], it, y_claim)) {
        ok = 0;
        break;
      }
      continue;
    }
    // Here all the rows of the current phase are done. The stats thread has
    // already folded [0, phase_end) into proba->stats in raster order; without
    // it, replay the phase's rows serially here (the original behaviour).
    if (ctx->stats_thread == NULL) {
      for (y = rows_replayed; y < ctx->phase_end; ++y) {
        VP8TokenReplayStats(&ctx->row_tokens[y], (proba_t*)proba->stats);
      }
      rows_replayed = ctx->phase_end;
    }
    if (ctx->phase_end == mb_h) break;  // pass complete
    FinalizeTokenProbas(proba);
    VP8CalculateLevelCosts(proba);  // refresh cost tables for rd-opt
    if (!WebPReportProgress(
            enc->pic, it->percent0 + pass_progress * ctx->phase_end / mb_h,
            &enc->percent)) {
      ok = 0;  // user abort: pic->error_code is set by WebPReportProgress()
    }
    WebPMonitorLock(ctx->monitor);
    if (!ok) {
      ctx->abort = 1;
    } else {
      ctx->phase_end += ctx->rows_per_refresh;
      if (ctx->phase_end > mb_h) ctx->phase_end = mb_h;
    }
    WebPMonitorBroadcast(ctx->monitor);
    WebPMonitorUnlock(ctx->monitor);
    if (!ok) break;
  }

  for (i = 1; i < ctx->num_threads; ++i) {
    ok &= winterface->Sync(&ctx->threads[i - 1]);
  }
  if (ctx->stats_thread != NULL) {  // it has replayed all rows (or seen abort)
    ok &= winterface->Sync(ctx->stats_thread);
  }
  VP8IteratorMergeMaxEdge(it);  // the workers merged theirs before Sync()
  for (i = 0; i < ctx->num_threads; ++i) {
    const VP8WavefrontWorker* const w = &ctx->workers[i];
    *size_p0 += w->size_p0;
    *distortion += w->distortion;
    if (is_last_pass) MergeSideInfoSums(side_sums, &w->side_sums);
  }
  // Merge the auto-filter statistics. The sums are fixed-point integers, so
  // the result does not depend on which worker processed which row. The
  // calling thread accumulated directly into 'enc->lf_stats'.
  if (is_last_pass && enc->lf_stats != NULL) {
    int s, j;
    for (i = 1; i < ctx->num_threads; ++i) {
      const VP8WavefrontWorker* const w = &ctx->workers[i];
      for (s = 0; s < NUM_MB_SEGMENTS; ++s) {
        for (j = 0; j < MAX_LF_LEVELS; ++j) {
          (*enc->lf_stats)[s][j] += w->lf_stats[s][j];
        }
      }
    }
  }
  if (!ok && enc->pic->error_code == VP8_ENC_OK) {
    WebPEncodingSetError(enc->pic, VP8_ENC_ERROR_OUT_OF_MEMORY);
  }
  return ok;
}

static void WavefrontDelete(VP8WavefrontCtx* const ctx) {
  if (ctx != NULL) {
    const WebPWorkerInterface* const winterface = WebPGetWorkerInterface();
    int i;
    if (ctx->threads != NULL) {
      for (i = 1; i < ctx->num_threads; ++i) {
        winterface->End(&ctx->threads[i - 1]);
      }
      WebPSafeFree(ctx->threads);
    }
    if (ctx->stats_thread != NULL) {
      winterface->End(ctx->stats_thread);
      WebPSafeFree(ctx->stats_thread);
    }
    if (ctx->row_tokens != NULL) {
      for (i = 0; i < ctx->enc->mb_h; ++i) {
        VP8TBufferClear(&ctx->row_tokens[i]);
      }
      WebPSafeFree(ctx->row_tokens);
    }
    WebPSafeFree(ctx->row_progress);
    WebPSafeFree(ctx->workers);
    WebPMonitorDelete(ctx->monitor);
    WebPSafeFree(ctx);
  }
}

// Creates the wavefront context, or returns NULL (e.g. when multi-threading
// is not wanted or not available): the caller then uses the single-threaded
// code path.
static VP8WavefrontCtx* WavefrontNew(VP8Encoder* const enc,
                                     int rows_per_refresh) {
  const WebPWorkerInterface* const winterface = WebPGetWorkerInterface();
  VP8WavefrontCtx* ctx = NULL;
  int num_threads;
  int i;
  if (enc->thread_level <= 0) return NULL;
  if (enc->mb_h < WAVEFRONT_MIN_MB_ROWS) return NULL;
  // Scale to the machine by default. WEBP_ENC_THREADS overrides the count
  // (e.g. to match a cgroup cpu quota, which the OS processor count ignores);
  // '1' effectively disables the wavefront.
  {
    const char* const env = getenv("WEBP_ENC_THREADS");
    num_threads = (env != NULL) ? atoi(env) : WavefrontDetectProcessors();
  }
  if (num_threads > WAVEFRONT_MAX_THREADS) num_threads = WAVEFRONT_MAX_THREADS;
  // a wavefront needs at least two rows per worker for a useful stagger
  if (num_threads > (enc->mb_h + 1) / 2) num_threads = (enc->mb_h + 1) / 2;
  if (num_threads < 1) num_threads = 1;

  ctx = (VP8WavefrontCtx*)WebPSafeCalloc(1ULL, sizeof(*ctx));
  if (ctx == NULL) return NULL;
  ctx->enc = enc;
  ctx->num_threads = num_threads;
  ctx->rows_per_refresh = rows_per_refresh;
  ctx->monitor = WebPMonitorNew();
  ctx->row_tokens =
      (VP8TBuffer*)WebPSafeCalloc(enc->mb_h, sizeof(*ctx->row_tokens));
  ctx->row_progress = (WavefrontProgress*)WebPSafeCalloc(
      enc->mb_h, sizeof(*ctx->row_progress));
  ctx->workers = (VP8WavefrontWorker*)WebPSafeCalloc(
      num_threads, sizeof(*ctx->workers));
  ctx->threads =
      (WebPWorker*)WebPSafeCalloc(num_threads - 1, sizeof(*ctx->threads));
  if (ctx->monitor == NULL || ctx->row_tokens == NULL ||
      ctx->row_progress == NULL || ctx->workers == NULL ||
      (num_threads > 1 && ctx->threads == NULL)) {
    goto Error;
  }
  for (i = 0; i < enc->mb_h; ++i) {
    VP8TBufferInit(&ctx->row_tokens[i], enc->mb_w * 16);
  }
  for (i = 0; i < num_threads; ++i) {
    ctx->workers[i].ctx = ctx;
  }
  for (i = 1; i < num_threads; ++i) {
    WebPWorker* const thread = &ctx->threads[i - 1];
    winterface->Init(thread);
    thread->data1 = &ctx->workers[i];
    thread->data2 = NULL;
    thread->hook = WavefrontWorkerHook;
    if (!winterface->Reset(thread)) {
      ctx->num_threads = i;  // only End() the threads reset so far
      goto Error;
    }
  }

  // Dedicated token-replay thread (overlaps the serial replay with the token
  // loop). Best-effort: on failure fall back to the serial barrier replay.
  if (num_threads >= 2) {
    ctx->stats_thread =
        (WebPWorker*)WebPSafeCalloc(1, sizeof(*ctx->stats_thread));
    if (ctx->stats_thread != NULL) {
      winterface->Init(ctx->stats_thread);
      ctx->stats_thread->data1 = ctx;
      ctx->stats_thread->data2 = NULL;
      ctx->stats_thread->hook = WavefrontStatsHook;
      if (!winterface->Reset(ctx->stats_thread)) {  // could not start it
        winterface->End(ctx->stats_thread);
        WebPSafeFree(ctx->stats_thread);
        ctx->stats_thread = NULL;
      }
    }
  }
  return ctx;

 Error:
  WavefrontDelete(ctx);
  return NULL;
}

// Largest power of two <= 'num_threads', capped at the VP8 maximum number of
// token partitions. The final entropy emit can use at most this many threads
// (one sequential boolean stream per partition); a wider wavefront still uses
// all its workers for token generation -- only the emit is bounded here.
static int WavefrontNumParts(int num_threads) {
  int n = 1;
  while (n * 2 <= num_threads && n < MAX_NUM_PARTITIONS) n *= 2;
  return n;
}

typedef struct {
  VP8WavefrontCtx* ctx;
  const uint8_t* probas;
  int part;
  int ok;
} VP8EmitJob;

// Emits every row assigned to partition 'part' (row y -> y & (num_parts - 1),
// matching the decoder) into that partition's bit-writer, in row order.
static int EmitPartitionBody(VP8WavefrontCtx* const ctx,
                             const uint8_t* const probas, int part) {
  VP8Encoder* const enc = ctx->enc;
  const int num_parts = enc->num_parts;
  VP8BitWriter* const bw = &enc->parts[part];
  int y;
  int ok = 1;
  for (y = part; ok && y < enc->mb_h; y += num_parts) {
    ok = VP8EmitTokens(&ctx->row_tokens[y], bw, probas, 1);
  }
  return ok;
}

static int EmitPartitionHook(void* arg1, void* arg2) {
  VP8EmitJob* const job = (VP8EmitJob*)arg1;
  (void)arg2;
  job->ok = EmitPartitionBody(job->ctx, job->probas, job->part);
  return job->ok;
}

// Writes the recorded tokens into the VP8 token partitions in parallel: each
// partition is an independent boolean stream, so partition 0 is emitted on the
// calling thread while partitions 1..num_parts-1 run on the wavefront's worker
// threads. num_parts <= num_threads by construction (see WavefrontNumParts).
static int WavefrontEmit(VP8WavefrontCtx* const ctx,
                         const uint8_t* const probas) {
  const WebPWorkerInterface* const winterface = WebPGetWorkerInterface();
  const int num_parts = ctx->enc->num_parts;
  VP8EmitJob jobs[MAX_NUM_PARTITIONS];
  int p;
  int ok;
  assert(num_parts >= 1 && num_parts <= MAX_NUM_PARTITIONS);
  assert(num_parts <= ctx->num_threads);
  for (p = 1; p < num_parts; ++p) {
    WebPWorker* const thread = &ctx->threads[p - 1];
    jobs[p].ctx = ctx;
    jobs[p].probas = probas;
    jobs[p].part = p;
    jobs[p].ok = 0;
    thread->hook = EmitPartitionHook;
    thread->data1 = &jobs[p];
    thread->data2 = NULL;
    winterface->Launch(thread);
  }
  ok = EmitPartitionBody(ctx, probas, 0);  // partition 0 on the caller
  for (p = 1; p < num_parts; ++p) {
    ok &= winterface->Sync(&ctx->threads[p - 1]) & jobs[p].ok;
  }
  return ok;
}

#endif  // WEBP_USE_THREAD

int VP8EncTokenLoop(VP8Encoder* const enc) {
  // Refresh the probas at row boundaries, roughly eight times per pass (and
  // not before MIN_COUNT macroblocks): the refresh schedule only depends on
  // the row so that the multi-threaded loop can reproduce it exactly.
  int rows_per_refresh = enc->mb_h >> 3;
  int num_pass_left = enc->config->pass;
  int remaining_progress = 40;  // percents
  const int do_search = enc->do_search;
  VP8EncIterator it;
  VP8EncProba* const proba = &enc->proba;
  const VP8RDLevel rd_opt = enc->rd_opt_level;
  const uint64_t pixel_count = (uint64_t)enc->mb_w * enc->mb_h * 384;
  PassStats stats;
#ifdef WEBP_USE_THREAD
  VP8WavefrontCtx* wavefront = NULL;
#endif
  int ok;

  InitPassStats(enc, &stats);

  {
    const int min_rows = (MIN_COUNT + enc->mb_w - 1) / enc->mb_w;
    if (rows_per_refresh < min_rows) rows_per_refresh = min_rows;
  }
#ifdef WEBP_USE_THREAD
  // returns NULL when multi-threading is off or not worth it
  wavefront = WavefrontNew(enc, rows_per_refresh);
  if (wavefront != NULL) {
    // Spread the recorded tokens over several VP8 token partitions so the final
    // entropy emit can run in parallel as well (row y -> partition
    // y & (num_parts - 1), as the decoder expects). Without the wavefront the
    // tokens are interleaved in a single buffer and must stay in one partition.
    enc->num_parts = WavefrontNumParts(wavefront->num_threads);
  }
#endif
  // Must run after the num_parts bump above: it sizes and initializes one
  // bit-writer per partition.
  ok = PreLoopInitialize(enc);
  if (!ok) {
#ifdef WEBP_USE_THREAD
    WavefrontDelete(wavefront);
#endif
    return 0;
  }

  assert(enc->num_parts >= 1);
  assert(enc->use_tokens);
  assert(proba->use_skip_proba == 0);
  assert(rd_opt >= RD_OPT_BASIC);  // otherwise, token-buffer won't be useful
  assert(num_pass_left > 0);

  while (ok && num_pass_left-- > 0) {
    const int is_last_pass = (fabs(stats.dq) <= DQ_LIMIT) ||
                             (num_pass_left == 0) ||
                             (enc->max_i4_header_bits == 0);
    uint64_t size_p0 = 0;
    uint64_t distortion = 0;
    SideInfoSums side_sums;
    // The final number of passes is not trivial to know in advance.
    const int pass_progress = remaining_progress / (2 + num_pass_left);
    remaining_progress -= pass_progress;
    VP8IteratorInit(enc, &it);
    SetLoopParams(enc, stats.q);
    if (is_last_pass) {
      ResetTokenStats(enc);
      VP8InitFilter(&it);  // don't collect stats until last pass (too costly)
    }
    memset(&side_sums, 0, sizeof(side_sums));
#ifdef WEBP_USE_THREAD
    if (wavefront != NULL) {
      ok = WavefrontOnePass(wavefront, &it, is_last_pass, rd_opt,
                            pass_progress, &size_p0, &distortion, &side_sums);
      if (!ok) break;
      goto PassDone;
    }
#endif
    VP8TBufferClear(&enc->tokens);
    do {
      VP8ModeScore info;
      if (it.x == 0 && it.y > 0 && (it.y % rows_per_refresh) == 0) {
        FinalizeTokenProbas(proba);
        VP8CalculateLevelCosts(proba);  // refresh cost tables for rd-opt
      }
      VP8IteratorImport(&it, NULL);
      VP8Decimate(&it, &info, rd_opt);
      ok = RecordTokens(&it, &info, &enc->tokens, proba->stats);
      if (!ok) {
        WebPEncodingSetError(enc->pic, VP8_ENC_ERROR_OUT_OF_MEMORY);
        break;
      }
      size_p0 += info.H;
      distortion += info.D;
      if (is_last_pass) {
        StoreSideInfo(&it, &side_sums);
        VP8StoreFilterStats(&it);
        VP8IteratorExport(&it);
        ok = VP8IteratorProgress(&it, pass_progress);
      }
      VP8IteratorSaveBoundary(&it);
    } while (ok && VP8IteratorNext(&it));
    if (!ok) break;
    VP8IteratorMergeMaxEdge(&it);

#ifdef WEBP_USE_THREAD
  PassDone:
#endif
    size_p0 += enc->segment_hdr.size;
    if (stats.do_size_search) {
      uint64_t size = FinalizeTokenProbas(&enc->proba);
#ifdef WEBP_USE_THREAD
      if (wavefront != NULL) {
        int y;
        for (y = 0; y < enc->mb_h; ++y) {
          size += VP8EstimateTokenSize(&wavefront->row_tokens[y],
                                       (const uint8_t*)proba->coeffs);
        }
      } else
#endif
      {
        size +=
            VP8EstimateTokenSize(&enc->tokens, (const uint8_t*)proba->coeffs);
      }
      size = (size + size_p0 + 1024) >> 11;  // -> size in bytes
      size += HEADER_SIZE_ESTIMATE;
      stats.value = (double)size;
    } else {  // compute and store PSNR
      stats.value = GetPSNR(distortion, pixel_count);
    }

#if (DEBUG_SEARCH > 0)
    printf(
        "#%2d metric:%.1lf -> %.1lf   last_q=%.2lf q=%.2lf dq=%.2lf "
        " range:[%.1f, %.1f]\n",
        num_pass_left, stats.last_value, stats.value, stats.last_q, stats.q,
        stats.dq, stats.qmin, stats.qmax);
#endif
    if (enc->max_i4_header_bits > 0 && size_p0 > PARTITION0_SIZE_LIMIT) {
      ++num_pass_left;
      enc->max_i4_header_bits >>= 1;  // strengthen header bit limitation...
      if (is_last_pass) {
        ResetSideInfo(&it);
      }
      continue;  // ...and start over
    }
    if (is_last_pass) {
      ApplySideInfoSums(enc, &side_sums);
      break;  // done
    }
    if (do_search) {
      ComputeNextQ(&stats);  // Adjust q
    }
  }
  if (ok) {
    if (!stats.do_size_search) {
      FinalizeTokenProbas(&enc->proba);
    }
#ifdef WEBP_USE_THREAD
    if (wavefront != NULL) {
      ok = WavefrontEmit(wavefront, (const uint8_t*)proba->coeffs);
    } else
#endif
    {
      ok = VP8EmitTokens(&enc->tokens, enc->parts + 0,
                         (const uint8_t*)proba->coeffs, 1);
    }
  }
#ifdef WEBP_USE_THREAD
  WavefrontDelete(wavefront);
#endif
  ok = ok && WebPReportProgress(enc->pic, enc->percent + remaining_progress,
                                &enc->percent);
  return PostLoopFinalize(&it, ok);
}

#else

int VP8EncTokenLoop(VP8Encoder* const enc) {
  (void)enc;
  return 0;  // we shouldn't be here.
}

#endif  // DISABLE_TOKEN_BUFFER

//------------------------------------------------------------------------------
