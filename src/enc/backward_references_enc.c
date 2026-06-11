// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Author: Jyrki Alakuijala (jyrki@google.com)
//

#include "src/enc/backward_references_enc.h"

#include <assert.h>
#include <string.h>

#include "src/dsp/cpu.h"
#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"
#include "src/enc/histogram_enc.h"
#include "src/enc/vp8i_enc.h"
#include "src/utils/color_cache_utils.h"
#include "src/utils/thread_utils.h"
#include "src/utils/utils.h"
#include "src/webp/encode.h"
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

#define MIN_BLOCK_SIZE 256  // minimum block size for backward references

// 1M window (4M bytes) minus 120 special codes for short distances.
#define WINDOW_SIZE ((1 << WINDOW_SIZE_BITS) - 120)

// Minimum number of pixels for which it is cheaper to encode a
// distance + length instead of each pixel as a literal.
#define MIN_LENGTH 4

// -----------------------------------------------------------------------------

static const uint8_t plane_to_code_lut[128] = {
    96,  73,  55,  39,  23, 13, 5,  1,  255, 255, 255, 255, 255, 255, 255, 255,
    101, 78,  58,  42,  26, 16, 8,  2,  0,   3,   9,   17,  27,  43,  59,  79,
    102, 86,  62,  46,  32, 20, 10, 6,  4,   7,   11,  21,  33,  47,  63,  87,
    105, 90,  70,  52,  37, 28, 18, 14, 12,  15,  19,  29,  38,  53,  71,  91,
    110, 99,  82,  66,  48, 35, 30, 24, 22,  25,  31,  36,  49,  67,  83,  100,
    115, 108, 94,  76,  64, 50, 44, 40, 34,  41,  45,  51,  65,  77,  95,  109,
    118, 113, 103, 92,  80, 68, 60, 56, 54,  57,  61,  69,  81,  93,  104, 114,
    119, 116, 111, 106, 97, 88, 84, 74, 72,  75,  85,  89,  98,  107, 112, 117};

extern int VP8LDistanceToPlaneCode(int xsize, int dist);
int VP8LDistanceToPlaneCode(int xsize, int dist) {
  const int yoffset = dist / xsize;
  const int xoffset = dist - yoffset * xsize;
  if (xoffset <= 8 && yoffset < 8) {
    return plane_to_code_lut[yoffset * 16 + 8 - xoffset] + 1;
  } else if (xoffset > xsize - 8 && yoffset < 7) {
    return plane_to_code_lut[(yoffset + 1) * 16 + 8 + (xsize - xoffset)] + 1;
  }
  return dist + 120;
}

// Returns the exact index where array1 and array2 are different. For an index
// inferior or equal to best_len_match, the return value just has to be strictly
// inferior to best_len_match. The current behavior is to return 0 if this index
// is best_len_match, and the index itself otherwise.
// If no two elements are the same, it returns max_limit.
static WEBP_INLINE int FindMatchLength(const uint32_t* const array1,
                                       const uint32_t* const array2,
                                       int best_len_match, int max_limit) {
  // Before 'expensive' linear match, check if the two arrays match at the
  // current best length index.
  if (array1[best_len_match] != array2[best_len_match]) return 0;

  return VP8LVectorMismatch(array1, array2, max_limit);
}

// -----------------------------------------------------------------------------
//  VP8LBackwardRefs

struct PixOrCopyBlock {
  PixOrCopyBlock* next;  // next block (or NULL)
  PixOrCopy* start;      // data start
  int size;              // currently used size
};

extern void VP8LClearBackwardRefs(VP8LBackwardRefs* const refs);
void VP8LClearBackwardRefs(VP8LBackwardRefs* const refs) {
  assert(refs != NULL);
  if (refs->tail != NULL) {
    *refs->tail = refs->free_blocks;  // recycle all blocks at once
  }
  refs->free_blocks = refs->refs;
  refs->tail = &refs->refs;
  refs->last_block = NULL;
  refs->refs = NULL;
}

void VP8LBackwardRefsClear(VP8LBackwardRefs* const refs) {
  assert(refs != NULL);
  VP8LClearBackwardRefs(refs);
  while (refs->free_blocks != NULL) {
    PixOrCopyBlock* const next = refs->free_blocks->next;
    WebPSafeFree(refs->free_blocks);
    refs->free_blocks = next;
  }
}

// Swaps the content of two VP8LBackwardRefs.
static void BackwardRefsSwap(VP8LBackwardRefs* const refs1,
                             VP8LBackwardRefs* const refs2) {
  const int point_to_refs1 =
      (refs1->tail != NULL && refs1->tail == &refs1->refs);
  const int point_to_refs2 =
      (refs2->tail != NULL && refs2->tail == &refs2->refs);
  const VP8LBackwardRefs tmp = *refs1;
  *refs1 = *refs2;
  *refs2 = tmp;
  if (point_to_refs2) refs1->tail = &refs1->refs;
  if (point_to_refs1) refs2->tail = &refs2->refs;
}

void VP8LBackwardRefsInit(VP8LBackwardRefs* const refs, int block_size) {
  assert(refs != NULL);
  memset(refs, 0, sizeof(*refs));
  refs->tail = &refs->refs;
  refs->block_size =
      (block_size < MIN_BLOCK_SIZE) ? MIN_BLOCK_SIZE : block_size;
}

VP8LRefsCursor VP8LRefsCursorInit(const VP8LBackwardRefs* const refs) {
  VP8LRefsCursor c;
  c.cur_block = refs->refs;
  if (refs->refs != NULL) {
    c.cur_pos = c.cur_block->start;
    c.last_pos = c.cur_pos + c.cur_block->size;
  } else {
    c.cur_pos = NULL;
    c.last_pos = NULL;
  }
  return c;
}

void VP8LRefsCursorNextBlock(VP8LRefsCursor* const c) {
  PixOrCopyBlock* const b = c->cur_block->next;
  c->cur_pos = (b == NULL) ? NULL : b->start;
  c->last_pos = (b == NULL) ? NULL : b->start + b->size;
  c->cur_block = b;
}

// Create a new block, either from the free list or allocated
static PixOrCopyBlock* BackwardRefsNewBlock(VP8LBackwardRefs* const refs) {
  PixOrCopyBlock* b = refs->free_blocks;
  if (b == NULL) {  // allocate new memory chunk
    const size_t total_size = sizeof(*b) + refs->block_size * sizeof(*b->start);
    b = (PixOrCopyBlock*)WebPSafeMalloc(1ULL, total_size);
    if (b == NULL) {
      refs->error |= 1;
      return NULL;
    }
    b->start = (PixOrCopy*)((uint8_t*)b + sizeof(*b));  // not always aligned
  } else {  // recycle from free-list
    refs->free_blocks = b->next;
  }
  *refs->tail = b;
  refs->tail = &b->next;
  refs->last_block = b;
  b->next = NULL;
  b->size = 0;
  return b;
}

// Return 1 on success, 0 on error.
static int BackwardRefsClone(const VP8LBackwardRefs* const from,
                             VP8LBackwardRefs* const to) {
  const PixOrCopyBlock* block_from = from->refs;
  VP8LClearBackwardRefs(to);
  while (block_from != NULL) {
    PixOrCopyBlock* const block_to = BackwardRefsNewBlock(to);
    if (block_to == NULL) return 0;
    memcpy(block_to->start, block_from->start,
           block_from->size * sizeof(PixOrCopy));
    block_to->size = block_from->size;
    block_from = block_from->next;
  }
  return 1;
}

extern void VP8LBackwardRefsCursorAdd(VP8LBackwardRefs* const refs,
                                      const PixOrCopy v);
void VP8LBackwardRefsCursorAdd(VP8LBackwardRefs* const refs,
                               const PixOrCopy v) {
  PixOrCopyBlock* b = refs->last_block;
  if (b == NULL || b->size == refs->block_size) {
    b = BackwardRefsNewBlock(refs);
    if (b == NULL) return;  // refs->error is set
  }
  b->start[b->size++] = v;
}

// -----------------------------------------------------------------------------
// Hash chains

int VP8LHashChainInit(VP8LHashChain* const p, int size) {
  assert(p->size == 0);
  assert(p->offset_length == NULL);
  assert(size > 0);
  p->offset_length = (uint32_t*)WebPSafeMalloc(size, sizeof(*p->offset_length));
  if (p->offset_length == NULL) return 0;
  p->size = size;

  return 1;
}

void VP8LHashChainClear(VP8LHashChain* const p) {
  assert(p != NULL);
  WebPSafeFree(p->offset_length);

  p->size = 0;
  p->offset_length = NULL;
}

// -----------------------------------------------------------------------------

static const uint32_t kHashMultiplierHi = 0xc6a4a793u;
static const uint32_t kHashMultiplierLo = 0x5bd1e996u;

static WEBP_UBSAN_IGNORE_UNSIGNED_OVERFLOW WEBP_INLINE uint32_t
GetPixPairHash64(const uint32_t* const argb) {
  uint32_t key;
  key = argb[1] * kHashMultiplierHi;
  key += argb[0] * kHashMultiplierLo;
  key = key >> (32 - HASH_BITS);
  return key;
}

// Returns the maximum number of hash chain lookups to do for a
// given compression quality. Return value in range [8, 86].
static int GetMaxItersForQuality(int quality) {
  return 8 + (quality * quality) / 128;
}

static int GetWindowSizeForHashChain(int quality, int xsize) {
  const int max_window_size = (quality > 75)   ? WINDOW_SIZE
                              : (quality > 50) ? (xsize << 8)
                              : (quality > 25) ? (xsize << 6)
                                               : (xsize << 4);
  assert(xsize > 0);
  return (max_window_size > WINDOW_SIZE) ? WINDOW_SIZE : max_window_size;
}

static WEBP_INLINE int MaxFindCopyLength(int len) {
  return (len < MAX_LENGTH) ? len : MAX_LENGTH;
}

// Finds the best match interval (an offset to the pixel and a length) at each
// position in [lo, from] (scanned in decreasing order), reading the hash
// chain links from 'chain' and storing the result in p->offset_length.
// Positions lower than 'lo' are not written to.
// 'pic' can be NULL, in which case no progress is reported and the function
// cannot fail. Returns 0 on user abort.
static int HashChainFillSearch(VP8LHashChain* const p,
                               const int32_t* const chain,
                               const uint32_t* const argb, int size, int xsize,
                               int iter_max, uint32_t window_size,
                               int low_effort, int lo, int from,
                               const WebPPicture* const pic, int percent_range,
                               int percent_start, int* const percent) {
  uint32_t base_position;
  for (base_position = from; base_position >= (uint32_t)lo;) {
    const int max_len = MaxFindCopyLength(size - 1 - base_position);
    const uint32_t* const argb_start = argb + base_position;
    int iter = iter_max;
    int best_length = 0;
    uint32_t best_distance = 0;
    uint32_t best_argb;
    const int min_pos =
        (base_position > window_size) ? base_position - window_size : 0;
    const int length_max = (max_len < 256) ? max_len : 256;
    uint32_t max_base_position;
    int pos;

    pos = chain[base_position];
    if (!low_effort) {
      int curr_length;
      // Heuristic: use the comparison with the above line as an initialization.
      if (base_position >= (uint32_t)xsize) {
        curr_length = FindMatchLength(argb_start - xsize, argb_start,
                                      best_length, max_len);
        if (curr_length > best_length) {
          best_length = curr_length;
          best_distance = xsize;
        }
        --iter;
      }
      // Heuristic: compare to the previous pixel.
      curr_length =
          FindMatchLength(argb_start - 1, argb_start, best_length, max_len);
      if (curr_length > best_length) {
        best_length = curr_length;
        best_distance = 1;
      }
      --iter;
      // Skip the for loop if we already have the maximum.
      if (best_length == MAX_LENGTH) pos = min_pos - 1;
    }
    best_argb = argb_start[best_length];

    for (; pos >= min_pos && --iter; pos = chain[pos]) {
      int curr_length;
      assert(base_position > (uint32_t)pos);

      if (argb[pos + best_length] != best_argb) continue;

      curr_length = VP8LVectorMismatch(argb + pos, argb_start, max_len);
      if (best_length < curr_length) {
        best_length = curr_length;
        best_distance = base_position - pos;
        best_argb = argb_start[best_length];
        // Stop if we have reached a good enough length.
        if (best_length >= length_max) break;
      }
    }
    // We have the best match but in case the two intervals continue matching
    // to the left, we have the best matches for the left-extended pixels.
    max_base_position = base_position;
    while (1) {
      assert(best_length <= MAX_LENGTH);
      assert(best_distance <= WINDOW_SIZE);
      p->offset_length[base_position] =
          (best_distance << MAX_LENGTH_BITS) | (uint32_t)best_length;
      --base_position;
      // Stop if we don't have a match or if we are out of bounds.
      if (best_distance == 0 || base_position < (uint32_t)lo) break;
      // Stop if we cannot extend the matching intervals to the left.
      if (base_position < best_distance ||
          argb[base_position - best_distance] != argb[base_position]) {
        break;
      }
      // Stop if we are matching at its limit because there could be a closer
      // matching interval with the same maximum length. Then again, if the
      // matching interval is as close as possible (best_distance == 1), we will
      // never find anything better so let's continue.
      if (best_length == MAX_LENGTH && best_distance != 1 &&
          base_position + MAX_LENGTH < max_base_position) {
        break;
      }
      if (best_length < MAX_LENGTH) {
        ++best_length;
        max_base_position = base_position;
      }
    }

    if (pic != NULL &&
        !WebPReportProgress(pic,
                            percent_start + percent_range *
                                                (size - 2 - base_position) /
                                                (size - 2),
                            percent)) {
      return 0;
    }
  }
  return 1;
}

// The match search is processed in chunks of this many positions, which can
// be handed out to workers. Matches do not get extended to the left across
// chunk boundaries (in the single-threaded search either), making the output
// identical whatever the number of threads (including one). The value
// balances multi-thread load distribution (favoring small chunks) against
// the compression cost of the boundaries (negligible at and above 64K
// positions, measurable at 32K and below).
#ifndef HASH_CHAIN_CHUNK_SIZE  /* overridable for experiments */
#define HASH_CHAIN_CHUNK_SIZE (1 << 16)
#endif
#define HASH_CHAIN_MAX_WORKERS 8

typedef struct {
  VP8LHashChain* p;
  const int32_t* chain;
  const uint32_t* argb;
  int size;
  int xsize;
  int iter_max;
  uint32_t window_size;
  int low_effort;
  int first_chunk;
  int num_workers;
} HashChainFillJob;

static int HashChainFillJobHook(void* arg1, void* arg2) {
  const HashChainFillJob* const job = (const HashChainFillJob*)arg1;
  int chunk;
  (void)arg2;
  for (chunk = job->first_chunk;
       1 + chunk * HASH_CHAIN_CHUNK_SIZE <= job->size - 2;
       chunk += job->num_workers) {
    const int lo = 1 + chunk * HASH_CHAIN_CHUNK_SIZE;
    int from = lo + HASH_CHAIN_CHUNK_SIZE - 1;
    if (from > job->size - 2) from = job->size - 2;
    HashChainFillSearch(job->p, job->chain, job->argb, job->size, job->xsize,
                        job->iter_max, job->window_size, job->low_effort, lo,
                        from, /*pic=*/NULL, /*percent_range=*/0,
                        /*percent_start=*/0, /*percent=*/NULL);
  }
  return 1;
}

// Multi-threaded version of the match search: the chain is copied so that the
// workers can write their match intervals to p->offset_length (which holds
// the chain links in the single-threaded version) independently.
// Returns 0 if multi-threading could not be used (missing memory or thread
// resources): the caller should then fall back to the single-threaded search.
static int HashChainFillSearchMT(VP8LHashChain* const p,
                                 const uint32_t* const argb, int size,
                                 int xsize, int iter_max, uint32_t window_size,
                                 int low_effort) {
  const int num_chunks =
      (size - 2 + HASH_CHAIN_CHUNK_SIZE - 1) / HASH_CHAIN_CHUNK_SIZE;
  const int num_workers = (num_chunks < HASH_CHAIN_MAX_WORKERS)
                              ? num_chunks
                              : HASH_CHAIN_MAX_WORKERS;
  const WebPWorkerInterface* const worker_interface = WebPGetWorkerInterface();
  WebPWorker workers[HASH_CHAIN_MAX_WORKERS];
  HashChainFillJob jobs[HASH_CHAIN_MAX_WORKERS];
  int i, ok = 1;
  int32_t* const chain =
      (int32_t*)WebPSafeMalloc(size, sizeof(*chain));
  if (chain == NULL) return 0;
  memcpy(chain, p->offset_length, size * sizeof(*chain));

  for (i = 0; i < num_workers; ++i) {
    jobs[i].p = p;
    jobs[i].chain = chain;
    jobs[i].argb = argb;
    jobs[i].size = size;
    jobs[i].xsize = xsize;
    jobs[i].iter_max = iter_max;
    jobs[i].window_size = window_size;
    jobs[i].low_effort = low_effort;
    jobs[i].first_chunk = i;
    jobs[i].num_workers = num_workers;
    worker_interface->Init(&workers[i]);
    workers[i].data1 = &jobs[i];
    workers[i].data2 = NULL;
    workers[i].hook = HashChainFillJobHook;
  }
  for (i = 0; ok && i < num_workers; ++i) {
    ok = worker_interface->Reset(&workers[i]);
  }
  if (ok) {
    p->offset_length[0] = p->offset_length[size - 1] = 0;
    for (i = 0; i < num_workers; ++i) {
      worker_interface->Launch(&workers[i]);
    }
    for (i = 0; i < num_workers; ++i) {
      ok &= worker_interface->Sync(&workers[i]);
    }
  }
  for (i = 0; i < num_workers; ++i) {
    worker_interface->End(&workers[i]);
  }
  WebPSafeFree(chain);
  return ok;
}

int VP8LHashChainFill(VP8LHashChain* const p, int quality,
                      const uint32_t* const argb, int xsize, int ysize,
                      int low_effort, int use_threads,
                      const WebPPicture* const pic, int percent_range,
                      int* const percent) {
  const int size = xsize * ysize;
  const int iter_max = GetMaxItersForQuality(quality);
  const uint32_t window_size = GetWindowSizeForHashChain(quality, xsize);
  int remaining_percent = percent_range;
  int percent_start = *percent;
  int pos;
  int argb_comp;
  int32_t* hash_to_first_index;
  // Temporarily use the p->offset_length as a hash chain.
  int32_t* chain = (int32_t*)p->offset_length;
  assert(size > 0);
  assert(p->size != 0);
  assert(p->offset_length != NULL);

  if (size <= 2) {
    p->offset_length[0] = p->offset_length[size - 1] = 0;
    return 1;
  }

  hash_to_first_index =
      (int32_t*)WebPSafeMalloc(HASH_SIZE, sizeof(*hash_to_first_index));
  if (hash_to_first_index == NULL) {
    return WebPEncodingSetError(pic, VP8_ENC_ERROR_OUT_OF_MEMORY);
  }

  percent_range = remaining_percent / 2;
  remaining_percent -= percent_range;

  // Set the int32_t array to -1.
  memset(hash_to_first_index, 0xff, HASH_SIZE * sizeof(*hash_to_first_index));
  // Fill the chain linking pixels with the same hash.
  argb_comp = (argb[0] == argb[1]);
  for (pos = 0; pos < size - 2;) {
    uint32_t hash_code;
    const int argb_comp_next = (argb[pos + 1] == argb[pos + 2]);
    if (argb_comp && argb_comp_next) {
      // Consecutive pixels with the same color will share the same hash.
      // We therefore use a different hash: the color and its repetition
      // length.
      uint32_t tmp[2];
      uint32_t len = 1;
      tmp[0] = argb[pos];
      // Figure out how far the pixels are the same.
      // The last pixel has a different 64 bit hash, as its next pixel does
      // not have the same color, so we just need to get to the last pixel equal
      // to its follower.
      while (pos + (int)len + 2 < size && argb[pos + len + 2] == argb[pos]) {
        ++len;
      }
      if (len > MAX_LENGTH) {
        // Skip the pixels that match for distance=1 and length>MAX_LENGTH
        // because they are linked to their predecessor and we automatically
        // check that in the main for loop below. Skipping means setting no
        // predecessor in the chain, hence -1.
        memset(chain + pos, 0xff, (len - MAX_LENGTH) * sizeof(*chain));
        pos += len - MAX_LENGTH;
        len = MAX_LENGTH;
      }
      // Process the rest of the hash chain.
      while (len) {
        tmp[1] = len--;
        hash_code = GetPixPairHash64(tmp);
        chain[pos] = hash_to_first_index[hash_code];
        hash_to_first_index[hash_code] = pos++;
      }
      argb_comp = 0;
    } else {
      // Just move one pixel forward.
      hash_code = GetPixPairHash64(argb + pos);
      chain[pos] = hash_to_first_index[hash_code];
      hash_to_first_index[hash_code] = pos++;
      argb_comp = argb_comp_next;
    }

    if (!WebPReportProgress(
            pic, percent_start + percent_range * pos / (size - 2), percent)) {
      WebPSafeFree(hash_to_first_index);
      return 0;
    }
  }
  // Process the penultimate pixel.
  chain[pos] = hash_to_first_index[GetPixPairHash64(argb + pos)];

  WebPSafeFree(hash_to_first_index);

  percent_start += percent_range;
  if (!WebPReportProgress(pic, percent_start, percent)) return 0;
  percent_range = remaining_percent;

  // Find the best match interval at each pixel, defined by an offset to the
  // pixel and a length. The right-most pixel cannot match anything to the right
  // (hence a best length of 0) and the left-most pixel nothing to the left
  // (hence an offset of 0).
  assert(size > 2);
  {
    const int num_chunks =
        (size - 2 + HASH_CHAIN_CHUNK_SIZE - 1) / HASH_CHAIN_CHUNK_SIZE;
    int chunk;
    if (use_threads > 0 && num_chunks > 1 &&
        HashChainFillSearchMT(p, argb, size, xsize, iter_max, window_size,
                              low_effort)) {
      return WebPReportProgress(pic, percent_start + percent_range, percent);
    }
    // Single-threaded search (or fall back to it on thread or memory
    // failure), over the exact same chunks: the output does not depend on
    // 'use_threads'. The chunks are processed from the highest positions down
    // so that the chain links (stored in place of the match intervals) are
    // still intact when a lower chunk reads them, and so that the reported
    // progress increases monotonically.
    p->offset_length[0] = p->offset_length[size - 1] = 0;
    for (chunk = num_chunks - 1; chunk >= 0; --chunk) {
      const int lo = 1 + chunk * HASH_CHAIN_CHUNK_SIZE;
      int from = lo + HASH_CHAIN_CHUNK_SIZE - 1;
      if (from > size - 2) from = size - 2;
      if (!HashChainFillSearch(p, chain, argb, size, xsize, iter_max,
                               window_size, low_effort, lo, from, pic,
                               percent_range, percent_start, percent)) {
        return 0;
      }
    }
  }

  return WebPReportProgress(pic, percent_start + percent_range, percent);
}

static WEBP_INLINE void AddSingleLiteral(uint32_t pixel, int use_color_cache,
                                         VP8LColorCache* const hashers,
                                         VP8LBackwardRefs* const refs) {
  PixOrCopy v;
  if (use_color_cache) {
    const uint32_t key = VP8LColorCacheGetIndex(hashers, pixel);
    if (VP8LColorCacheLookup(hashers, key) == pixel) {
      v = PixOrCopyCreateCacheIdx(key);
    } else {
      v = PixOrCopyCreateLiteral(pixel);
      VP8LColorCacheSet(hashers, key, pixel);
    }
  } else {
    v = PixOrCopyCreateLiteral(pixel);
  }
  VP8LBackwardRefsCursorAdd(refs, v);
}

static int BackwardReferencesRle(int xsize, int ysize,
                                 const uint32_t* const argb, int cache_bits,
                                 VP8LBackwardRefs* const refs) {
  const int pix_count = xsize * ysize;
  int i, k;
  const int use_color_cache = (cache_bits > 0);
  VP8LColorCache hashers;

  if (use_color_cache && !VP8LColorCacheInit(&hashers, cache_bits)) {
    return 0;
  }
  VP8LClearBackwardRefs(refs);
  // Add first pixel as literal.
  AddSingleLiteral(argb[0], use_color_cache, &hashers, refs);
  i = 1;
  while (i < pix_count) {
    const int max_len = MaxFindCopyLength(pix_count - i);
    const int rle_len = FindMatchLength(argb + i, argb + i - 1, 0, max_len);
    const int prev_row_len =
        (i < xsize) ? 0
                    : FindMatchLength(argb + i, argb + i - xsize, 0, max_len);
    if (rle_len >= prev_row_len && rle_len >= MIN_LENGTH) {
      VP8LBackwardRefsCursorAdd(refs, PixOrCopyCreateCopy(1, rle_len));
      // We don't need to update the color cache here since it is always the
      // same pixel being copied, and that does not change the color cache
      // state.
      i += rle_len;
    } else if (prev_row_len >= MIN_LENGTH) {
      VP8LBackwardRefsCursorAdd(refs, PixOrCopyCreateCopy(xsize, prev_row_len));
      if (use_color_cache) {
        for (k = 0; k < prev_row_len; ++k) {
          VP8LColorCacheInsert(&hashers, argb[i + k]);
        }
      }
      i += prev_row_len;
    } else {
      AddSingleLiteral(argb[i], use_color_cache, &hashers, refs);
      i++;
    }
  }
  if (use_color_cache) VP8LColorCacheClear(&hashers);
  return !refs->error;
}

static int BackwardReferencesLz77(int xsize, int ysize,
                                  const uint32_t* const argb, int cache_bits,
                                  const VP8LHashChain* const hash_chain,
                                  VP8LBackwardRefs* const refs) {
  int i;
  int i_last_check = -1;
  int ok = 0;
  int cc_init = 0;
  const int use_color_cache = (cache_bits > 0);
  const int pix_count = xsize * ysize;
  VP8LColorCache hashers;

  if (use_color_cache) {
    cc_init = VP8LColorCacheInit(&hashers, cache_bits);
    if (!cc_init) goto Error;
  }
  VP8LClearBackwardRefs(refs);
  for (i = 0; i < pix_count;) {
    // Alternative#1: Code the pixels starting at 'i' using backward reference.
    int offset = 0;
    int len = 0;
    int j;
    VP8LHashChainFindCopy(hash_chain, i, &offset, &len);
    if (len >= MIN_LENGTH) {
      const int len_ini = len;
      int max_reach = 0;
      const int j_max =
          (i + len_ini >= pix_count) ? pix_count - 1 : i + len_ini;
      // Only start from what we have not checked already.
      i_last_check = (i > i_last_check) ? i : i_last_check;
      // We know the best match for the current pixel but we try to find the
      // best matches for the current pixel AND the next one combined.
      // The naive method would use the intervals:
      // [i,i+len) + [i+len, length of best match at i+len)
      // while we check if we can use:
      // [i,j) (where j<=i+len) + [j, length of best match at j)
      for (j = i_last_check + 1; j <= j_max; ++j) {
        const int len_j = VP8LHashChainFindLength(hash_chain, j);
        const int reach =
            j + (len_j >= MIN_LENGTH ? len_j : 1);  // 1 for single literal.
        if (reach > max_reach) {
          len = j - i;
          max_reach = reach;
          if (max_reach >= pix_count) break;
        }
      }
    } else {
      len = 1;
    }
    // Go with literal or backward reference.
    assert(len > 0);
    if (len == 1) {
      AddSingleLiteral(argb[i], use_color_cache, &hashers, refs);
    } else {
      VP8LBackwardRefsCursorAdd(refs, PixOrCopyCreateCopy(offset, len));
      if (use_color_cache) {
        for (j = i; j < i + len; ++j) VP8LColorCacheInsert(&hashers, argb[j]);
      }
    }
    i += len;
  }

  ok = !refs->error;
Error:
  if (cc_init) VP8LColorCacheClear(&hashers);
  return ok;
}

// Compute an LZ77 by forcing matches to happen within a given distance cost.
// We therefore limit the algorithm to the lowest 32 values in the PlaneCode
// definition.
#define WINDOW_OFFSETS_SIZE_MAX 32
static int BackwardReferencesLz77Box(int xsize, int ysize,
                                     const uint32_t* const argb, int cache_bits,
                                     const VP8LHashChain* const hash_chain_best,
                                     VP8LHashChain* hash_chain,
                                     VP8LBackwardRefs* const refs) {
  int i;
  const int pix_count = xsize * ysize;
  uint16_t* counts;
  int window_offsets[WINDOW_OFFSETS_SIZE_MAX] = {0};
  int window_offsets_new[WINDOW_OFFSETS_SIZE_MAX] = {0};
  int window_offsets_size = 0;
  int window_offsets_new_size = 0;
  uint16_t* const counts_ini =
      (uint16_t*)WebPSafeMalloc(xsize * ysize, sizeof(*counts_ini));
  int best_offset_prev = -1, best_length_prev = -1;
  if (counts_ini == NULL) return 0;

  // counts[i] counts how many times a pixel is repeated starting at position i.
  i = pix_count - 2;
  counts = counts_ini + i;
  counts[1] = 1;
  for (; i >= 0; --i, --counts) {
    if (argb[i] == argb[i + 1]) {
      // Max out the counts to MAX_LENGTH.
      counts[0] = counts[1] + (counts[1] != MAX_LENGTH);
    } else {
      counts[0] = 1;
    }
  }

  // Figure out the window offsets around a pixel. They are stored in a
  // spiraling order around the pixel as defined by VP8LDistanceToPlaneCode.
  {
    int x, y;
    for (y = 0; y <= 6; ++y) {
      for (x = -6; x <= 6; ++x) {
        const int offset = y * xsize + x;
        int plane_code;
        // Ignore offsets that bring us after the pixel.
        if (offset <= 0) continue;
        plane_code = VP8LDistanceToPlaneCode(xsize, offset) - 1;
        if (plane_code >= WINDOW_OFFSETS_SIZE_MAX) continue;
        window_offsets[plane_code] = offset;
      }
    }
    // For narrow images, not all plane codes are reached, so remove those.
    for (i = 0; i < WINDOW_OFFSETS_SIZE_MAX; ++i) {
      if (window_offsets[i] == 0) continue;
      window_offsets[window_offsets_size++] = window_offsets[i];
    }
    // Given a pixel P, find the offsets that reach pixels unreachable from P-1
    // with any of the offsets in window_offsets[].
    for (i = 0; i < window_offsets_size; ++i) {
      int j;
      int is_reachable = 0;
      for (j = 0; j < window_offsets_size && !is_reachable; ++j) {
        is_reachable |= (window_offsets[i] == window_offsets[j] + 1);
      }
      if (!is_reachable) {
        window_offsets_new[window_offsets_new_size] = window_offsets[i];
        ++window_offsets_new_size;
      }
    }
  }

  hash_chain->offset_length[0] = 0;
  for (i = 1; i < pix_count; ++i) {
    int ind;
    int best_length = VP8LHashChainFindLength(hash_chain_best, i);
    int best_offset;
    int do_compute = 1;

    if (best_length >= MAX_LENGTH) {
      // Do not recompute the best match if we already have a maximal one in the
      // window.
      best_offset = VP8LHashChainFindOffset(hash_chain_best, i);
      for (ind = 0; ind < window_offsets_size; ++ind) {
        if (best_offset == window_offsets[ind]) {
          do_compute = 0;
          break;
        }
      }
    }
    if (do_compute) {
      // Figure out if we should use the offset/length from the previous pixel
      // as an initial guess and therefore only inspect the offsets in
      // window_offsets_new[].
      const int use_prev =
          (best_length_prev > 1) && (best_length_prev < MAX_LENGTH);
      const int num_ind =
          use_prev ? window_offsets_new_size : window_offsets_size;
      best_length = use_prev ? best_length_prev - 1 : 0;
      best_offset = use_prev ? best_offset_prev : 0;
      // Find the longest match in a window around the pixel.
      for (ind = 0; ind < num_ind; ++ind) {
        int curr_length = 0;
        int j = i;
        int j_offset =
            use_prev ? i - window_offsets_new[ind] : i - window_offsets[ind];
        if (j_offset < 0 || argb[j_offset] != argb[i]) continue;
        // The longest match is the sum of how many times each pixel is
        // repeated.
        do {
          const int counts_j_offset = counts_ini[j_offset];
          const int counts_j = counts_ini[j];
          if (counts_j_offset != counts_j) {
            curr_length +=
                (counts_j_offset < counts_j) ? counts_j_offset : counts_j;
            break;
          }
          // The same color is repeated counts_pos times at j_offset and j.
          curr_length += counts_j_offset;
          j_offset += counts_j_offset;
          j += counts_j_offset;
        } while (curr_length <= MAX_LENGTH && j < pix_count &&
                 argb[j_offset] == argb[j]);
        if (best_length < curr_length) {
          best_offset =
              use_prev ? window_offsets_new[ind] : window_offsets[ind];
          if (curr_length >= MAX_LENGTH) {
            best_length = MAX_LENGTH;
            break;
          } else {
            best_length = curr_length;
          }
        }
      }
    }

    assert(i + best_length <= pix_count);
    assert(best_length <= MAX_LENGTH);
    if (best_length <= MIN_LENGTH) {
      hash_chain->offset_length[i] = 0;
      best_offset_prev = 0;
      best_length_prev = 0;
    } else {
      hash_chain->offset_length[i] =
          (best_offset << MAX_LENGTH_BITS) | (uint32_t)best_length;
      best_offset_prev = best_offset;
      best_length_prev = best_length;
    }
  }
  hash_chain->offset_length[0] = 0;
  WebPSafeFree(counts_ini);

  return BackwardReferencesLz77(xsize, ysize, argb, cache_bits, hash_chain,
                                refs);
}

// -----------------------------------------------------------------------------

static void BackwardReferences2DLocality(int xsize,
                                         const VP8LBackwardRefs* const refs) {
  VP8LRefsCursor c = VP8LRefsCursorInit(refs);
  while (VP8LRefsCursorOk(&c)) {
    if (PixOrCopyIsCopy(c.cur_pos)) {
      const int dist = c.cur_pos->argb_or_distance;
      const int transformed_dist = VP8LDistanceToPlaneCode(xsize, dist);
      c.cur_pos->argb_or_distance = transformed_dist;
    }
    VP8LRefsCursorNext(&c);
  }
}

// Evaluate optimal cache bits for the local color cache.
// The input *best_cache_bits sets the maximum cache bits to use (passing 0
// implies disabling the local color cache). The local color cache is also
// disabled for the lower (<= 25) quality.
// Returns 0 in case of memory error.
static int CalculateBestCacheSize(const uint32_t* argb, int quality,
                                  const VP8LBackwardRefs* const refs,
                                  int* const best_cache_bits) {
  int i;
  const int cache_bits_max = (quality <= 25) ? 0 : *best_cache_bits;
  uint64_t entropy_min = WEBP_UINT64_MAX;
  int cc_init[MAX_COLOR_CACHE_BITS + 1] = {0};
  VP8LColorCache hashers[MAX_COLOR_CACHE_BITS + 1];
  VP8LRefsCursor c = VP8LRefsCursorInit(refs);
  VP8LHistogram* histos[MAX_COLOR_CACHE_BITS + 1] = {NULL};
  int ok = 0;

  assert(cache_bits_max >= 0 && cache_bits_max <= MAX_COLOR_CACHE_BITS);

  if (cache_bits_max == 0) {
    *best_cache_bits = 0;
    // Local color cache is disabled.
    return 1;
  }

  // Allocate data.
  for (i = 0; i <= cache_bits_max; ++i) {
    histos[i] = VP8LAllocateHistogram(i);
    if (histos[i] == NULL) goto Error;
    VP8LHistogramInit(histos[i], i, /*init_arrays=*/1);
    if (i == 0) continue;
    cc_init[i] = VP8LColorCacheInit(&hashers[i], i);
    if (!cc_init[i]) goto Error;
  }

  // Find the cache_bits giving the lowest entropy. The search is done in a
  // brute-force way as the function (entropy w.r.t cache_bits) can be
  // anything in practice.
  while (VP8LRefsCursorOk(&c)) {
    const PixOrCopy* const v = c.cur_pos;
    if (PixOrCopyIsLiteral(v)) {
      const uint32_t pix = *argb++;
      const uint32_t a = (pix >> 24) & 0xff;
      const uint32_t r = (pix >> 16) & 0xff;
      const uint32_t g = (pix >> 8) & 0xff;
      const uint32_t b = (pix >> 0) & 0xff;
      // The keys of the caches can be derived from the longest one.
      int key = VP8LHashPix(pix, 32 - cache_bits_max);
      // Do not use the color cache for cache_bits = 0.
      ++histos[0]->blue[b];
      ++histos[0]->literal[g];
      ++histos[0]->red[r];
      ++histos[0]->alpha[a];
      // Deal with cache_bits > 0.
      for (i = cache_bits_max; i >= 1; --i, key >>= 1) {
        if (VP8LColorCacheLookup(&hashers[i], key) == pix) {
          ++histos[i]->literal[NUM_LITERAL_CODES + NUM_LENGTH_CODES + key];
        } else {
          VP8LColorCacheSet(&hashers[i], key, pix);
          ++histos[i]->blue[b];
          ++histos[i]->literal[g];
          ++histos[i]->red[r];
          ++histos[i]->alpha[a];
        }
      }
    } else {
      int code, extra_bits, extra_bits_value;
      // We should compute the contribution of the (distance,length)
      // histograms but those are the same independently from the cache size.
      // As those constant contributions are in the end added to the other
      // histogram contributions, we can ignore them, except for the length
      // prefix that is part of the 'literal' histogram.
      int len = PixOrCopyLength(v);
      uint32_t argb_prev = *argb ^ 0xffffffffu;
      VP8LPrefixEncode(len, &code, &extra_bits, &extra_bits_value);
      for (i = 0; i <= cache_bits_max; ++i) {
        ++histos[i]->literal[NUM_LITERAL_CODES + code];
      }
      // Update the color caches.
      do {
        if (*argb != argb_prev) {
          // Efficiency: insert only if the color changes.
          int key = VP8LHashPix(*argb, 32 - cache_bits_max);
          for (i = cache_bits_max; i >= 1; --i, key >>= 1) {
            hashers[i].colors[key] = *argb;
          }
          argb_prev = *argb;
        }
        argb++;
      } while (--len != 0);
    }
    VP8LRefsCursorNext(&c);
  }

  for (i = 0; i <= cache_bits_max; ++i) {
    const uint64_t entropy = VP8LHistogramEstimateBits(histos[i]);
    if (i == 0 || entropy < entropy_min) {
      entropy_min = entropy;
      *best_cache_bits = i;
    }
  }
  ok = 1;
Error:
  for (i = 0; i <= cache_bits_max; ++i) {
    if (cc_init[i]) VP8LColorCacheClear(&hashers[i]);
    VP8LFreeHistogram(histos[i]);
  }
  return ok;
}

// Update (in-place) backward references for specified cache_bits.
static int BackwardRefsWithLocalCache(const uint32_t* const argb,
                                      int cache_bits,
                                      VP8LBackwardRefs* const refs) {
  int pixel_index = 0;
  VP8LColorCache hashers;
  VP8LRefsCursor c = VP8LRefsCursorInit(refs);
  if (!VP8LColorCacheInit(&hashers, cache_bits)) return 0;

  while (VP8LRefsCursorOk(&c)) {
    PixOrCopy* const v = c.cur_pos;
    if (PixOrCopyIsLiteral(v)) {
      const uint32_t argb_literal = v->argb_or_distance;
      const int ix = VP8LColorCacheContains(&hashers, argb_literal);
      if (ix >= 0) {
        // hashers contains argb_literal
        *v = PixOrCopyCreateCacheIdx(ix);
      } else {
        VP8LColorCacheInsert(&hashers, argb_literal);
      }
      ++pixel_index;
    } else {
      // refs was created without local cache, so it can not have cache indexes.
      int k;
      assert(PixOrCopyIsCopy(v));
      for (k = 0; k < v->len; ++k) {
        VP8LColorCacheInsert(&hashers, argb[pixel_index++]);
      }
    }
    VP8LRefsCursorNext(&c);
  }
  VP8LColorCacheClear(&hashers);
  return 1;
}

static VP8LBackwardRefs* GetBackwardReferencesLowEffort(
    int width, int height, const uint32_t* const argb, int* const cache_bits,
    const VP8LHashChain* const hash_chain, VP8LBackwardRefs* const refs_lz77) {
  *cache_bits = 0;
  if (!BackwardReferencesLz77(width, height, argb, 0, hash_chain, refs_lz77)) {
    return NULL;
  }
  BackwardReferences2DLocality(width, refs_lz77);
  return refs_lz77;
}

extern int VP8LBackwardReferencesTraceBackwards(
    int xsize, int ysize, const uint32_t* const argb, int cache_bits,
    const VP8LHashChain* const hash_chain,
    const VP8LBackwardRefs* const refs_src, VP8LBackwardRefs* const refs_dst);

// One LZ77 variant evaluation: computes the backward references of the
// variant and their bit costs with and without a color cache. The jobs are
// independent of each other and can run on separate threads; the (strictly
// ordered) comparison of their costs is done by the caller afterwards.
typedef struct {
  // Input.
  int lz77_type;
  int width, height;
  const uint32_t* argb;
  int quality;
  int cache_bits_max;
  int do_no_cache;
  const VP8LHashChain* hash_chain;
  // Output.
  VP8LBackwardRefs refs_raw;     // references without a color cache
  VP8LBackwardRefs refs_cached;  // references with the best color cache
  VP8LHashChain hash_chain_box;  // only used by kLZ77Box
  int cache_bits;                // best color cache size (0 = none)
  uint64_t bit_cost[2];          // cost with ([0]) and without ([1]) cache
  int ok;
} LZ77Job;

static int RunLZ77Job(LZ77Job* const job) {
  const int width = job->width, height = job->height;
  const uint32_t* const argb = job->argb;
  VP8LHistogram* const histo = VP8LAllocateHistogram(MAX_COLOR_CACHE_BITS);
  uint64_t bit_cost = 0u;
  int res = 0;
  job->ok = 0;
  if (histo == NULL) return 0;
  switch (job->lz77_type) {
    case kLZ77RLE:
      res = BackwardReferencesRle(width, height, argb, 0, &job->refs_raw);
      break;
    case kLZ77Standard:
      // Compute LZ77 with no cache (0 bits), as the ideal LZ77 with a color
      // cache is not that different in practice.
      res = BackwardReferencesLz77(width, height, argb, 0, job->hash_chain,
                                   &job->refs_raw);
      break;
    case kLZ77Box:
      if (!VP8LHashChainInit(&job->hash_chain_box, width * height)) goto End;
      res = BackwardReferencesLz77Box(width, height, argb, 0, job->hash_chain,
                                      &job->hash_chain_box, &job->refs_raw);
      break;
    default:
      assert(0);
  }
  if (!res) goto End;

  // Start with the no color cache case.
  if (job->do_no_cache) {
    VP8LHistogramCreate(histo, &job->refs_raw, /*cache_bits=*/0);
    bit_cost = VP8LHistogramEstimateBits(histo);
    job->bit_cost[1] = bit_cost;
  }
  // Try with a color cache.
  job->cache_bits = job->cache_bits_max;
  if (!CalculateBestCacheSize(argb, job->quality, &job->refs_raw,
                              &job->cache_bits)) {
    goto End;
  }
  if (!BackwardRefsClone(&job->refs_raw, &job->refs_cached)) goto End;
  if (job->cache_bits > 0) {
    if (!BackwardRefsWithLocalCache(argb, job->cache_bits,
                                    &job->refs_cached)) {
      goto End;
    }
  }
  if (job->do_no_cache && job->cache_bits == 0) {
    // No need to re-compute bit_cost as it was computed without a cache.
    job->bit_cost[0] = bit_cost;
  } else {
    VP8LHistogramCreate(histo, &job->refs_cached, job->cache_bits);
    job->bit_cost[0] = VP8LHistogramEstimateBits(histo);
  }
  job->ok = 1;

 End:
  VP8LFreeHistogram(histo);
  return job->ok;
}

#ifdef WEBP_USE_THREAD
static int LZ77JobHook(void* arg1, void* arg2) {
  (void)arg2;
  return RunLZ77Job((LZ77Job*)arg1);
}
#endif

static int GetBackwardReferences(int width, int height,
                                 const uint32_t* const argb, int quality,
                                 int lz77_types_to_try, int cache_bits_max,
                                 int do_no_cache, int use_threads,
                                 const VP8LHashChain* const hash_chain,
                                 VP8LBackwardRefs* const refs,
                                 int* const cache_bits_best) {
  int i, lz77_type;
  // Index 0 is for a color cache, index 1 for no cache (if needed).
  int lz77_types_best[2] = {0, 0};
  uint64_t bit_costs_best[2] = {WEBP_UINT64_MAX, WEBP_UINT64_MAX};
  const VP8LHashChain* hash_chain_box = NULL;
  VP8LBackwardRefs* const refs_tmp = &refs[do_no_cache ? 2 : 1];
  int status = 0;
  LZ77Job jobs[3];
  int num_jobs = 0;

  for (lz77_type = 1; lz77_types_to_try;
       lz77_types_to_try &= ~lz77_type, lz77_type <<= 1) {
    LZ77Job* job;
    if ((lz77_types_to_try & lz77_type) == 0) continue;
    job = &jobs[num_jobs++];
    memset(job, 0, sizeof(*job));
    job->lz77_type = lz77_type;
    job->width = width;
    job->height = height;
    job->argb = argb;
    job->quality = quality;
    job->cache_bits_max = cache_bits_max;
    job->do_no_cache = do_no_cache;
    job->hash_chain = hash_chain;
    VP8LBackwardRefsInit(&job->refs_raw, refs[0].block_size);
    VP8LBackwardRefsInit(&job->refs_cached, refs[0].block_size);
  }
  assert(num_jobs > 0);

#ifdef WEBP_USE_THREAD
  if (use_threads && num_jobs > 1) {
    // Run the last jobs on worker threads, the first one on this thread.
    const WebPWorkerInterface* const winterface = WebPGetWorkerInterface();
    WebPWorker threads[2];
    int num_threads = 0;
    int main_ok;
    for (i = 1; i < num_jobs; ++i) {
      WebPWorker* const thread = &threads[i - 1];
      winterface->Init(thread);
      thread->data1 = &jobs[i];
      thread->data2 = NULL;
      thread->hook = LZ77JobHook;
      if (!winterface->Reset(thread)) break;
      ++num_threads;
      winterface->Launch(thread);
    }
    for (i = num_threads + 1; i < num_jobs; ++i) {
      RunLZ77Job(&jobs[i]);  // threads that could not start
    }
    main_ok = RunLZ77Job(&jobs[0]);
    for (i = 0; i < num_threads; ++i) {
      winterface->Sync(&threads[i]);
      winterface->End(&threads[i]);
    }
    if (!main_ok) goto Error;
  } else
#else
  (void)use_threads;
#endif
  {
    for (i = 0; i < num_jobs; ++i) {
      if (!RunLZ77Job(&jobs[i])) goto Error;
    }
  }

  // Compare the variants, in the same order as the single-threaded
  // evaluation.
  for (i = 0; i < num_jobs; ++i) {
    LZ77Job* const job = &jobs[i];
    int j;
    if (!job->ok) goto Error;
    // Start with the no color cache case.
    for (j = 1; j >= 0; --j) {
      if (j == 1 && !do_no_cache) continue;
      if (job->bit_cost[j] < bit_costs_best[j]) {
        BackwardRefsSwap((j == 1) ? &job->refs_raw : &job->refs_cached,
                         &refs[j]);
        bit_costs_best[j] = job->bit_cost[j];
        lz77_types_best[j] = job->lz77_type;
        if (j == 0) *cache_bits_best = job->cache_bits;
      }
    }
    if (job->lz77_type == kLZ77Box) hash_chain_box = &job->hash_chain_box;
  }
  assert(lz77_types_best[0] > 0);
  assert(!do_no_cache || lz77_types_best[1] > 0);

  // Improve on simple LZ77 but only for high quality (TraceBackwards is
  // costly).
  for (i = 1; i >= 0; --i) {
    if (i == 1 && !do_no_cache) continue;
    if ((lz77_types_best[i] == kLZ77Standard ||
         lz77_types_best[i] == kLZ77Box) &&
        quality >= 25) {
      const VP8LHashChain* const hash_chain_tmp =
          (lz77_types_best[i] == kLZ77Standard) ? hash_chain : hash_chain_box;
      const int cache_bits = (i == 1) ? 0 : *cache_bits_best;
      uint64_t bit_cost_trace;
      VP8LHistogram* const histo = VP8LAllocateHistogram(MAX_COLOR_CACHE_BITS);
      if (histo == NULL) goto Error;
      if (!VP8LBackwardReferencesTraceBackwards(width, height, argb, cache_bits,
                                                hash_chain_tmp, &refs[i],
                                                refs_tmp)) {
        VP8LFreeHistogram(histo);
        goto Error;
      }
      VP8LHistogramCreate(histo, refs_tmp, cache_bits);
      bit_cost_trace = VP8LHistogramEstimateBits(histo);
      VP8LFreeHistogram(histo);
      if (bit_cost_trace < bit_costs_best[i]) {
        BackwardRefsSwap(refs_tmp, &refs[i]);
      }
    }

    BackwardReferences2DLocality(width, &refs[i]);

    if (i == 1 && lz77_types_best[0] == lz77_types_best[1] &&
        *cache_bits_best == 0) {
      // If the best cache size is 0 and we have the same best LZ77, just copy
      // the data over and stop here.
      if (!BackwardRefsClone(&refs[1], &refs[0])) goto Error;
      break;
    }
  }
  status = 1;

Error:
  for (i = 0; i < num_jobs; ++i) {
    VP8LBackwardRefsClear(&jobs[i].refs_raw);
    VP8LBackwardRefsClear(&jobs[i].refs_cached);
    VP8LHashChainClear(&jobs[i].hash_chain_box);
  }
  return status;
}

int VP8LGetBackwardReferences(
    int width, int height, const uint32_t* const argb, int quality,
    int low_effort, int lz77_types_to_try, int cache_bits_max, int do_no_cache,
    int use_threads, const VP8LHashChain* const hash_chain,
    VP8LBackwardRefs* const refs, int* const cache_bits_best,
    const WebPPicture* const pic, int percent_range, int* const percent) {
  if (low_effort) {
    VP8LBackwardRefs* refs_best;
    *cache_bits_best = cache_bits_max;
    refs_best = GetBackwardReferencesLowEffort(
        width, height, argb, cache_bits_best, hash_chain, refs);
    if (refs_best == NULL) {
      return WebPEncodingSetError(pic, VP8_ENC_ERROR_OUT_OF_MEMORY);
    }
    // Set it in first position.
    BackwardRefsSwap(refs_best, &refs[0]);
  } else {
    if (!GetBackwardReferences(width, height, argb, quality, lz77_types_to_try,
                               cache_bits_max, do_no_cache, use_threads,
                               hash_chain, refs, cache_bits_best)) {
      return WebPEncodingSetError(pic, VP8_ENC_ERROR_OUT_OF_MEMORY);
    }
  }

  return WebPReportProgress(pic, *percent + percent_range, percent);
}
