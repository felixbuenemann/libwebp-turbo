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

#ifndef WEBP_SHARPYUV_SHARPYUV_DSP_H_
#define WEBP_SHARPYUV_SHARPYUV_DSP_H_

#include "./sharpyuv.h"
#include "./sharpyuv_cpu.h"
#include "webp/types.h"

extern uint64_t (*SharpYuvUpdateY)(const uint16_t* src, const uint16_t* ref,
                                   uint16_t* dst, int len, int bit_depth);
extern void (*SharpYuvUpdateRGB)(const int16_t* src, const int16_t* ref,
                                 int16_t* dst, int len);
extern void (*SharpYuvFilterRow)(const int16_t* A, const int16_t* B, int len,
                                 const uint16_t* best_y, uint16_t* out,
                                 int bit_depth);
// Converts a row of planar RGB samples ('src': R row then G row then B row,
// each 'w' samples) to a gamma-space gray row in 'dst'.
extern void (*SharpYuvUpdateW)(const uint16_t* src, uint16_t* dst, int w,
                               int bit_depth,
                               SharpYuvTransferFunctionType transfer_type);
// Downscales by 2x (in linear space) two rows of planar RGB samples
// 'src1'/'src2' (each plane '2 * uv_w' samples) and stores r/g/b minus their
// gray value in the three 'uv_w'-sized planes of 'dst'.
extern void (*SharpYuvUpdateChroma)(const uint16_t* src1, const uint16_t* src2,
                                    int16_t* dst, int uv_w, int bit_depth,
                                    SharpYuvTransferFunctionType
                                        transfer_type);

// Plain-C versions of the two functions above, used by the SIMD versions as
// fallback for the transfer functions they do not vectorize.
void SharpYuvUpdateW_C(const uint16_t* src, uint16_t* dst, int w,
                       int bit_depth,
                       SharpYuvTransferFunctionType transfer_type);
void SharpYuvUpdateChroma_C(const uint16_t* src1, const uint16_t* src2,
                            int16_t* dst, int uv_w, int bit_depth,
                            SharpYuvTransferFunctionType transfer_type);

void SharpYuvInitDsp(void);

#endif  // WEBP_SHARPYUV_SHARPYUV_DSP_H_
