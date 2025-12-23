/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "cfu.h"
#include <stdio.h>

// ---- CFU conv performance knobs ----

// 0 = always write A/B (usually faster on VexRiscv unless you have strong temporal locality)
// 1 = compare+skip writes if unchanged
#ifndef CFU_CONV_USE_WRITE_CACHE
#define CFU_CONV_USE_WRITE_CACHE 0
#endif

// If ALL weights of the current (m0 tile) are zero at some local-k, then B at that local-k
// does not affect any output in this tile; we can skip writing B for that local-k.
#ifndef CFU_CONV_SKIP_B_IF_ALLZERO_K
#define CFU_CONV_SKIP_B_IF_ALLZERO_K 1
#endif

namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  // GEMM dims
  const int M = output_depth;
  const int N = output_height * output_width;
  const int K = filter_height * filter_width * input_depth;

  // ---- Keep your original max sizes ----
  static constexpr int IM2COL_MAX_K = 8192;
  static constexpr int IM2COL_MAX_N = 1024;

  // Fixed physical tile size used by your TPU buffers.
  static constexpr int TM = 256;
  static constexpr int TN = 256;
  static constexpr int TK = 256;

  // im2col buffer (built once per batch; later we only pack what we need)
  int8_t im2col[IM2COL_MAX_K][IM2COL_MAX_N];

  // Per-output-channel correction term: sum(filter)*input_offset
  int32_t offset_correction[2048];

  // Accumulator for one (m0,n0) tile
  int32_t tile_acc[TM * TN];

#if CFU_CONV_USE_WRITE_CACHE
  static uint32_t lastA[16384];
  static uint32_t lastB[16384];
  static bool last_inited = false;
  if (!last_inited) {
    for (int i = 0; i < 16384; ++i) {
      lastA[i] = 0xFFFFFFFFu;
      lastB[i] = 0xFFFFFFFFu;
    }
    last_inited = true;
  }
#endif

  for (int batch = 0; batch < batches; ++batch) {
    // -------------------------
    // Build im2col for this batch (columns 0..N-1)
    // -------------------------
    int col_idx = 0;
    int stride_height_sum = 0;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = stride_height_sum - pad_height;
      int stride_width_sum = 0;

      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = stride_width_sum - pad_width;

        int row_idx = 0;
        int in_y = in_y_origin;

        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          int in_x = in_x_origin;

          for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
              int8_t val = 0;
              const bool inside =
                  (in_x >= 0) && (in_x < input_width) &&
                  (in_y >= 0) && (in_y < input_height);

              if (inside) {
                const int8_t real_val =
                    input_data[Offset(input_shape, batch, in_y, in_x, in_channel)];
                val = real_val;
              } else {
                // padding uses q = input_zero_point = -input_offset
                val = static_cast<int8_t>(-input_offset);
              }

              if (row_idx < IM2COL_MAX_K && col_idx < IM2COL_MAX_N) {
                im2col[row_idx][col_idx] = val;
              }
              row_idx++;
            }
            in_x += dilation_width_factor;
          }
          in_y += dilation_height_factor;
        }

        col_idx++;
        stride_width_sum += stride_width;
      }
      stride_height_sum += stride_height;
    }

    // -------------------------
    // Precompute offset correction for each output channel
    // -------------------------
    for (int m = 0; m < M; ++m) {
      const int8_t* fbase = filter_data + Offset(filter_shape, m, 0, 0, 0);
      int32_t sum = 0;
      for (int kk = 0; kk < K; ++kk) sum += fbase[kk];
      offset_correction[m] = sum * input_offset;
    }

    // -------------------------
    // Tiled GEMM using CFU/TPU
    // -------------------------
    for (int m0 = 0; m0 < M; m0 += TM) {
      const int m_tile = std::min(TM, M - m0);
      // Key fix: run only effective M on TPU for the last tile (4-row aligned)
      const int tm_eff_aligned = (m_tile + 3) & ~3;  // multiple of 4

      for (int n0 = 0; n0 < N; n0 += TN) {
        const int n_tile = std::min(TN, N - n0);
        const int tn_eff_aligned = (n_tile + 3) & ~3;  // multiple of 4

        // Accumulate across K blocks.
        for (int k0 = 0; k0 < K; k0 += TK) {
          const int k_tile = std::min(TK, K - k0);
          const int tk_eff_aligned = (k_tile + 3) & ~3;  // multiple of 4

          // 1) send weights -> A buffer, build local-k allzero mask
          uint8_t k_nonzero[TK];
          for (int i = 0; i < TK; ++i) k_nonzero[i] = 0;
          bool block_has_any_weight = false;

          uint32_t addrA = 0;
          for (int tm = 0; tm < tm_eff_aligned; tm += 4) {
            const int gm0 = m0 + tm + 0;
            const int gm1 = m0 + tm + 1;
            const int gm2 = m0 + tm + 2;
            const int gm3 = m0 + tm + 3;

            const int8_t* f0 = (gm0 < M) ? (filter_data + Offset(filter_shape, gm0, 0, 0, 0)) : nullptr;
            const int8_t* f1 = (gm1 < M) ? (filter_data + Offset(filter_shape, gm1, 0, 0, 0)) : nullptr;
            const int8_t* f2 = (gm2 < M) ? (filter_data + Offset(filter_shape, gm2, 0, 0, 0)) : nullptr;
            const int8_t* f3 = (gm3 < M) ? (filter_data + Offset(filter_shape, gm3, 0, 0, 0)) : nullptr;

            for (int tk = 0; tk < tk_eff_aligned; tk += 4) {
              for (int r = 0; r < 4; ++r) {
                const int local_k = tk + r;          // 0..TK-1
                const int cur_k = k0 + local_k;      // 0..K-1 (or padded)

                uint32_t packed_val = 0;
                if (cur_k < K) {
                  const uint8_t v0 = (f0) ? static_cast<uint8_t>(f0[cur_k]) : 0;
                  const uint8_t v1 = (f1) ? static_cast<uint8_t>(f1[cur_k]) : 0;
                  const uint8_t v2 = (f2) ? static_cast<uint8_t>(f2[cur_k]) : 0;
                  const uint8_t v3 = (f3) ? static_cast<uint8_t>(f3[cur_k]) : 0;
                  packed_val = (static_cast<uint32_t>(v0) << 24) |
                               (static_cast<uint32_t>(v1) << 16) |
                               (static_cast<uint32_t>(v2) <<  8) |
                               (static_cast<uint32_t>(v3) <<  0);
                }

                if (packed_val != 0) {
                  k_nonzero[local_k] = 1;
                  block_has_any_weight = true;
                }

#if CFU_CONV_USE_WRITE_CACHE
                if (packed_val != lastA[addrA]) {
                  cfu_op0(0, packed_val, addrA);
                  lastA[addrA] = packed_val;
                }
#else
                cfu_op0(0, packed_val, addrA);
#endif
                addrA++;
              }
            }
          }

          // If all weights in this K-block are zero, skip B+TPU+read entirely.
          if (!block_has_any_weight) {
            if (k0 == 0) {
              // first block: clear tile_acc to 0 for valid region
              int idx_base = 0;
              for (int i = 0; i < m_tile; ++i) {
                for (int j = 0; j < n_tile; j+=4) {
                  tile_acc[idx_base + j] = 0;
                  tile_acc[idx_base + j + 1] = 0;
                  tile_acc[idx_base + j + 2] = 0;
                  tile_acc[idx_base + j + 3] = 0;
                }
                idx_base += n_tile;
              }
            }
            continue;
          }

          // 2) send im2col -> B buffer (only TN_eff x TK_eff)
          uint32_t addrB = 0;
          for (int tn = 0; tn < tn_eff_aligned; tn += 4) {
            for (int tk = 0; tk < tk_eff_aligned; tk += 4) {
              for (int r = 0; r < 4; ++r) {
                const int local_k = tk + r;
                const int cur_k = k0 + local_k;

#if CFU_CONV_SKIP_B_IF_ALLZERO_K
                if (!k_nonzero[local_k]) {
                  addrB++;
                  continue;
                }
#endif
                uint32_t packed_val = 0;

                if (cur_k < K) {
                  const int cur_n = n0 + tn;
                  if (cur_n + 3 < N) {
                    // Fast path: 4 bytes contiguous (tn is multiple of 4)
                    packed_val = *reinterpret_cast<const uint32_t*>(im2col[cur_k] + cur_n);
                    packed_val = __builtin_bswap32(packed_val);
                  } else {
                    // Tail-safe packing
                    uint8_t b[4];
                    for (int t = 0; t < 4; ++t) {
                      const int nn = cur_n + t;
                      b[t] = (nn < N) ? static_cast<uint8_t>(im2col[cur_k][nn])
                                      : static_cast<uint8_t>(-input_offset);
                    }
                    packed_val = (static_cast<uint32_t>(b[0]) << 24) |
                                 (static_cast<uint32_t>(b[1]) << 16) |
                                 (static_cast<uint32_t>(b[2]) <<  8) |
                                 (static_cast<uint32_t>(b[3]) <<  0);
                  }
                } else {
                  packed_val = 0;
                }

#if CFU_CONV_USE_WRITE_CACHE
                if (packed_val != lastB[addrB]) {
                  cfu_op0(0, packed_val, (1u << 24) + addrB);
                  lastB[addrB] = packed_val;
                }
#else
                cfu_op0(0, packed_val, (1u << 24) + addrB);
#endif
                addrB++;
              }
            }
          }

          // 3) compute on TPU with effective dims
          cfu_op0(1, (uint32_t)tm_eff_aligned | ((uint32_t)tn_eff_aligned << 16),
                  (uint32_t)tk_eff_aligned);

          // 4) read back only valid region (m_tile x n_tile)
          const uint32_t num_row_blocks = (uint32_t)(tm_eff_aligned >> 2);
          uint32_t idx_base = 0;
          for (uint32_t i = 0; i < (uint32_t)m_tile; ++i) {
            const uint32_t block_row = i >> 2;
            const uint32_t inner_row = i & 3;
            uint32_t block_base = 0;
            for (uint32_t j = 0; j < (uint32_t)n_tile; j+=4) {
              const uint32_t buffer_index =
                  ((block_base + block_row) << 2) + inner_row;
              int32_t ret =
                  static_cast<int32_t>(cfu_op0(2, buffer_index, 0));
              uint32_t idx = idx_base + j;
              if (k0 == 0) tile_acc[idx] = ret;
              else         tile_acc[idx] += ret;
              ret =
                  static_cast<int32_t>(cfu_op0(2, buffer_index, 1));
              idx++;
              if (k0 == 0) tile_acc[idx] = ret;
              else         tile_acc[idx] += ret;
              ret =
                  static_cast<int32_t>(cfu_op0(2, buffer_index, 2));
              idx++;
              if (k0 == 0) tile_acc[idx] = ret;
              else         tile_acc[idx] += ret;
              ret =
                  static_cast<int32_t>(cfu_op0(2, buffer_index, 3));
              idx++;
              if (k0 == 0) tile_acc[idx] = ret;
              else         tile_acc[idx] += ret;
              block_base += num_row_blocks;
            }
            idx_base += (uint32_t)TN;
          }
        }  // k0 loop end

        // -------------------------
        // Postprocess + write outputs for this tile
        // acc = dot(filter, input) + sum(filter)*input_offset + bias
        // -------------------------
        // Precompute output base offsets (channel 0) for columns in this tile.
        // This avoids per-element divisions/mods and repeated Offset() calls.
        int out_y = n0 / output_width;
        int out_x = n0 - out_y * output_width;
        int out_base[TN];  // only first n_tile entries used
        for (int j = 0; j < n_tile; ++j) {
          out_base[j] = Offset(output_shape, batch, out_y, out_x, 0);
          ++out_x;
          if (out_x == output_width) {
            out_x = 0;
            ++out_y;
          }
        }
        uint32_t row_base = 0;
        if (bias_data) {
          for (int i = 0; i < m_tile; ++i) {
            const int out_c = m0 + i;
            const int32_t corr = offset_correction[out_c];
            const int32_t bias = bias_data[out_c];
            const int32_t mult = output_multiplier[out_c];
            const int32_t shift = output_shift[out_c];
            for (int j = 0; j < n_tile; j+=4) {
              int32_t acc = tile_acc[row_base + (uint32_t)j] + corr + bias;
              acc = MultiplyByQuantizedMultiplier(acc, mult, shift);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[out_base[j] + out_c] = static_cast<int8_t>(acc);
              acc = tile_acc[row_base + (uint32_t)j+1] + corr + bias;
              acc = MultiplyByQuantizedMultiplier(acc, mult, shift);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[out_base[j+1] + out_c] = static_cast<int8_t>(acc);
              acc = tile_acc[row_base + (uint32_t)j+2] + corr + bias;
              acc = MultiplyByQuantizedMultiplier(acc, mult, shift);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[out_base[j+2] + out_c] = static_cast<int8_t>(acc);
              acc = tile_acc[row_base + (uint32_t)j+3] + corr + bias;
              acc = MultiplyByQuantizedMultiplier(acc, mult, shift);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[out_base[j+3] + out_c] = static_cast<int8_t>(acc);
            }
            row_base += (uint32_t)TN;
          }
        } else {
          for (int i = 0; i < m_tile; ++i) {
            const int out_c = m0 + i;
            const int32_t corr = offset_correction[out_c];
            const int32_t mult = output_multiplier[out_c];
            const int32_t shift = output_shift[out_c];
            for (int j = 0; j < n_tile; j+=4) {
              int32_t acc = tile_acc[row_base + (uint32_t)j] + corr;
              acc = MultiplyByQuantizedMultiplier(acc, mult, shift);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[out_base[j] + out_c] = static_cast<int8_t>(acc);
              acc = tile_acc[row_base + (uint32_t)j+1] + corr;
              acc = MultiplyByQuantizedMultiplier(acc, mult, shift);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[out_base[j+1] + out_c] = static_cast<int8_t>(acc);
              acc = tile_acc[row_base + (uint32_t)j+2] + corr;
              acc = MultiplyByQuantizedMultiplier(acc, mult, shift);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[out_base[j+2] + out_c] = static_cast<int8_t>(acc);
              acc = tile_acc[row_base + (uint32_t)j+3] + corr;
              acc = MultiplyByQuantizedMultiplier(acc, mult, shift);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[out_base[j+3] + out_c] = static_cast<int8_t>(acc);
            }
            row_base += (uint32_t)TN;
          }
        }
}  // n0 loop end
    }    // m0 loop end
  }      // batch loop end
}

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter (KEEP stock reference implementation)
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              const bool inside =
                  (in_x >= 0) && (in_x < input_width) &&
                  (in_y >= 0) && (in_y < input_height);

              if (!inside) continue;

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) acc += bias_data[out_channel];
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
