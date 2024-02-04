/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <wholememory/device_reference.cuh>
#include <wholememory/global_reference.h>
#include <wholememory/tensor_description.h>

#include "cuda_macros.hpp"
#include "error.hpp"
#include "wholememory/integer_utils.hpp"

#include "wholememory_ops/functions/gather_scatter_func.cuh"
namespace wholememory_ops {

template <typename EmbeddingT, typename IndexT, typename OutputT, int ALIGNMENT = 1>
__global__ void old_gather_func_kernel(wholememory_gref_t embedding_gref,
                                       wholememory_matrix_description_t embedding_desc,
                                       const IndexT* indices,
                                       int64_t indice_count,
                                       OutputT* output,
                                       wholememory_matrix_description_t output_desc)
{
  int64_t output_idx = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  wholememory::device_reference<EmbeddingT> embedding_dev_ref(embedding_gref);
  int thread_idx           = threadIdx.x;
  int embedding_size       = embedding_desc.sizes[1];
  int64_t embedding_stride = embedding_desc.stride;
  int64_t output_stride    = output_desc.stride;

  for (; output_idx < indice_count; output_idx += static_cast<int64_t>(gridDim.x) * blockDim.y) {
    IndexT embedding_table_idx = indices[output_idx];
    if (embedding_table_idx < 0) return;
    typed_data_vector<EmbeddingT, ALIGNMENT> embeddings;
    typed_data_vector<OutputT, ALIGNMENT> outputs;
    OutputT* output_ptr = output + output_desc.storage_offset + output_stride * output_idx;
    int64_t embedding_offset =
      embedding_desc.storage_offset + embedding_table_idx * embedding_stride;

    for (int emb_idx = thread_idx * ALIGNMENT; emb_idx < embedding_size;
         emb_idx += ALIGNMENT * blockDim.x) {
      mov_data<sizeof(EmbeddingT) * ALIGNMENT>(&embeddings,
                                               &embedding_dev_ref[embedding_offset + emb_idx]);
#pragma unroll
      for (int sub_idx = 0; sub_idx < ALIGNMENT; sub_idx++) {
        typed_data_vector_at(outputs, sub_idx) =
          convert_type<EmbeddingT, OutputT>(typed_data_vector_at(embeddings, sub_idx));
      }
      mov_data<sizeof(OutputT) * ALIGNMENT>(output_ptr + emb_idx, &outputs);
    }
  }
}

template <typename EmbeddingT, typename IndexT, typename OutputT>
void old_gather_temp_func(wholememory_gref_t embedding_gref,
                          wholememory_matrix_description_t embedding_desc,
                          void* indices,
                          int64_t indice_count,
                          void* output,
                          wholememory_matrix_description_t output_desc,
                          int64_t block_count,
                          cudaStream_t stream)
{
  WHOLEMEMORY_EXPECTS(output_desc.sizes[0] == indice_count,
                      "gather_func, output shape[0]=%ld, but indice_count=%ld",
                      output_desc.sizes[0],
                      indice_count);
  if (indice_count == 0 || embedding_desc.sizes[1] == 0) return;
  int wm_alignment   = determine_wholememory_alignment_elt_count(embedding_desc);
  int mm_alignment   = determine_memory_alignment_elt_count(output, output_desc);
  int alignment      = std::min<int>(wm_alignment, mm_alignment);
  int embedding_size = embedding_desc.sizes[1];
  int thread_x       = wholememory::div_rounding_up_safe<int>(embedding_size, alignment);
  thread_x           = std::min(thread_x, 256);
  int thread_y       = 1;
  if (thread_x < 64) {
    int power2_thread_x = 1;
    for (; power2_thread_x < thread_x; power2_thread_x *= 2)
      ;
    thread_x = power2_thread_x;
    thread_y = 64 / thread_x;
  }
  //   int64_t block_count_64 = (indice_count + thread_y - 1) / thread_y;
  //   int block_count = block_count_64 >= INT_MAX ? INT_MAX / 4 : static_cast<int>(block_count_64);
  dim3 block_dim(thread_x, thread_y, 1);
  void (*kernel_fn)(wholememory_gref_t,
                    wholememory_matrix_description_t,
                    const IndexT*,
                    int64_t,
                    OutputT*,
                    wholememory_matrix_description_t) = nullptr;
  switch (alignment) {
    case 16: {
      kernel_fn = old_gather_func_kernel<EmbeddingT, IndexT, OutputT, 16>;
      break;
    }
    case 8: {
      kernel_fn = old_gather_func_kernel<EmbeddingT, IndexT, OutputT, 8>;
      break;
    }
    case 4: {
      kernel_fn = old_gather_func_kernel<EmbeddingT, IndexT, OutputT, 4>;
      break;
    }
    case 2: {
      kernel_fn = old_gather_func_kernel<EmbeddingT, IndexT, OutputT, 2>;
      break;
    }
    case 1: {
      kernel_fn = old_gather_func_kernel<EmbeddingT, IndexT, OutputT, 1>;
      break;
    }
    default: {
      WHOLEMEMORY_FAIL("gather func alignment=%d.", alignment);
      return;
    }
  }
  kernel_fn<<<block_count, block_dim, 0, stream>>>(embedding_gref,
                                                   embedding_desc,
                                                   static_cast<const IndexT*>(indices),
                                                   indice_count,
                                                   static_cast<OutputT*>(output),
                                                   output_desc);
  WM_CUDA_CHECK(cudaGetLastError());
}

template <typename EmbeddingT, typename IndexT, typename OutputT>
void new_gather_temp_func(wholememory_gref_t embedding_gref,
                          wholememory_matrix_description_t embedding_desc,
                          void* indices,
                          int64_t indice_count,
                          void* output,
                          wholememory_matrix_description_t output_desc,
                          int64_t block_count,
                          cudaStream_t stream)
{
  WHOLEMEMORY_EXPECTS(output_desc.sizes[0] == indice_count,
                      "gather_func, output shape[0]=%ld, but indice_count=%ld",
                      output_desc.sizes[0],
                      indice_count);
  if (indice_count == 0 || embedding_desc.sizes[1] == 0) return;
  int wm_alignment = determine_wholememory_alignment_elt_count(embedding_desc);
  int mm_alignment = determine_memory_alignment_elt_count(output, output_desc);
  int alignment    = std::min<int>(wm_alignment, mm_alignment);
  // int embedding_size = embedding_desc.sizes[1];
  // int thread_num       = wholememory::div_rounding_up_safe<int>(embedding_size, alignment);
  // thread_num           = std::min(thread_num, 512);
  // int64_t block_count = indice_count >= 1024 ? 1024 : static_cast<int>(indice_count);

  void (*kernel_fn)(wholememory_gref_t,
                    wholememory_matrix_description_t,
                    const IndexT*,
                    int64_t,
                    OutputT*,
                    wholememory_matrix_description_t) = nullptr;
  switch (alignment) {
    case 16: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 16>;
      break;
    }
    case 8: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 8>;
      break;
    }
    case 4: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 4>;
      break;
    }
    case 2: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 2>;
      break;
    }
    case 1: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 1>;
      break;
    }
    default: {
      WHOLEMEMORY_FAIL("gather func alignment=%d.", alignment);
      return;
    }
  }
  int block_size = 1024;
  //   int block_count = indice_count > 1568 ? 1568 : indice_count;
  kernel_fn<<<block_count, block_size, 0, stream>>>(embedding_gref,
                                                    embedding_desc,
                                                    static_cast<const IndexT*>(indices),
                                                    indice_count,
                                                    static_cast<OutputT*>(output),
                                                    output_desc);
  WM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace wholememory_ops
