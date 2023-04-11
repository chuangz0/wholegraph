#pragma once
#include "error.hpp"
#include <wholememory/env_func_ptrs.h>
#include <wholememory/tensor_description.h>

namespace graph_ops {

// aggregator
// 0: sum
// 1: mean
// 2: gcn
template <typename T = float, int AGG = 0>
__global__ void SpmmCsrNoWeightForwardSimpleKernel(const int* csr_row_ptr,
                                                   const int* csr_col_ind,
                                                   const T* x,
                                                   int64_t embedding_dim,
                                                   int64_t embedding_stride,
                                                   int64_t input_count,
                                                   T* output,
                                                   int64_t output_stride)
{
  int row_idx       = blockIdx.x;
  int emb_idx       = threadIdx.x;
  int row_ptr_start = csr_row_ptr[row_idx];
  int row_ptr_end   = csr_row_ptr[row_idx + 1];
  int agg_count     = row_ptr_end - row_ptr_start;
  for (; emb_idx < embedding_dim; emb_idx += blockDim.x) {
    float agg_value = 0.0f;
    if (AGG == 2) { agg_value += (float)x[(int64_t)row_idx * embedding_stride + emb_idx]; }

    for (int row_ptr = row_ptr_start; row_ptr < row_ptr_end; row_ptr++) {
      int col_idx = csr_col_ind[row_ptr];
      assert(col_idx >= 0 && col_idx < input_count);
      float value = (float)x[(int64_t)col_idx * embedding_stride + emb_idx];
      agg_value += value;
    }
    if (AGG == 1) {
      if (agg_count > 0) { agg_value /= agg_count; }
    }
    if (AGG == 2) { agg_value /= (agg_count + 1); }
    output[(int64_t)row_idx * output_stride + emb_idx] = (T)agg_value;
  }
}

template <typename T>
void spmm_csr_no_weight_forward_func(void* csr_row_ptr,
                                     wholememory_array_description_t csr_row_ptr_desc,
                                     void* csr_col_ptr,
                                     wholememory_array_description_t csr_col_ptr_desc,
                                     void* feature_ptr,
                                     wholememory_matrix_description_t feature_desc,
                                     int aggregator,
                                     void* output_ptr,
                                     wholememory_matrix_description_t output_desc,
                                     cudaStream_t stream)
{
  WHOLEMEMORY_EXPECTS(csr_row_ptr_desc.dtype == WHOLEMEMORY_DT_INT,
                      "spmm_csr_no_weight_forward_func(). "
                      "csr_row_ptr_desc.dtype != WHOLEMEMORY_DT_INT, "
                      "csr_row_ptr_desc.dtype = %d",
                      csr_row_ptr_desc.dtype);
  WHOLEMEMORY_EXPECTS(csr_col_ptr_desc.dtype == WHOLEMEMORY_DT_INT,
                      "spmm_csr_no_weight_forward_func(). "
                      "csr_col_ptr_desc.dtype != WHOLEMEMORY_DT_INT, "
                      "csr_col_ptr_desc.dtype = %d",
                      csr_col_ptr_desc.dtype);

  int input_count          = csr_row_ptr_desc.size - 1;
  int64_t embedding_dim    = feature_desc.sizes[1];
  int64_t embedding_stride = feature_desc.stride;
  int64_t output_stride    = embedding_stride;

  int block_count  = input_count;
  int thread_count = embedding_dim;
  if (embedding_dim > 512) { thread_count = 512; }

  if (aggregator == 0) {
    SpmmCsrNoWeightForwardSimpleKernel<T, 0>
      <<<block_count, thread_count, 0, stream>>>((const int*)csr_row_ptr,
                                                 (const int*)csr_col_ptr,
                                                 (const T*)feature_ptr,
                                                 embedding_dim,
                                                 embedding_stride,
                                                 input_count,
                                                 (T*)output_ptr,
                                                 output_stride);
  } else if (aggregator == 1) {
    SpmmCsrNoWeightForwardSimpleKernel<T, 1>
      <<<block_count, thread_count, 0, stream>>>((const int*)csr_row_ptr,
                                                 (const int*)csr_col_ptr,
                                                 (const T*)feature_ptr,
                                                 embedding_dim,
                                                 embedding_stride,
                                                 input_count,
                                                 (T*)output_ptr,
                                                 output_stride);
  } else if (aggregator == 2) {
    SpmmCsrNoWeightForwardSimpleKernel<T, 2>
      <<<block_count, thread_count, 0, stream>>>((const int*)csr_row_ptr,
                                                 (const int*)csr_col_ptr,
                                                 (const T*)feature_ptr,
                                                 embedding_dim,
                                                 embedding_stride,
                                                 input_count,
                                                 (T*)output_ptr,
                                                 output_stride);
  }
}

}  // namespace graph_ops
