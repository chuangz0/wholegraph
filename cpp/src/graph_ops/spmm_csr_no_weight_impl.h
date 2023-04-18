#pragma once
#include <cuda_runtime_api.h>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace graph_ops {
wholememory_error_code_t spmm_csr_no_weight_forward_mapped(
  void* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_desc,
  void* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_desc,
  void* feature_ptr,
  wholememory_matrix_description_t feature_desc,
  int aggregator,
  void* output_ptr,
  wholememory_matrix_description_t output_desc,
  cudaStream_t stream);

wholememory_error_code_t spmm_csr_no_weight_backward_mapped(
  void* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_desc,
  void* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_desc,
  void* input_grad_ptr,
  wholememory_tensor_description_t input_grad_tensor_desc,
  int aggregator,
  void* output_grad_feature_ptr,
  wholememory_tensor_description_t output_grad_feature_tensor_desc,
  cudaStream_t stream);
}  // namespace graph_ops
