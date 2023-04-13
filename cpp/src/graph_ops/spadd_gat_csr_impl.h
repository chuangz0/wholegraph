#pragma once
#include <cuda_runtime_api.h>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace graph_ops {
wholememory_error_code_t spadd_gat_csr_forward_impl(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  int* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  void* edge_weight_left_ptr,
  wholememory_matrix_description_t edge_weight_left_matrix_desc,
  void* edge_weight_right_ptr,
  wholememory_matrix_description_t edge_weight_right_matrix_desc,
  void* output_score_ptr,
  wholememory_matrix_description_t output_score_matrix_desc,
  cudaStream_t stream);

wholememory_error_code_t spadd_gat_csr_backward_impl(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  int* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  void* grad_score_ptr,
  wholememory_matrix_description_t grad_score_matrix_desc,
  void* output_grad_edge_weight_left_ptr,
  wholememory_matrix_description_t edge_weight_left_matrix_desc,
  void* output_grad_edge_weight_right_ptr,
  wholememory_matrix_description_t edge_weight_right_matrix_desc,
  cudaStream_t stream);

}  // namespace graph_ops
