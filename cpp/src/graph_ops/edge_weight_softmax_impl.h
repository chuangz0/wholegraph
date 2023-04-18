#pragma once
#include <cuda_runtime_api.h>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace graph_ops {
wholememory_error_code_t edge_weight_softmax_forward_impl(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  void* edge_weight_ptr,
  wholememory_matrix_description_t edge_weight_matrix_desc,
  void* output_edge_weight_ptr,
  wholememory_matrix_description_t output_edge_weight_matrix_desc,
  cudaStream_t stream);
wholememory_error_code_t edge_weight_softmax_backward_impl(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  void* edge_weight_ptr,
  wholememory_matrix_description_t edge_weight_matrix_desc,
  void* grad_edge_weight_softmax_ptr,
  wholememory_matrix_description_t grad_edge_weight_softmax_matrix_desc,
  void* output_grad_edge_weight_ptr,
  wholememory_matrix_description_t output_grad_edge_weight_matrix_desc,
  cudaStream_t stream);

}  // namespace graph_ops
