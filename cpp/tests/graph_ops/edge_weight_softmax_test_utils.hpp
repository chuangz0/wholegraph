#pragma once
#include <wholememory/tensor_description.h>

namespace graph_ops::testing {
void host_edge_weight_softmax_forward(void* host_csr_row_ptr,
                                      wholememory_array_description_t csr_row_ptr_array_desc,
                                      void* host_weight_ptr,
                                      wholememory_matrix_description_t weight_matrix_desc,
                                      void* host_ref_output_weight_ptr,
                                      wholememory_matrix_description_t output_weight_matrix_desc);
void host_edge_weight_softmax_backward(
  void* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  void* host_weight_ptr,
  wholememory_matrix_description_t weight_matrix_desc,
  void* host_grad_weight_softmax_ptr,
  wholememory_matrix_description_t grad_weight_softmax_matrix_desc,
  void* host_ref_output_grad_weight_ptr,
  wholememory_matrix_description_t output_grad_weight_matrix_desc);
}  // namespace graph_ops::testing
