#pragma once
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory_tensor.h>

namespace graph_ops::testing {
void host_gspmm_csr_weighted_forward(void* host_csr_row_ptr,
                                     wholememory_array_description_t csr_row_ptr_array_desc,
                                     void* host_csr_col_ptr,
                                     wholememory_array_description_t csr_col_ptr_array_desc,
                                     void* host_edge_weight_ptr,
                                     wholememory_tensor_description_t edge_weight_tensor_desc,
                                     void* host_feature_ptr,
                                     wholememory_tensor_description_t feature_tensor_desc,
                                     void* host_ref_output_feature_ptr,
                                     wholememory_tensor_description_t output_feature_tensor_desc);
void host_gspmm_csr_weighted_backward(
  void* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  void* host_csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  void* host_edge_weight_ptr,
  wholememory_tensor_description_t edge_weight_tensor_desc,
  void* host_feature_ptr,
  wholememory_tensor_description_t feature_tensor_desc,
  void* host_input_grad_feature_ptr,
  wholememory_tensor_description_t input_grad_feature_tensor_desc,
  void* host_ref_output_grad_edge_weight_ptr,
  wholememory_tensor_description_t output_grad_edge_weight_tensor_desc,
  void* host_ref_output_grad_feature_ptr,
  wholememory_tensor_description_t output_grad_feature_tensor_desc);

}  // namespace graph_ops::testing
