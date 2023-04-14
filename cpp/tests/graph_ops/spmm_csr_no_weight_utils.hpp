#pragma once
#include <wholememory/tensor_description.h>

namespace graph_ops::testing {
void gen_local_csr_graph(
  int row_num,
  int col_num,
  int graph_edge_num,
  void* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_desc,
  void* host_csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_desc,
  void* host_csr_weight_ptr                           = nullptr,
  wholememory_array_description_t csr_weight_ptr_desc = wholememory_array_description_t{});

void gen_features(void* feature_ptr, wholememory_matrix_description_t feature_desc);
void gen_features(void* feature_ptr, wholememory_tensor_description_t feature_desc);
void host_spmm_csr_no_weight_forward(void* host_csr_row_ptr,
                                     wholememory_array_description_t csr_row_ptr_desc,
                                     void* host_csr_col_ptr,
                                     wholememory_array_description_t csr_col_ptr_desc,
                                     void* input_feature_ptr,
                                     wholememory_matrix_description_t input_feature_desc,
                                     int aggregator,
                                     void* host_output_feature,
                                     wholememory_matrix_description_t output_feature_desc);

void host_check_float_matrix_same(void* input,
                                  wholememory_matrix_description_t input_matrix_desc,
                                  void* input_ref,
                                  wholememory_matrix_description_t input_ref_matrix_desc);
void host_check_float_tensor_same(void* input,
                                  wholememory_tensor_description_t input_tensor_desc,
                                  void* input_ref,
                                  wholememory_tensor_description_t input_ref_tensor_desc);
}  // namespace graph_ops::testing