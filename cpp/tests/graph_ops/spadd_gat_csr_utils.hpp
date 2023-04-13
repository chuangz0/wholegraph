#pragma once
#include <wholememory/tensor_description.h>

namespace graph_ops::testing {
void host_spadd_gat_csr_forward(void* host_csr_row_ptr,
                                wholememory_array_description_t csr_row_ptr_array_desc,
                                void* host_csr_col_ptr,
                                wholememory_array_description_t csr_col_ptr_array_desc,
                                void* host_weight_left_ptr,
                                wholememory_matrix_description_t weight_left_matrix_desc,
                                void* host_weight_right_ptr,
                                wholememory_matrix_description_t weight_right_matrix_desc,
                                void* host_output_score_ptr,
                                wholememory_matrix_description_t output_score_matrix_desc);
void host_spadd_gat_csr_backward(void* host_csr_row_ptr,
                                 wholememory_array_description_t csr_row_ptr_array_desc,
                                 void* host_csr_col_ptr,
                                 wholememory_array_description_t csr_col_ptr_array_desc,
                                 void* host_grad_score_ptr,
                                 wholememory_matrix_description_t grad_score_matrix_desc,
                                 void* host_ref_output_grad_weight_left_ptr,
                                 wholememory_matrix_description_t weight_left_matrix_desc,
                                 void* host_ref_output_grad_weight_right_ptr,
                                 wholememory_matrix_description_t weight_right_matrix_desc);
}  // namespace graph_ops::testing
