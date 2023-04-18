#pragma once
#include <wholememory/tensor_description.h>

namespace graph_ops::testing {
void host_csr_add_self_loop(void* host_csr_row_ptr,
                            wholememory_array_description_t csr_row_ptr_array_desc,
                            void* host_csr_col_ptr,
                            wholememory_array_description_t csr_col_ptr_array_desc,
                            void* host_ref_output_csr_row_ptr,
                            wholememory_array_description_t output_csr_row_ptr_array_desc,
                            void* host_ref_output_csr_col_ptr,
                            wholememory_array_description_t output_csr_col_ptr_array_desc);

}  // namespace graph_ops::testing
