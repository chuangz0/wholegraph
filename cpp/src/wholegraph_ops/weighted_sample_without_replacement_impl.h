#pragma once

#include <wholememory/env_func_ptrs.h>
#include <wholememory/global_reference.h>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace wholegraph_ops {

wholememory_error_code_t wholegraph_csr_weighted_sample_without_replacement_mapped(
  wholememory_gref_t wm_csr_row_ptr,
  wholememory_array_description_t wm_csr_row_ptr_desc,
  wholememory_gref_t wm_csr_col_ptr,
  wholememory_array_description_t wm_csr_col_ptr_desc,
  wholememory_gref_t wm_csr_weight_ptr,
  wholememory_array_description_t wm_csr_weight_ptr_desc,
  void* center_nodes,
  wholememory_array_description_t center_nodes_desc,
  int max_sample_count,
  void* output_sample_offset,
  wholememory_array_description_t output_sample_offset_desc,
  memory_context_t* output_dest_memory_context,
  memory_context_t* output_center_localid_memory_context,
  memory_context_t* output_edge_gid_memory_context,
  unsigned long long random_seed,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream);

}  // namespace wholegraph_ops