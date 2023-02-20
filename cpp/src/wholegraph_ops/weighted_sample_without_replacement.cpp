#include <wholememory/wholegraph_op.h>

#include <wholegraph_ops/weighted_sample_without_replacement_impl.h>

#include "error.hpp"

wholememory_error_code_t wholegraph_csr_weighted_sample_without_replacement(
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
  cudaStream_t stream)
{
  return wholegraph_ops::wholegraph_csr_weighted_sample_without_replacement_mapped(
    wm_csr_row_ptr,
    wm_csr_row_ptr_desc,
    wm_csr_col_ptr,
    wm_csr_col_ptr_desc,
    wm_csr_weight_ptr,
    wm_csr_weight_ptr_desc,
    center_nodes,
    center_nodes_desc,
    max_sample_count,
    output_sample_offset,
    output_sample_offset_desc,
    output_dest_memory_context,
    output_center_localid_memory_context,
    output_edge_gid_memory_context,
    random_seed,
    p_env_fns,
    stream);
}