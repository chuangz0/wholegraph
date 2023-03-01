#pragma once

#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>
#include <wholememory/wholememory_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Unweighted sample without replacement kernel op
 * @param wm_csr_row_ptr_tensor : Wholememory Tensor of graph csr_row_ptr 
 * @param wm_csr_col_ptr_tensor : Wholememory Tensor of graph csr_col_ptr
 * @param center_nodes_tensor : None Wholememory Tensor of center node to sample
 * @param max_sample_count : maximum sample count
 * @param output_sample_offset_tensor : pointer to output sample offset
 * @param output_dest_memory_context : memory context to output dest nodes
 * @param output_center_localid_memory_context : memory context to output center local id
 * @param output_edge_gid_memory_context : memory context to output edge global id
 * @param random_seed: random number generator seed
 * @param p_env_fns : pointers to environment functions.
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */

wholememory_error_code_t wholegraph_csr_unweighted_sample_without_replacement(
  wholememory_tensor_t wm_csr_row_ptr_tensor,
  wholememory_tensor_t wm_csr_col_ptr_tensor,
  wholememory_tensor_t center_nodes_tensor,
  int max_sample_count,
  wholememory_tensor_t output_sample_offset_tensor,
  memory_context_t* output_dest_memory_context,
  memory_context_t* output_center_localid_memory_context,
  memory_context_t* output_edge_gid_memory_context,
  unsigned long long random_seed,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream);

/**
 * Unweighted sample without replacement kernel op
 * @param wm_csr_row_ptr_tensor : Wholememory Tensor of graph csr_row_ptr 
 * @param wm_csr_col_ptr_tensor : Wholememory Tensor of graph csr_col_ptr
 * @param wm_csr_weight_ptr_tensor : Wholememory Tensor of graph edge weight
 * @param center_nodes_tensor : None Wholememory Tensor of center node to sample
 * @param max_sample_count : maximum sample count
 * @param output_sample_offset_tensor : pointer to output sample offset
 * @param output_dest_memory_context : memory context to output dest nodes
 * @param output_center_localid_memory_context : memory context to output center local id
 * @param output_edge_gid_memory_context : memory context to output edge global id
 * @param random_seed: random number generator seed
 * @param p_env_fns : pointers to environment functions.
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */

wholememory_error_code_t wholegraph_csr_weighted_sample_without_replacement(
  wholememory_tensor_t wm_csr_row_ptr_tensor,
  wholememory_tensor_t wm_csr_col_ptr_tensor,
  wholememory_tensor_t wm_csr_weight_ptr_tensor,
  wholememory_tensor_t center_nodes_tensor,
  int max_sample_count,
  wholememory_tensor_t output_sample_offset_tensor,
  memory_context_t* output_dest_memory_context,
  memory_context_t* output_center_localid_memory_context,
  memory_context_t* output_edge_gid_memory_context,
  unsigned long long random_seed,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream);

#ifdef __cplusplus
}
#endif