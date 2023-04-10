#pragma once
#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>
#include <wholememory/wholememory_tensor.h>
#define SUM_AGGREGATOR  0
#define MEAN_AGGREGATOR 1
#define GCN_AGGREGATOR  2

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Append Unique op
 * @param target_nodes_tensor : Wholememory Tensor of graph csr_row_ptr
 * @param neighbor_nodes_tensor : Wholememory Tensor of graph csr_col_ptr
 * @param output_unique_node_memory_context : memory context to output dest nodes
 * @param output_neighbor_raw_to_unique_mapping_tensor : pointer to output sample offset, optional
 * output
 * @param p_env_fns : pointers to environment functions.
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */

wholememory_error_code_t graph_append_unique(
  wholememory_tensor_t target_nodes_tensor,
  wholememory_tensor_t neighbor_nodes_tensor,
  void* output_unique_node_memory_context,
  wholememory_tensor_t output_neighbor_raw_to_unique_mapping_tensor,
  wholememory_env_func_t* p_env_fns,
  void* stream);

/**
 * Spmm CSR no Weight Forward Op
 * @param csr_row_ptr : Wholememory Tensor of local graph csr_row_ptr
 * @param csr_col_ind : Wholememory Tensor of local graph csr_col_ptr
 * @param features : Wholememory Tensor of features
 * @param aggregator : aggreagtor type
 * @param output_features: Wholememory Tensor of output features
 * @param p_env_fns : pointers to environment functions.
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */

wholememory_error_code_t spmm_csr_no_weight_forward(wholememory_tensor_t csr_row_ptr,
                                                    wholememory_tensor_t csr_col_ptr,
                                                    wholememory_tensor_t features,
                                                    int64_t aggregator,
                                                    wholememory_tensor_t output_features,
                                                    void* stream);

/**
 * Spmm CSR no Weight Forward Op
 * @param csr_row_ptr : Wholememory Tensor of local graph csr_row_ptr
 * @param csr_col_ind : Wholememory Tensor of local graph csr_col_ptr
 * @param grad_output : Wholememory Tensor of grad_output
 * @param aggregator : aggreagtor type
 * @param output_features: Wholememory Tensor of output features
 * @param p_env_fns : pointers to environment functions.
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t spmm_csr_no_weight_backward(wholememory_tensor_t csr_row_ptr,
                                                     wholememory_tensor_t csr_col_ptr,
                                                     wholememory_tensor_t grad_output,
                                                     int64_t aggregator,
                                                     wholememory_tensor_t output_features,
                                                     wholememory_env_func_t* p_env_fns,
                                                     void* stream);

#ifdef __cplusplus
}
#endif
