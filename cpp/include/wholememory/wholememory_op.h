#pragma once

#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Gather Op
 * @param wholememory_handle : WholeMemory handle to embedding table.
 * @param wholememory_desc : matrix descriptor to embedding table.
 * @param indices : indices to gather
 * @param indices_desc : array descriptor of indices
 * @param output : pointer to output memory
 * @param output_desc : matrix descriptor of output
 * @param p_env_fns : pointers to environment functions.
 * @param stream : cudaStream_t to use.
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_gather(wholememory_handle_t wholememory_handle,
                                            wholememory_matrix_description_t wholememory_desc,
                                            void* indices,
                                            wholememory_array_description_t indices_desc,
                                            void* output,
                                            wholememory_matrix_description_t output_desc,
                                            wholememory_env_func_t* p_env_fns,
                                            cudaStream_t stream);

/**
 * Scatter Op
 * @param input : pointer to input matrix to be scattered to WholeMemory
 * @param input_desc : matrix descriptor of input
 * @param indices : indices to scatter to
 * @param indices_desc : array descriptor of indices
 * @param wholememory_handle : WholeMemory handle to embedding table.
 * @param wholememory_desc : matrix descriptor to embedding table.
 * @param p_env_fns : pointers to environment functions.
 * @param stream : cudaStream_t to use.
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_scatter(void* input,
                                             wholememory_matrix_description_t input_desc,
                                             void* indices,
                                             wholememory_array_description_t indices_desc,
                                             wholememory_handle_t wholememory_handle,
                                             wholememory_matrix_description_t wholememory_desc,
                                             wholememory_env_func_t* p_env_fns,
                                             cudaStream_t stream);

#ifdef __cplusplus
}
#endif
