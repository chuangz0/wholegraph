#pragma once

#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>
#include <wholememory/wholememory_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Gather Op
 * @param wholememory_tensor : WholeMemory Tensor of embedding table.
 * @param indices_tensor : indices to gather from, should NOT be WholeMemory Tensor
 * @param output_tensor : output tensor to gather to, should NOT be WholeMemoryTensor
 * @param p_env_fns : pointers to environment functions.
 * @param stream : cudaStream_t to use.
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_gather(wholememory_tensor_t wholememory_tensor,
                                            wholememory_tensor_t indices_tensor,
                                            wholememory_tensor_t output_tensor,
                                            wholememory_env_func_t* p_env_fns,
                                            cudaStream_t stream);

/**
 * Scatter Op
 * @param input_tensor : input tensor tor scatter from, should NOT be WholeMemory Tensor
 * @param indices_tensor : indices to scatter to, should NOT be WholeMemory Tensor
 * @param wholememory_tensor : WholeMemory Tensor of embedding table.
 * @param p_env_fns : pointers to environment functions.
 * @param stream : cudaStream_t to use.
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_scatter(wholememory_tensor_t input_tensor,
                                             wholememory_tensor_t indices_tensor,
                                             wholememory_tensor_t wholememory_tensor,
                                             wholememory_env_func_t* p_env_fns,
                                             cudaStream_t stream);

#ifdef __cplusplus
}
#endif
