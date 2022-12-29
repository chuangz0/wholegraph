#pragma once

#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>
#include <wholememory/tensor_description.h>

/**
 * Gather Op
 * @param wholememory_handle : WholeMemory handle to embedding table.
 * @param wholememory_desc : matrix descriptor to embedding table.
 * @param indices : indices to gather
 * @param indice_desc : array descriptor of indice
 * @param output : pointer to output memory
 * @param output_desc : matrix descriptor of output
 * @param p_env_fns : pointers to environment functions.
 * @param stream : cudaStream_t to use.
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_gather(wholememory_handle_t wholememory_handle,
                                            wholememory_matrix_description_t wholememory_desc,
                                            void* indices,
                                            wholememory_array_description_t indice_desc,
                                            void* output,
                                            wholememory_matrix_description_t output_desc,
                                            wholememory_env_func_t* p_env_fns,
                                            cudaStream_t stream);