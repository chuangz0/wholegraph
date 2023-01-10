#include <wholememory/wholememory_op.h>

#include <wholememory_ops/scatter_op_impl.h>

#include "error.hpp"

wholememory_error_code_t wholememory_scatter(void* input,
                                             wholememory_matrix_description_t input_desc,
                                             void* indices,
                                             wholememory_array_description_t indices_desc,
                                             wholememory_handle_t wholememory_handle,
                                             wholememory_matrix_description_t wholememory_desc,
                                             wholememory_env_func_t* p_env_fns,
                                             cudaStream_t stream)
{
  auto memory_type = wholememory_get_memory_type(wholememory_handle);
  if (memory_type == WHOLEMEMORY_MT_DISTRIBUTED) {
    return wholememory_ops::wholememory_scatter_nccl(input,
                                                     input_desc,
                                                     indices,
                                                     indices_desc,
                                                     wholememory_handle,
                                                     wholememory_desc,
                                                     p_env_fns,
                                                     stream);
  }

  WHOLEMEMORY_EXPECTS_NOTHROW(
    memory_type == WHOLEMEMORY_MT_CHUNKED || memory_type == WHOLEMEMORY_MT_CONTINUOUS,
    "Memory type not supported.");

  return wholememory_ops::wholememory_scatter_mapped(input,
                                                     input_desc,
                                                     indices,
                                                     indices_desc,
                                                     wholememory_handle,
                                                     wholememory_desc,
                                                     p_env_fns,
                                                     stream);
}