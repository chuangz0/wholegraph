#include <wholememory/wholememory_op.h>

#include <wholememory_ops/gather_op_impl.h>

#include "error.hpp"

wholememory_error_code_t wholememory_gather(wholememory_handle_t wholememory_handle,
                                            wholememory_matrix_description_t wholememory_desc,
                                            void* indices,
                                            wholememory_array_description_t indice_desc,
                                            void* output,
                                            wholememory_matrix_description_t output_desc,
                                            wholememory_env_func_t* p_env_fns,
                                            cudaStream_t stream) {
  auto memory_type = wholememory_get_memory_type(wholememory_handle);
  if (memory_type == WHOLEMEMORY_MT_DISTRIBUTED) {
    return wholememory_ops::wholememory_gather_nccl(wholememory_handle,
                                                    wholememory_desc,
                                                    indices,
                                                    indice_desc,
                                                    output,
                                                    output_desc,
                                                    p_env_fns,
                                                    stream);
  }

  WHOLEMEMORY_EXPECTS_NOTHROW(memory_type == WHOLEMEMORY_MT_CHUNKED || memory_type == WHOLEMEMORY_MT_CONTINUOUS,
                              "Memory type not supported.");

  return wholememory_ops::wholememory_gather_mapped(wholememory_handle,
                                                    wholememory_desc,
                                                    indices,
                                                    indice_desc,
                                                    output,
                                                    output_desc,
                                                    p_env_fns,
                                                    stream);
}