#pragma once

#include <wholememory/global_reference.h>
#include <wholememory/wholememory.h>

namespace wholememory_ops {

wholememory_error_code_t wholememory_gather_mapped(
  wholememory_gref_t wholememory_gref,
  wholememory_matrix_description_t wholememory_desc,
  void* indices,
  wholememory_array_description_t indice_desc,
  void* output,
  wholememory_matrix_description_t output_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream);

wholememory_error_code_t wholememory_gather_nccl(wholememory_handle_t wholememory_handle,
                                                 wholememory_matrix_description_t wholememory_desc,
                                                 void* indices,
                                                 wholememory_array_description_t indice_desc,
                                                 void* output,
                                                 wholememory_matrix_description_t output_desc,
                                                 wholememory_env_func_t* p_env_fns,
                                                 cudaStream_t stream);

}  // namespace wholememory_ops
