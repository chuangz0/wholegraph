#pragma once

#include <wholememory/global_reference.h>
#include <wholememory/wholememory.h>

namespace wholememory_ops {

wholememory_error_code_t wholememory_scatter_mapped(
  void* input,
  wholememory_matrix_description_t input_desc,
  void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_gref_t wholememory_gref,
  wholememory_matrix_description_t wholememory_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream);

wholememory_error_code_t wholememory_scatter_nccl(void* input,
                                                  wholememory_matrix_description_t input_desc,
                                                  void* indices,
                                                  wholememory_array_description_t indices_desc,
                                                  wholememory_handle_t wholememory_handle,
                                                  wholememory_matrix_description_t wholememory_desc,
                                                  wholememory_env_func_t* p_env_fns,
                                                  cudaStream_t stream);

}  // namespace wholememory_ops
