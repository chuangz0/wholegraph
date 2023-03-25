#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "cuda_macros.hpp"
#include "wholememory_ops/functions/gather_scatter_func.h"

namespace wholememory_ops {

wholememory_error_code_t wholememory_gather_mapped(
  wholememory_gref_t wholememory_gref,
  wholememory_matrix_description_t wholememory_desc,
  void* indices,
  wholememory_array_description_t indice_desc,
  void* output,
  wholememory_matrix_description_t output_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  WHOLEMEMORY_RETURN_ON_FAIL(gather_func(
    wholememory_gref, wholememory_desc, indices, indice_desc, output, output_desc, stream));
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops