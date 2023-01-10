#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "wholememory_ops/functions/gather_scatter_func.h"

namespace wholememory_ops {

wholememory_error_code_t wholememory_scatter_mapped(
  void* input,
  wholememory_matrix_description_t input_desc,
  void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_handle_t wholememory_handle,
  wholememory_matrix_description_t wholememory_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  wholememory_gref_t wholememory_gref;
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_get_global_reference(&wholememory_gref, wholememory_handle));
  return scatter_func(
    input, input_desc, indices, indices_desc, wholememory_gref, wholememory_desc, stream);
}

}  // namespace wholememory_ops