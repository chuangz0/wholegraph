#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "wholememory_ops/functions/gather_scatter_func.h"

namespace wholememory_ops {

wholememory_error_code_t wholememory_gather_mapped(
  wholememory_handle_t wholememory_handle,
  wholememory_matrix_description_t wholememory_desc,
  void* indices,
  wholememory_array_description_t indice_desc,
  void* output,
  wholememory_matrix_description_t output_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  wholememory_gref_t wholememory_gref;
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_get_global_reference(&wholememory_gref, wholememory_handle));
  return gather_func(
    wholememory_gref, wholememory_desc, indices, indice_desc, output, output_desc, stream);
}

}  // namespace wholememory_ops