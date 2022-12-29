#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "wholememory_ops/register.hpp"
#include "gather_scatter_func.cuh"

namespace wholememory_ops {

REGISTER_DISPATCH_THREE_TYPES(WholeMemoryGatherMapped, gather_func, HALF_FLOAT_DOUBLE, SINT3264, HALF_FLOAT_DOUBLE)

wholememory_error_code_t wholememory_gather_mapped(wholememory_handle_t wholememory_handle,
                                                   wholememory_matrix_description_t wholememory_desc,
                                                   void *indices,
                                                   wholememory_array_description_t indice_desc,
                                                   void *output,
                                                   wholememory_matrix_description_t output_desc,
                                                   wholememory_env_func_t *p_env_fns,
                                                   cudaStream_t stream) {
  wholememory_gref_t wholememory_gref;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_global_reference(&wholememory_gref, wholememory_handle));

  try {
    DISPATCH_THREE_TYPES(wholememory_desc.dtype,
                         indice_desc.dtype,
                         output_desc.dtype,
                         WholeMemoryGatherMapped,
                         wholememory_gref,
                         wholememory_desc,
                         static_cast<char *>(indices)
                             + indice_desc.storage_offset * wholememory_dtype_get_element_size(indice_desc.dtype),
                         indice_desc.size,
                         output,
                         output_desc,
                         stream);
  } catch (const raft::cuda_error& rle) {
    // WHOLEMEMORY_FAIL_NOTHROW("%s", rle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops