#include <wholememory/wholememory_op.h>

#include <wholememory_ops/gather_op_impl.h>

#include "error.hpp"
#include "logger.hpp"

wholememory_error_code_t wholememory_gather(wholememory_tensor_t wholememory_tensor,
                                            void* indices,
                                            wholememory_array_description_t indices_desc,
                                            void* output,
                                            wholememory_matrix_description_t output_desc,
                                            wholememory_env_func_t* p_env_fns,
                                            cudaStream_t stream)
{
  bool const has_handle                 = wholememory_tensor_has_handle(wholememory_tensor);
  wholememory_memory_type_t memory_type = WHOLEMEMORY_MT_NONE;
  if (has_handle) {
    memory_type =
      wholememory_get_memory_type(wholememory_tensor_get_memory_handle(wholememory_tensor));
  }
  wholememory_matrix_description_t matrix_description;
  wholememory_tensor_description_t tensor_description;
  wholememory_tensor_get_tensor_description(&tensor_description, wholememory_tensor);
  if (!wholememory_convert_tensor_desc_to_matrix(&matrix_description, &tensor_description)) {
    WHOLEMEMORY_ERROR("Input wholememory_tensor should be 2D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (has_handle && memory_type == WHOLEMEMORY_MT_DISTRIBUTED) {
    return wholememory_ops::wholememory_gather_nccl(
      wholememory_tensor_get_memory_handle(wholememory_tensor),
      matrix_description,
      indices,
      indices_desc,
      output,
      output_desc,
      p_env_fns,
      stream);
  }

  WHOLEMEMORY_EXPECTS_NOTHROW(!has_handle || memory_type == WHOLEMEMORY_MT_CHUNKED ||
                                memory_type == WHOLEMEMORY_MT_CONTINUOUS,
                              "Memory type not supported.");

  wholememory_gref_t gref;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_tensor_get_global_reference(&gref, wholememory_tensor));

  return wholememory_ops::wholememory_gather_mapped(
    gref, matrix_description, indices, indices_desc, output, output_desc, p_env_fns, stream);
}