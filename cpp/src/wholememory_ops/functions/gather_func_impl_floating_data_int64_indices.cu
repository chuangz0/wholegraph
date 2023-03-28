#include "gather_scatter_func.cuh"

#include <wholememory/wholememory.h>

#include "logger.hpp"
#include "wholememory_ops/register.hpp"

namespace wholememory_ops {

template <typename EmbeddingT, typename OutputT>
void gather_floating_int64_temp_func(wholememory_gref_t embedding_gref,
                                     wholememory_matrix_description_t embedding_desc,
                                     void* indices,
                                     int64_t indice_count,
                                     void* output,
                                     wholememory_matrix_description_t output_desc,
                                     cudaStream_t stream)
{
  gather_temp_func<EmbeddingT, int64_t, OutputT>(
    embedding_gref, embedding_desc, indices, indice_count, output, output_desc, stream);
}

REGISTER_DISPATCH_TWO_TYPES(GatherFuncFloatingInt64,
                            gather_floating_int64_temp_func,
                            HALF_FLOAT_DOUBLE,
                            HALF_FLOAT_DOUBLE)

wholememory_error_code_t gather_floating_int64_func(wholememory_gref_t embedding_gref,
                                                    wholememory_matrix_description_t embedding_desc,
                                                    void* indices,
                                                    wholememory_array_description_t indices_desc,
                                                    void* output,
                                                    wholememory_matrix_description_t output_desc,
                                                    cudaStream_t stream)
{
  try {
    WHOLEMEMORY_CHECK(wholememory_dtype_is_floating_number(embedding_desc.dtype));
    WHOLEMEMORY_CHECK(wholememory_dtype_is_floating_number(output_desc.dtype));
    WHOLEMEMORY_CHECK(indices_desc.dtype == WHOLEMEMORY_DT_INT64);
    DISPATCH_TWO_TYPES(
      embedding_desc.dtype,
      output_desc.dtype,
      GatherFuncFloatingInt64,
      embedding_gref,
      embedding_desc,
      static_cast<char*>(indices) +
        indices_desc.storage_offset * wholememory_dtype_get_element_size(indices_desc.dtype),
      indices_desc.size,
      output,
      output_desc,
      stream);
  } catch (const wholememory::cuda_error& wle) {
    WHOLEMEMORY_ERROR("gather CUDA LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    WHOLEMEMORY_ERROR("gather CUDA LOGIC Error %s\n", le.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops