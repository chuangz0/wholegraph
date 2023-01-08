#include "gather_scatter_func.cuh"

#include <wholememory/wholememory.h>

#include "logger.hpp"
#include "wholememory_ops/register.hpp"

namespace wholememory_ops {

template<typename InputT, typename EmbeddingT>
void scatter_floating_int32_temp_func(const void *input,
                                      wholememory_matrix_description_t input_desc,
                                      void *indices,
                                      int64_t indice_count,
                                      wholememory_gref_t embedding_gref,
                                      wholememory_matrix_description_t embedding_desc,
                                      cudaStream_t stream) {
  scatter_temp_func<InputT, int32_t, EmbeddingT>(input,
                                                 input_desc,
                                                 indices,
                                                 indice_count,
                                                 embedding_gref,
                                                 embedding_desc,
                                                 stream);
}

REGISTER_DISPATCH_TWO_TYPES(ScatterFuncFloatingInt32,
                            scatter_floating_int32_temp_func,
                            HALF_FLOAT_DOUBLE,
                            HALF_FLOAT_DOUBLE)

wholememory_error_code_t scatter_floating_int32_func(const void *input,
                                                     wholememory_matrix_description_t input_desc,
                                                     void *indices,
                                                     wholememory_array_description_t indices_desc,
                                                     wholememory_gref_t embedding_gref,
                                                     wholememory_matrix_description_t embedding_desc,
                                                     cudaStream_t stream) {
  try {
    WHOLEMEMORY_CHECK(wholememory_dtype_is_floating_number(embedding_desc.dtype));
    WHOLEMEMORY_CHECK(wholememory_dtype_is_floating_number(input_desc.dtype));
    WHOLEMEMORY_CHECK(indices_desc.dtype == WHOLEMEMORY_DT_INT);
    DISPATCH_TWO_TYPES(input_desc.dtype,
                       embedding_desc.dtype,
                       ScatterFuncFloatingInt32,
                       input,
                       input_desc,
                       static_cast<char *>(indices)
                           + indices_desc.storage_offset * wholememory_dtype_get_element_size(indices_desc.dtype),
                       indices_desc.size,
                       embedding_gref,
                       embedding_desc,
                       stream);
  } catch (const wholememory::cuda_error &wle) {
    WHOLEMEMORY_ERROR("scatter CUDA LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error &le) {
    WHOLEMEMORY_ERROR("scatter LOGIC Error %s\n", le.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops