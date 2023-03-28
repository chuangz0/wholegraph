#include "gather_scatter_func.h"

#include "cuda_macros.hpp"
#include "error.hpp"
#include "logger.hpp"

namespace wholememory_ops {

wholememory_error_code_t scatter_integer_int32_func(const void* input,
                                                    wholememory_matrix_description_t input_desc,
                                                    void* indices,
                                                    wholememory_array_description_t indices_desc,
                                                    wholememory_gref_t embedding_gref,
                                                    wholememory_matrix_description_t embedding_desc,
                                                    cudaStream_t stream);
wholememory_error_code_t scatter_integer_int64_func(const void* input,
                                                    wholememory_matrix_description_t input_desc,
                                                    void* indices,
                                                    wholememory_array_description_t indices_desc,
                                                    wholememory_gref_t embedding_gref,
                                                    wholememory_matrix_description_t embedding_desc,
                                                    cudaStream_t stream);
wholememory_error_code_t scatter_floating_int32_func(
  const void* input,
  wholememory_matrix_description_t input_desc,
  void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_gref_t embedding_gref,
  wholememory_matrix_description_t embedding_desc,
  cudaStream_t stream);
wholememory_error_code_t scatter_floating_int64_func(
  const void* input,
  wholememory_matrix_description_t input_desc,
  void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_gref_t embedding_gref,
  wholememory_matrix_description_t embedding_desc,
  cudaStream_t stream);

wholememory_error_code_t scatter_func(const void* input,
                                      wholememory_matrix_description_t input_desc,
                                      void* indices,
                                      wholememory_array_description_t indices_desc,
                                      wholememory_gref_t embedding_gref,
                                      wholememory_matrix_description_t embedding_desc,
                                      cudaStream_t stream)
{
  try {
    bool embedding_is_float = wholememory_dtype_is_floating_number(embedding_desc.dtype);
    WHOLEMEMORY_CHECK(embedding_is_float ||
                      wholememory_dtype_is_integer_number(embedding_desc.dtype));
    bool input_is_float = wholememory_dtype_is_floating_number(input_desc.dtype);
    WHOLEMEMORY_CHECK(input_is_float || wholememory_dtype_is_integer_number(input_desc.dtype));
    WHOLEMEMORY_EXPECTS(
      embedding_is_float == input_is_float,
      "embedding and output should be same number type, e.g. floating number or integer number.");
    if (indices_desc.size == 0) { return WHOLEMEMORY_SUCCESS; }
    wholememory_error_code_t (*p_scatter_func)(const void*,
                                               wholememory_matrix_description_t,
                                               void*,
                                               wholememory_array_description_t,
                                               wholememory_gref_t,
                                               wholememory_matrix_description_t,
                                               cudaStream_t) = nullptr;
    if (embedding_is_float) {
      if (indices_desc.dtype == WHOLEMEMORY_DT_INT) {
        p_scatter_func = scatter_floating_int32_func;
      } else {
        p_scatter_func = scatter_floating_int64_func;
      }
    } else {
      if (indices_desc.dtype == WHOLEMEMORY_DT_INT) {
        p_scatter_func = scatter_integer_int32_func;
      } else {
        p_scatter_func = scatter_integer_int64_func;
      }
    }
    return p_scatter_func(
      input, input_desc, indices, indices_desc, embedding_gref, embedding_desc, stream);
  } catch (const wholememory::cuda_error& wle) {
    WHOLEMEMORY_ERROR("scatter CUDA LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    WHOLEMEMORY_ERROR("scatter LOGIC Error %s\n", le.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops