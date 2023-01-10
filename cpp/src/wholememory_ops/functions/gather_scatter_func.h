#pragma once

#include <wholememory/global_reference.h>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace wholememory_ops {

wholememory_error_code_t gather_func(wholememory_gref_t embedding_gref,
                                     wholememory_matrix_description_t embedding_desc,
                                     void* indices,
                                     wholememory_array_description_t indices_desc,
                                     void* output,
                                     wholememory_matrix_description_t output_desc,
                                     cudaStream_t stream);

wholememory_error_code_t scatter_func(const void* input,
                                      wholememory_matrix_description_t input_desc,
                                      void* indices,
                                      wholememory_array_description_t indices_desc,
                                      wholememory_gref_t embedding_gref,
                                      wholememory_matrix_description_t embedding_desc,
                                      cudaStream_t stream);

}  // namespace wholememory_ops