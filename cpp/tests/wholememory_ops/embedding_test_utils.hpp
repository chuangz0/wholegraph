#pragma once

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace wholememory_ops::testing {

void random_init_local_embedding_table(wholememory_handle_t embedding_handle,
                                       wholememory_matrix_description_t embedding_desc);

void random_init_indice(void* indice,
                        wholememory_array_description_t indice_desc);

}  // namespace wholememory_ops::testing