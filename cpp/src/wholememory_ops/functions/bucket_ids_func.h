#pragma once

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace wholememory_ops {

wholememory_error_code_t bucket_ids_for_ranks(void* indices,
                                              wholememory_array_description_t indice_desc,
                                              int64_t* dev_rank_id_count_ptr,
                                              size_t embedding_entry_count_per_rank,
                                              int world_size,
                                              cudaDeviceProp* prop,
                                              cudaStream_t stream);

}  // namespace wholememory_ops