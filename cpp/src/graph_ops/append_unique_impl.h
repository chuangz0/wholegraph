#pragma once
#include <wholememory/env_func_ptrs.h>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace graph_ops{
wholememory_error_code_t graph_append_unique(
void* target_nodes_ptr,
wholememory_array_description_t target_nodes_desc,
void* neighbor_nodes_ptr,
wholememory_array_description_t neighbor_nodes_desc,
memory_context_t* output_unique_node_memory_context,
void* output_neighbor_raw_to_unique_mapping_ptr, 
wholememory_array_description_t output_neighbor_raw_to_unique_mapping_desc,
wholememory_env_func_t * p_env_fns, 
cudaStream_t stream);
}