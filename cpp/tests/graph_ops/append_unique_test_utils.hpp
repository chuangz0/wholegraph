#pragma once
#include <wholememory/tensor_description.h>

namespace graph_ops::testing
{
void gen_node_ids(void* host_target_nodes_ptr, wholememory_array_description_t node_desc, int64_t range, bool unique);
void host_append_unique(void* target_nodes_ptr,
                        wholememory_array_description_t target_nodes_desc,
                        void* neighbor_nodes_ptr,
                        wholememory_array_description_t neighbor_nodes_desc,
                        int* host_total_unique_count,
                        void** host_output_unique_nodes_ptr,
                        void** host_output_neighbor_raw_to_unique_ptr,
                        wholememory_array_description_t output_neighbor_raw_to_unique_desc);
}  // namespace graph_ops::testing
