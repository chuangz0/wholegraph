#include <wholememory/graph_op.h>
#include <graph_ops/append_unique_impl.h>
#include "error.hpp"
#include "logger.hpp"

wholememory_error_code_t append_unique(wholememory_tensor_t target_nodes_tensor,
                                       wholememory_tensor_t neighbor_nodes_tensor,
                                       memory_context_t* output_unique_node_memory_context,
                                       wholememory_tensor_t output_neighbor_raw_to_unique_mapping_tensor, 
                                       wholememory_env_func_t * p_env_fns, 
                                       cudaStream_t stream) {
  auto target_nodes_tensor_description = *wholememory_tensor_get_tensor_description(target_nodes_tensor);
  if (target_nodes_tensor_description.dim != 1) {
    WHOLEMEMORY_ERROR("target_nodes_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto neighbor_nodes_tensor_description = *wholememory_tensor_get_tensor_description(neighbor_nodes_tensor);
  if (neighbor_nodes_tensor_description.dim != 1) {
    WHOLEMEMORY_ERROR("neighbor_nodes_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto output_neighbor_raw_to_unique_mapping_tensor_description = *wholememory_tensor_get_tensor_description(output_neighbor_raw_to_unique_mapping_tensor);
  if (output_neighbor_raw_to_unique_mapping_tensor_description.dim != 1) {
    WHOLEMEMORY_ERROR("output_neighbor_raw_to_unique_mapping_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_array_description_t target_nodes_array_desc, neighbor_nodes_array_desc, output_neighbor_raw_to_unique_mapping_array_desc;

  if (!wholememory_convert_tensor_desc_to_array(&target_nodes_array_desc,
                                                &target_nodes_tensor_description)) {
    WHOLEMEMORY_ERROR("Input target_nodes_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  if (!wholememory_convert_tensor_desc_to_array(&neighbor_nodes_array_desc,
                                                &neighbor_nodes_tensor_description)) {
    WHOLEMEMORY_ERROR("Input neighbor_nodes_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  if (!wholememory_convert_tensor_desc_to_array(&output_neighbor_raw_to_unique_mapping_array_desc,
                                                &output_neighbor_raw_to_unique_mapping_tensor_description)) {
    WHOLEMEMORY_ERROR("Output output_neighbor_raw_to_unique_mapping_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  if (target_nodes_array_desc.dtype != neighbor_nodes_array_desc.dtype) { 
    WHOLEMEMORY_ERROR("target_nodes_dtype should be the same with neighbor_nodes_dtype");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  void* target_nodes_ptr = wholememory_tensor_get_data_pointer(target_nodes_tensor);
  void* neighbor_nodes_ptr = wholememory_tensor_get_data_pointer(neighbor_nodes_tensor);
  void* output_neighbor_raw_to_unique_mapping_ptr = wholememory_tensor_get_data_pointer(output_neighbor_raw_to_unique_mapping_tensor);
  return graph_ops::graph_append_unique(
    target_nodes_ptr,
    target_nodes_array_desc,
    neighbor_nodes_ptr,
    neighbor_nodes_array_desc,
    output_unique_node_memory_context,
    output_neighbor_raw_to_unique_mapping_ptr,
    output_neighbor_raw_to_unique_mapping_array_desc,
    p_env_fns,
    stream);
}