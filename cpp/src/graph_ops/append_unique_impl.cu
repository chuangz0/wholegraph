#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>
#include "append_unique_func.cuh"

#include "wholememory_ops/register.hpp"


namespace graph_ops
{

REGISTER_DISPATCH_ONE_TYPE(GraphAppendUnique, graph_append_unique_func, SINT3264)
wholememory_error_code_t graph_append_unique(
  void* target_nodes_ptr,
  wholememory_array_description_t target_nodes_desc,
  void* neighbor_nodes_ptr,
  wholememory_array_description_t neighbor_nodes_desc,
  void* output_unique_node_memory_context,
  void* output_neighbor_raw_to_unique_mapping_ptr,
  wholememory_array_description_t output_neighbor_raw_to_unique_mapping_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream) {

    try {
      DISPATCH_ONE_TYPE(target_nodes_desc.dtype,
                        GraphAppendUnique,
                        target_nodes_ptr,
                        target_nodes_desc,
                        neighbor_nodes_ptr,
                        neighbor_nodes_desc,
                        output_unique_node_memory_context,
                        output_neighbor_raw_to_unique_mapping_ptr,
                        output_neighbor_raw_to_unique_mapping_desc,
                        p_env_fns,
                        stream);

    } catch (const raft::cuda_error& rle) {
      // WHOLEMEMORY_FAIL_NOTHROW("%s", rle.what());
      return WHOLEMEMORY_LOGIC_ERROR;
    } catch (const wholememory::logic_error& le) {
      return WHOLEMEMORY_LOGIC_ERROR;
    } catch (...) {
      return WHOLEMEMORY_LOGIC_ERROR;
    }
    return WHOLEMEMORY_SUCCESS;
  }

} // namespace graph_ops
