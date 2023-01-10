#include "exchange_embeddings_nccl_func.h"

#include <vector>

#include <wholememory/communicator.hpp>

#include "logger.hpp"

namespace wholememory_ops {

wholememory_error_code_t exchange_embeddings_nccl_func(const void* dev_local_gather_buffer_ptr,
                                                       const int64_t* host_send_to_rank_count_ptr,
                                                       const int64_t* host_recv_from_rank_count_ptr,
                                                       void* dev_embedding_recv_buffer_ptr,
                                                       size_t embedding_size,
                                                       wholememory_comm_t wm_comm,
                                                       cudaStream_t stream)
{
  try {
    int world_size;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, wm_comm));
    std::vector<size_t> embedding_send_counts(world_size), embedding_send_displs(world_size);
    std::vector<size_t> embedding_recv_counts(world_size), embedding_recv_displs(world_size);
    size_t send_disp = 0, recv_disp = 0;
    for (int i = 0; i < world_size; i++) {
      embedding_send_displs[i] = send_disp;
      embedding_recv_displs[i] = recv_disp;
      size_t send_count        = host_send_to_rank_count_ptr[i] * embedding_size;
      size_t recv_count        = host_recv_from_rank_count_ptr[i] * embedding_size;
      embedding_send_counts[i] = send_count;
      embedding_recv_counts[i] = recv_count;
      send_disp += send_count;
      recv_disp += recv_count;
    }
    wm_comm->alltoallv(dev_local_gather_buffer_ptr,
                       dev_embedding_recv_buffer_ptr,
                       embedding_send_counts.data(),
                       embedding_send_displs.data(),
                       embedding_recv_counts.data(),
                       embedding_recv_displs.data(),
                       WHOLEMEMORY_DT_INT8,
                       stream);
    WHOLEMEMORY_EXPECTS(wm_comm->sync_stream(stream) == WHOLEMEMORY_SUCCESS,
                        "Embedding AllToAllV failed.");
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("exchange_embeddings_nccl_func LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
