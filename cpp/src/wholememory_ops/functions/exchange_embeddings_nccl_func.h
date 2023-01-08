#pragma once

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

#include <wholememory_ops/temp_memory_handle.hpp>
#include <wholememory_ops/thrust_allocator.hpp>

namespace wholememory_ops {

/**
 * Exchange embeddings between ranks
 * @param dev_local_gather_buffer_ptr : local buffer to send
 * @param host_send_to_rank_count_ptr : id count that current rank send to other ranks
 * @param host_recv_from_rank_count_ptr : id count that current rank receive from each rank
 * @param dev_embedding_recv_buffer_ptr : local buffer to receive embedding data
 * @param embedding_size : embedding size in bytes.
 * @param wm_comm : WholeMemory communicator
 * @param stream : CUDA stream to use
 * @return : WHOLEMEMORY_SUCCESS on success, others on failure.
 */
wholememory_error_code_t exchange_embeddings_nccl_func(const void* dev_local_gather_buffer_ptr,
                                                       const int64_t* host_send_to_rank_count_ptr,
                                                       const int64_t* host_recv_from_rank_count_ptr,
                                                       void* dev_embedding_recv_buffer_ptr,
                                                       size_t embedding_size,
                                                       wholememory_comm_t wm_comm,
                                                       cudaStream_t stream);

}  // namespace wholememory_ops