#pragma once

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

#include <wholememory_ops/temp_memory_handle.hpp>
#include <wholememory_ops/thrust_allocator.hpp>

namespace wholememory_ops {

/**
 * exchange ids using nccl
 * @param indices_before_sort : pointer to indices array to sort
 * @param indices_desc : indices array description, should have storage offset = 0, indice can be int32 or int64
 * @param host_recv_rank_id_count_ptr : pointer to int64_t array of received id count from each rank.
 * @param host_rank_id_count_ptr : pointer to int64_t array of id count to send to each rank.
 * @param host_rank_id_offset_ptr : pointer to int64_t array of offsets of id array to send to each rank
 * @param dev_recv_indices_buffer_handle : temp_memory_handle to create buffer for received indices.
 * @param indices_after_sort : pointer to sorted indices array
 * @param raw_indices : pointer to allocated int64_t array to storage raw indices mapping of sort
 * @param wm_comm : WholeMemory communicator
 * @param p_thrust_allocator : thrust allocator
 * @param stream : CUDA stream to use.
 * @return : WHOLEMEMORY_SUCCESS on success, others on failure
 */
wholememory_error_code_t exchange_ids_func(const void *indices_before_sort,
                                           wholememory_array_description_t indices_desc,
                                           const int64_t *host_recv_rank_id_count_ptr,
                                           const int64_t *host_rank_id_count_ptr,
                                           const int64_t *host_rank_id_offset_ptr,
                                           temp_memory_handle *dev_recv_indices_buffer_handle,
                                           void* indices_after_sort,
                                           int64_t *raw_indices,
                                           wholememory_comm_t wm_comm,
                                           wm_thrust_allocator *p_thrust_allocator,
                                           cudaStream_t stream);

}  // namespace wholememory_ops