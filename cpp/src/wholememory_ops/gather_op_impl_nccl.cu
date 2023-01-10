#include <cuda_runtime_api.h>

#include <raft/util/integer_utils.hpp>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "logger.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/memory_handle.hpp"
#include "wholememory_ops/functions/bucket_ids_func.h"
#include "wholememory_ops/functions/exchange_embeddings_nccl_func.h"
#include "wholememory_ops/functions/exchange_ids_nccl_func.h"
#include "wholememory_ops/functions/gather_scatter_func.h"
#include "wholememory_ops/register.hpp"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"

namespace wholememory_ops {

wholememory_error_code_t wholememory_gather_nccl(wholememory_handle_t wholememory_handle,
                                                 wholememory_matrix_description_t wholememory_desc,
                                                 void* indices,
                                                 wholememory_array_description_t indice_desc,
                                                 void* output,
                                                 wholememory_matrix_description_t output_desc,
                                                 wholememory_env_func_t* p_env_fns,
                                                 cudaStream_t stream)
{
  try {
    if (wholememory_desc.storage_offset < 0 ||
        wholememory_desc.storage_offset + wholememory_desc.sizes[0] > wholememory_desc.stride) {
      return WHOLEMEMORY_INVALID_INPUT;
    }

    wm_thrust_allocator thrust_allocator(p_env_fns);

    size_t embedding_size_per_rank;
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_get_partition_plan(&embedding_size_per_rank, wholememory_handle));

    size_t element_size         = wholememory_dtype_get_element_size(wholememory_desc.dtype);
    size_t embedding_entry_size = element_size * wholememory_desc.stride;

    WHOLEMEMORY_EXPECTS_NOTHROW(
      embedding_size_per_rank % embedding_entry_size == 0,
      "embedding_size_per_rank=%ld is not multiple of embedding_entry_size=%ldx%ld",
      embedding_size_per_rank,
      element_size,
      wholememory_desc.stride);

    size_t embedding_entry_count_per_rank = embedding_size_per_rank / embedding_entry_size;

    wholememory_comm_t wm_comm;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(&wm_comm, wholememory_handle));

    int world_rank, world_size;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, wm_comm));
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_rank(&world_rank, wm_comm));

    // Exchange node count
    temp_memory_handle dev_rank_id_count(p_env_fns), host_rank_id_count(p_env_fns),
      host_recv_rank_id_count(p_env_fns);
    int64_t* dev_rank_id_count_ptr =
      static_cast<int64_t*>(dev_rank_id_count.device_malloc(world_size, WHOLEMEMORY_DT_INT64));
    int64_t* host_rank_id_count_ptr =
      static_cast<int64_t*>(host_rank_id_count.host_malloc(world_size, WHOLEMEMORY_DT_INT64));
    int64_t* host_recv_rank_id_count_ptr =
      static_cast<int64_t*>(host_recv_rank_id_count.host_malloc(world_size, WHOLEMEMORY_DT_INT64));

    WHOLEMEMORY_RETURN_ON_FAIL(bucket_ids_for_ranks(indices,
                                                    indice_desc,
                                                    dev_rank_id_count_ptr,
                                                    embedding_entry_count_per_rank,
                                                    world_size,
                                                    &wholememory_handle->device_prop,
                                                    stream));

    WM_CUDA_CHECK(cudaGetLastError());

    temp_memory_handle host_rank_id_offset(p_env_fns);
    temp_memory_handle dev_sorted_indice(p_env_fns);
    temp_memory_handle dev_recv_indice_buffer(p_env_fns);
    temp_memory_handle dev_raw_indice(p_env_fns);
    temp_memory_handle dev_local_gather_buffer(p_env_fns);
    temp_memory_handle dev_embedding_recv_buffer(p_env_fns);
    int64_t* host_rank_id_offset_ptr =
      static_cast<int64_t*>(host_rank_id_offset.host_malloc(world_size + 1, WHOLEMEMORY_DT_INT64));
    int64_t* dev_raw_indice_ptr =
      static_cast<int64_t*>(dev_raw_indice.device_malloc(indice_desc.size, WHOLEMEMORY_DT_INT64));
    void* dev_sorted_indice_ptr =
      dev_sorted_indice.device_malloc(indice_desc.size, indice_desc.dtype);
    WM_CUDA_CHECK(cudaMemcpyAsync(host_rank_id_count_ptr,
                                  dev_rank_id_count_ptr,
                                  sizeof(int64_t) * world_size,
                                  cudaMemcpyDeviceToHost,
                                  stream));
    WM_CUDA_CHECK(cudaGetLastError());
    WM_CUDA_CHECK(cudaStreamSynchronize(stream));
    wm_comm->host_alltoall(
      host_rank_id_count_ptr, host_recv_rank_id_count_ptr, 1, WHOLEMEMORY_DT_INT64);
    host_rank_id_offset_ptr[0] = 0;
    for (int i = 0; i < world_size; i++) {
      host_rank_id_offset_ptr[i + 1] = host_rank_id_offset_ptr[i] + host_rank_id_count_ptr[i];
    }
    WHOLEMEMORY_EXPECTS(wm_comm->sync_stream() == WHOLEMEMORY_SUCCESS,
                        "Rank id count AllToAll failed.");
    int64_t total_recv_count = 0;
    for (int i = 0; i < world_size; i++) {
      total_recv_count += host_recv_rank_id_count_ptr[i];
    }
    void* dev_local_gather_buffer_ptr = dev_local_gather_buffer.device_malloc(
      wholememory_desc.sizes[0] * total_recv_count, output_desc.dtype);
    void* dev_embedding_recv_buffer_ptr = dev_embedding_recv_buffer.device_malloc(
      wholememory_desc.sizes[0] * indice_desc.size, output_desc.dtype);
    void* indice_ptr =
      static_cast<char*>(indices) +
      wholememory_dtype_get_element_size(indice_desc.dtype) * indice_desc.storage_offset;
    // Exchange ids
    WHOLEMEMORY_RETURN_ON_FAIL(exchange_ids_func(indice_ptr,
                                                 indice_desc,
                                                 host_recv_rank_id_count_ptr,
                                                 host_rank_id_count_ptr,
                                                 host_rank_id_offset_ptr,
                                                 &dev_recv_indice_buffer,
                                                 dev_sorted_indice_ptr,
                                                 dev_raw_indice_ptr,
                                                 wm_comm,
                                                 &thrust_allocator,
                                                 stream));

    // Local Gather
    size_t local_mem_offset, local_mem_size;
    void* local_fake_ptr = nullptr;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_local_memory(
      &local_fake_ptr, &local_mem_size, &local_mem_offset, wholememory_handle));
    local_fake_ptr = static_cast<char*>(local_fake_ptr) - local_mem_offset;
    wholememory_gref_t local_fake_gref =
      wholememory_create_continuous_global_reference(local_fake_ptr);
    int64_t local_buffer_size[2] = {wholememory_desc.sizes[0], total_recv_count};
    wholememory_matrix_description_t local_gather_buffer_desc = wholememory_create_matrix_desc(
      local_buffer_size, wholememory_desc.sizes[0], 0, output_desc.dtype);
    auto dev_recv_indice_desc =
      wholememory_create_array_desc(total_recv_count, 0, indice_desc.dtype);
    WHOLEMEMORY_RETURN_ON_FAIL(gather_func(local_fake_gref,
                                           wholememory_desc,
                                           dev_recv_indice_buffer.pointer(),
                                           dev_recv_indice_desc,
                                           dev_local_gather_buffer_ptr,
                                           local_gather_buffer_desc,
                                           stream));
    // AllToAllV for embeddings
    size_t embedding_size =
      wholememory_desc.sizes[0] * wholememory_dtype_get_element_size(output_desc.dtype);
    WHOLEMEMORY_RETURN_ON_FAIL(exchange_embeddings_nccl_func(dev_local_gather_buffer_ptr,
                                                             host_recv_rank_id_count_ptr,
                                                             host_rank_id_count_ptr,
                                                             dev_embedding_recv_buffer_ptr,
                                                             embedding_size,
                                                             wm_comm,
                                                             stream));
    // Local reorder
    wholememory_gref_t output_gref = wholememory_create_continuous_global_reference(output);
    wholememory_matrix_description_t local_recv_buffer_desc =
      wholememory_create_matrix_desc(output_desc.sizes, output_desc.sizes[0], 0, output_desc.dtype);
    auto raw_indice_desc = wholememory_create_array_desc(indice_desc.size, 0, WHOLEMEMORY_DT_INT64);
    WHOLEMEMORY_RETURN_ON_FAIL(scatter_func(dev_embedding_recv_buffer_ptr,
                                            local_recv_buffer_desc,
                                            dev_raw_indice_ptr,
                                            raw_indice_desc,
                                            output_gref,
                                            output_desc,
                                            stream));
    WM_CUDA_CHECK(cudaGetLastError());
    WM_CUDA_CHECK(cudaStreamSynchronize(stream));
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("CUDA logic Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }

  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops