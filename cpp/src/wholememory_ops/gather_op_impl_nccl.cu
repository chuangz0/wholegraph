#include <cuda_runtime_api.h>

#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "wholememory/memory_handle.hpp"
#include "gather_scatter_func.cuh"
#include "wholememory/communicator.hpp"
#include "wholememory_ops/register.hpp"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"

namespace wholememory_ops {

template <typename IndexT>
__global__ void count_for_ranks_kernel(const IndexT* indices,
                                       size_t indice_count,
                                       int64_t* dev_rank_id_count_ptr,
                                       size_t embedding_entry_count_per_rank,
                                       int world_size) {
  extern __shared__ int rank_count_shared[];
  for (int idx = threadIdx.x; idx < world_size; idx += blockDim.x) {
    rank_count_shared[idx] = 0;
  }
  __syncthreads();
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < indice_count;
       idx += blockDim.x * gridDim.x) {
    IndexT node_idx = indices[idx];
    int rank = node_idx / embedding_entry_count_per_rank;
    assert(rank >= 0 && rank < world_size);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    atomicAdd_block(&rank_count_shared[rank], 1);
#else
    atomicAdd(&rank_count_shared[rank], 1);
#endif
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < world_size; idx += blockDim.x) {
    atomicAdd(reinterpret_cast<unsigned long long*>(dev_rank_id_count_ptr) + idx,
              static_cast<unsigned long long>(rank_count_shared[idx]));
  }
}

template <typename IndexT>
void count_for_ranks_temp_fn(void* indices,
                             wholememory_array_description_t indice_desc,
                             int64_t* dev_rank_id_count_ptr,
                             size_t embedding_entry_count_per_rank,
                             int world_size,
                             int sm_count,
                             cudaStream_t stream) {
  static constexpr int BLOCK_SIZE = 128;
  int block_count = raft::div_rounding_up_unsafe(indice_desc.size, BLOCK_SIZE);
  block_count = std::min(block_count, sm_count * 2);
  IndexT* indices_ptr = static_cast<IndexT*>(indices);
  indices_ptr += indice_desc.storage_offset;
  count_for_ranks_kernel<<<block_count, BLOCK_SIZE, sizeof(int) * world_size, stream>>>(indices_ptr,
                                                                                        indice_desc.size,
                                                                                        dev_rank_id_count_ptr,
                                                                                        embedding_entry_count_per_rank,
                                                                                        world_size);
}

REGISTER_DISPATCH_ONE_TYPE(CountForRanks, count_for_ranks_temp_fn, SINT3264)

wholememory_error_code_t count_for_ranks(void* indices,
                                         wholememory_array_description_t indice_desc,
                                         int64_t* dev_rank_id_count_ptr,
                                         size_t embedding_entry_count_per_rank,
                                         int world_size,
                                         cudaDeviceProp* prop,
                                         cudaStream_t stream) {
  try {
    CUDA_CHECK(cudaMemsetAsync(dev_rank_id_count_ptr, 0, sizeof(int64_t) * world_size, stream));
    if (indice_desc.size == 0) {
      return WHOLEMEMORY_SUCCESS;
    }
    DISPATCH_ONE_TYPE(indice_desc.dtype,
                      CountForRanks,
                      indices,
                      indice_desc,
                      dev_rank_id_count_ptr,
                      embedding_entry_count_per_rank,
                      world_size,
                      prop->multiProcessorCount,
                      stream);
    CUDA_CHECK(cudaGetLastError());
  } catch (raft::cuda_error& rce) {
    return WHOLEMEMORY_CUDA_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

template<typename IndexT>
void exchange_ids_temp_func(void* indices_before_sort,
                            int64_t indice_count,
                            wholememory_dtype_t index_type,
                            int64_t* reverse_indice,
                            const int64_t* host_recv_rank_id_count_ptr,
                            const int64_t* host_rank_id_count_ptr,
                            const int64_t* host_rank_id_offset_ptr,
                            int world_size,
                            temp_memory_handle* dev_recv_indice_buffer,
                            wholememory_comm_t wm_comm,
                            wm_thrust_allocator* p_thrust_allocator,
                            cudaStream_t stream) {
  WHOLEMEMORY_CHECK(index_type == WHOLEMEMORY_DT_INT || index_type == WHOLEMEMORY_DT_INT64);
  wm_thrust_allocator& allocator = *p_thrust_allocator;
  thrust::sequence(thrust::cuda::par(allocator).on(stream), reverse_indice,
                   reverse_indice + indice_count, 0);
  IndexT* sorted_indice = static_cast<IndexT*>(indices_before_sort);
  thrust::sort_by_key(thrust::cuda::par(allocator).on(stream), sorted_indice,
                      sorted_indice + indice_count, reverse_indice);
  int64_t total_recv_count = 0;
  std::vector<size_t> host_recv_offset(world_size);
  for (int i = 0; i < world_size; i++) {
    host_recv_offset[i] = total_recv_count;
    total_recv_count += host_recv_rank_id_count_ptr[i];
  }
  IndexT *dev_recv_indice_buffer_ptr =
      static_cast<IndexT *>(dev_recv_indice_buffer->device_malloc(total_recv_count, index_type));
  raft::comms::datatype_t raft_comm_dt = (index_type == WHOLEMEMORY_DT_INT) ? raft::comms::datatype_t::INT32 : raft::comms::datatype_t::INT64;
  wm_comm->raft_nccl_comm->alltoallv(sorted_indice,
                                     dev_recv_indice_buffer_ptr,
                                     reinterpret_cast<const size_t*>(host_rank_id_count_ptr),
                                     reinterpret_cast<const size_t*>(host_rank_id_offset_ptr),
                                     reinterpret_cast<const size_t*>(host_recv_rank_id_count_ptr),
                                     host_recv_offset.data(),
                                     raft_comm_dt,
                                     stream);
  wm_comm->raft_nccl_comm->sync_stream(stream);
}

REGISTER_DISPATCH_ONE_TYPE(NCCLExchangeIDs, exchange_ids_temp_func, SINT3264)

REGISTER_DISPATCH_THREE_TYPES(NCCLLocalGather, gather_func, HALF_FLOAT_DOUBLE, SINT3264, HALF_FLOAT_DOUBLE)

template<typename DataTypeT, typename IndexT>
void local_scatter_temp_func(const void *input,
                             wholememory_matrix_description_t input_desc,
                             void *indices,
                             int64_t indice_count,
                             wholememory_gref_t embedding_gref,
                             wholememory_matrix_description_t embedding_desc,
                             cudaStream_t stream) {
  scatter_func<DataTypeT, IndexT, DataTypeT>(input,
                                             input_desc,
                                             indices,
                                             indice_count,
                                             embedding_gref,
                                             embedding_desc,
                                             stream);
}

REGISTER_DISPATCH_TWO_TYPES(NCCLLocalScatter, local_scatter_temp_func, HALF_FLOAT_DOUBLE, SINT3264)

wholememory_error_code_t wholememory_gather_nccl(wholememory_handle_t wholememory_handle,
                                                 wholememory_matrix_description_t wholememory_desc,
                                                 void *indices,
                                                 wholememory_array_description_t indice_desc,
                                                 void *output,
                                                 wholememory_matrix_description_t output_desc,
                                                 wholememory_env_func_t *p_env_fns,
                                                 cudaStream_t stream) {
  if (wholememory_desc.storage_offset < 0
      || wholememory_desc.storage_offset + wholememory_desc.sizes[0] > wholememory_desc.stride) {
    return WHOLEMEMORY_INVALID_INPUT;
  }

  wm_thrust_allocator thrust_allocator(p_env_fns);

  size_t embedding_size_per_rank;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_partition_plan(&embedding_size_per_rank, wholememory_handle));

  size_t element_size = wholememory_dtype_get_element_size(wholememory_desc.dtype);
  size_t embedding_entry_size = element_size * wholememory_desc.stride;

  WHOLEMEMORY_EXPECTS_NOTHROW(embedding_size_per_rank % embedding_entry_size == 0,
                              "embedding_size_per_rank=%ld is not multiple of embedding_entry_size=%ldx%ld",
                              embedding_size_per_rank, element_size, wholememory_desc.stride);

  size_t embedding_entry_count_per_rank = embedding_size_per_rank / embedding_entry_size;

  wholememory_comm_t wm_comm;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(&wm_comm, wholememory_handle));

  int world_rank, world_size;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, wm_comm));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_rank(&world_rank, wm_comm));

  // Exchange node count
  temp_memory_handle dev_rank_id_count(p_env_fns), host_rank_id_count(p_env_fns), host_recv_rank_id_count(p_env_fns);
  int64_t *dev_rank_id_count_ptr =
      static_cast<int64_t *>(dev_rank_id_count.device_malloc(world_size, WHOLEMEMORY_DT_INT64));
  int64_t *host_rank_id_count_ptr =
      static_cast<int64_t *>(host_rank_id_count.host_malloc(world_size, WHOLEMEMORY_DT_INT64));
  int64_t *host_recv_rank_id_count_ptr =
      static_cast<int64_t *>(host_recv_rank_id_count.host_malloc(world_size, WHOLEMEMORY_DT_INT64));

  WHOLEMEMORY_RETURN_ON_FAIL(count_for_ranks(indices,
                                             indice_desc,
                                             dev_rank_id_count_ptr,
                                             embedding_entry_count_per_rank,
                                             world_size,
                                             &wholememory_handle->device_prop,
                                             stream));

  CUDA_CHECK(cudaGetLastError());

  temp_memory_handle host_rank_id_offset(p_env_fns);
  temp_memory_handle dev_sorted_indice(p_env_fns);
  temp_memory_handle dev_recv_indice_buffer(p_env_fns);
  temp_memory_handle dev_reverse_indice(p_env_fns);
  temp_memory_handle dev_local_gather_buffer(p_env_fns);
  temp_memory_handle dev_embedding_recv_buffer(p_env_fns);
  int64_t *host_rank_id_offset_ptr =
      static_cast<int64_t *>(host_rank_id_offset.host_malloc(world_size + 1, WHOLEMEMORY_DT_INT64));
  int64_t *dev_reverse_indice_ptr =
      static_cast<int64_t *>(dev_reverse_indice.device_malloc(indice_desc.size, WHOLEMEMORY_DT_INT64));
  void* dev_sorted_indice_ptr = dev_sorted_indice.device_malloc(indice_desc.size, indice_desc.dtype);
  try {
    CUDA_CHECK(cudaMemcpyAsync(host_rank_id_count_ptr,
                               dev_rank_id_count_ptr,
                               sizeof(int64_t) * world_size,
                               cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    wm_comm->raft_nccl_comm->host_alltoall(host_rank_id_count_ptr,
                                           host_recv_rank_id_count_ptr,
                                           1,
                                           raft::comms::datatype_t::INT64);
    host_rank_id_offset_ptr[0] = 0;
    for (int i = 0; i < world_size; i++) {
      host_rank_id_offset_ptr[i + 1] = host_rank_id_offset_ptr[i] + host_rank_id_count_ptr[i];
    }
    WHOLEMEMORY_EXPECTS(wm_comm->raft_nccl_comm->sync_stream() == raft::comms::status_t::SUCCESS,
                        "Rank id count AllToAll failed.");
    int64_t total_recv_count = 0;
    for (int i = 0; i < world_size; i++) {
      total_recv_count += host_recv_rank_id_count_ptr[i];
    }
    void *dev_local_gather_buffer_ptr =
        dev_local_gather_buffer.device_malloc(wholememory_desc.sizes[0] * total_recv_count, output_desc.dtype);
    void *dev_embedding_recv_buffer_ptr =
        dev_embedding_recv_buffer.device_malloc(wholememory_desc.sizes[0] * indice_desc.size, output_desc.dtype);
    void *indice_ptr = static_cast<char *>(indices)
        + wholememory_dtype_get_element_size(indice_desc.dtype) * indice_desc.storage_offset;
    // Exchange ids
    CUDA_CHECK(cudaMemcpyAsync(dev_sorted_indice_ptr, indice_ptr,
                               wholememory_get_memory_size_from_array(&indice_desc),
                               cudaMemcpyDeviceToDevice, stream));
    DISPATCH_ONE_TYPE(indice_desc.dtype,
                      NCCLExchangeIDs,
                      dev_sorted_indice_ptr,
                      indice_desc.size,
                      indice_desc.dtype,
                      dev_reverse_indice_ptr,
                      host_recv_rank_id_count_ptr,
                      host_rank_id_count_ptr,
                      host_rank_id_offset_ptr,
                      world_size,
                      &dev_recv_indice_buffer,
                      wm_comm,
                      &thrust_allocator,
                      stream);
    // Local Gather
    size_t local_mem_offset, local_mem_size;
    void* local_fake_ptr = nullptr;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_local_memory(&local_fake_ptr,
                                                            &local_mem_size,
                                                            &local_mem_offset,
                                                            wholememory_handle));
    local_fake_ptr = static_cast<char*>(local_fake_ptr) - local_mem_offset;
    wholememory_gref_t local_fake_gref = wholememory_create_continuous_global_reference(local_fake_ptr);
    int64_t local_buffer_size[2] = {wholememory_desc.sizes[0], total_recv_count};
    wholememory_matrix_description_t local_gather_buffer_desc =
        wholememory_create_matrix_desc(local_buffer_size, wholememory_desc.sizes[0], 0, output_desc.dtype);
    DISPATCH_THREE_TYPES(wholememory_desc.dtype,
                         indice_desc.dtype,
                         output_desc.dtype,
                         NCCLLocalGather,
                         local_fake_gref,
                         wholememory_desc,
                         dev_recv_indice_buffer.pointer(),
                         total_recv_count,
                         dev_local_gather_buffer_ptr,
                         local_gather_buffer_desc,
                         stream);
    // AllToAllV for embeddings
    std::vector<size_t> embedding_send_counts(world_size), embedding_send_displs(world_size);
    std::vector<size_t> embedding_recv_counts(world_size), embedding_recv_displs(world_size);
    size_t send_disp = 0, recv_disp = 0;
    size_t embedding_size = wholememory_desc.sizes[0] * wholememory_dtype_get_element_size(output_desc.dtype);
    for (int i = 0; i < world_size; i++) {
      embedding_send_displs[i] = send_disp;
      embedding_recv_displs[i] = recv_disp;
      size_t send_count = host_recv_rank_id_count_ptr[i] * embedding_size;
      size_t recv_count = host_rank_id_count_ptr[i] * embedding_size;
      embedding_send_counts[i] = send_count;
      embedding_recv_counts[i] = recv_count;
      send_disp += send_count;
      recv_disp += recv_count;
    }
    wm_comm->raft_nccl_comm->alltoallv(dev_local_gather_buffer_ptr,
                                       dev_embedding_recv_buffer_ptr,
                                       embedding_send_counts.data(),
                                       embedding_send_displs.data(),
                                       embedding_recv_counts.data(),
                                       embedding_recv_displs.data(),
                                       raft::comms::datatype_t::CHAR,
                                       stream);
    WHOLEMEMORY_EXPECTS(wm_comm->raft_nccl_comm->sync_stream(stream) == raft::comms::status_t::SUCCESS,
                        "Embedding AllToAllV failed.");
    // Local reorder
    wholememory_gref_t output_gref = wholememory_create_continuous_global_reference(output);
    wholememory_matrix_description_t local_recv_buffer_desc =
        wholememory_create_matrix_desc(output_desc.sizes, output_desc.sizes[0], 0, output_desc.dtype);
    DISPATCH_TWO_TYPES(output_desc.dtype,
                       WHOLEMEMORY_DT_INT64,
                       NCCLLocalScatter,
                       dev_embedding_recv_buffer_ptr,
                       local_recv_buffer_desc,
                       dev_reverse_indice_ptr,
                       indice_desc.size,
                       output_gref,
                       output_desc,
                       stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } catch (raft::logic_error& rle) {
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (raft::cuda_error& rce) {
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (wholememory::logic_error& wle) {
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }

  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops