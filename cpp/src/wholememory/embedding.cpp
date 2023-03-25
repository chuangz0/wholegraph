#include <wholememory/embedding.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory_op.h>

#include "cuda_macros.hpp"
#include "embedding_optimizer.hpp"
#include "error.hpp"
#include "logger.hpp"
#include "wholememory_ops/functions/embedding_cache_func.h"
#include "wholememory_ops/functions/exchange_embeddings_nccl_func.h"
#include "wholememory_ops/functions/exchange_ids_nccl_func.h"
#include "wholememory_ops/functions/gather_cached_func.h"
#include "wholememory_ops/functions/gather_scatter_func.h"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"

#ifdef __cplusplus
extern "C" {
#endif

struct wholememory_embedding_ {
  wholememory_tensor_t allocated_embedding          = nullptr;
  wholememory_tensor_t user_embedding               = nullptr;  // subtensor of allocated_embedding
  wholememory_embedding_cache_policy_t cache_policy = nullptr;
  wholememory_embedding_optimizer_t optimizer       = nullptr;
};

#ifdef __cplusplus
}
#endif

namespace wholememory {

class embedding_base : public wholememory_embedding_ {
 public:
  embedding_base()          = default;
  virtual ~embedding_base() = default;
  wholememory_error_code_t allocate(wholememory_matrix_description_t* embedding_description,
                                    wholememory_comm_t comm,
                                    wholememory_memory_type_t memory_type,
                                    wholememory_memory_location_t memory_location,
                                    wholememory_embedding_cache_policy_t policy,
                                    wholememory_embedding_optimizer_t opt) noexcept;
  void deallocate() noexcept;
  virtual wholememory_error_code_t gather(wholememory_tensor_t indices,
                                          wholememory_tensor_t output,
                                          bool adjust_cache,
                                          wholememory_env_func_t* p_env_fns,
                                          cudaStream_t stream) noexcept = 0;
  // virtual wholememory_error_code_t apply_gradients();
 protected:
  wholememory_comm_t raw_embedding_comm_        = nullptr;
  wholememory::embedding_cache_base* cache_ptr_ = nullptr;
};

wholememory_error_code_t embedding_base::allocate(
  wholememory_matrix_description_t* embedding_description,
  wholememory_comm_t comm,
  wholememory_memory_type_t memory_type,
  wholememory_memory_location_t memory_location,
  wholememory_embedding_cache_policy_t policy,
  wholememory_embedding_optimizer_t opt) noexcept
{
  cache_policy        = policy;
  optimizer           = opt;
  raw_embedding_comm_ = comm;
  wholememory_tensor_description_t padded_embedding_tensor_description;
  try {
    if (cache_policy != nullptr) {
      WHOLEMEMORY_CHECK_NOTHROW(cache_policy->cache_comm != nullptr);
      if (cache_policy->cache_comm != comm) {
        cache_ptr_ = new wholememory::local_cache_for_global(cache_policy);
      } else {
        cache_ptr_ = new wholememory::device_cache_for_host(cache_policy);
      }
      WHOLEMEMORY_RETURN_ON_FAIL(
        cache_ptr_->get_embedding_requirement(&padded_embedding_tensor_description,
                                              *embedding_description,
                                              comm,
                                              memory_type,
                                              memory_location));
    } else {
      wholememory_copy_matrix_desc_to_tensor(&padded_embedding_tensor_description,
                                             embedding_description);
      int64_t const embedding_dim = embedding_description->sizes[1];
      size_t const element_size = wholememory_dtype_get_element_size(embedding_description->dtype);
      int64_t const align_count = 16 / element_size;
      int64_t const embedding_stride                     = embedding_dim % align_count == 0
                                                             ? embedding_dim
                                                             : (embedding_dim / align_count + 1) * align_count;
      padded_embedding_tensor_description.storage_offset = 0;
      padded_embedding_tensor_description.strides[0]     = embedding_stride;
      padded_embedding_tensor_description.strides[1]     = 1;
    }
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_create_tensor(&allocated_embedding,
                                                         &padded_embedding_tensor_description,
                                                         comm,
                                                         memory_type,
                                                         memory_location));
    int64_t starts[2] = {0, 0};
    int64_t ends[2]   = {embedding_description->sizes[0], embedding_description->sizes[1]};
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_tensor_get_subtensor(allocated_embedding, &starts[0], &ends[0], &user_embedding));
    if (cache_ptr_ != nullptr) { WHOLEMEMORY_RETURN_ON_FAIL(cache_ptr_->allocate(user_embedding)); }
    if (optimizer != nullptr) {
      // TODO: optimizer states
      WHOLEMEMORY_CHECK_NOTHROW(false);
      return WHOLEMEMORY_NOT_IMPLEMENTED;
    }
  } catch (std::bad_alloc& sba) {
    WHOLEMEMORY_ERROR("bad_alloc");
    return WHOLEMEMORY_OUT_OF_MEMORY;
  } catch (...) {
    WHOLEMEMORY_ERROR("Unknown error");
    return WHOLEMEMORY_UNKNOW_ERROR;
  }

  return WHOLEMEMORY_SUCCESS;
}

void embedding_base::deallocate() noexcept
{
  if (optimizer != nullptr) {
    // TODO: optimizer states
    WHOLEMEMORY_CHECK_NOTHROW(false);
  }
  if (cache_ptr_ != nullptr) {
    delete cache_ptr_;
    cache_ptr_ = nullptr;
  }
  WHOLEMEMORY_CHECK_NOTHROW(wholememory_destroy_tensor(user_embedding) == WHOLEMEMORY_SUCCESS);
  WHOLEMEMORY_CHECK_NOTHROW(wholememory_destroy_tensor(allocated_embedding) == WHOLEMEMORY_SUCCESS);
}

class noncached_embedding : public embedding_base {
 public:
  noncached_embedding()          = default;
  virtual ~noncached_embedding() = default;
  wholememory_error_code_t gather(wholememory_tensor_t indices,
                                  wholememory_tensor_t output,
                                  bool adjust_cache,
                                  wholememory_env_func_t* p_env_fns,
                                  cudaStream_t stream) noexcept override;
};

wholememory_error_code_t noncached_embedding::gather(wholememory_tensor_t indices,
                                                     wholememory_tensor_t output,
                                                     bool adjust_cache,
                                                     wholememory_env_func_t* p_env_fns,
                                                     cudaStream_t stream) noexcept
{
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_gather(allocated_embedding, indices, output, p_env_fns, stream));
  return WHOLEMEMORY_SUCCESS;
}

class device_cached_host_embedding : public embedding_base {
 public:
  device_cached_host_embedding()          = default;
  virtual ~device_cached_host_embedding() = default;
  wholememory_error_code_t gather(wholememory_tensor_t indices,
                                  wholememory_tensor_t output,
                                  bool adjust_cache,
                                  wholememory_env_func_t* p_env_fns,
                                  cudaStream_t stream) noexcept override;
};

wholememory_error_code_t device_cached_host_embedding::gather(wholememory_tensor_t indices,
                                                              wholememory_tensor_t output,
                                                              bool adjust_cache,
                                                              wholememory_env_func_t* p_env_fns,
                                                              cudaStream_t stream) noexcept
{
  auto* indice_desc    = wholememory_tensor_get_tensor_description(indices);
  auto* output_desc    = wholememory_tensor_get_tensor_description(output);
  auto* embedding_desc = wholememory_tensor_get_tensor_description(allocated_embedding);
  WHOLEMEMORY_CHECK_NOTHROW(indice_desc->dim == 1);
  wholememory_ops::temp_memory_handle host_recv_rank_id_count_handle(p_env_fns),
    host_rank_id_count_handle(p_env_fns);
  wholememory_ops::temp_memory_handle dev_recv_indices_buffer_handle(p_env_fns);
  wholememory_ops::temp_memory_handle dev_raw_indice_handle(p_env_fns);
  size_t const embedding_entry_count_per_rank =
    wholememory_tensor_get_entry_per_partition(allocated_embedding);
  wholememory_ops::wm_thrust_allocator thrust_allocator(p_env_fns);
  int world_size = -1, world_rank = -1;
  int64_t* host_recv_rank_id_count_ptr = nullptr;
  int64_t* host_rank_id_count_ptr      = nullptr;
  int64_t* dev_raw_indice_ptr          = nullptr;
  int64_t total_recv_count             = 0;
  if (adjust_cache || cache_policy->cache_memory_type == WHOLEMEMORY_MT_DISTRIBUTED) {
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, raw_embedding_comm_));
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_rank(&world_rank, raw_embedding_comm_));
    host_recv_rank_id_count_ptr = static_cast<int64_t*>(
      host_recv_rank_id_count_handle.pinned_malloc(world_size, WHOLEMEMORY_DT_INT64));
    host_rank_id_count_ptr = static_cast<int64_t*>(
      host_rank_id_count_handle.pinned_malloc(world_size, WHOLEMEMORY_DT_INT64));
    dev_raw_indice_ptr = static_cast<int64_t*>(
      dev_raw_indice_handle.device_malloc(indice_desc->sizes[0], WHOLEMEMORY_DT_INT64));
    wholememory_array_description_t indice_array_desc;
    WHOLEMEMORY_CHECK_NOTHROW(
      wholememory_convert_tensor_desc_to_array(&indice_array_desc, indice_desc));
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_ops::bucket_and_exchange_ids_func(wholememory_tensor_get_data_pointer(indices),
                                                    indice_array_desc,
                                                    host_recv_rank_id_count_ptr,
                                                    host_rank_id_count_ptr,
                                                    &dev_recv_indices_buffer_handle,
                                                    dev_raw_indice_ptr,
                                                    embedding_entry_count_per_rank,
                                                    raw_embedding_comm_,
                                                    &thrust_allocator,
                                                    p_env_fns,
                                                    stream));
    if (adjust_cache) {
      total_recv_count = 0;
      for (int i = 0; i < world_size; i++) {
        total_recv_count += host_recv_rank_id_count_ptr[i];
      }
      auto update_indice_desc =
        wholememory_create_array_desc(total_recv_count, 0, indice_desc->dtype);
      WHOLEMEMORY_RETURN_ON_FAIL(
        wholememory_ops::update_cache_direct_same_comm(dev_recv_indices_buffer_handle.pointer(),
                                                       update_indice_desc,
                                                       allocated_embedding,
                                                       cache_ptr_->get_cache_local_data(),
                                                       cache_ptr_->get_cache_set_coverage(),
                                                       p_env_fns,
                                                       stream));
      WM_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));
      WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_barrier(cache_policy->cache_comm));
    }
  }
  if (cache_policy->cache_memory_type == WHOLEMEMORY_MT_DISTRIBUTED) {
    // Local Gather
    total_recv_count = 0;
    for (int i = 0; i < world_size; i++) {
      total_recv_count += host_recv_rank_id_count_ptr[i];
    }
    wholememory_ops::temp_memory_handle dev_local_gather_buffer(p_env_fns);
    wholememory_ops::temp_memory_handle dev_embedding_recv_buffer(p_env_fns);
    void* dev_local_gather_buffer_ptr = dev_local_gather_buffer.device_malloc(
      embedding_desc->sizes[1] * total_recv_count, output_desc->dtype);
    void* dev_embedding_recv_buffer_ptr = dev_embedding_recv_buffer.device_malloc(
      embedding_desc->sizes[1] * indice_desc->sizes[0], output_desc->dtype);
    wholememory_tensor_t local_raw_tensor;
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_tensor_map_local_tensor(allocated_embedding, &local_raw_tensor));
    wholememory_gref_t local_raw_gref;
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_tensor_get_global_reference(local_raw_tensor, &local_raw_gref));

    wholememory_tensor_t cached_embedding_local_tensor =
      cache_ptr_->get_cache_local_data()->cache_line_data_;
    wholememory_gref_t cached_embedding_gref;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_tensor_get_global_reference(
      cached_embedding_local_tensor, &cached_embedding_gref));
    wholememory_gref_t cache_line_tag_gref;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_tensor_get_global_reference(
      cache_ptr_->get_cache_local_data()->cache_line_tag_, &cache_line_tag_gref));
    int64_t const rank_start_gid =
      wholememory_tensor_get_entry_per_partition(allocated_embedding) * world_rank;
    wholememory_tensor_description_t recv_indices_desc;
    auto recv_indices_array_desc =
      wholememory_create_array_desc(total_recv_count, 0, indice_desc->dtype);
    wholememory_copy_array_desc_to_tensor(&recv_indices_desc, &recv_indices_array_desc);
    wholememory_tensor_description_t local_gather_desc = *output_desc;
    local_gather_desc.sizes[0]                         = total_recv_count;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_ops::gather_cached_func(
      local_raw_gref,
      wholememory_tensor_get_tensor_description(local_raw_tensor),
      cached_embedding_gref,
      wholememory_tensor_get_tensor_description(cached_embedding_local_tensor),
      cache_line_tag_gref,
      dev_recv_indices_buffer_handle.pointer(),
      &recv_indices_desc,
      dev_local_gather_buffer_ptr,
      &local_gather_desc,
      cache_ptr_->get_cache_set_coverage(),
      rank_start_gid,
      rank_start_gid,
      stream));
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_destroy_tensor(local_raw_tensor));
    // AllToAllV
    wholememory_comm_t wm_comm;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(
      &wm_comm, wholememory_tensor_get_memory_handle(allocated_embedding)));
    size_t const embedding_size =
      embedding_desc->sizes[1] * wholememory_dtype_get_element_size(output_desc->dtype);
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_ops::exchange_embeddings_nccl_func(dev_local_gather_buffer_ptr,
                                                     host_recv_rank_id_count_ptr,
                                                     host_rank_id_count_ptr,
                                                     dev_embedding_recv_buffer_ptr,
                                                     embedding_size,
                                                     wm_comm,
                                                     stream));
    WM_CUDA_DEBUG_SYNC_STREAM(stream);
    // Local reorder
    wholememory_gref_t output_gref =
      wholememory_create_continuous_global_reference(wholememory_tensor_get_data_pointer(output));
    wholememory_matrix_description_t local_recv_buffer_desc = wholememory_create_matrix_desc(
      output_desc->sizes, output_desc->sizes[1], 0, output_desc->dtype);
    auto raw_indice_desc =
      wholememory_create_array_desc(indice_desc->sizes[0], 0, WHOLEMEMORY_DT_INT64);
    int64_t total_need_scatter_count = 0;
    for (int i = 0; i < world_size; i++) {
      total_need_scatter_count += host_rank_id_count_ptr[i];
    }
    local_recv_buffer_desc.sizes[0] = total_need_scatter_count;
    raw_indice_desc.size            = total_need_scatter_count;
    wholememory_matrix_description_t output_matrix_desc;
    WHOLEMEMORY_CHECK_NOTHROW(
      wholememory_convert_tensor_desc_to_matrix(&output_matrix_desc, output_desc));
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_ops::scatter_func(dev_embedding_recv_buffer_ptr,
                                                             local_recv_buffer_desc,
                                                             dev_raw_indice_ptr,
                                                             raw_indice_desc,
                                                             output_gref,
                                                             output_matrix_desc,
                                                             stream));
    WM_CUDA_DEBUG_SYNC_STREAM(stream);
  } else {
    wholememory_gref_t global_raw_gref, global_cached_gref, global_cached_line_tag_gref;
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_tensor_get_global_reference(allocated_embedding, &global_raw_gref));
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_tensor_get_global_reference(
      cache_ptr_->cache_line_data_wm_tensor_, &global_cached_gref));
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_tensor_get_global_reference(
      cache_ptr_->cache_line_tag_wm_tensor_, &global_cached_line_tag_gref));
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_ops::gather_cached_func(
      global_raw_gref,
      wholememory_tensor_get_tensor_description(allocated_embedding),
      global_cached_gref,
      wholememory_tensor_get_tensor_description(cache_ptr_->cache_line_data_wm_tensor_),
      global_cached_line_tag_gref,
      wholememory_tensor_get_data_pointer(indices),
      indice_desc,
      wholememory_tensor_get_data_pointer(output),
      output_desc,
      cache_ptr_->get_cache_set_coverage(),
      0,
      0,
      stream));
  }
  return WHOLEMEMORY_SUCCESS;
}

class local_cached_global_readonly_embedding : public embedding_base {
 public:
  local_cached_global_readonly_embedding()          = default;
  virtual ~local_cached_global_readonly_embedding() = default;
  wholememory_error_code_t gather(wholememory_tensor_t indices,
                                  wholememory_tensor_t output,
                                  bool adjust_cache,
                                  wholememory_env_func_t* p_env_fns,
                                  cudaStream_t stream) noexcept override;
};

wholememory_error_code_t local_cached_global_readonly_embedding::gather(
  wholememory_tensor_t indices,
  wholememory_tensor_t output,
  bool adjust_cache,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream) noexcept
{
  auto* indice_desc = wholememory_tensor_get_tensor_description(indices);
  auto* output_desc = wholememory_tensor_get_tensor_description(output);
  WHOLEMEMORY_CHECK_NOTHROW(indice_desc->dim == 1);
  wholememory_ops::temp_memory_handle host_recv_rank_id_count_handle(p_env_fns),
    host_rank_id_count_handle(p_env_fns);
  wholememory_ops::temp_memory_handle dev_recv_indices_buffer_handle(p_env_fns);
  wholememory_ops::temp_memory_handle dev_raw_indice_handle(p_env_fns);
  size_t const embedding_entry_count_per_rank =
    wholememory_tensor_get_entry_per_partition(cache_ptr_->access_count_wm_tensor_);
  wholememory_ops::wm_thrust_allocator thrust_allocator(p_env_fns);
  int cache_world_size = -1, cache_world_rank = -1;
  int64_t* host_recv_rank_id_count_ptr = nullptr;
  int64_t* host_rank_id_count_ptr      = nullptr;
  int64_t* dev_raw_indice_ptr          = nullptr;
  int64_t total_recv_count             = 0;
  // WHOLEMEMORY_MT_DISTRIBUTED is actully not supported now
  if (adjust_cache || cache_policy->cache_memory_type == WHOLEMEMORY_MT_DISTRIBUTED) {
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_communicator_get_size(&cache_world_size, cache_policy->cache_comm));
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_communicator_get_rank(&cache_world_rank, cache_policy->cache_comm));
    host_recv_rank_id_count_ptr = static_cast<int64_t*>(
      host_recv_rank_id_count_handle.pinned_malloc(cache_world_size, WHOLEMEMORY_DT_INT64));
    host_rank_id_count_ptr = static_cast<int64_t*>(
      host_rank_id_count_handle.pinned_malloc(cache_world_size, WHOLEMEMORY_DT_INT64));
    dev_raw_indice_ptr = static_cast<int64_t*>(
      dev_raw_indice_handle.device_malloc(indice_desc->sizes[0], WHOLEMEMORY_DT_INT64));
    wholememory_array_description_t indice_array_desc;
    WHOLEMEMORY_CHECK_NOTHROW(
      wholememory_convert_tensor_desc_to_array(&indice_array_desc, indice_desc));
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_ops::bucket_and_exchange_ids_func(wholememory_tensor_get_data_pointer(indices),
                                                    indice_array_desc,
                                                    host_recv_rank_id_count_ptr,
                                                    host_rank_id_count_ptr,
                                                    &dev_recv_indices_buffer_handle,
                                                    dev_raw_indice_ptr,
                                                    embedding_entry_count_per_rank,
                                                    cache_policy->cache_comm,
                                                    &thrust_allocator,
                                                    p_env_fns,
                                                    stream));
    if (adjust_cache) {
      total_recv_count = 0;
      for (int i = 0; i < cache_world_size; i++) {
        total_recv_count += host_recv_rank_id_count_ptr[i];
      }
      auto update_indice_desc =
        wholememory_create_array_desc(total_recv_count, 0, indice_desc->dtype);
      WHOLEMEMORY_RETURN_ON_FAIL(
        wholememory_ops::update_cache_different_comm(dev_recv_indices_buffer_handle.pointer(),
                                                     update_indice_desc,
                                                     allocated_embedding,
                                                     cache_policy->cache_comm,
                                                     embedding_entry_count_per_rank,
                                                     cache_ptr_->get_cache_local_data(),
                                                     cache_ptr_->get_cache_set_coverage(),
                                                     p_env_fns,
                                                     stream));
      WM_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));
      WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_barrier(cache_policy->cache_comm));
    }
  }
  wholememory_gref_t cached_gref, cached_line_tag_gref;
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_tensor_get_global_reference(cache_ptr_->cache_line_data_wm_tensor_, &cached_gref));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_tensor_get_global_reference(
    cache_ptr_->cache_line_tag_wm_tensor_, &cached_line_tag_gref));
  wholememory_ops::temp_memory_handle dev_miss_ids_handle(p_env_fns);
  void* dev_miss_ids_ptr =
    dev_miss_ids_handle.device_malloc(indice_desc->sizes[0], indice_desc->dtype);
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_ops::try_gather_cached_func(
    cached_gref,
    wholememory_tensor_get_tensor_description(cache_ptr_->cache_line_data_wm_tensor_),
    cached_line_tag_gref,
    wholememory_tensor_get_data_pointer(indices),
    indice_desc,
    nullptr,
    dev_miss_ids_ptr,
    wholememory_tensor_get_data_pointer(output),
    output_desc,
    cache_ptr_->get_cache_set_coverage(),
    0,
    stream));
  wholememory_tensor_t missed_indices_tensor;
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_make_tensor_from_pointer(&missed_indices_tensor, dev_miss_ids_ptr, indice_desc));
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_gather(allocated_embedding, missed_indices_tensor, output, p_env_fns, stream));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_destroy_tensor(missed_indices_tensor));

  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory

#ifdef __cplusplus
extern "C" {
#endif

wholememory_error_code_t wholememory_create_embedding_optimizer(
  wholememory_embedding_optimizer_t* optimizer, wholememory_optimizer_type_t optimizer_type)
{
  return wholememory::create_embedding_optimizer(optimizer, optimizer_type);
}

wholememory_error_code_t wholememory_optimizer_set_parameter(
  wholememory_embedding_optimizer_t optimizer, const char* parameter_name, void* value)
{
  return wholememory::optimizer_set_parameter(optimizer, parameter_name, value);
}

void wholememory_destroy_embedding_optimizer(wholememory_embedding_optimizer_t optimizer)
{
  wholememory::destroy_embedding_optimizer(optimizer);
}

wholememory_error_code_t wholememory_create_embedding_cache_policy(
  wholememory_embedding_cache_policy_t* cache_policy,
  wholememory_comm_t cache_level_comm,
  wholememory_memory_type_t memory_type,
  wholememory_memory_location_t memory_location,
  wholememory_access_type_t access_type,
  float cache_ratio)
{
  if (cache_ratio > 1.0F || cache_ratio < 1.0F / 512) {
    WHOLEMEMORY_ERROR("cache_ratio should in range [1/512, 1.0]");
    return WHOLEMEMORY_INVALID_VALUE;
  }
  auto* embedding_cache_policy                  = new wholememory_embedding_cache_policy_;
  embedding_cache_policy->cache_comm            = cache_level_comm;
  embedding_cache_policy->cache_memory_type     = memory_type;
  embedding_cache_policy->cache_memory_location = memory_location;
  embedding_cache_policy->access_type           = access_type;
  embedding_cache_policy->cache_ratio           = cache_ratio;
  *cache_policy                                 = embedding_cache_policy;
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_destroy_embedding_cache_policy(
  wholememory_embedding_cache_policy_t cache_policy)
{
  delete cache_policy;
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_create_embedding(
  wholememory_embedding_t* wholememory_embedding,
  wholememory_matrix_description_t* embedding_description,
  wholememory_comm_t comm,
  wholememory_memory_type_t memory_type,
  wholememory_memory_location_t memory_location,
  wholememory_embedding_optimizer_t optimizer,
  wholememory_embedding_cache_policy_t cache_policy)
{
  wholememory::embedding_base* embedding_impl_ptr = nullptr;
  int embedding_world_size                        = 1;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&embedding_world_size, comm));
  if (cache_policy != nullptr) {
    if (cache_policy->cache_comm == comm) {
      if (cache_policy->cache_memory_location != WHOLEMEMORY_ML_DEVICE) {
        WHOLEMEMORY_ERROR(
          "Cache has same communicator with raw embedding, should be device cached host embedding,"
          " but cache memory location is not WHOLEMEMORY_ML_DEVICE.");
        return WHOLEMEMORY_INVALID_INPUT;
      }
      if (cache_policy->cache_memory_type < memory_type) {
        WHOLEMEMORY_ERROR(
          "For device cached host memory, raw embedding should cover cache's address modes.");
        return WHOLEMEMORY_INVALID_INPUT;
      }
      embedding_impl_ptr = new wholememory::device_cached_host_embedding();
    } else {
      int const cache_world_size = 1;
      WHOLEMEMORY_RETURN_ON_FAIL(
        wholememory_communicator_get_size(&embedding_world_size, cache_policy->cache_comm));
      WHOLEMEMORY_CHECK_NOTHROW(cache_world_size <= embedding_world_size);
      if (cache_policy->cache_memory_type == WHOLEMEMORY_MT_DISTRIBUTED) {
        WHOLEMEMORY_ERROR(
          "For local cached global readonly embedding, cache_memory_type should be chunked or "
          "continuous.");
        return WHOLEMEMORY_INVALID_INPUT;
      }
      if (cache_policy->access_type != WHOLEMEMORY_AT_READONLY) {
        WHOLEMEMORY_ERROR(
          "Only ReadOnly access type supported for local cached global readonly embedding.");
        return WHOLEMEMORY_INVALID_INPUT;
      }
      if (optimizer != nullptr) {
        WHOLEMEMORY_ERROR("optimizer not supported for local cached global readonly embedding.");
        return WHOLEMEMORY_INVALID_INPUT;
      }
      embedding_impl_ptr = new wholememory::local_cached_global_readonly_embedding();
    }
  } else {
    embedding_impl_ptr = new wholememory::noncached_embedding();
  }

  WHOLEMEMORY_RETURN_ON_FAIL(embedding_impl_ptr->allocate(
    embedding_description, comm, memory_type, memory_location, cache_policy, optimizer));

  *wholememory_embedding = static_cast<wholememory_embedding_t>(embedding_impl_ptr);
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_destroy_embedding(
  wholememory_embedding_t wholememory_embedding)
{
  if (wholememory_embedding == nullptr) { return WHOLEMEMORY_INVALID_INPUT; }
  auto* embedding_impl_ptr = static_cast<wholememory::embedding_base*>(wholememory_embedding);
  delete embedding_impl_ptr;
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_embedding_gather(wholememory_embedding_t wholememory_embedding,
                                                      wholememory_tensor_t indices,
                                                      wholememory_tensor_t output,
                                                      bool adjust_cache,
                                                      wholememory_env_func_t* p_env_fns,
                                                      cudaStream_t stream)
{
  auto* embedding_impl_ptr = static_cast<wholememory::embedding_base*>(wholememory_embedding);
  return embedding_impl_ptr->gather(indices, output, adjust_cache, p_env_fns, stream);
}

wholememory_tensor_t wholememory_embedding_get_embedding_tensor(
  wholememory_embedding_t wholememory_embedding)
{
  wholememory::embedding_base* embedding_impl_ptr =
    static_cast<wholememory::embedding_base*>(wholememory_embedding);
  return embedding_impl_ptr->user_embedding;
}

wholememory_error_code_t wholememory_embedding_init_optimizer_states(
  wholememory_embedding_t wholememory_embedding)
{
  // TODO: implemented it.
  return WHOLEMEMORY_NOT_IMPLEMENTED;
}

const char** wholememory_embedding_get_optimizer_state_names(
  wholememory_embedding_t wholememory_embedding)
{
  // TODO: implemented it.
  return nullptr;
}

wholememory_tensor_t wholememory_embedding_get_optimizer_state(
  wholememory_embedding_t wholememory_embedding, const char* name)
{
  // TODO: implemented it.
  return nullptr;
}

wholememory_error_code_t wholememory_embedding_flush_cache(
  wholememory_embedding_t wholememory_embedding)
{
  // TODO: implemented it.
  return WHOLEMEMORY_NOT_IMPLEMENTED;
}

#ifdef __cplusplus
}
#endif
