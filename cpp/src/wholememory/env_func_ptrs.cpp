#include <wholememory/env_func_ptrs.hpp>

#include <raft/util/cudart_utils.hpp>

#include <mutex>

#include "error.hpp"

namespace wholememory {

void* empty_create_memory_context_func(void* /*global_context*/) {
  return nullptr;
}

void empty_destroy_memory_context_func(void* /*memory_context*/, void* /*global_context*/) {
}

void* default_host_malloc_func(wholememory_tensor_description_t* tensor_description,
                               void* /*memory_context*/,
                               void* /*global_context*/) {

  void* ptr = malloc(wholememory_get_memory_size_from_tensor(tensor_description));
  return ptr;
}

void default_host_free_func(void* ptr, void* /*memory_context*/, void* /*global_context*/) {
  free(ptr);
}

void *default_device_malloc_func(wholememory_tensor_description_t *tensor_description,
                                 void * /*memory_context*/,
                                 void * /*global_context*/) {

  void* ptr = nullptr;
  try {
    CUDA_CHECK(cudaMalloc(&ptr, wholememory_get_memory_size_from_tensor(tensor_description)));
  } catch (raft::cuda_error& rce) {
    WHOLEMEMORY_FAIL_NOTHROW("cudaMalloc failed, %s.\n", rce.what());
  }
  return ptr;
}

void default_device_free_func(void* ptr, void* /*memory_context*/, void* /*global_context*/) {
  try {
    CUDA_CHECK(cudaFree(ptr));
  } catch (raft::cuda_error& rce) {
    WHOLEMEMORY_FAIL_NOTHROW("cudaFree failed, %s.\n", rce.what());
  }
}

void *default_pinned_malloc_func(wholememory_tensor_description_t *tensor_description,
                                 void * /*memory_context*/,
                                 void * /*global_context*/) {

  void* ptr = nullptr;
  try {
    CUDA_CHECK(cudaMallocHost(&ptr, wholememory_get_memory_size_from_tensor(tensor_description)));
  } catch (raft::cuda_error& rce) {
    WHOLEMEMORY_FAIL_NOTHROW("cudaMallocHost failed, %s.\n", rce.what());
  }
  return ptr;
}

void default_pinned_free_func(void* ptr, void* /*memory_context*/, void* /*global_context*/) {
  try {
    CUDA_CHECK(cudaFreeHost(ptr));
  } catch (raft::cuda_error& rce) {
    WHOLEMEMORY_FAIL_NOTHROW("cudaFreeHost failed, %s.\n", rce.what());
  }
}

static wholememory_env_func_t default_env_func = {
    .temporary_fns = {
        .create_memory_context_fn = empty_create_memory_context_func,
        .destroy_memory_context_fn = empty_destroy_memory_context_func,
        .host_malloc_fn = default_host_malloc_func,
        .host_free_fn = default_host_free_func,
        .device_malloc_fn = default_device_malloc_func,
        .device_free_fn = default_device_free_func,
        .pinned_malloc_fn = default_pinned_malloc_func,
        .pinned_free_fn = default_pinned_free_func,
        .global_context = nullptr,
    },
    .output_fns = {
        .create_memory_context_fn = empty_create_memory_context_func,
        .destroy_memory_context_fn = empty_destroy_memory_context_func,
        .host_malloc_fn = default_host_malloc_func,
        .host_free_fn = default_host_free_func,
        .device_malloc_fn = default_device_malloc_func,
        .device_free_fn = default_device_free_func,
        .pinned_malloc_fn = default_pinned_malloc_func,
        .pinned_free_fn = default_pinned_free_func,
        .global_context = nullptr,
    }
};

wholememory_env_func_t* get_default_env_func() {
  return &default_env_func;
}

class CachedAllocator {
 public:
  CachedAllocator() {
    WHOLEMEMORY_FAIL_NOTHROW("Not implemented.");
  };
  ~CachedAllocator() {
    DropCaches();
  }
  void* MallocHost() {
    return nullptr;
  }
  void* MallocDevice() {
    return nullptr;
  }
  void* MallocPinned() {
    return nullptr;
  }
  void FreeHost() {

  }
  void FreeDevice() {

  }
  void FreePinned() {

  }
  void DropCaches() {

  }
 private:
  std::mutex mu_;
};

#define K_MAX_DEVICE_COUNT (16)

static CachedAllocator* p_cached_allocators[K_MAX_DEVICE_COUNT] = {nullptr};

wholememory_env_func_t* get_cached_env_func() {
  WHOLEMEMORY_FAIL_NOTHROW("Not implemented.");
}

void drop_env_func_cache() {
  WHOLEMEMORY_FAIL_NOTHROW("Not implemented.");
}

}  // namespace wholememory