#include <wholememory/env_func_ptrs.hpp>

#include <mutex>

#include "cuda_macros.hpp"
#include "error.hpp"
#include "initialize.hpp"

namespace wholememory {

void empty_create_memory_context_func(memory_context_t* memory_context, void* /*global_context*/)
{
  wholememory_initialize_tensor_desc(&memory_context->desc);
  memory_context->context = nullptr;
}

void empty_destroy_memory_context_func(memory_context_t* /*memory_context*/,
                                       void* /*global_context*/)
{
}

void* default_host_malloc_func(wholememory_tensor_description_t* tensor_description,
                               memory_context_t* memory_context,
                               void* /*global_context*/)
{
  void* ptr               = malloc(wholememory_get_memory_size_from_tensor(tensor_description));
  memory_context->desc    = *tensor_description;
  memory_context->context = ptr;
  return ptr;
}

void default_host_free_func(memory_context_t* memory_context, void* /*global_context*/)
{
  free(memory_context->context);
  wholememory_initialize_tensor_desc(&memory_context->desc);
  memory_context->context = nullptr;
}

void* default_device_malloc_func(wholememory_tensor_description_t* tensor_description,
                                 memory_context_t* memory_context,
                                 void* /*global_context*/)
{
  void* ptr = nullptr;
  try {
    WM_CUDA_CHECK(cudaMalloc(&ptr, wholememory_get_memory_size_from_tensor(tensor_description)));
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("cudaMalloc failed, %s.\n", wce.what());
  }
  memory_context->desc    = *tensor_description;
  memory_context->context = ptr;
  return ptr;
}

void default_device_free_func(memory_context_t* memory_context, void* /*global_context*/)
{
  try {
    WM_CUDA_CHECK(cudaFree(memory_context->context));
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("cudaFree failed, %s.\n", wce.what());
  }
  wholememory_initialize_tensor_desc(&memory_context->desc);
  memory_context->context = nullptr;
}

void* default_pinned_malloc_func(wholememory_tensor_description_t* tensor_description,
                                 memory_context_t* memory_context,
                                 void* /*global_context*/)
{
  void* ptr = nullptr;
  try {
    WM_CUDA_CHECK(
      cudaMallocHost(&ptr, wholememory_get_memory_size_from_tensor(tensor_description)));
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("cudaMallocHost failed, %s.\n", wce.what());
  }
  memory_context->desc    = *tensor_description;
  memory_context->context = ptr;
  return ptr;
}

void default_pinned_free_func(memory_context_t* memory_context, void* /*global_context*/)
{
  try {
    WM_CUDA_CHECK(cudaFreeHost(memory_context->context));
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("cudaFreeHost failed, %s.\n", wce.what());
  }
  wholememory_initialize_tensor_desc(&memory_context->desc);
  memory_context->context = nullptr;
}

static wholememory_env_func_t default_env_func = {
  .temporary_fns =
    {
      .create_memory_context_fn  = empty_create_memory_context_func,
      .destroy_memory_context_fn = empty_destroy_memory_context_func,
      .host_malloc_fn            = default_host_malloc_func,
      .host_free_fn              = default_host_free_func,
      .device_malloc_fn          = default_device_malloc_func,
      .device_free_fn            = default_device_free_func,
      .pinned_malloc_fn          = default_pinned_malloc_func,
      .pinned_free_fn            = default_pinned_free_func,
      .global_context            = nullptr,
    },
  .output_fns = {
    .host_malloc_fn   = default_host_malloc_func,
    .host_free_fn     = default_host_free_func,
    .device_malloc_fn = default_device_malloc_func,
    .device_free_fn   = default_device_free_func,
    .pinned_malloc_fn = default_pinned_malloc_func,
    .pinned_free_fn   = default_pinned_free_func,
    .global_context   = nullptr,
  }};

wholememory_env_func_t* get_default_env_func() { return &default_env_func; }

class CachedAllocator {
 public:
  CachedAllocator() { WHOLEMEMORY_FAIL_NOTHROW("Not implemented."); };
  ~CachedAllocator() { DropCaches(); }
  void* MallocHost() { return nullptr; }
  void* MallocDevice() { return nullptr; }
  void* MallocPinned() { return nullptr; }
  void FreeHost() {}
  void FreeDevice() {}
  void FreePinned() {}
  void DropCaches() {}

 private:
  std::mutex mu_;
};

#define K_MAX_DEVICE_COUNT (16)

static CachedAllocator* p_cached_allocators[K_MAX_DEVICE_COUNT] = {nullptr};

wholememory_env_func_t* get_cached_env_func() { WHOLEMEMORY_FAIL_NOTHROW("Not implemented."); }

void drop_env_func_cache() { WHOLEMEMORY_FAIL_NOTHROW("Not implemented."); }

}  // namespace wholememory

#ifdef __cplusplus
extern "C" {
#endif

cudaDeviceProp* get_device_prop(int dev_id) { return wholememory::get_device_prop(dev_id); }

#ifdef __cplusplus
}
#endif