#include "torch_env_func_ptrs.h"

#include <torch/script.h>

#include "torch_utils.h"

namespace wholegraph_torch {

void *torch_host_malloc_func(wholememory_tensor_description_t *tensor_description,
                             memory_context_t *memory_context,
                             void * /*global_context*/)
{
  return torch_common_malloc_func(tensor_description, memory_context, false, false);
}

void *torch_device_malloc_func(wholememory_tensor_description_t *tensor_description,
                               memory_context_t *memory_context,
                               void * /*global_context*/)
{
  return torch_common_malloc_func(tensor_description, memory_context, true);
}

void *torch_pinned_malloc_func(wholememory_tensor_description_t *tensor_description,
                               memory_context_t *memory_context,
                               void * /*global_context*/)
{
  return torch_common_malloc_func(tensor_description, memory_context, false, true);
}

static wholememory_env_func_t pytorch_env_func = {
    .temporary_fns =
        {
            .create_memory_context_fn  = create_torch_memory_context_func,
            .destroy_memory_context_fn = destroy_torch_memory_context_func,
            .host_malloc_fn            = torch_host_malloc_func,
            .host_free_fn              = torch_common_free_func,
            .device_malloc_fn          = torch_device_malloc_func,
            .device_free_fn            = torch_common_free_func,
            .pinned_malloc_fn          = torch_pinned_malloc_func,
            .pinned_free_fn            = torch_common_free_func,
            .global_context            = nullptr,
        },
    .output_fns = {
        .host_malloc_fn            = torch_host_malloc_func,
        .host_free_fn              = torch_common_free_func,
        .device_malloc_fn          = torch_device_malloc_func,
        .device_free_fn            = torch_common_free_func,
        .pinned_malloc_fn          = torch_pinned_malloc_func,
        .pinned_free_fn            = torch_common_free_func,
        .global_context            = nullptr,
    }};

wholememory_env_func_t* get_pytorch_env_func() {
  return &pytorch_env_func;
}

}  // namespace wholegraph_torch