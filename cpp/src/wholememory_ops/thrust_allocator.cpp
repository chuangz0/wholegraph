#include "thrust_allocator.hpp"

#include <raft/util/integer_utils.hpp>

#include "error.hpp"

namespace wholememory_ops {

wm_thrust_allocator::~wm_thrust_allocator() { deallocate_all(); }

wm_thrust_allocator::value_type* wm_thrust_allocator::allocate(std::ptrdiff_t mem_size)
{
  static const std::ptrdiff_t kThrustAlignSize = 256;
  mem_size = std::max<std::ptrdiff_t>(kThrustAlignSize, mem_size);
  mem_size = raft::div_rounding_up_unsafe(mem_size, kThrustAlignSize) * kThrustAlignSize;
  auto* memory_context =
    fns->temporary_fns.create_memory_context_fn(fns->temporary_fns.global_context);
  wholememory_tensor_description_t tensor_description;
  wholememory_initialize_tensor_desc(&tensor_description);
  tensor_description.dim      = 1;
  tensor_description.dtype    = WHOLEMEMORY_DT_INT64;
  tensor_description.sizes[0] = mem_size / sizeof(int64_t);
  auto* ptr                   = static_cast<value_type*>(fns->temporary_fns.device_malloc_fn(
    &tensor_description, memory_context, fns->temporary_fns.global_context));
  mem_ptr_to_context_map.emplace(ptr, memory_context);
  return ptr;
}

void wm_thrust_allocator::deallocate(value_type* p, size_t /*mem_size*/)
{
  auto it = mem_ptr_to_context_map.find(p);
  WHOLEMEMORY_CHECK_NOTHROW(it != mem_ptr_to_context_map.end());
  fns->temporary_fns.device_free_fn(it->first, it->second, fns->temporary_fns.global_context);
  fns->temporary_fns.destroy_memory_context_fn(it->second, fns->temporary_fns.global_context);
  mem_ptr_to_context_map.erase(p);
}

void wm_thrust_allocator::deallocate_all()
{
  while (!mem_ptr_to_context_map.empty()) {
    auto it = mem_ptr_to_context_map.begin();
    fns->temporary_fns.device_free_fn(it->first, it->second, fns->temporary_fns.global_context);
    fns->temporary_fns.destroy_memory_context_fn(it->second, fns->temporary_fns.global_context);
    mem_ptr_to_context_map.erase(it->first);
  }
}

}  // namespace wholememory_ops