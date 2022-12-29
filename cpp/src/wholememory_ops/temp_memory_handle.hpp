#pragma once

#include <wholememory/env_func_ptrs.h>
#include <wholememory/tensor_description.h>

namespace wholememory_ops {

class temp_memory_handle {
 public:
  explicit temp_memory_handle(wholememory_env_func_t* env_fns) {
    temp_mem_fns_ = &env_fns->temporary_fns;
    memory_context_ = temp_mem_fns_->create_memory_context_fn(temp_mem_fns_->global_context);
  }
  temp_memory_handle() = delete;
  ~temp_memory_handle() {
    free_memory();
    temp_mem_fns_->destroy_memory_context_fn(memory_context_, temp_mem_fns_->global_context);
    memory_context_ = nullptr;
  }
  void* device_malloc(size_t elt_count, wholememory_dtype_t data_type) {
    free_memory();
    wholememory_tensor_description_t tensor_description;
    get_tensor_description(&tensor_description, elt_count, data_type);
    ptr_ = temp_mem_fns_->device_malloc_fn(&tensor_description, memory_context_, temp_mem_fns_->global_context);
    free_fn_ = temp_mem_fns_->device_free_fn;
    return ptr_;
  }
  void* host_malloc(size_t elt_count, wholememory_dtype_t data_type) {
    free_memory();
    wholememory_tensor_description_t tensor_description;
    get_tensor_description(&tensor_description, elt_count, data_type);
    ptr_ = temp_mem_fns_->host_malloc_fn(&tensor_description, memory_context_, temp_mem_fns_->global_context);
    free_fn_ = temp_mem_fns_->host_free_fn;
    return ptr_;
  }
  void* pinned_malloc(size_t elt_count, wholememory_dtype_t data_type) {
    free_memory();
    wholememory_tensor_description_t tensor_description;
    get_tensor_description(&tensor_description, elt_count, data_type);
    ptr_ = temp_mem_fns_->pinned_malloc_fn(&tensor_description, memory_context_, temp_mem_fns_->global_context);
    free_fn_ = temp_mem_fns_->pinned_free_fn;
    return ptr_;
  }
  void* pointer() const {
    return ptr_;
  }
  void free_memory() {
    if (ptr_ != nullptr) {
      free_fn_(ptr_, memory_context_, temp_mem_fns_->global_context);
      temp_mem_fns_->destroy_memory_context_fn(memory_context_, temp_mem_fns_->global_context);
      free_fn_ = nullptr;
      ptr_ = nullptr;
    }
  }
 private:
  static void get_tensor_description(wholememory_tensor_description_t *tensor_description,
                                     size_t elt_count,
                                     wholememory_dtype_t data_type) {
    wholememory_initialize_tensor_desc(tensor_description);
    tensor_description->dim = 1;
    tensor_description->storage_offset = 0;
    tensor_description->dtype = data_type;
    tensor_description->sizes[0] = elt_count;
    tensor_description->strides[0] = 1;
  }

  wholememory_memory_func_t* temp_mem_fns_ = nullptr;
  void* memory_context_ = nullptr;

  wholememory_free_func_t free_fn_ = nullptr;
  void* ptr_ = nullptr;
};

}  // namespace wholememory_ops