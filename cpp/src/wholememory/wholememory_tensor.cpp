#include "wholememory/wholememory_tensor.h"

#include <cstdlib>

#include "logger.hpp"

#ifdef __cplusplus
extern "C" {
#endif

struct wholememory_tensor_ {
  union {
    wholememory_handle_t wholememory_handle;
    void* data_ptr;
  };
  wholememory_tensor_description_t tensor_description;
  bool is_wholememory;
  bool own_handle;
};

wholememory_error_code_t wholememory_create_tensor(
  wholememory_tensor_t* p_wholememory_tensor,
  wholememory_tensor_description_t* tensor_description,
  wholememory_comm_t comm,
  wholememory_memory_type_t memory_type,
  wholememory_memory_location_t memory_location)
{
  if (p_wholememory_tensor == nullptr) {
    WHOLEMEMORY_ERROR("p_wholememory_tensor is nullptr");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description == nullptr) {
    WHOLEMEMORY_ERROR("tensor_description is nullptr");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description->dim <= 0 || tensor_description->dim > 2) {
    WHOLEMEMORY_ERROR("tensor_description->dim=%d", tensor_description->dim);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description->storage_offset != 0) {
    WHOLEMEMORY_ERROR("tensor_description->storage_offset=%ld", tensor_description->storage_offset);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  int const dim = tensor_description->dim;
  if (tensor_description->strides[dim - 1] != 1) {
    WHOLEMEMORY_ERROR("tensor_description->strides[dim - 1]", tensor_description->strides[dim - 1]);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description->dim == 2 &&
      tensor_description->strides[0] != tensor_description->sizes[1]) {
    WHOLEMEMORY_ERROR("tensor_description->strides[0]=%ld, but tensor_description->sizes[1]=%ld",
                      tensor_description->strides[0],
                      tensor_description->sizes[1]);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description->dtype <= WHOLEMEMORY_DT_UNKNOWN ||
      tensor_description->dtype >= WHOLEMEMORY_DT_COUNT) {
    WHOLEMEMORY_ERROR("tensor_description is unknown");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  size_t elt_count   = wholememory_get_memory_element_count_from_tensor(tensor_description);
  size_t elt_size    = wholememory_dtype_get_element_size(tensor_description->dtype);
  size_t malloc_size = elt_count * elt_size;
  size_t granularity = elt_size * tensor_description->strides[0];

  auto* wholememory_tensor = static_cast<wholememory_tensor_*>(malloc(sizeof(wholememory_tensor_)));

  wholememory_tensor->tensor_description = *tensor_description;
  wholememory_tensor->own_handle         = true;
  wholememory_tensor->is_wholememory     = true;
  *p_wholememory_tensor                  = wholememory_tensor;
  auto ret_code = wholememory_malloc(&wholememory_tensor->wholememory_handle,
                                     malloc_size,
                                     comm,
                                     memory_type,
                                     memory_location,
                                     granularity);
  if (ret_code != WHOLEMEMORY_SUCCESS) { free(wholememory_tensor); }
  return ret_code;
}

wholememory_error_code_t wholememory_destroy_tensor(wholememory_tensor_t wholememory_tensor)
{
  if (wholememory_tensor->own_handle) {
    if (wholememory_tensor->is_wholememory) {
      WHOLEMEMORY_RETURN_ON_FAIL(wholememory_free(wholememory_tensor->wholememory_handle));
    } else {
      free(wholememory_tensor->data_ptr);
    }
  }
  free(wholememory_tensor);
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_make_tensor_from_pointer(
  wholememory_tensor_t* p_wholememory_tensor,
  void* data_ptr,
  wholememory_tensor_description_t* tensor_description)
{
  if (data_ptr == nullptr || p_wholememory_tensor == nullptr || tensor_description == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description->dim <= 0 || tensor_description->dim > 2) {
    WHOLEMEMORY_ERROR("tensor_description->dim=%d", tensor_description->dim);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  int const dim = tensor_description->dim;
  if (tensor_description->strides[dim - 1] != 1) {
    WHOLEMEMORY_ERROR("tensor_description->strides[dim - 1]", tensor_description->strides[dim - 1]);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description->dtype <= WHOLEMEMORY_DT_UNKNOWN ||
      tensor_description->dtype >= WHOLEMEMORY_DT_COUNT) {
    WHOLEMEMORY_ERROR("tensor_description is unknown");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto* wholememory_tensor = static_cast<wholememory_tensor_*>(malloc(sizeof(wholememory_tensor_)));
  wholememory_tensor->data_ptr           = data_ptr;
  wholememory_tensor->tensor_description = *tensor_description;
  wholememory_tensor->own_handle         = false;
  wholememory_tensor->is_wholememory     = false;
  *p_wholememory_tensor                  = wholememory_tensor;
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_make_tensor_from_handle(
  wholememory_tensor_t* p_wholememory_tensor,
  wholememory_handle_t wholememory_handle,
  wholememory_tensor_description_t* tensor_description)
{
  if (wholememory_handle == nullptr || p_wholememory_tensor == nullptr ||
      tensor_description == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description->dim <= 0 || tensor_description->dim > 2) {
    WHOLEMEMORY_ERROR("tensor_description->dim=%d", tensor_description->dim);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  int const dim = tensor_description->dim;
  if (tensor_description->strides[dim - 1] != 1) {
    WHOLEMEMORY_ERROR("tensor_description->strides[dim - 1]", tensor_description->strides[dim - 1]);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description->dtype <= WHOLEMEMORY_DT_UNKNOWN ||
      tensor_description->dtype >= WHOLEMEMORY_DT_COUNT) {
    WHOLEMEMORY_ERROR("tensor_description is unknown");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto* wholememory_tensor = static_cast<wholememory_tensor_*>(malloc(sizeof(wholememory_tensor_)));
  wholememory_tensor->wholememory_handle = wholememory_handle;
  wholememory_tensor->tensor_description = *tensor_description;
  wholememory_tensor->own_handle         = false;
  wholememory_tensor->is_wholememory     = true;
  *p_wholememory_tensor                  = wholememory_tensor;
  return WHOLEMEMORY_SUCCESS;
}

bool wholememory_tensor_has_handle(wholememory_tensor_t wholememory_tensor)
{
  return wholememory_tensor->is_wholememory;
}

wholememory_handle_t wholememory_tensor_get_memory_handle(wholememory_tensor_t wholememory_tensor)
{
  if (wholememory_tensor->is_wholememory) { return wholememory_tensor->wholememory_handle; }
  return nullptr;
}

void wholememory_tensor_get_tensor_description(wholememory_tensor_description_t* tensor_description,
                                               wholememory_tensor_t wholememory_tensor)
{
  *tensor_description = wholememory_tensor->tensor_description;
}

wholememory_error_code_t wholememory_tensor_get_global_reference(
  wholememory_gref_t* wholememory_gref, wholememory_tensor_t wholememory_tensor)
{
  if (wholememory_gref == nullptr || wholememory_tensor == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (wholememory_tensor->is_wholememory) {
    return wholememory_get_global_reference(wholememory_gref,
                                            wholememory_tensor->wholememory_handle);
  }
  *wholememory_gref = wholememory_create_continuous_global_reference(wholememory_tensor->data_ptr);
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_tensor_get_subtensor(
  wholememory_tensor_t* p_sub_wholememory_tensor,
  wholememory_tensor_t wholememory_tensor,
  int64_t* starts,
  int64_t* ends)
{
  if (p_sub_wholememory_tensor == nullptr || wholememory_tensor == nullptr || starts == nullptr ||
      ends == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (wholememory_tensor->tensor_description.dim > 2) { return WHOLEMEMORY_NOT_IMPLEMENTED; }
  int const dim      = wholememory_tensor->tensor_description.dim;
  int64_t offsets[2] = {0, 0};
  if (dim == 1) {
    offsets[0] = wholememory_tensor->tensor_description.storage_offset;
  } else {
    offsets[0] = wholememory_tensor->tensor_description.storage_offset /
                 wholememory_tensor->tensor_description.strides[0];
    offsets[1] = wholememory_tensor->tensor_description.storage_offset %
                 wholememory_tensor->tensor_description.strides[0];
  }
  int64_t new_size[2] = {0, 0};
  int64_t new_offset  = wholememory_tensor->tensor_description.storage_offset;
  for (int i = 0; i < dim; i++) {
    int64_t starts_i = starts[i];
    int64_t ends_i   = ends[i];
    if (starts[i] == -1) starts_i = 0;
    if (ends[i] == -1) ends_i = wholememory_tensor->tensor_description.sizes[i];
    if (ends_i <= starts_i) return WHOLEMEMORY_INVALID_INPUT;
    if (starts_i >= wholememory_tensor->tensor_description.sizes[i])
      return WHOLEMEMORY_INVALID_INPUT;
    if (ends_i <= 0) return WHOLEMEMORY_INVALID_INPUT;
    new_offset += wholememory_tensor->tensor_description.strides[i] * starts_i;
    new_size[i] = ends_i - starts_i;
  }
  auto* sub_wholememory_tensor =
    static_cast<wholememory_tensor_*>(malloc(sizeof(wholememory_tensor_)));
  *sub_wholememory_tensor                                   = *wholememory_tensor;
  sub_wholememory_tensor->own_handle                        = false;
  sub_wholememory_tensor->tensor_description.storage_offset = new_offset;
  sub_wholememory_tensor->tensor_description.dim            = dim;
  sub_wholememory_tensor->tensor_description.dtype =
    sub_wholememory_tensor->tensor_description.dtype;
  for (int i = 0; i < dim; i++) {
    sub_wholememory_tensor->tensor_description.sizes[i] = new_size[i];
    sub_wholememory_tensor->tensor_description.strides[i] =
      wholememory_tensor->tensor_description.strides[i];
  }
  *p_sub_wholememory_tensor = sub_wholememory_tensor;

  return WHOLEMEMORY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
