#include <wholememory/tensor_description.h>

size_t wholememory_dtype_get_element_size(wholememory_dtype_t dtype) {
  switch(dtype) {
    case WHOLEMEMORY_DT_INT8:   return 1;
    case WHOLEMEMORY_DT_INT16:
    case WHOLEMEMORY_DT_BF16:
    case WHOLEMEMORY_DT_HALF:   return 2;
    case WHOLEMEMORY_DT_INT:
    case WHOLEMEMORY_DT_FLOAT:  return 4;
    case WHOLEMEMORY_DT_INT64:
    case WHOLEMEMORY_DT_DOUBLE: return 8;
    default:  return -1;
  }
}

wholememory_array_description_t wholememory_create_array_desc(int64_t size,
                                                              int64_t storage_offset,
                                                              wholememory_dtype_t dtype) {
  wholememory_array_description_t wm_array_desc;
  wm_array_desc.size = size;
  wm_array_desc.storage_offset = storage_offset;
  wm_array_desc.dtype = dtype;
  return wm_array_desc;
}

wholememory_matrix_description_t wholememory_create_matrix_desc(int64_t sizes[2],
                                                                int64_t stride,
                                                                int64_t storage_offset,
                                                                wholememory_dtype_t dtype) {
  wholememory_matrix_description_t wm_matrix_desc;
  wm_matrix_desc.sizes[0] = sizes[0];
  wm_matrix_desc.sizes[1] = sizes[1];
  wm_matrix_desc.stride = stride;
  wm_matrix_desc.storage_offset = storage_offset;
  wm_matrix_desc.dtype = dtype;
  return wm_matrix_desc;
}

void wholememory_initialize_tensor_desc(wholememory_tensor_description_t* p_tensor_description) {
  p_tensor_description->dim = 0;
  for (int i = 0; i < WHOLEMEMORY_MAX_TENSOR_DIM; i++) {
    p_tensor_description->sizes[i] = 1;
    p_tensor_description->strides[i] = 1;
  }
  p_tensor_description->storage_offset = 0;
  p_tensor_description->dtype = WHOLEMEMORY_DT_UNKNOWN;
}

void wholememory_copy_array_desc_to_tensor(wholememory_tensor_description_t* p_tensor_description,
                                           wholememory_array_description_t* p_array_description) {
  wholememory_initialize_tensor_desc(p_tensor_description);
  p_tensor_description->dim = 1;
  p_tensor_description->sizes[0] = p_array_description->size;
  p_tensor_description->dtype = p_array_description->dtype;
  p_tensor_description->storage_offset = p_array_description->storage_offset;
}

void wholememory_copy_matrix_desc_to_tensor(wholememory_tensor_description_t* p_tensor_description,
                                            wholememory_matrix_description_t* p_matrix_description) {
  wholememory_initialize_tensor_desc(p_tensor_description);
  p_tensor_description->dim = 2;
  p_tensor_description->sizes[0] = p_matrix_description->sizes[0];
  p_tensor_description->sizes[1] = p_matrix_description->sizes[1];
  p_tensor_description->strides[1] = p_matrix_description->stride;
  p_tensor_description->dtype = p_matrix_description->dtype;
  p_tensor_description->storage_offset = p_matrix_description->storage_offset;
}

bool wholememory_convert_tensor_desc_to_array(wholememory_array_description_t* p_array_description,
                                              wholememory_tensor_description_t* p_tensor_description) {
  if (p_tensor_description->dtype <= WHOLEMEMORY_DT_UNKNOWN
      || p_tensor_description->dtype >= WHOLEMEMORY_DT_COUNT)
    return false;
  if (p_tensor_description->dim != 1) return false;
  if (p_tensor_description->strides[0] != 1) return false;
  p_array_description->dtype = p_tensor_description->dtype;
  p_array_description->storage_offset = p_tensor_description->storage_offset;
  p_array_description->size = p_tensor_description->sizes[0];
  return true;
}

bool wholememory_convert_tensor_desc_to_matrix(wholememory_matrix_description_t* p_matrix_description,
                                               wholememory_tensor_description_t* p_tensor_description) {
  if (p_tensor_description->dtype <= WHOLEMEMORY_DT_UNKNOWN
      || p_tensor_description->dtype >= WHOLEMEMORY_DT_COUNT)
    return false;
  if (p_tensor_description->dim != 2) return false;
  if (p_tensor_description->strides[0] != 1) return false;
  p_matrix_description->dtype = p_tensor_description->dtype;
  p_matrix_description->storage_offset = p_tensor_description->storage_offset;
  p_matrix_description->sizes[0] = p_tensor_description->sizes[0];
  p_matrix_description->sizes[1] = p_tensor_description->sizes[1];
  p_matrix_description->stride = p_tensor_description->strides[1];
  return true;
}

int64_t wholememory_get_memory_element_count_from_array(wholememory_array_description_t* p_array_description) {
  return p_array_description->size;
}

int64_t wholememory_get_memory_size_from_array(wholememory_array_description_t* p_array_description) {
  return wholememory_get_memory_element_count_from_array(p_array_description)
      * wholememory_dtype_get_element_size(p_array_description->dtype);
}

int64_t wholememory_get_memory_element_count_from_matrix(wholememory_matrix_description_t* p_matrix_description) {
  return p_matrix_description->sizes[1] * p_matrix_description->stride;
}

int64_t wholememory_get_memory_size_from_matrix(wholememory_matrix_description_t* p_matrix_description) {
  return wholememory_get_memory_element_count_from_matrix(p_matrix_description)
      * wholememory_dtype_get_element_size(p_matrix_description->dtype);
}

int64_t wholememory_get_memory_element_count_from_tensor(wholememory_tensor_description_t* p_tensor_description) {
  if (p_tensor_description->dim == 0) return 1;
  if (p_tensor_description->dim < 0 || p_tensor_description->dim >= WHOLEMEMORY_MAX_TENSOR_DIM) return -1;
  int dim = p_tensor_description->dim;
  return p_tensor_description->strides[dim - 1] * p_tensor_description->sizes[dim - 1];
}

int64_t wholememory_get_memory_size_from_tensor(wholememory_tensor_description_t* p_tensor_description) {
  return wholememory_get_memory_element_count_from_tensor(p_tensor_description)
      * wholememory_dtype_get_element_size(p_tensor_description->dtype);
}