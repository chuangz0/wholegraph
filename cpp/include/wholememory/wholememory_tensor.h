#pragma once

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wholememory_tensor_* wholememory_tensor_t;

/**
 * Create WholeMemory Tensor
 * @param wholememory_tensor : returned WholeMemory Tensor handle
 * @param tensor_description : description of the WholeMemory Tensor, should be 1-D or 2-D
 * continuous tensor without offset.
 * @param comm : WholeMemory Communicator
 * @param memory_type : Memory Type of the underlying WholeMemory
 * @param memory_location : Memory Location of the underlying WholeMemory
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_create_tensor(
  wholememory_tensor_t* wholememory_tensor,
  wholememory_tensor_description_t* tensor_description,
  wholememory_comm_t comm,
  wholememory_memory_type_t memory_type,
  wholememory_memory_location_t memory_location);

/**
 * Destroy WholeMemory Tensor
 * @param wholememory_tensor : WholeMemory Tensor to destroy
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_destroy_tensor(wholememory_tensor_t wholememory_tensor);

/**
 * Get WholeMemory handle from WholeMemory Tensor
 * @param wholememory_tensor : WholeMemory Tensor
 * @return : WholeMemory handle
 */
wholememory_handle_t wholememory_tensor_get_memory_handle(wholememory_tensor_t wholememory_tensor);

/**
 * Get tensor description from WholeMemory Tensor
 * @param tensor_description : pointer to returned tensor_description
 * @param wholememory_tensor : WholeMemory Tensor
 */
void wholememory_tensor_get_tensor_description(wholememory_tensor_description_t* tensor_description,
                                               wholememory_tensor_t wholememory_tensor);

/**
 * Get sub tensor of a WholeMemory Tensor
 * @param sub_wholememory_tensor : pointer to returned sub tensor
 * @param wholememory_tensor : WholeMemory Tensor
 * @param starts : starts of each dim, length should be the dim of wholememory_tensor.
 * @param ends : ends of each dim, length should be the dim of wholememory_tensor
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_tensor_get_subtensor(
  wholememory_tensor_t* sub_wholememory_tensor,
  wholememory_tensor_t wholememory_tensor,
  int64_t* starts,
  int64_t* ends);

#ifdef __cplusplus
}
#endif
