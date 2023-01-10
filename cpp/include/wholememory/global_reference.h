#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Global reference of a WholeMemory object
 */
typedef struct wholememory_global_reference {
  void* pointer; /*!< pointer to data for CONTINUOUS WholeMemory or pointer to data pointer array
                    for CHUNKED WholeMemory */
  size_t
    stride; /*!< must be 0 for CONTINUOUS WholeMemory or memory size in byte for each pointer */
} wholememory_gref_t;

/**
 * @brief Create global reference for continuous memory
 * @param ptr : pointer to the memory
 * @return : wholememory_gref_t
 */
wholememory_gref_t wholememory_create_continuous_global_reference(void* ptr);

#ifdef __cplusplus
}
#endif