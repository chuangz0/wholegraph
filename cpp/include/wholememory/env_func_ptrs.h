#pragma once

#include <wholememory/tensor_description.h>

/**
 * Function pointers for memory allocation.
 * Input tensor memory should be allocated and use void* pointer to the memory and
 * wholememory_array_description_t or wholememory_matrix_description_t to specify the shape Output
 * tensor with fixed size should be the same as Input tensor. Output tensor with shape determined by
 * Op should has void* memory_context input and allocated by wholememory_malloc_func_t functions.
 */

/**
 * Function pointer to create temporary memory context.
 */
typedef void* (*wholememory_create_memory_context_func_t)(void* global_context);

typedef void (*wholememory_destroy_memory_context_func_t)(void* memory_context,
                                                          void* global_context);

typedef void* (*wholememory_malloc_func_t)(wholememory_tensor_description_t*,
                                           void* memory_context,
                                           void* global_context);

typedef void (*wholememory_free_func_t)(void* ptr, void* memory_context, void* global_context);

typedef struct wholememory_memory_func_ {
  wholememory_create_memory_context_func_t create_memory_context_fn;
  wholememory_destroy_memory_context_func_t destroy_memory_context_fn;
  wholememory_malloc_func_t host_malloc_fn;
  wholememory_free_func_t host_free_fn;
  wholememory_malloc_func_t device_malloc_fn;
  wholememory_free_func_t device_free_fn;
  wholememory_malloc_func_t pinned_malloc_fn;
  wholememory_free_func_t pinned_free_fn;
  void* global_context;
} wholememory_memory_func_t;

typedef struct wholememory_env_func_ {
  wholememory_memory_func_t temporary_fns; /* function pointers to create temporary memory */
  wholememory_memory_func_t output_fns;    /* function pointers to create Op output memory */
} wholememory_env_func_t;