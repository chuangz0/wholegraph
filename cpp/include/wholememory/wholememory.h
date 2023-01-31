#pragma once

#include <unistd.h>

#include <wholememory/global_reference.h>

#ifdef __cplusplus
extern "C" {
#endif

enum wholememory_error_code_t {
  WHOLEMEMORY_SUCCESS = 0,         /* success */
  WHOLEMEMORY_UNKNOW_ERROR,        /* unknown error */
  WHOLEMEMORY_NOT_IMPLEMENTED,     /* method is not implemented */
  WHOLEMEMORY_LOGIC_ERROR,         /* logic error */
  WHOLEMEMORY_CUDA_ERROR,          /* CUDA error */
  WHOLEMEMORY_COMMUNICATION_ERROR, /* communication error */
  WHOLEMEMORY_INVALID_INPUT,       /* input is invalid, e.g. nullptr */
  WHOLEMEMORY_INVALID_VALUE,       /* input value is invalid */
};

#define WHOLEMEMORY_RETURN_ON_FAIL(X)               \
  do {                                              \
    auto err = X;                                   \
    if (err != WHOLEMEMORY_SUCCESS) { return err; } \
  } while (0)

enum wholememory_memory_type_t {
  WHOLEMEMORY_MT_NONE = 0,
  WHOLEMEMORY_MT_CONTINUOUS,
  WHOLEMEMORY_MT_CHUNKED,
  WHOLEMEMORY_MT_DISTRIBUTED,
};

enum wholememory_memory_location_t {
  WHOLEMEMORY_ML_NONE = 0,
  WHOLEMEMORY_ML_DEVICE,
  WHOLEMEMORY_ML_HOST,
};

wholememory_error_code_t wholememory_init(unsigned int flags);

wholememory_error_code_t wholememory_finalize();

/* Opaque handle to communicator */
typedef struct wholememory_comm_* wholememory_comm_t;

#define WHOLEMEMORY_UNIQUE_ID_BYTES (128)
struct wholememory_unique_id_t {
  char internal[WHOLEMEMORY_UNIQUE_ID_BYTES];
};

wholememory_error_code_t wholememory_create_unique_id(wholememory_unique_id_t* unique_id);

wholememory_error_code_t wholememory_create_communicator(wholememory_comm_t* comm,
                                                         wholememory_unique_id_t unique_id,
                                                         int rank,
                                                         int size);

wholememory_error_code_t wholememory_destroy_communicator(wholememory_comm_t comm);

wholememory_error_code_t wholememory_communicator_get_rank(int* rank, wholememory_comm_t comm);

wholememory_error_code_t wholememory_communicator_get_size(int* size, wholememory_comm_t comm);

wholememory_error_code_t wholememory_communicator_barrier(wholememory_comm_t comm);

typedef struct wholememory_handle_* wholememory_handle_t;

wholememory_error_code_t wholememory_malloc(wholememory_handle_t* wholememory_handle_ptr,
                                            size_t total_size,
                                            wholememory_comm_t comm,
                                            wholememory_memory_type_t memory_type,
                                            wholememory_memory_location_t memory_location,
                                            size_t data_granularity);

wholememory_error_code_t wholememory_free(wholememory_handle_t wholememory_handle);

wholememory_error_code_t wholememory_get_communicator(wholememory_comm_t* comm,
                                                      wholememory_handle_t wholememory_handle);

wholememory_memory_type_t wholememory_get_memory_type(wholememory_handle_t wholememory_handle);

wholememory_memory_location_t wholememory_get_memory_location(
  wholememory_handle_t wholememory_handle);

size_t wholememory_get_total_size(wholememory_handle_t wholememory_handle);

wholememory_error_code_t wholememory_get_local_memory(void** local_ptr,
                                                      size_t* local_size,
                                                      size_t* local_offset,
                                                      wholememory_handle_t wholememory_handle);

wholememory_error_code_t wholememory_get_rank_memory(void** rank_memory_ptr,
                                                     size_t* rank_memory_size,
                                                     size_t* rank_memory_offset,
                                                     int rank,
                                                     wholememory_handle_t wholememory_handle);

wholememory_error_code_t wholememory_get_global_pointer(void** global_ptr,
                                                        wholememory_handle_t wholememory_handle);

wholememory_error_code_t wholememory_get_global_reference(wholememory_gref_t* wholememory_gref,
                                                          wholememory_handle_t wholememory_handle);

wholememory_error_code_t wholememory_determine_partition_plan(size_t* size_per_rank,
                                                              size_t total_size,
                                                              size_t data_granularity,
                                                              int world_size);

wholememory_error_code_t wholememory_determine_entry_partition_plan(size_t* entry_per_rank,
                                                                    size_t total_entry_count,
                                                                    int world_size);

wholememory_error_code_t wholememory_get_partition_plan(size_t* size_per_rank,
                                                        wholememory_handle_t wholememory_handle);

int fork_get_device_count();

wholememory_error_code_t wholememory_load_from_file(wholememory_handle_t wholememory_handle,
                                                    size_t memory_offset,
                                                    size_t memory_entry_size,
                                                    size_t file_entry_size,
                                                    const char** file_prefix,
                                                    int file_count);

wholememory_error_code_t wholememory_load_hdfs_support();

wholememory_error_code_t wholememory_load_from_hdfs_file(wholememory_handle_t wholememory_handle,
                                                         size_t memory_offset,
                                                         size_t memory_entry_size,
                                                         size_t file_entry_size,
                                                         const char* hdfs_host,
                                                         int hdfs_port,
                                                         const char* hdfs_user,
                                                         const char* hdfs_path,
                                                         const char* hdfs_prefix);

#ifdef __cplusplus
}
#endif
