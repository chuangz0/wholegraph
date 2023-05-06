#pragma once

#include <wholememory/wholememory.h>

#include <cuda_runtime_api.h>

#include "communicator.hpp"

namespace wholememory {

class wholememory_impl;

}

struct wholememory_handle_ {
  int handle_id;
  wholememory::wholememory_impl* impl = nullptr;
  ~wholememory_handle_();
};

namespace wholememory {

wholememory_error_code_t create_wholememory(wholememory_handle_t* wholememory_handle_ptr,
                                            size_t total_size,
                                            wholememory_comm_t comm,
                                            wholememory_memory_type_t memory_type,
                                            wholememory_memory_location_t memory_location,
                                            size_t data_granularity) noexcept;

wholememory_error_code_t destroy_wholememory_with_comm_locked(
  wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t destroy_wholememory(wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_communicator_from_handle(
  wholememory_comm_t* comm, wholememory_handle_t wholememory_handle) noexcept;

wholememory_memory_type_t get_memory_type(wholememory_handle_t wholememory_handle) noexcept;

wholememory_memory_location_t get_memory_location(wholememory_handle_t wholememory_handle) noexcept;

size_t get_total_size(wholememory_handle_t wholememory_handle) noexcept;

size_t get_data_granularity(wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_local_memory_from_handle(
  void** local_ptr,
  size_t* local_size,
  size_t* local_offset,
  wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_rank_memory_from_handle(
  void** rank_memory_ptr,
  size_t* rank_memory_size,
  size_t* rank_memory_offset,
  int rank,
  wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_global_pointer_from_handle(
  void** global_ptr, wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_global_reference_from_handle(
  wholememory_gref_t* wholememory_gref, wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t determine_partition_plan(size_t* size_per_rank,
                                                  size_t total_size,
                                                  size_t data_granularity,
                                                  int world_size) noexcept;

size_t determine_entry_partition_plan(size_t total_entry_count, int world_size) noexcept;

wholememory_error_code_t get_partition_plan_from_handle(
  size_t* size_per_rank, wholememory_handle_t wholememory_handle) noexcept;

}  // namespace wholememory
