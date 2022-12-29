#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <nccl.h>

#include <raft/comms/nccl_comms.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <mutex>
#include <vector>

#include <wholememory/wholememory.h>

struct wholememory_comm_ {
  wholememory_comm_(ncclComm_t nccl_comm,
                    int num_ranks,
                    int rank,
                    rmm::cuda_stream_view stream) {
    world_rank = rank;
    world_size = num_ranks;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    comm_stream = stream;
    raw_nccl_comm = nccl_comm;
    CUDA_CHECK(cudaEventCreate(&cuda_event));
    raft_nccl_comm = std::make_unique<raft::comms::nccl_comms>(nccl_comm, num_ranks, rank, stream);
  }
  ~wholememory_comm_(){
    raft_nccl_comm.reset();
    if (cuda_event != nullptr) {
      cudaEventDestroy(cuda_event);
      cuda_event = nullptr;
    }
  }
  std::unique_ptr<raft::comms::nccl_comms> raft_nccl_comm;
  cudaStream_t comm_stream = nullptr;
  cudaEvent_t cuda_event = nullptr;
  ncclComm_t raw_nccl_comm = nullptr;

  int world_rank = 0;
  int world_size = 1;

  int intra_node_first_rank = -1;
  int intra_node_rank = -1;
  int intra_node_rank_num = 0;
  int intra_node_first_rank_pid = -1;

  int comm_id = -1;

  int dev_id = -1;

  size_t alloc_granularity = 2 * 1024 * 1024UL;

  std::mutex mu;
  std::map<int, wholememory_handle_t> wholememory_map;
} __attribute__((aligned(64)));

template <typename T>
inline bool wm_comm_check_all_same(wholememory_comm_t comm, const T& t) {
  std::unique_ptr<T[]> t_array(new T[comm->world_size]());
  comm->raft_nccl_comm->host_allgather(&t, t_array.get(), sizeof(T), raft::comms::datatype_t::CHAR);
  for (int r = 0; r < comm->world_size; r++) {
    if (t_array.get()[r] != t) return false;
  }
  return true;
}

template <>
inline bool wm_comm_check_all_same(wholememory_comm_t comm, const std::string& str) {
  size_t str_len = str.size();
  if (!wm_comm_check_all_same(comm, str_len)) return false;
  std::string cat_str;
  cat_str.resize(str_len * comm->world_size, '\0');
  comm->raft_nccl_comm->host_allgather(str.data(), cat_str.data(), str_len, raft::comms::datatype_t::CHAR);
  for (int r = 0; r < comm->world_size; r++) {
    if (std::strncmp(str.data(), cat_str.data() + r * str_len, str_len) != 0) return false;
  }
  return true;
}

#define WM_COMM_CHECK_ALL_SAME(comm, data)               \
do {                                                     \
  if (!wm_comm_check_all_same(comm, data)) {             \
    WHOLEMEMORY_FATAL("COMM_CHECK_ALL_SAME failed.");    \
  }                                                      \
} while(0)

namespace wholememory {

wholememory_error_code_t create_unique_id(wholememory_unique_id_t *unique_id) noexcept;

wholememory_error_code_t create_communicator(wholememory_comm_t *comm,
                                             wholememory_unique_id_t unique_id,
                                             int rank,
                                             int size) noexcept;

wholememory_error_code_t destroy_communicator_locked(wholememory_comm_t comm) noexcept;

wholememory_error_code_t destroy_communicator(wholememory_comm_t comm) noexcept;

wholememory_error_code_t destroy_all_communicators() noexcept;

wholememory_error_code_t communicator_get_rank(int *rank,
                                               wholememory_comm_t comm) noexcept;

wholememory_error_code_t communicator_get_size(int* size,
                                               wholememory_comm_t comm) noexcept;

void communicator_barrier(wholememory_comm_t comm);

bool is_intranode_communicator(wholememory_comm_t comm) noexcept;

std::string get_temporary_directory_path(wholememory_comm_t comm);

std::string get_shm_prefix(wholememory_comm_t comm);

}