#include "communicator.hpp"

#include <unistd.h>

#include <cuda.h>

#include <memory>
#include <raft/core/comms.hpp>
#include <raft/comms/detail/util.hpp>
#include <raft/comms/nccl_comms.hpp>

#include "logger.hpp"
#include "memory_handle.hpp"
#include "raft/core/cu_macros.hpp"

namespace wholememory {

static std::mutex comm_mu;
static std::map<int, wholememory_comm_t> communicator_map;

enum wm_comm_op : int32_t {
  WM_COMM_OP_STARTING = 0xEEC0EE,
  WM_COMM_OP_EXCHANGE_ID,
  WM_COMM_OP_WAIT_CREATE_TEMPDIR,
  WM_COMM_OP_DESTROY_ALL_HANDLES,
  WM_COMM_OP_DESTROY_COMM,
};

wholememory_error_code_t create_unique_id(wholememory_unique_id_t *unique_id) noexcept {
  raft::comms::get_unique_id(unique_id->internal, sizeof(unique_id->internal));
  return WHOLEMEMORY_SUCCESS;
}

static constexpr int HOST_NAME_MAX_LENGTH = 1024;
static constexpr int BOOT_ID_MAX_LENGTH = 1024;

struct host_info {
  char host_name[HOST_NAME_MAX_LENGTH];
  char boot_id[BOOT_ID_MAX_LENGTH];
  dev_t shm_dev;
  bool operator==(const host_info& rhs) const {
    if (std::strncmp(host_name, rhs.host_name, HOST_NAME_MAX_LENGTH) != 0) return false;
    if (std::strncmp(boot_id, rhs.boot_id, BOOT_ID_MAX_LENGTH) != 0) return false;
    if (shm_dev != rhs.shm_dev) return false;
    return true;
  }
} __attribute__((aligned(128)));

struct rank_info {
  host_info rank_host_info;
  pid_t pid;
  int rank;
  int size;
};

static void get_host_name(char *hostname, int maxlen, const char delim) {
  if (gethostname(hostname, maxlen) != 0) {
    strncpy(hostname, "unknown", maxlen);
    WHOLEMEMORY_FATAL("gethostname failed.");
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1))
    i++;
  hostname[i] = '\0';
}

/* Get the hostname and boot id
 * Equivalent of:
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 * This string can be overridden by using the WHOLEGRAPH_HOSTID env var.
 */
void get_boot_id(char *host_id, size_t len) {
  char *env_host_id;
  int offset = 0;

#define BOOTID_FILE "/proc/sys/kernel/random/boot_id"

  if ((env_host_id = getenv("WHOLEMEMORY_BOOTID")) != nullptr) {
    WHOLEMEMORY_LOG(LEVEL_INFO, "WHOLEMEMORY_BOOTID set by environment to %s", env_host_id);
    strncpy(host_id, env_host_id, len - 1);
    offset = strlen(env_host_id);
  } else {
    FILE *file = fopen(BOOTID_FILE, "r");
    if (file != nullptr) {
      char *p;
      if (fscanf(file, "%ms", &p) == 1) {
        strncpy(host_id + offset, p, len - offset - 1);
        offset += strlen(p);
        free(p);
      }
    }
    fclose(file);
  }

#undef HOSTID_FILE

  host_id[offset] = '\0';
}

void get_shm_devid(dev_t* shm_dev) {
  struct stat statbuf {};
  WHOLEMEMORY_CHECK(stat("/dev/shm", &statbuf) == 0);
  *shm_dev = statbuf.st_dev;
}

void get_host_info(host_info* phi) {
  bzero(phi, sizeof(host_info));
  get_host_name(&phi->host_name[0], HOST_NAME_MAX_LENGTH, '\0');
  get_boot_id(&phi->boot_id[0], BOOT_ID_MAX_LENGTH);
  get_shm_devid(&phi->shm_dev);
}

void exchange_rank_info(wholememory_comm_t wm_comm) {
  rank_info ri;
  get_host_info(&ri.rank_host_info);
  ri.rank = wm_comm->world_rank;
  ri.size = wm_comm->world_size;
  ri.pid = getpid();

  std::unique_ptr<rank_info[]> p_rank_info(new rank_info[ri.size]);
  wm_comm->raft_nccl_comm->host_allgather(&ri, p_rank_info.get(), sizeof(rank_info), raft::comms::datatype_t::CHAR);
  wm_comm->intra_node_first_rank = -1;
  wm_comm->intra_node_rank_num = 0;
  wm_comm->intra_node_rank = -1;
  for (int r = 0; r < wm_comm->world_size; r++) {
    WHOLEMEMORY_CHECK(r == p_rank_info.get()[r].rank);
    if (ri.rank_host_info == p_rank_info.get()[r].rank_host_info) {
      if (r == wm_comm->world_rank) {
        wm_comm->intra_node_rank = wm_comm->intra_node_rank_num;
      }
      if (wm_comm->intra_node_rank_num == 0) {
        wm_comm->intra_node_first_rank_pid = p_rank_info.get()[r].pid;
        wm_comm->intra_node_first_rank = r;
      }
      wm_comm->intra_node_rank_num++;
    }
  }
}

void negotiate_communicator_id_locked(wholememory_comm_t wm_comm) {
  WM_COMM_CHECK_ALL_SAME(wm_comm, WM_COMM_OP_EXCHANGE_ID);
  int id = 0;
  bool all_same = false;
  std::vector<int> rank_ids(wm_comm->world_size);
  while (!all_same) {
    while (communicator_map.find(id) != communicator_map.end()) id++;
    wm_comm->raft_nccl_comm->host_allgather(&id, rank_ids.data(), 1, raft::comms::datatype_t::INT32);
    int max_id = -1;
    all_same = true;
    for (int i = 0; i < wm_comm->world_size; i++) {
      if (rank_ids[i] > max_id) max_id = rank_ids[i];
      if (rank_ids[i] != id) all_same = false;
    }
    id = max_id;
  }
  wm_comm->comm_id = id;
  communicator_map.insert(std::pair<int, wholememory_comm_t>(id, wm_comm));
}

std::string get_temporary_directory_path(wholememory_comm_t comm) {
  const char *sock_prefix = getenv("WHOLEMEMORY_TMPPREFIX");
  std::string wholememory_prefix_str = "/tmp/wmtmp";
  if (sock_prefix != nullptr) {
    wholememory_prefix_str = sock_prefix;
  }
  char temp_path_buffer[128];
  (void) std::snprintf(temp_path_buffer, 128, "_comm_id_%d_pid_%d",
                       comm->comm_id, comm->intra_node_first_rank_pid);
  wholememory_prefix_str.append(temp_path_buffer);
  return wholememory_prefix_str;
}

std::string get_shm_prefix(wholememory_comm_t comm) {
  char temp_path_buffer[128];
  (void) std::snprintf(temp_path_buffer, 128, "wmshm_comm_id_%d_pid_%d",
                       comm->comm_id, comm->intra_node_first_rank_pid);
  std::string wholememory_prefix_str = temp_path_buffer;
  return wholememory_prefix_str;
}

void maybe_create_temp_dir(wholememory_comm_t wm_comm) {
  if (!is_intranode_communicator(wm_comm)) return;
  if (wm_comm->intra_node_rank == 0) {
    auto temp_path = get_temporary_directory_path(wm_comm);
    WHOLEMEMORY_CHECK(mkdir(temp_path.c_str(), 0700) == 0);
  }
  WM_COMM_CHECK_ALL_SAME(wm_comm, WM_COMM_OP_WAIT_CREATE_TEMPDIR);
}

void maybe_remove_temp_dir(wholememory_comm_t wm_comm) {
  if (!is_intranode_communicator(wm_comm)) return;
  if (wm_comm->intra_node_rank == 0) {
    auto temp_path = get_temporary_directory_path(wm_comm);
    WHOLEMEMORY_CHECK(remove(get_temporary_directory_path(wm_comm).c_str()) == 0);
  }
}

static size_t get_alloc_granularity(int dev_id) {
  size_t granularity = 0;
  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof(prop));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
  CUmemAllocationGranularity_flags flags = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED;
  prop.location.id = dev_id;
  CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, flags));
  return granularity;
}

void determine_alloc_granularity(wholememory_comm_t comm) {
  size_t granularity = get_alloc_granularity(comm->dev_id);
  std::vector<size_t> all_granularitys(comm->world_size);
  comm->raft_nccl_comm->host_allgather(&granularity, all_granularitys.data(), 1, raft::comms::datatype_t::UINT64);
  size_t max_granularity = granularity;
  for (auto g: all_granularitys) {
    if (g > max_granularity) {
      max_granularity = g;
    }
  }
  comm->alloc_granularity = max_granularity;
}

wholememory_error_code_t create_communicator(wholememory_comm_t *comm,
                                             wholememory_unique_id_t unique_id,
                                             int world_rank,
                                             int world_size) noexcept {
  try {
    std::unique_lock<std::mutex> mlock(comm_mu);
    ncclComm_t nccl_comm;
    RAFT_NCCL_TRY(ncclCommInitRank(&nccl_comm, world_size, (ncclUniqueId &) unique_id, world_rank));
    cudaStream_t cuda_stream;
    CUDA_CHECK(cudaStreamCreate(&cuda_stream));
    rmm::cuda_stream_view rmm_stream(cuda_stream);
    auto *wm_comm = new wholememory_comm_(nccl_comm, world_size, world_rank, rmm_stream);
    *comm = wm_comm;
    WM_COMM_CHECK_ALL_SAME(wm_comm, WM_COMM_OP_STARTING);

    exchange_rank_info(wm_comm);

    negotiate_communicator_id_locked(wm_comm);

    maybe_create_temp_dir(wm_comm);

    determine_alloc_granularity(wm_comm);

    return WHOLEMEMORY_SUCCESS;
  } catch (const raft::cuda_error& rce) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", rce.what());
  } catch (const raft::logic_error& rle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", rle.what());
  } catch (const wholememory::logic_error& wle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wle.what());
  } catch (const raft::exception& re) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
  } catch (const std::bad_alloc& sba) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", sba.what());
  } catch (...) {
    WHOLEMEMORY_FAIL_NOTHROW("Unknown exception.");
  }
}

void destroy_all_wholememory(wholememory_comm_t comm) noexcept {
  try {
    std::unique_lock<std::mutex> mlock(comm->mu);
    WM_COMM_CHECK_ALL_SAME(comm, WM_COMM_OP_DESTROY_ALL_HANDLES);
    WM_COMM_CHECK_ALL_SAME(comm, comm->wholememory_map.size());
    while (!comm->wholememory_map.empty()) {
      auto id_wm = comm->wholememory_map.begin();
      destroy_wholememory_with_comm_locked(id_wm->second);
    }
  } catch (const wholememory::logic_error& wle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wle.what());
  } catch (const raft::cuda_error& rce) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", rce.what());
  } catch (const raft::logic_error& rle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", rle.what());
  } catch (const raft::exception& re) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
  } catch (...) {
    WHOLEMEMORY_FAIL_NOTHROW("Unknown exception.");
  }
}

wholememory_error_code_t destroy_communicator_locked(wholememory_comm_t comm) noexcept {
  try {
    if (communicator_map.find(comm->comm_id) == communicator_map.end()) {
      return WHOLEMEMORY_INVALID_INPUT;
    }
    destroy_all_wholememory(comm);
    WM_COMM_CHECK_ALL_SAME(comm, WM_COMM_OP_DESTROY_COMM);
    communicator_map.erase(comm->comm_id);
    auto *raw_nccl_comm = comm->raw_nccl_comm;
    auto *cuda_stream = comm->comm_stream;

    maybe_remove_temp_dir(comm);

    delete comm;
    RAFT_NCCL_TRY(ncclCommDestroy(raw_nccl_comm));
    CUDA_CHECK(cudaStreamDestroy(cuda_stream));

    return WHOLEMEMORY_SUCCESS;
  } catch (const raft::cuda_error& rce) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", rce.what());
  } catch (const raft::logic_error& rle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", rle.what());
  } catch (const wholememory::logic_error& wle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wle.what());
  } catch (const raft::exception& re) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
  } catch (...) {
    WHOLEMEMORY_FAIL_NOTHROW("Unknown exception.");
  }
}

wholememory_error_code_t destroy_communicator(wholememory_comm_t comm) noexcept {
  std::unique_lock<std::mutex> mlock(comm_mu);
  return destroy_communicator_locked(comm);
}

wholememory_error_code_t destroy_all_communicators() noexcept {
  std::unique_lock<std::mutex> mlock(comm_mu);
  while (!communicator_map.empty()) {
    auto id_comm = communicator_map.begin();
    destroy_communicator_locked(id_comm->second);
  }
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t communicator_get_rank(int *rank,
                                               wholememory_comm_t comm) noexcept {
  *rank = comm->world_rank;
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t communicator_get_size(int* size,
                                               wholememory_comm_t comm) noexcept {
  *size = comm->world_size;
  return WHOLEMEMORY_SUCCESS;
}

void communicator_barrier(wholememory_comm_t comm) {
  comm->raft_nccl_comm->barrier();
}

bool is_intranode_communicator(wholememory_comm_t comm) noexcept {
  return comm->intra_node_rank_num == comm->world_size;
}

}