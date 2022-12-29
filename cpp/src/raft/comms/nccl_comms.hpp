#pragma once

#include <raft/core/handle.hpp>

#include <raft/comms/comms.hpp>

#include "raft/comms/detail/nccl_comms.hpp"

#include <iostream>
#include <nccl.h>

namespace raft {
namespace comms {

using nccl_comms = detail::nccl_comms;

/**
 * Function to construct comms_t and inject it on a handle_t. This
 * is used for convenience in the Python layer.
 *
 * @param handle raft::handle_t for injecting the comms
 * @param nccl_comm initialized NCCL communicator to use for collectives
 * @param num_ranks number of ranks in communicator clique
 * @param rank rank of local instance
 */
inline void build_comms_pure_nccl(handle_t* handle, ncclComm_t nccl_comm, int num_ranks, int rank)
{
  cudaStream_t stream = handle->get_stream();

  auto communicator = std::make_shared<comms_t>(
      std::unique_ptr<comms_iface>(new raft::comms::nccl_comms(nccl_comm, num_ranks, rank, stream)));
  handle->set_comms(communicator);
}

// FIXME: remove duplicate from raft/comms/std_comms.hpp
inline void nccl_unique_id_from_char(ncclUniqueId* id, char* uniqueId, int size)
{
  memcpy(id->internal, uniqueId, size);
}

inline void get_unique_id(char* uid, int size)
{
  ncclUniqueId id;
  ncclGetUniqueId(&id);

  memcpy(uid, id.internal, size);
}

}  // namespace comms
}  // end namespace raft
