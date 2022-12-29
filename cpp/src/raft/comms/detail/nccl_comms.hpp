#pragma once

#include "raft/comms/detail/nccl_comms.hpp"

#include <raft/comms/comms.hpp>
#include <raft/comms/detail/util.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/error.hpp>

#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

#include <nccl.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdlib.h>
#include <thread>
#include <time.h>

#include "rmm/pinned_uvector.hpp"

namespace raft {
namespace comms {
namespace detail {

class nccl_comms : public comms_iface {
 public:
  nccl_comms() = delete;

  /**
   * @brief Constructor for collective + point-to-point operation.
   * @param nccl_comm initialized nccl comm
   * @param ucp_worker initialized ucp_worker instance
   * @param eps shared pointer to array of ucp endpoints
   * @param num_ranks number of ranks in the cluster
   * @param rank rank of the current worker
   * @param stream cuda stream for synchronizing and ordering collective operations
   * @param subcomms_ucp use ucp for subcommunicators
   */
  nccl_comms(ncclComm_t nccl_comm,
             int num_ranks,
             int rank,
             rmm::cuda_stream_view rmm_stream)
      : nccl_comm_(nccl_comm),
        rmm_stream_(rmm_stream.value()),
        status_(rmm_stream),
        num_ranks_(num_ranks),
        rank_(rank),
        host_send_buffer_(HOST_BUFFER_SIZE_PER_RANK * num_ranks, rmm_stream),
        host_recv_buffer_(HOST_BUFFER_SIZE_PER_RANK * num_ranks, rmm_stream)
  {
    initialize();
  };

  void initialize()
  {
    status_.set_value_to_zero_async(rmm_stream_);
    buf_ = status_.data();
  }

  ~nccl_comms()
  {
  }

  int get_size() const { return num_ranks_; }

  int get_rank() const { return rank_; }

  std::unique_ptr<comms_iface> comm_split(int color, int key) const
  {
    rmm::device_uvector<int> d_colors(get_size(), rmm_stream_);
    rmm::device_uvector<int> d_keys(get_size(), rmm_stream_);

    update_device(d_colors.data() + get_rank(), &color, 1, rmm_stream_);
    update_device(d_keys.data() + get_rank(), &key, 1, rmm_stream_);

    allgather(d_colors.data() + get_rank(), d_colors.data(), 1, datatype_t::INT32, rmm_stream_);
    allgather(d_keys.data() + get_rank(), d_keys.data(), 1, datatype_t::INT32, rmm_stream_);
    this->sync_stream(rmm_stream_);

    std::vector<int> h_colors(get_size());
    std::vector<int> h_keys(get_size());

    update_host(h_colors.data(), d_colors.data(), get_size(), rmm_stream_);
    update_host(h_keys.data(), d_keys.data(), get_size(), rmm_stream_);

    RAFT_CUDA_TRY(cudaStreamSynchronize(rmm_stream_));

    std::vector<int> subcomm_ranks{};

    for (int i = 0; i < get_size(); ++i) {
      if (h_colors[i] == color) {
        subcomm_ranks.push_back(i);
      }
    }

    ncclUniqueId id{};
    if (get_rank() == subcomm_ranks[0]) {  // root of the new subcommunicator
      RAFT_NCCL_TRY(ncclGetUniqueId(&id));
      std::vector<request_t> requests(subcomm_ranks.size() - 1);
      for (size_t i = 1; i < subcomm_ranks.size(); ++i) {
        isend(&id, sizeof(ncclUniqueId), subcomm_ranks[i], color, requests.data() + (i - 1));
      }
      waitall(requests.size(), requests.data());
    } else {
      request_t request{};
      irecv(&id, sizeof(ncclUniqueId), subcomm_ranks[0], color, &request);
      waitall(1, &request);
    }
    // FIXME: this seems unnecessary, do more testing and remove this
    barrier();

    ncclComm_t nccl_comm;
    RAFT_NCCL_TRY(ncclCommInitRank(&nccl_comm, subcomm_ranks.size(), id, key));
    return std::unique_ptr<comms_iface>(
        new nccl_comms(nccl_comm, subcomm_ranks.size(), key, rmm_stream_));
  }

  void barrier() const
  {
    allreduce(buf_, buf_, 1, datatype_t::INT32, op_t::SUM, rmm_stream_);

    ASSERT(sync_stream(rmm_stream_) == status_t::SUCCESS,
           "ERROR: syncStream failed. This can be caused by a failed rank_.");
  }

  void isend(const void* send_buf, size_t send_size, int dest, int tag, request_t* request) const
  {
    RAFT_FAIL("nccl_comms doesn't support isend.");
  }

  void irecv(void* recv_buf, size_t recv_size, int source, int tag, request_t* request) const
  {
    RAFT_FAIL("nccl_comms doesn't support irecv.");
  }

  void waitall(int count, request_t array_of_requests[]) const
  {
    RAFT_FAIL("nccl_comms doesn't support waitall.");
  }

  void allreduce(const void* sendbuff,
                 void* recvbuff,
                 size_t count,
                 datatype_t datatype,
                 op_t op,
                 cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(ncclAllReduce(
        sendbuff, recvbuff, count, get_nccl_datatype(datatype), get_nccl_op(op), nccl_comm_, stream));
  }

  void host_allreduce(const void* sendbuff,
                      void* recvbuff,
                      size_t count,
                      datatype_t datatype,
                      op_t op) const
  {
    const size_t datatype_size = get_datatype_size(datatype);
    const size_t max_elt_count = host_send_buffer_.size() / datatype_size;
    for (size_t offset = 0; offset < count; offset += max_elt_count) {
      size_t elt_count = (count - offset > max_elt_count) ? max_elt_count : count - offset;
      std::memcpy(host_send_buffer_.data(), static_cast<const char*>(sendbuff) + datatype_size * offset, elt_count * datatype_size);
      RAFT_NCCL_TRY(ncclAllReduce(
          host_send_buffer_.data(), host_recv_buffer_.data(), elt_count, get_nccl_datatype(datatype), get_nccl_op(op), nccl_comm_, rmm_stream_.value()));
      rmm_stream_.synchronize();
      std::memcpy(static_cast<char*>(recvbuff) + datatype_size * offset, host_recv_buffer_.data(), elt_count * datatype_size);
    }
  }

  void bcast(void* buff, size_t count, datatype_t datatype, int root, cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(
        ncclBroadcast(buff, buff, count, get_nccl_datatype(datatype), root, nccl_comm_, stream));
  }

  void bcast(const void* sendbuff,
             void* recvbuff,
             size_t count,
             datatype_t datatype,
             int root,
             cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(ncclBroadcast(
        sendbuff, recvbuff, count, get_nccl_datatype(datatype), root, nccl_comm_, stream));
  }

  void host_bcast(const void *sendbuff,
                  void *recvbuff,
                  size_t count,
                  datatype_t datatype,
                  int root) const
  {
    const size_t datatype_size = get_datatype_size(datatype);
    const size_t max_elt_count = host_send_buffer_.size() / datatype_size;
    for (size_t offset = 0; offset < count; offset += max_elt_count) {
      size_t elt_count = (count - offset > max_elt_count) ? max_elt_count : count - offset;
      std::memcpy(host_send_buffer_.data(), static_cast<const char*>(sendbuff) + datatype_size * offset, elt_count * datatype_size);
      RAFT_NCCL_TRY(ncclBroadcast(
          host_send_buffer_.data(), host_recv_buffer_.data(), elt_count, get_nccl_datatype(datatype), root, nccl_comm_, rmm_stream_.value()));
      rmm_stream_.synchronize();
      std::memcpy(static_cast<char*>(recvbuff) + datatype_size * offset, host_recv_buffer_.data(), elt_count * datatype_size);
    }
  }

  void host_bcast(void* buff, size_t count, datatype_t datatype, int root) const
  {
    host_bcast(buff, buff, count, datatype, root);
  }

  void reduce(const void* sendbuff,
              void* recvbuff,
              size_t count,
              datatype_t datatype,
              op_t op,
              int root,
              cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(ncclReduce(sendbuff,
                             recvbuff,
                             count,
                             get_nccl_datatype(datatype),
                             get_nccl_op(op),
                             root,
                             nccl_comm_,
                             stream));
  }

  void host_reduce(const void *sendbuff,
                   void *recvbuff,
                   size_t count,
                   datatype_t datatype,
                   op_t op,
                   int root) const
  {
    const size_t datatype_size = get_datatype_size(datatype);
    const size_t max_elt_count = host_send_buffer_.size() / datatype_size;
    for (size_t offset = 0; offset < count; offset += max_elt_count) {
      size_t elt_count = (count - offset > max_elt_count) ? max_elt_count : count - offset;
      std::memcpy(host_send_buffer_.data(),
                  static_cast<const char *>(sendbuff) + datatype_size * offset,
                  elt_count * datatype_size);
      RAFT_NCCL_TRY(ncclReduce(
          host_send_buffer_.data(),
          host_recv_buffer_.data(),
          elt_count,
          get_nccl_datatype(datatype),
          get_nccl_op(op),
          root,
          nccl_comm_,
          rmm_stream_.value()));
      rmm_stream_.synchronize();
      if (get_rank() == root) {
        std::memcpy(static_cast<char *>(recvbuff) + datatype_size * offset,
                    host_recv_buffer_.data(),
                    elt_count * datatype_size);
      }
    }
  }

  void allgather(const void* sendbuff,
                 void* recvbuff,
                 size_t sendcount,
                 datatype_t datatype,
                 cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(ncclAllGather(
        sendbuff, recvbuff, sendcount, get_nccl_datatype(datatype), nccl_comm_, stream));
  }

  void host_allgather(const void *sendbuff,
                      void *recvbuff,
                      size_t sendcount,
                      datatype_t datatype) const
  {
    const size_t datatype_size = get_datatype_size(datatype);
    const size_t max_elt_count = HOST_BUFFER_SIZE_PER_RANK / datatype_size;
    for (size_t offset = 0; offset < sendcount; offset += max_elt_count) {
      size_t elt_count = (sendcount - offset > max_elt_count) ? max_elt_count : sendcount - offset;
      std::memcpy(host_send_buffer_.data(),
                  static_cast<const char *>(sendbuff) + datatype_size * offset,
                  elt_count * datatype_size);
      RAFT_NCCL_TRY(ncclAllGather(
          host_send_buffer_.data(), host_recv_buffer_.data(), sendcount, get_nccl_datatype(datatype), nccl_comm_, rmm_stream_.value()));
      rmm_stream_.synchronize();
      for (int i = 0; i < get_size(); i++) {
        std::memcpy(static_cast<char *>(recvbuff) + datatype_size * offset + i * sendcount * datatype_size,
                    host_recv_buffer_.data() + i * elt_count * datatype_size,
                    elt_count * datatype_size);
      }
    }
  }

  void allgatherv(const void* sendbuf,
                  void* recvbuf,
                  const size_t* recvcounts,
                  const size_t* displs,
                  datatype_t datatype,
                  cudaStream_t stream) const
  {
    // From: "An Empirical Evaluation of Allgatherv on Multi-GPU Systems" -
    // https://arxiv.org/pdf/1812.05964.pdf Listing 1 on page 4.
    RAFT_EXPECTS(num_ranks_ <= 2048,
                 "# NCCL operations between ncclGroupStart & ncclGroupEnd shouldn't exceed 2048.");
    RAFT_NCCL_TRY(ncclGroupStart());
    for (int root = 0; root < num_ranks_; ++root) {
      size_t dtype_size = get_datatype_size(datatype);
      RAFT_NCCL_TRY(ncclBroadcast(sendbuf,
                                  static_cast<char*>(recvbuf) + displs[root] * dtype_size,
                                  recvcounts[root],
                                  get_nccl_datatype(datatype),
                                  root,
                                  nccl_comm_,
                                  stream));
    }
    RAFT_NCCL_TRY(ncclGroupEnd());
  }

  void host_allgatherv(const void *sendbuf,
                       void *recvbuf,
                       const size_t *recvcounts,
                       const size_t *displs,
                       datatype_t datatype) const
  {
    size_t dtype_size = get_datatype_size(datatype);
    for (int root = 0; root < num_ranks_; ++root) {
      host_bcast(sendbuf,
                 static_cast<char *>(recvbuf) + displs[root] * dtype_size,
                 recvcounts[root],
                 datatype,
                 root);
    }
  }

  void gather(const void* sendbuff,
              void* recvbuff,
              size_t sendcount,
              datatype_t datatype,
              int root,
              cudaStream_t stream) const
  {
    size_t dtype_size = get_datatype_size(datatype);
    RAFT_NCCL_TRY(ncclGroupStart());
    if (get_rank() == root) {
      for (int r = 0; r < get_size(); ++r) {
        RAFT_NCCL_TRY(ncclRecv(static_cast<char*>(recvbuff) + sendcount * r * dtype_size,
                               sendcount,
                               get_nccl_datatype(datatype),
                               r,
                               nccl_comm_,
                               stream));
      }
    }
    RAFT_NCCL_TRY(
        ncclSend(sendbuff, sendcount, get_nccl_datatype(datatype), root, nccl_comm_, stream));
    RAFT_NCCL_TRY(ncclGroupEnd());
  }

  void host_gather(const void *sendbuff,
                   void *recvbuff,
                   size_t sendcount,
                   datatype_t datatype,
                   int root) const {
    const size_t datatype_size = get_datatype_size(datatype);
    const size_t max_elt_count = HOST_BUFFER_SIZE_PER_RANK / datatype_size;
    for (size_t offset = 0; offset < sendcount; offset += max_elt_count) {
      size_t elt_count = (sendcount - offset > max_elt_count) ? max_elt_count : sendcount - offset;
      std::memcpy(host_send_buffer_.data(),
                  static_cast<const char *>(sendbuff) + datatype_size * offset,
                  elt_count * datatype_size);
      gather(host_send_buffer_.data(), host_recv_buffer_.data(), sendcount, datatype, root, rmm_stream_.value());
      rmm_stream_.synchronize();
      if (rank_ == root) {
        for (int i = 0; i < num_ranks_; i++) {
          std::memcpy(static_cast<char *>(recvbuff) + datatype_size * offset + i * sendcount * datatype_size,
                      host_recv_buffer_.data() + i * elt_count * datatype_size,
                      elt_count * datatype_size);
        }
      }
    }
  }

  void gatherv(const void* sendbuff,
               void* recvbuff,
               size_t sendcount,
               const size_t* recvcounts,
               const size_t* displs,
               datatype_t datatype,
               int root,
               cudaStream_t stream) const
  {
    size_t dtype_size = get_datatype_size(datatype);
    RAFT_NCCL_TRY(ncclGroupStart());
    if (get_rank() == root) {
      for (int r = 0; r < get_size(); ++r) {
        RAFT_NCCL_TRY(ncclRecv(static_cast<char*>(recvbuff) + displs[r] * dtype_size,
                               recvcounts[r],
                               get_nccl_datatype(datatype),
                               r,
                               nccl_comm_,
                               stream));
      }
    }
    RAFT_NCCL_TRY(
        ncclSend(sendbuff, sendcount, get_nccl_datatype(datatype), root, nccl_comm_, stream));
    RAFT_NCCL_TRY(ncclGroupEnd());
  }

  void reducescatter(const void* sendbuff,
                     void* recvbuff,
                     size_t recvcount,
                     datatype_t datatype,
                     op_t op,
                     cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(ncclReduceScatter(sendbuff,
                                    recvbuff,
                                    recvcount,
                                    get_nccl_datatype(datatype),
                                    get_nccl_op(op),
                                    nccl_comm_,
                                    stream));
  }
  void alltoall(const void* sendbuff,
                void* recvbuff,
                size_t sendcount,
                datatype_t datatype,
                cudaStream_t stream) const {
    size_t dtype_size = get_datatype_size(datatype);
    RAFT_NCCL_TRY(ncclGroupStart());
    for (int r = 0; r < get_size(); ++r) {
      RAFT_NCCL_TRY(ncclRecv(static_cast<char*>(recvbuff) + sendcount * r * dtype_size,
                             sendcount,
                             get_nccl_datatype(datatype),
                             r,
                             nccl_comm_,
                             stream));
    }
    for (int r = 0; r < get_size(); ++r) {
      RAFT_NCCL_TRY(ncclSend(static_cast<const char *>(sendbuff) + sendcount * r * dtype_size,
                             sendcount,
                             get_nccl_datatype(datatype),
                             r,
                             nccl_comm_,
                             stream));
    }
    RAFT_NCCL_TRY(ncclGroupEnd());
  }

  void host_alltoall(const void *sendbuff,
                     void *recvbuff,
                     size_t sendcount,
                     datatype_t datatype) const {
    const size_t datatype_size = get_datatype_size(datatype);
    const size_t max_elt_count = HOST_BUFFER_SIZE_PER_RANK / datatype_size;
    for (size_t offset = 0; offset < sendcount; offset += max_elt_count) {
      size_t elt_count = (sendcount - offset > max_elt_count) ? max_elt_count : sendcount - offset;
      for (int i = 0; i < num_ranks_; i++) {
        std::memcpy(host_send_buffer_.data() + i * elt_count * datatype_size,
                    static_cast<const char *>(sendbuff) + datatype_size * offset + i * sendcount * datatype_size,
                    elt_count * datatype_size);
      }
      alltoall(host_send_buffer_.data(), host_recv_buffer_.data(), sendcount, datatype, rmm_stream_.value());
      rmm_stream_.synchronize();
      for (int i = 0; i < num_ranks_; i++) {
        std::memcpy(static_cast<char *>(recvbuff) + datatype_size * offset + i * sendcount * datatype_size,
                    host_recv_buffer_.data() + i * elt_count * datatype_size,
                    elt_count * datatype_size);
      }
    }
  }

  void alltoallv(const void *sendbuff,
                 void *recvbuff,
                 const size_t *sendcounts,
                 const size_t *senddispls,
                 const size_t *recvcounts,
                 const size_t *recvdispls,
                 datatype_t datatype,
                 cudaStream_t stream) const
  {
    size_t dtype_size = get_datatype_size(datatype);
    RAFT_NCCL_TRY(ncclGroupStart());
    for (int r = 0; r < get_size(); ++r) {
      RAFT_NCCL_TRY(ncclRecv(static_cast<char*>(recvbuff) + recvdispls[r] * dtype_size,
                             recvcounts[r],
                             get_nccl_datatype(datatype),
                             r,
                             nccl_comm_,
                             stream));
    }
    for (int r = 0; r < get_size(); ++r) {
      RAFT_NCCL_TRY(ncclSend(static_cast<const char *>(sendbuff) + senddispls[r] * dtype_size,
                             sendcounts[r],
                             get_nccl_datatype(datatype),
                             r,
                             nccl_comm_,
                             stream));
    }
    RAFT_NCCL_TRY(ncclGroupEnd());
  }


  status_t sync_stream(cudaStream_t stream) const { return nccl_sync_stream(nccl_comm_, stream); }

  status_t sync_stream() const { return sync_stream(rmm_stream_.value()); }

  // if a thread is sending & receiving at the same time, use device_sendrecv to avoid deadlock
  void device_send(const void* send_buf, size_t send_size, int dest, cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(ncclSend(send_buf, send_size, ncclUint8, dest, nccl_comm_, stream));
  }

  // if a thread is sending & receiving at the same time, use device_sendrecv to avoid deadlock
  void device_recv(void* recv_buf, size_t recv_size, int source, cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(ncclRecv(recv_buf, recv_size, ncclUint8, source, nccl_comm_, stream));
  }

  void device_sendrecv(const void* sendbuf,
                       size_t sendsize,
                       int dest,
                       void* recvbuf,
                       size_t recvsize,
                       int source,
                       cudaStream_t stream) const
  {
    // ncclSend/ncclRecv pair needs to be inside ncclGroupStart/ncclGroupEnd to avoid deadlock
    RAFT_NCCL_TRY(ncclGroupStart());
    RAFT_NCCL_TRY(ncclSend(sendbuf, sendsize, ncclUint8, dest, nccl_comm_, stream));
    RAFT_NCCL_TRY(ncclRecv(recvbuf, recvsize, ncclUint8, source, nccl_comm_, stream));
    RAFT_NCCL_TRY(ncclGroupEnd());
  }

  void device_multicast_sendrecv(const void* sendbuf,
                                 std::vector<size_t> const& sendsizes,
                                 std::vector<size_t> const& sendoffsets,
                                 std::vector<int> const& dests,
                                 void* recvbuf,
                                 std::vector<size_t> const& recvsizes,
                                 std::vector<size_t> const& recvoffsets,
                                 std::vector<int> const& sources,
                                 cudaStream_t stream) const
  {
    // ncclSend/ncclRecv pair needs to be inside ncclGroupStart/ncclGroupEnd to avoid deadlock
    RAFT_NCCL_TRY(ncclGroupStart());
    for (size_t i = 0; i < sendsizes.size(); ++i) {
      RAFT_NCCL_TRY(ncclSend(static_cast<const char*>(sendbuf) + sendoffsets[i],
                             sendsizes[i],
                             ncclUint8,
                             dests[i],
                             nccl_comm_,
                             stream));
    }
    for (size_t i = 0; i < recvsizes.size(); ++i) {
      RAFT_NCCL_TRY(ncclRecv(static_cast<char*>(recvbuf) + recvoffsets[i],
                             recvsizes[i],
                             ncclUint8,
                             sources[i],
                             nccl_comm_,
                             stream));
    }
    RAFT_NCCL_TRY(ncclGroupEnd());
  }

  void group_start() const { RAFT_NCCL_TRY(ncclGroupStart()); }

  void group_end() const { RAFT_NCCL_TRY(ncclGroupEnd()); }

 private:
  ncclComm_t nccl_comm_;
  rmm::cuda_stream_view rmm_stream_;

  rmm::device_scalar<int32_t> status_;

  int num_ranks_;
  int rank_;

  mutable rmm::pinned_uvector<char> host_send_buffer_;
  mutable rmm::pinned_uvector<char> host_recv_buffer_;
  static constexpr size_t HOST_BUFFER_SIZE_PER_RANK = 1LL * 1024 * 1024;
  int32_t* buf_;
};

}  // namespace detail
}  // end namespace comms
}  // end namespace raft
