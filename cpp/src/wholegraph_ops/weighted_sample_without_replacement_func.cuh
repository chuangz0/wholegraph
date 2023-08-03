/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <cub/device/device_radix_sort.cuh>
#include <driver_types.h>
#include <raft/matrix/select_k.cuh>
#include <random>
#include <thrust/scan.h>

#include <raft/util/integer_utils.hpp>
#include <wholememory/device_reference.cuh>
#include <wholememory/env_func_ptrs.h>
#include <wholememory/global_reference.h>
#include <wholememory/tensor_description.h>

#include "raft/matrix/detail/select_warpsort.cuh"
#include "raft/util/cuda_dev_essentials.cuh"
#include "wholememory_ops/output_memory_handle.hpp"
#include "wholememory_ops/raft_random.cuh"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"

#include "block_radix_topk.cuh"
#include "cuda_macros.hpp"
#include "error.hpp"
#include "sample_comm.cuh"

namespace wholegraph_ops {

#define USE_RAFT_WARP_SORT  1
#define USE_RAFT_RADIX_TOPK 1

template <typename WeightType>
__device__ __forceinline__ float gen_key_from_weight(const WeightType weight, PCGenerator& rng)
{
  float u              = -rng.next_float(1.0f, 0.5f);
  uint64_t random_num2 = 0;
  int seed_count       = -1;
  do {
    rng.next(random_num2);
    seed_count++;
  } while (!random_num2);
  int one_bit = __clzll(random_num2) + seed_count * 64;
  u *= exp2f(-one_bit);
  float logk = (log1pf(u) / logf(2.0)) * (1.0f / (float)weight);
  // u = random_uniform(0,1), logk = 1/weight *logf(u)
  return logk;
}

template <typename IdType,
          typename LocalIdType,
          typename WeightType,
          typename WeightKeyType,
          typename WMIdType,
          typename WMOffsetType,
          typename WMWeightType,
          unsigned int BLOCK_SIZE,
          bool NeedRandom = true,
          bool Ascending  = false>
__launch_bounds__(BLOCK_SIZE) __global__ void weighted_sample_without_replacement_large_kernel(
  wholememory_gref_t wm_csr_row_ptr,
  wholememory_array_description_t wm_csr_row_ptr_desc,
  wholememory_gref_t wm_csr_col_ptr,
  wholememory_array_description_t wm_csr_col_ptr_desc,
  wholememory_gref_t wm_csr_weight_ptr,
  wholememory_array_description_t wm_csr_weight_ptr_desc,
  const IdType* input_nodes,
  const int input_node_count,
  const int max_sample_count,
  unsigned long long random_seed,
  const int* sample_offset,
  wholememory_array_description_t sample_offset_desc,
  const int* target_neighbor_offset,
  WMIdType* output,
  int* src_lid,
  int64_t* out_edge_gid,
  WeightKeyType* weight_keys_buff)
{
  int input_idx = blockIdx.x;
  if (input_idx >= input_node_count) return;
  int gidx = threadIdx.x + blockIdx.x * BLOCK_SIZE;

  wholememory::device_reference<WMOffsetType> csr_row_ptr_gen(wm_csr_row_ptr);
  wholememory::device_reference<WMIdType> csr_col_ptr_gen(wm_csr_col_ptr);
  wholememory::device_reference<WMWeightType> csr_weight_ptr_gen(wm_csr_weight_ptr);
  IdType nid         = input_nodes[input_idx];
  int64_t start      = csr_row_ptr_gen[nid];
  int64_t end        = csr_row_ptr_gen[nid + 1];
  int neighbor_count = (int)(end - start);

  WeightKeyType* weight_keys_local_buff = weight_keys_buff + target_neighbor_offset[input_idx];
  int offset                            = sample_offset[input_idx];
  if (neighbor_count <= max_sample_count) {
    for (int sample_id = threadIdx.x; sample_id < neighbor_count; sample_id += BLOCK_SIZE) {
      int neighbor_idx           = sample_id;
      int original_neighbor_idx  = neighbor_idx;
      IdType gid                 = csr_col_ptr_gen[start + original_neighbor_idx];
      output[offset + sample_id] = gid;
      if (src_lid) src_lid[offset + sample_id] = (LocalIdType)input_idx;
      if (out_edge_gid)
        out_edge_gid[offset + sample_id] = static_cast<int64_t>(start + original_neighbor_idx);
    }
    return;
  }

  PCGenerator rng(random_seed, (uint64_t)gidx, (uint64_t)0);
  for (int id = threadIdx.x; id < neighbor_count; id += BLOCK_SIZE) {
    WeightType thread_weight = csr_weight_ptr_gen[start + id];
    weight_keys_local_buff[id] =
      NeedRandom ? static_cast<WeightKeyType>(gen_key_from_weight(thread_weight, rng))
                 : (static_cast<WeightKeyType>(thread_weight));
  }

  __syncthreads();

  WeightKeyType topk_val;
  bool topk_is_unique;

  using BlockRadixSelectT =
    std::conditional_t<Ascending,
                       BlockRadixTopKGlobalMemory<WeightKeyType, BLOCK_SIZE, false>,
                       BlockRadixTopKGlobalMemory<WeightKeyType, BLOCK_SIZE, true>>;
  __shared__ typename BlockRadixSelectT::TempStorage share_storage;

  BlockRadixSelectT{share_storage}.radixTopKGetThreshold(
    weight_keys_local_buff, max_sample_count, neighbor_count, topk_val, topk_is_unique);
  __shared__ int cnt;

  if (threadIdx.x == 0) { cnt = 0; }
  __syncthreads();

  for (int i = threadIdx.x; i < max_sample_count; i += BLOCK_SIZE) {
    if (src_lid) src_lid[offset + i] = (LocalIdType)input_idx;
  }

  // We use atomicAdd 1 operations instead of binaryScan to calculate the write
  // index, since we do not need to keep the relative positions of element.

  if (topk_is_unique) {
    for (int neighbor_idx = threadIdx.x; neighbor_idx < neighbor_count;
         neighbor_idx += BLOCK_SIZE) {
      WeightKeyType key = weight_keys_local_buff[neighbor_idx];
      bool has_topk     = Ascending ? (key <= topk_val) : (key >= topk_val);

      if (has_topk) {
        int write_index                = atomicAdd(&cnt, 1);
        LocalIdType local_original_idx = neighbor_idx;
        output[offset + write_index]   = csr_col_ptr_gen[start + local_original_idx];
        if (out_edge_gid)
          out_edge_gid[offset + write_index] = static_cast<IdType>(start + local_original_idx);
      }
    }
  } else {
    for (int neighbor_idx = threadIdx.x; neighbor_idx < neighbor_count;
         neighbor_idx += BLOCK_SIZE) {
      WeightKeyType key = weight_keys_local_buff[neighbor_idx];
      bool has_topk     = Ascending ? (key < topk_val) : (key > topk_val);

      if (has_topk) {
        int write_index                = atomicAdd(&cnt, 1);
        LocalIdType local_original_idx = neighbor_idx;
        output[offset + write_index]   = csr_col_ptr_gen[start + local_original_idx];
        if (out_edge_gid)
          out_edge_gid[offset + write_index] = static_cast<IdType>(start + local_original_idx);
      }
    }
    __syncthreads();
    for (int neighbor_idx = threadIdx.x; neighbor_idx < neighbor_count;
         neighbor_idx += BLOCK_SIZE) {
      WeightKeyType key = weight_keys_local_buff[neighbor_idx];
      bool has_topk     = (key == topk_val);

      if (has_topk) {
        int write_index = atomicAdd(&cnt, 1);
        if (write_index >= max_sample_count) break;
        LocalIdType local_original_idx = neighbor_idx;
        output[offset + write_index]   = csr_col_ptr_gen[start + local_original_idx];
        if (out_edge_gid)
          out_edge_gid[offset + write_index] = static_cast<IdType>(start + local_original_idx);
      }
    }
  }
}

template <typename T, typename IdxT>
__device__ __host__ void set_buf_pointers(T* buf1,
                                          IdxT* idx_buf1,
                                          T* buf2,
                                          IdxT* idx_buf2,
                                          int pass,
                                          const T*& in_buf,
                                          const IdxT*& in_idx_buf,
                                          T*& out_buf,
                                          IdxT*& out_idx_buf)
{
  if (pass == 0) {
    in_buf      = buf1;
    in_idx_buf  = nullptr;
    out_buf     = nullptr;
    out_idx_buf = nullptr;

  } else if (pass % 2 == 0) {
    in_buf      = buf2;
    in_idx_buf  = idx_buf2;
    out_buf     = buf1;
    out_idx_buf = idx_buf1;
  } else {
    in_buf      = buf1;
    in_idx_buf  = idx_buf1;
    out_buf     = buf2;
    out_idx_buf = idx_buf2;
  }
}

template <typename IdType,
          typename LocalIdType,
          typename WeightType,
          typename WeightKeyType,
          typename WMIdType,
          typename WMOffsetType,
          typename WMWeightType,
          unsigned int BLOCK_SIZE,
          int BitsPerPass,
          bool NeedRandom = true>
__launch_bounds__(BLOCK_SIZE) __global__
  void weighted_sample_without_replacement_large_raft_radix_kernel(
    wholememory_gref_t wm_csr_row_ptr,
    wholememory_array_description_t wm_csr_row_ptr_desc,
    wholememory_gref_t wm_csr_col_ptr,
    wholememory_array_description_t wm_csr_col_ptr_desc,
    wholememory_gref_t wm_csr_weight_ptr,
    wholememory_array_description_t wm_csr_weight_ptr_desc,
    const IdType* input_nodes,
    const int input_node_count,
    const int max_sample_count,
    unsigned long long random_seed,
    const int* sample_offset,
    wholememory_array_description_t sample_offset_desc,
    const int* target_neighbor_offset,
    WMIdType* output,
    int* src_lid,
    int64_t* out_edge_gid,
    WeightKeyType* weight_keys_buff0,
    LocalIdType* local_idx_buff0,
    WeightKeyType* weight_keys_buff1,
    LocalIdType* local_idx_buff1,
    WeightKeyType* weight_keys_out,
    LocalIdType* local_idx_out,
    const bool select_min = false)
{
  int input_idx = blockIdx.x;
  if (input_idx >= input_node_count) return;
  int gidx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  wholememory::device_reference<WMOffsetType> csr_row_ptr_gen(wm_csr_row_ptr);
  wholememory::device_reference<WMIdType> csr_col_ptr_gen(wm_csr_col_ptr);
  wholememory::device_reference<WMWeightType> csr_weight_ptr_gen(wm_csr_weight_ptr);
  IdType nid         = input_nodes[input_idx];
  int64_t start      = csr_row_ptr_gen[nid];
  int64_t end        = csr_row_ptr_gen[nid + 1];
  int neighbor_count = (int)(end - start);
  int offset         = sample_offset[input_idx];
  if (neighbor_count <= max_sample_count) {
    for (int sample_id = threadIdx.x; sample_id < neighbor_count; sample_id += BLOCK_SIZE) {
      int neighbor_idx           = sample_id;
      int original_neighbor_idx  = neighbor_idx;
      IdType gid                 = csr_col_ptr_gen[start + original_neighbor_idx];
      output[offset + sample_id] = gid;
      if (src_lid) src_lid[offset + sample_id] = (LocalIdType)input_idx;
      if (out_edge_gid)
        out_edge_gid[offset + sample_id] = static_cast<int64_t>(start + original_neighbor_idx);
    }
    return;
  }
  PCGenerator rng(random_seed, (uint64_t)gidx, (uint64_t)0);
  int buff_offset = target_neighbor_offset[input_idx];
  weight_keys_buff0 += buff_offset;
  local_idx_buff0 += buff_offset;
  weight_keys_buff1 += buff_offset;
  local_idx_buff1 += buff_offset;
  weight_keys_out += input_idx * max_sample_count;
  local_idx_out += input_idx * max_sample_count;

  //
  for (int id = threadIdx.x; id < neighbor_count; id += BLOCK_SIZE) {
    WeightType thread_weight = csr_weight_ptr_gen[start + id];
    weight_keys_buff0[id]    = NeedRandom
                                 ? static_cast<WeightKeyType>(gen_key_from_weight(thread_weight, rng))
                                 : (static_cast<WeightKeyType>(thread_weight));
    local_idx_buff0[id]      = id;
  }
  // __syncthreads();

  constexpr int num_buckets =
    raft::matrix::detail::select::radix::impl::calc_num_buckets<BitsPerPass>();
  __shared__ raft::matrix::detail::select::radix::impl::Counter<WeightKeyType, LocalIdType> counter;
  __shared__ LocalIdType histogram[num_buckets];
  if (threadIdx.x == 0) {
    counter.k              = max_sample_count;
    counter.len            = neighbor_count;
    counter.previous_len   = neighbor_count;
    counter.kth_value_bits = 0;
    counter.out_cnt        = 0;
    counter.out_back_cnt   = 0;
  }
  __syncthreads();
  const WeightKeyType* in_buf   = nullptr;
  const LocalIdType* in_idx_buf = nullptr;
  WeightKeyType* out_buf        = nullptr;
  LocalIdType* out_idx_buf      = nullptr;
  constexpr int num_passes =
    raft::matrix::detail::select::radix::impl::calc_num_passes<WeightKeyType, BitsPerPass>();
  for (int pass = 0; pass < num_passes; ++pass) {
    set_buf_pointers(weight_keys_buff0,
                     local_idx_buff0,
                     weight_keys_buff1,
                     local_idx_buff1,
                     pass,
                     in_buf,
                     in_idx_buf,
                     out_buf,
                     out_idx_buf);
    LocalIdType current_len = counter.len;
    LocalIdType current_k   = counter.k;
    raft::matrix::detail::select::radix::impl::
      filter_and_histogram_for_one_block<WeightKeyType, LocalIdType, BitsPerPass>(in_buf,
                                                                                  in_idx_buf,
                                                                                  out_buf,
                                                                                  out_idx_buf,
                                                                                  weight_keys_out,
                                                                                  local_idx_out,
                                                                                  &counter,
                                                                                  histogram,
                                                                                  select_min,
                                                                                  pass);
    __syncthreads();

    raft::matrix::detail::select::radix::impl::scan<LocalIdType, BitsPerPass, BLOCK_SIZE>(
      histogram);
    __syncthreads();

    raft::matrix::detail::select::radix::impl::
      choose_bucket<WeightKeyType, LocalIdType, BitsPerPass>(&counter, histogram, current_k, pass);
    if (threadIdx.x == 0) { counter.previous_len = current_len; }
    __syncthreads();

    if (counter.len == counter.k || pass == num_passes - 1) {
      raft::matrix::detail::select::radix::impl::
        last_filter<WeightKeyType, LocalIdType, BitsPerPass>(
          pass == 0 ? weight_keys_buff0 : out_buf,
          pass == 0 ? local_idx_buff0 : out_idx_buf,
          weight_keys_out,
          local_idx_out,
          current_len,
          max_sample_count,
          &counter,
          select_min,
          pass);
      break;
    }
  }
  // topk  idx in local_idx_out
  __syncthreads();
  for (int sample_id = threadIdx.x; sample_id < max_sample_count; sample_id += BLOCK_SIZE) {
    int original_neighbor_idx  = local_idx_out[sample_id];
    IdType gid                 = csr_col_ptr_gen[start + original_neighbor_idx];
    output[offset + sample_id] = gid;
    if (src_lid) src_lid[offset + sample_id] = (LocalIdType)input_idx;
    if (out_edge_gid)
      out_edge_gid[offset + sample_id] = static_cast<int64_t>(start + original_neighbor_idx);
  }
}

template <typename IdType, typename WMOffsetType, bool NeedNeighbor = false>
__global__ void get_sample_count_and_neighbor_count_without_replacement_kernel(
  wholememory_gref_t wm_csr_row_ptr,
  wholememory_array_description_t wm_csr_row_ptr_desc,
  const IdType* input_nodes,
  const int input_node_count,
  int* tmp_sample_count_mem_pointer,
  int* tmp_neighbor_count_mem_pointer,
  const int max_sample_count)
{
  int gidx      = threadIdx.x + blockIdx.x * blockDim.x;
  int input_idx = gidx;
  if (input_idx >= input_node_count) return;
  IdType nid = input_nodes[input_idx];
  wholememory::device_reference<WMOffsetType> wm_csr_row_ptr_dev_ref(wm_csr_row_ptr);
  int64_t start      = wm_csr_row_ptr_dev_ref[nid];
  int64_t end        = wm_csr_row_ptr_dev_ref[nid + 1];
  int neighbor_count = (int)(end - start);
  // sample_count <= 0 means sample all.
  int sample_count = neighbor_count;
  if (max_sample_count > 0) { sample_count = min(neighbor_count, max_sample_count); }
  tmp_sample_count_mem_pointer[input_idx] = sample_count;
  if (NeedNeighbor) {
    tmp_neighbor_count_mem_pointer[input_idx] =
      (neighbor_count <= max_sample_count) ? 0 : neighbor_count;
  }
}

// A-RES algorithmn
// https://en.wikipedia.org/wiki/Reservoir_sampling#Algorithm_A-Res
// max_sample_count should <=(BLOCK_SIZE*ITEMS_PER_THREAD*/4)  otherwise,need to
// change the template parameters of BlockRadixTopK.
template <typename IdType,
          typename LocalIdType,
          typename WeightType,
          typename WMIdType,
          typename WMOffsetType,
          typename WMWeightType,
          unsigned int ITEMS_PER_THREAD,
          unsigned int BLOCK_SIZE,
          bool NeedRandom = true,
          bool Ascending  = false>
__launch_bounds__(BLOCK_SIZE) __global__ void weighted_sample_without_replacement_kernel(
  wholememory_gref_t wm_csr_row_ptr,
  wholememory_array_description_t wm_csr_row_ptr_desc,
  wholememory_gref_t wm_csr_col_ptr,
  wholememory_array_description_t wm_csr_col_ptr_desc,
  wholememory_gref_t wm_csr_weight_ptr,
  wholememory_array_description_t wm_csr_weight_ptr_desc,
  const IdType* input_nodes,
  const int input_node_count,
  const int max_sample_count,
  unsigned long long random_seed,
  const int* sample_offset,
  wholememory_array_description_t sample_offset_desc,
  WMIdType* output,
  int* src_lid,
  int64_t* out_edge_gid)
{
  int input_idx = blockIdx.x;
  if (input_idx >= input_node_count) return;
  int gidx = threadIdx.x + blockIdx.x * BLOCK_SIZE;

  wholememory::device_reference<WMOffsetType> csr_row_ptr_gen(wm_csr_row_ptr);
  wholememory::device_reference<WMIdType> csr_col_ptr_gen(wm_csr_col_ptr);
  wholememory::device_reference<WMWeightType> csr_weight_ptr_gen(wm_csr_weight_ptr);

  IdType nid         = input_nodes[input_idx];
  int64_t start      = csr_row_ptr_gen[nid];
  int64_t end        = csr_row_ptr_gen[nid + 1];
  int neighbor_count = (int)(end - start);
  int offset         = sample_offset[input_idx];
  if (neighbor_count <= max_sample_count) {
    for (int sample_id = threadIdx.x; sample_id < neighbor_count; sample_id += BLOCK_SIZE) {
      int neighbor_idx           = sample_id;
      int original_neighbor_idx  = neighbor_idx;
      IdType gid                 = csr_col_ptr_gen[start + original_neighbor_idx];
      output[offset + sample_id] = gid;
      if (src_lid) src_lid[offset + sample_id] = (LocalIdType)input_idx;
      if (out_edge_gid)
        out_edge_gid[offset + sample_id] = static_cast<int64_t>(start + original_neighbor_idx);
    }
    return;
  } else {
    PCGenerator rng(random_seed, (uint64_t)gidx, (uint64_t)0);

    float weight_keys[ITEMS_PER_THREAD];
    int neighbor_idxs[ITEMS_PER_THREAD];

    using BlockRadixTopKT =
      std::conditional_t<Ascending,
                         BlockRadixTopKRegister<float, BLOCK_SIZE, ITEMS_PER_THREAD, false, int>,
                         BlockRadixTopKRegister<float, BLOCK_SIZE, ITEMS_PER_THREAD, true, int>>;

    __shared__ typename BlockRadixTopKT::TempStorage sort_tmp_storage;

    const int tx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      int idx = BLOCK_SIZE * i + tx;
      if (idx < neighbor_count) {
        WeightType thread_weight = csr_weight_ptr_gen[start + idx];
        weight_keys[i] =
          NeedRandom ? gen_key_from_weight(thread_weight, rng) : (float)thread_weight;
        neighbor_idxs[i] = idx;
      }
    }
    const int valid_count = (neighbor_count < (BLOCK_SIZE * ITEMS_PER_THREAD))
                              ? neighbor_count
                              : (BLOCK_SIZE * ITEMS_PER_THREAD);
    BlockRadixTopKT{sort_tmp_storage}.radixTopKToStriped(
      weight_keys, neighbor_idxs, max_sample_count, valid_count);
    __syncthreads();
    const int stride = BLOCK_SIZE * ITEMS_PER_THREAD - max_sample_count;

    for (int idx_offset = ITEMS_PER_THREAD * BLOCK_SIZE; idx_offset < neighbor_count;
         idx_offset += stride) {
#pragma unroll
      for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int local_idx = BLOCK_SIZE * i + tx - max_sample_count;
        // [0,BLOCK_SIZE*ITEMS_PER_THREAD-max_sample_count)
        int target_idx = idx_offset + local_idx;
        if (local_idx >= 0 && target_idx < neighbor_count) {
          WeightType thread_weight = csr_weight_ptr_gen[start + target_idx];
          weight_keys[i] =
            NeedRandom ? gen_key_from_weight(thread_weight, rng) : (float)thread_weight;
          neighbor_idxs[i] = target_idx;
        }
      }
      const int iter_valid_count = ((neighbor_count - idx_offset) >= stride)
                                     ? (BLOCK_SIZE * ITEMS_PER_THREAD)
                                     : (max_sample_count + neighbor_count - idx_offset);
      BlockRadixTopKT{sort_tmp_storage}.radixTopKToStriped(
        weight_keys, neighbor_idxs, max_sample_count, iter_valid_count);
      __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      int idx = i * BLOCK_SIZE + tx;
      if (idx < max_sample_count) {
        if (src_lid) src_lid[offset + idx] = (LocalIdType)input_idx;
        LocalIdType local_original_idx = neighbor_idxs[i];
        output[offset + idx]           = csr_col_ptr_gen[start + local_original_idx];
        if (out_edge_gid)
          out_edge_gid[offset + idx] = static_cast<int64_t>(start + local_original_idx);
      }
    }
  }
}

struct null_store_t {};
// to  avoid queue.store()  store keys or values in output.

struct null_store_op {
  template <typename Type, typename... UnusedArgs>
  constexpr auto operator()(const Type& in, UnusedArgs...) const
  {
    return null_store_t{};
  }
};

template <template <int, bool, typename, typename> class WarpSortClass,
          int Capacity,
          typename IdType,
          typename LocalIdType,
          typename WeightType,
          typename WMIdType,
          typename WMOffsetType,
          typename WMWeightType,
          bool NEED_RANDOM = true,
          bool ASCENDING   = false>
__launch_bounds__(256) __global__ void weighted_sample_without_replacement_raft_kernel(
  wholememory_gref_t wm_csr_row_ptr,
  wholememory_array_description_t wm_csr_row_ptr_desc,
  wholememory_gref_t wm_csr_col_ptr,
  wholememory_array_description_t wm_csr_col_ptr_desc,
  wholememory_gref_t wm_csr_weight_ptr,
  wholememory_array_description_t wm_csr_weight_ptr_desc,
  const IdType* input_nodes,
  const int input_node_count,
  const int max_sample_count,
  unsigned long long random_seed,
  const int* sample_offset,
  wholememory_array_description_t sample_offset_desc,
  WMIdType* output,
  int* src_lid,
  int64_t* out_edge_gid)
{
  int input_idx = blockIdx.x;
  if (input_idx >= input_node_count) return;
  int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  wholememory::device_reference<WMOffsetType> csr_row_ptr_gen(wm_csr_row_ptr);
  wholememory::device_reference<WMIdType> csr_col_ptr_gen(wm_csr_col_ptr);
  wholememory::device_reference<WMWeightType> csr_weight_ptr_gen(wm_csr_weight_ptr);

  IdType nid         = input_nodes[input_idx];
  int64_t start      = csr_row_ptr_gen[nid];
  int64_t end        = csr_row_ptr_gen[nid + 1];
  int neighbor_count = static_cast<int>(end - start);
  int offset         = sample_offset[input_idx];
  if (neighbor_count <= max_sample_count) {
    for (int sample_id = threadIdx.x; sample_id < neighbor_count; sample_id += blockDim.x) {
      int neighbor_idx           = sample_id;
      int original_neighbor_idx  = neighbor_idx;
      IdType gid                 = csr_col_ptr_gen[start + original_neighbor_idx];
      output[offset + sample_id] = gid;
      if (src_lid) src_lid[offset + sample_id] = input_idx;
      if (out_edge_gid)
        out_edge_gid[offset + sample_id] = static_cast<int64_t>(start + original_neighbor_idx);
    }
    return;
  } else {
    // assert k<=Capacity
    extern __shared__ __align__(256) uint8_t smem_buf_bytes[];
    using bq_t = raft::matrix::detail::select::warpsort::
      block_sort<WarpSortClass, Capacity, ASCENDING, float, int>;

    uint8_t* warp_smem = bq_t::queue_t::mem_required(blockDim.x) > 0 ? smem_buf_bytes : nullptr;
    bq_t queue(max_sample_count, warp_smem);
    PCGenerator rng(random_seed, static_cast<uint64_t>(gidx), static_cast<uint64_t>(0));
    const int per_thread_lim = neighbor_count + raft::laneId();
    for (int idx = threadIdx.x; idx < per_thread_lim; idx += blockDim.x) {
      float weight_key = WarpSortClass<Capacity, ASCENDING, float, int>::kDummy;
      if (idx < neighbor_count) {
        WeightType thread_weight = csr_weight_ptr_gen[start + idx];
        weight_key = NEED_RANDOM ? gen_key_from_weight(thread_weight, rng) : (float)thread_weight;
      }
      queue.add(weight_key, idx);
    }
    queue.done(smem_buf_bytes);

    __syncthreads();
    int* topk_idx_smem = reinterpret_cast<int*>(smem_buf_bytes);
    queue.store(static_cast<null_store_t*>(nullptr), topk_idx_smem, null_store_op{});
    __syncthreads();
    for (int idx = threadIdx.x; idx < max_sample_count; idx += blockDim.x) {
      LocalIdType local_original_idx = static_cast<LocalIdType>(topk_idx_smem[idx]);
      if (src_lid) { src_lid[offset + idx] = input_idx; }
      output[offset + idx] = csr_col_ptr_gen[start + local_original_idx];
      if (out_edge_gid) {
        out_edge_gid[offset + idx] = static_cast<int64_t>(start + local_original_idx);
      }
    }
  };
}

template <template <int, bool, typename, typename> class WarpSortClass,
          int Capacity,
          typename IdType,
          typename LocalIdType,
          typename WeightType,
          typename WMIdType,
          typename WMOffsetType,
          typename WMWeightType,
          bool NEED_RANDOM = true,
          bool ASCENDING   = false>
void launch_kernel(wholememory_gref_t wm_csr_row_ptr,
                   wholememory_array_description_t wm_csr_row_ptr_desc,
                   wholememory_gref_t wm_csr_col_ptr,
                   wholememory_array_description_t wm_csr_col_ptr_desc,
                   wholememory_gref_t wm_csr_weight_ptr,
                   wholememory_array_description_t wm_csr_weight_ptr_desc,
                   const IdType* input_nodes,
                   const int input_node_count,
                   const int max_sample_count,
                   unsigned long long random_seed,
                   const int* sample_offset,
                   wholememory_array_description_t sample_offset_desc,
                   WMIdType* output,
                   int* src_lid,
                   int64_t* out_edge_gid,
                   int block_dim,
                   int smem_size,
                   cudaStream_t stream)
{
  const int capacity = raft::bound_by_power_of_two(max_sample_count);
  if constexpr (Capacity > 8) {
    if (capacity < Capacity) {
      return launch_kernel<WarpSortClass,
                           Capacity / 2,
                           IdType,
                           LocalIdType,
                           WeightType,
                           WMIdType,
                           WMOffsetType,
                           WMWeightType,
                           NEED_RANDOM,
                           ASCENDING>(wm_csr_row_ptr,
                                      wm_csr_row_ptr_desc,
                                      wm_csr_col_ptr,
                                      wm_csr_col_ptr_desc,
                                      wm_csr_weight_ptr,
                                      wm_csr_weight_ptr_desc,
                                      input_nodes,
                                      input_node_count,
                                      max_sample_count,
                                      random_seed,
                                      sample_offset,
                                      sample_offset_desc,
                                      output,
                                      src_lid,
                                      out_edge_gid,
                                      block_dim,
                                      smem_size,
                                      stream);
    }
  }
  WHOLEMEMORY_EXPECTS(
    capacity <= Capacity, "Requested max_sample_count is too large (%d)", max_sample_count);
  smem_size = std::max<int>(smem_size, WarpSortClass<1, true, float, int>::mem_required(block_dim));
  weighted_sample_without_replacement_raft_kernel<WarpSortClass,
                                                  Capacity,
                                                  IdType,
                                                  LocalIdType,
                                                  WeightType,
                                                  WMIdType,
                                                  WMOffsetType,
                                                  WMWeightType,
                                                  NEED_RANDOM,
                                                  ASCENDING>
    <<<input_node_count, block_dim, smem_size, stream>>>(wm_csr_row_ptr,
                                                         wm_csr_row_ptr_desc,
                                                         wm_csr_col_ptr,
                                                         wm_csr_col_ptr_desc,
                                                         wm_csr_weight_ptr,
                                                         wm_csr_weight_ptr_desc,
                                                         input_nodes,
                                                         input_node_count,
                                                         max_sample_count,
                                                         random_seed,
                                                         sample_offset,
                                                         sample_offset_desc,
                                                         output,
                                                         src_lid,
                                                         out_edge_gid);
}

template <int Capacity, bool Ascending, class T, class IdxT>
using WarpSortClassT =
  raft::matrix::detail::select::warpsort::warp_sort_filtered<Capacity, Ascending, T, IdxT>;

template <typename IdType, typename WMIdType, typename WeightType>
void wholegraph_csr_weighted_sample_without_replacement_func(
  wholememory_gref_t wm_csr_row_ptr,
  wholememory_array_description_t wm_csr_row_ptr_desc,
  wholememory_gref_t wm_csr_col_ptr,
  wholememory_array_description_t wm_csr_col_ptr_desc,
  wholememory_gref_t wm_csr_weight_ptr,
  wholememory_array_description_t wm_csr_weight_ptr_desc,
  void* center_nodes,
  wholememory_array_description_t center_nodes_desc,
  int max_sample_count,
  void* output_sample_offset,
  wholememory_array_description_t output_sample_offset_desc,
  void* output_dest_memory_context,
  void* output_center_localid_memory_context,
  void* output_edge_gid_memory_context,
  unsigned long long random_seed,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  int center_node_count = center_nodes_desc.size;

  WHOLEMEMORY_EXPECTS(wm_csr_row_ptr_desc.dtype == WHOLEMEMORY_DT_INT64,
                      "wholegraph_csr_unweighted_sample_without_replacement_func(). "
                      "wm_csr_row_ptr_desc.dtype != WHOLEMEMORY_DT_INT, "
                      "wm_csr_row_ptr_desc.dtype = %d",
                      wm_csr_row_ptr_desc.dtype);

  WHOLEMEMORY_EXPECTS(output_sample_offset_desc.dtype == WHOLEMEMORY_DT_INT,
                      "wholegraph_csr_unweighted_sample_without_replacement_func(). "
                      "output_sample_offset_desc.dtype != WHOLEMEMORY_DT_INT, "
                      "output_sample_offset_desc.dtype = %d",
                      output_sample_offset_desc.dtype);
#if USE_RAFT_WARP_SORT
  constexpr int SAMPLE_COUNT_THRESHOLD = raft::matrix::detail::select::warpsort::kMaxCapacity;
  //  constexpr int SAMPLE_COUNT_THRESHOLD =16;
#else
  constexpr int SAMPLE_COUNT_THRESHOLD = 1024;
#endif
  const bool need_neighbor_count = max_sample_count > SAMPLE_COUNT_THRESHOLD;

  wholememory_ops::temp_memory_handle gen_buffer_tmh(p_env_fns);
  int* tmp_sample_count_mem_pointer =
    (int*)gen_buffer_tmh.device_malloc(center_node_count + 1, WHOLEMEMORY_DT_INT);

  wholememory_ops::temp_memory_handle gen_neighbor_count_buffer_tmh(p_env_fns);
  int* tmp_neighbor_counts_mem_pointer = nullptr;

  int thread_x    = 32;
  int block_count = raft::div_rounding_up_safe<int>(center_node_count, thread_x);

  if (need_neighbor_count) {
    tmp_neighbor_counts_mem_pointer =
      (int*)gen_neighbor_count_buffer_tmh.device_malloc(center_node_count + 1, WHOLEMEMORY_DT_INT);
    get_sample_count_and_neighbor_count_without_replacement_kernel<IdType, int64_t, true>
      <<<block_count, thread_x, 0, stream>>>(wm_csr_row_ptr,
                                             wm_csr_row_ptr_desc,
                                             (const IdType*)center_nodes,
                                             center_node_count,
                                             tmp_sample_count_mem_pointer,
                                             tmp_neighbor_counts_mem_pointer,
                                             max_sample_count);
  } else {
    get_sample_count_and_neighbor_count_without_replacement_kernel<IdType, int64_t, false>
      <<<block_count, thread_x, 0, stream>>>(wm_csr_row_ptr,
                                             wm_csr_row_ptr_desc,
                                             (const IdType*)center_nodes,
                                             center_node_count,
                                             tmp_sample_count_mem_pointer,
                                             tmp_neighbor_counts_mem_pointer,
                                             max_sample_count);
  }

  // prefix sum
  wholememory_ops::wm_thrust_allocator thrust_allocator(p_env_fns);
  thrust::exclusive_scan(thrust::cuda::par(thrust_allocator).on(stream),
                         tmp_sample_count_mem_pointer,
                         tmp_sample_count_mem_pointer + center_node_count + 1,
                         (int*)output_sample_offset);

  int count;
  WM_CUDA_CHECK(cudaMemcpyAsync(&count,
                                ((int*)output_sample_offset) + center_node_count,
                                sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream));
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));

  wholememory_ops::output_memory_handle gen_output_dest_buffer_mh(p_env_fns,
                                                                  output_dest_memory_context);
  WMIdType* output_dest_node_ptr =
    (WMIdType*)gen_output_dest_buffer_mh.device_malloc(count, wm_csr_col_ptr_desc.dtype);

  int* output_center_localid_ptr = nullptr;
  if (output_center_localid_memory_context) {
    wholememory_ops::output_memory_handle gen_output_center_localid_buffer_mh(
      p_env_fns, output_center_localid_memory_context);
    output_center_localid_ptr =
      (int*)gen_output_center_localid_buffer_mh.device_malloc(count, WHOLEMEMORY_DT_INT);
  }

  int64_t* output_edge_gid_ptr = nullptr;
  if (output_edge_gid_memory_context) {
    wholememory_ops::output_memory_handle gen_output_edge_gid_buffer_mh(
      p_env_fns, output_edge_gid_memory_context);
    output_edge_gid_ptr =
      (int64_t*)gen_output_edge_gid_buffer_mh.device_malloc(count, WHOLEMEMORY_DT_INT64);
  }

  if (max_sample_count > SAMPLE_COUNT_THRESHOLD) {
    wholememory_ops::wm_thrust_allocator tmp_thrust_allocator(p_env_fns);
    thrust::exclusive_scan(thrust::cuda::par(tmp_thrust_allocator).on(stream),
                           tmp_neighbor_counts_mem_pointer,
                           tmp_neighbor_counts_mem_pointer + center_node_count + 1,
                           tmp_neighbor_counts_mem_pointer);
    int* tmp_neighbor_counts_offset = tmp_neighbor_counts_mem_pointer;
    int target_neighbor_counts;
    WM_CUDA_CHECK(cudaMemcpyAsync(&target_neighbor_counts,
                                  ((int*)tmp_neighbor_counts_mem_pointer) + center_node_count,
                                  sizeof(int),
                                  cudaMemcpyDeviceToHost,
                                  stream));
    WM_CUDA_CHECK(cudaStreamSynchronize(stream));
#if USE_RAFT_RADIX_TOPK

    // printf(
    //   "**************** use large kernel  "
    //   "weighted_sample_without_replacement_large_raft_radix_kernel***********************\n");

    wholememory_ops::temp_memory_handle gen_weights_buffer0_tmh(p_env_fns);
    WeightType* tmp_weights_buffer0_mem_pointer =
      (WeightType*)gen_weights_buffer0_tmh.device_malloc(target_neighbor_counts,
                                                         wm_csr_weight_ptr_desc.dtype);
    wholememory_ops::temp_memory_handle gen_weights_buffer1_tmh(p_env_fns);
    WeightType* tmp_weights_buffer1_mem_pointer =
      (WeightType*)gen_weights_buffer1_tmh.device_malloc(target_neighbor_counts,
                                                         wm_csr_weight_ptr_desc.dtype);
    wholememory_ops::temp_memory_handle gen_weights_buffer_out_tmh(p_env_fns);
    WeightType* tmp_weights_buffer_out_mem_pointer =
      (WeightType*)gen_weights_buffer_out_tmh.device_malloc(center_node_count * max_sample_count,
                                                            wm_csr_weight_ptr_desc.dtype);

    auto local_idx_dtype = wholememory_dtype_t::WHOLEMEMORY_DT_INT;
    wholememory_ops::temp_memory_handle local_idx_buffer0_tmh(p_env_fns);
    int* local_idx_buffer0_mem_pointer = static_cast<int*>(
      local_idx_buffer0_tmh.device_malloc(target_neighbor_counts, local_idx_dtype));
    wholememory_ops::temp_memory_handle local_idx_buffer1_tmh(p_env_fns);
    int* local_idx_buffer1_mem_pointer = static_cast<int*>(
      local_idx_buffer1_tmh.device_malloc(target_neighbor_counts, local_idx_dtype));
    wholememory_ops::temp_memory_handle local_idx_buffer_out_tmh(p_env_fns);
    int* local_idx_buffer_out_mem_pointer =
      static_cast<int*>(local_idx_buffer_out_tmh.device_malloc(center_node_count * max_sample_count,
                                                               local_idx_dtype));
    constexpr int BLOCK_SIZE  = 256;
    constexpr int BitsPerPass = 8;
    weighted_sample_without_replacement_large_raft_radix_kernel<IdType,
                                                                int,
                                                                WeightType,
                                                                WeightType,
                                                                WMIdType,
                                                                int64_t,
                                                                WeightType,
                                                                BLOCK_SIZE,
                                                                BitsPerPass>
      <<<center_node_count, BLOCK_SIZE, 0, stream>>>(wm_csr_row_ptr,
                                                     wm_csr_row_ptr_desc,
                                                     wm_csr_col_ptr,
                                                     wm_csr_col_ptr_desc,
                                                     wm_csr_weight_ptr,
                                                     wm_csr_weight_ptr_desc,
                                                     (const IdType*)center_nodes,
                                                     center_node_count,
                                                     max_sample_count,
                                                     random_seed,
                                                     (const int*)output_sample_offset,
                                                     output_sample_offset_desc,
                                                     tmp_neighbor_counts_offset,
                                                     (WMIdType*)output_dest_node_ptr,
                                                     (int*)output_center_localid_ptr,
                                                     (int64_t*)output_edge_gid_ptr,
                                                     tmp_weights_buffer0_mem_pointer,
                                                     local_idx_buffer0_mem_pointer,
                                                     tmp_weights_buffer1_mem_pointer,
                                                     local_idx_buffer1_mem_pointer,
                                                     tmp_weights_buffer_out_mem_pointer,
                                                     local_idx_buffer_out_mem_pointer,
                                                     false);
#else
    wholememory_ops::temp_memory_handle gen_weights_buffer_tmh(p_env_fns);
    WeightType* tmp_weights_buffer_mem_pointer = (WeightType*)gen_weights_buffer_tmh.device_malloc(
      target_neighbor_counts, wm_csr_weight_ptr_desc.dtype);

    constexpr int BLOCK_SIZE = 256;
    weighted_sample_without_replacement_large_kernel<IdType,
                                                     int,
                                                     WeightType,
                                                     WeightType,
                                                     WMIdType,
                                                     int64_t,
                                                     WeightType,
                                                     BLOCK_SIZE>
      <<<center_node_count, BLOCK_SIZE, 0, stream>>>(wm_csr_row_ptr,
                                                     wm_csr_row_ptr_desc,
                                                     wm_csr_col_ptr,
                                                     wm_csr_col_ptr_desc,
                                                     wm_csr_weight_ptr,
                                                     wm_csr_weight_ptr_desc,
                                                     (const IdType*)center_nodes,
                                                     center_node_count,
                                                     max_sample_count,
                                                     random_seed,
                                                     (const int*)output_sample_offset,
                                                     output_sample_offset_desc,
                                                     tmp_neighbor_counts_offset,
                                                     (WMIdType*)output_dest_node_ptr,
                                                     (int*)output_center_localid_ptr,
                                                     (int64_t*)output_edge_gid_ptr,
                                                     tmp_weights_buffer_mem_pointer);
#endif
    WM_CUDA_CHECK(cudaGetLastError());
    WM_CUDA_CHECK(cudaStreamSynchronize(stream));
    return;
  }

  if (max_sample_count <= 0) {
    sample_all_kernel<IdType, int, WMIdType, int64_t>
      <<<center_node_count, 64, 0, stream>>>(wm_csr_row_ptr,
                                             wm_csr_row_ptr_desc,
                                             wm_csr_col_ptr,
                                             wm_csr_col_ptr_desc,
                                             (const IdType*)center_nodes,
                                             center_node_count,
                                             (const int*)output_sample_offset,
                                             output_sample_offset_desc,
                                             (WMIdType*)output_dest_node_ptr,
                                             (int*)output_center_localid_ptr,
                                             (int64_t*)output_edge_gid_ptr);

    WM_CUDA_CHECK(cudaGetLastError());
    WM_CUDA_CHECK(cudaStreamSynchronize(stream));
    return;
  }

#if USE_RAFT_WARP_SORT
  // https://github.com/rapidsai/raft/blob/branch-23.08/cpp/include/raft/matrix/detail/select_warpsort.cuh#L788
  // TODO:   Calculate the opt_launch_parameters according to center_node_count and the average
  // neighbours.

  constexpr int Capacity    = SAMPLE_COUNT_THRESHOLD;
  const int capacity        = raft::bound_by_power_of_two(max_sample_count);
  constexpr int block_dim   = 128;
  constexpr int num_of_warp = block_dim / raft::WarpSize;
  int smem_size = raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<float, int>(
    num_of_warp, max_sample_count);
  smem_size = raft::max(static_cast<int>(max_sample_count * sizeof(int)),
                        smem_size);  // store values of topk-result

  launch_kernel<WarpSortClassT, Capacity, IdType, int, WeightType, WMIdType, int64_t, WeightType>(
    wm_csr_row_ptr,
    wm_csr_row_ptr_desc,
    wm_csr_col_ptr,
    wm_csr_col_ptr_desc,
    wm_csr_weight_ptr,
    wm_csr_weight_ptr_desc,
    (const IdType*)center_nodes,
    center_node_count,
    max_sample_count,
    random_seed,
    (const int*)output_sample_offset,
    output_sample_offset_desc,
    (WMIdType*)output_dest_node_ptr,
    (int*)output_center_localid_ptr,
    (int64_t*)output_edge_gid_ptr,
    block_dim,
    smem_size,
    stream);

#else

  using weighted_sample_fun_type = void (*)(wholememory_gref_t,
                                            wholememory_array_description_t,
                                            wholememory_gref_t,
                                            wholememory_array_description_t,
                                            wholememory_gref_t,
                                            wholememory_array_description_t,
                                            const IdType*,
                                            const int,
                                            const int,
                                            unsigned long long,
                                            const int*,
                                            wholememory_array_description_t,
                                            WMIdType*,
                                            int*,
                                            int64_t*);

  static const weighted_sample_fun_type func_array[4] = {
    weighted_sample_without_replacement_kernel<IdType,
                                               int,
                                               WeightType,
                                               WMIdType,
                                               int64_t,
                                               WeightType,
                                               4,
                                               128>,
    weighted_sample_without_replacement_kernel<IdType,
                                               int,
                                               WeightType,
                                               WMIdType,
                                               int64_t,
                                               WeightType,
                                               4,
                                               256>,
    weighted_sample_without_replacement_kernel<IdType,
                                               int,
                                               WeightType,
                                               WMIdType,
                                               int64_t,
                                               WeightType,
                                               8,
                                               256>,
    weighted_sample_without_replacement_kernel<IdType,
                                               int,
                                               WeightType,
                                               WMIdType,
                                               int64_t,
                                               WeightType,
                                               8,
                                               512>,
  };

  // 128,256,512,1024
  // Maximum one-fourth ratio , however it  may not be a good way to choose a
  // fun.

  const int block_sizes[4] = {128, 256, 256, 512};
  auto choose_fun_idx      = [](int max_sample_count) {
    if (max_sample_count <= 128) {
      // return (max_sample_count - 1) / 32;
      return 0;
    }
    if (max_sample_count <= 256) { return 1; }
    if (max_sample_count <= 512) { return 2; }
    return 3;
  };
  int func_idx = choose_fun_idx(max_sample_count);

  int block_size = block_sizes[func_idx];

  func_array[func_idx]<<<center_node_count, block_size, 0, stream>>>(
    wm_csr_row_ptr,
    wm_csr_row_ptr_desc,
    wm_csr_col_ptr,
    wm_csr_col_ptr_desc,
    wm_csr_weight_ptr,
    wm_csr_weight_ptr_desc,
    (const IdType*)center_nodes,
    center_node_count,
    max_sample_count,
    random_seed,
    (const int*)output_sample_offset,
    output_sample_offset_desc,
    (WMIdType*)output_dest_node_ptr,
    (int*)output_center_localid_ptr,
    (int64_t*)output_edge_gid_ptr);
#endif
  WM_CUDA_CHECK(cudaGetLastError());
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace wholegraph_ops
