#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdint.h>

#include "wholegraph_ops//block_radix_topk.cuh"

namespace wholememory_ops {

__device__ __forceinline__ unsigned int WarpMatchLocalIDPairSync(int targets, int key)
{
  int xor_target       = __shfl_xor_sync(0xFFFFFFFF, targets, 0x10);
  bool first_half_lane = (threadIdx.x & 0x10) == 0;
  int match_value      = 0;
  unsigned int match_flag, tmp_match_flag;

  match_value    = first_half_lane ? key : xor_target;
  tmp_match_flag = __match_any_sync(0xFFFFFFFF, match_value);
  if (first_half_lane) match_flag = tmp_match_flag >> 16;

  match_value    = first_half_lane ? key : targets;
  tmp_match_flag = __match_any_sync(0xFFFFFFFF, match_value);
  if (first_half_lane) match_flag |= (tmp_match_flag & 0xFFFF0000);

  match_value    = !first_half_lane ? key : targets;
  tmp_match_flag = __match_any_sync(0xFFFFFFFF, match_value);
  if (!first_half_lane) match_flag = (tmp_match_flag & 0xFFFF);

  match_value    = !first_half_lane ? key : xor_target;
  tmp_match_flag = __match_any_sync(0xFFFFFFFF, match_value);
  if (!first_half_lane) match_flag |= (tmp_match_flag & 0xFFFF) << 16;

  return match_flag;
}

__device__ __forceinline__ int WarpFindMaxScaleSync(int scale)
{
#if __CUDA_ARCH__ >= 800
  return __reduce_max_sync(0xFFFFFFFF, scale);
#else
  for (int delta = 16; delta > 0; delta /= 2) {
    scale = max(__shfl_down_sync(0xFFFFFFFF, scale, delta, 32), scale);
  }
  return __shfl_sync(0xFFFFFFFF, scale, 0, 32);
#endif
}

class CacheLineInfo {
 public:
  __device__ __forceinline__ CacheLineInfo() {}
  __device__ __forceinline__ void LoadTag(uint16_t* tag_ptr) { tag_ = tag_ptr[threadIdx.x]; }
  __device__ __forceinline__ void LoadInfo(uint16_t* tag_ptr, uint16_t* count_ptr)
  {
    tag_       = tag_ptr[threadIdx.x];
    lfu_count_ = count_ptr[threadIdx.x];
  }
  __device__ __forceinline__ void StoreInfo(uint16_t* tag_ptr, uint16_t* count_ptr) const
  {
    tag_ptr[threadIdx.x]   = tag_;
    count_ptr[threadIdx.x] = lfu_count_;
  }
  __device__ __forceinline__ bool IsValid() const { return (tag_ & kValidMask) != 0U; }
  __device__ __forceinline__ bool IsInValid() const { return !IsValid(); }
  __device__ __forceinline__ bool IsModified() const { return (tag_ & kModifiedMask) != 0U; };
  __device__ __forceinline__ int LocalID() const
  {
    return IsValid() ? (int)(tag_ & kLocalIDMask) : -1;
  }
  __device__ __forceinline__ int ScaleSync() const
  {
    return __ballot_sync(0xFFFFFFFF, lfu_count_ & kScaleMask);
  }
  __device__ __forceinline__ int64_t LfuCountSync() const
  {
    int const scale = ScaleSync();
    int64_t count   = (lfu_count_ & kCountMask);
    count <<= scale;
    count += (1ULL << scale) - 1;
    return count;
  }
  /**
   * Check if local_id is in CacheSet
   * @param local_id : local_id
   * @return : CacheLine Id if key in cache, else -1.
   */
  __device__ __forceinline__ int KeyIndexSync(int local_id) const
  {
    bool is_key_in_cache_line = IsValid() && LocalID() == local_id;
    uint32_t mask             = __ballot_sync(0xFFFFFFFF, static_cast<int>(is_key_in_cache_line));
    // __ffs(0) returns 0
    return __ffs(mask) - 1;
  }
  /**
   * Set new counter for LFU, if invalid, use -1
   * @param new_lfu_count : new LFU count
   */
  __device__ __forceinline__ void SetScaleLfuCountSync(int64_t new_lfu_count)
  {
    int scale     = (new_lfu_count >= 0) ? 64 - __clzll(new_lfu_count) : 0;
    scale         = max(scale, kScaledCounterBits) - kScaledCounterBits;
    int max_scale = WarpFindMaxScaleSync(scale);
    // printf("threadIdx.x=%d, new_lfu_count=%ld, scale=%d, max_scale=%d\n", threadIdx.x,
    // new_lfu_count, scale, max_scale);
    int scale_lfu_count = new_lfu_count >> max_scale;
    scale_lfu_count |= ((max_scale >> threadIdx.x) & 1) << kScaledCounterBits;
    lfu_count_ = scale_lfu_count;
  }
  //
  __device__ __forceinline__ void SetLocalID(int local_id)
  {
    if (local_id >= 0) {
      if (IsInValid() || local_id != LocalID()) ClearModify();
      tag_ &= ~(kLocalIDMask | kValidMask);
      tag_ |= (local_id | kValidMask);
    } else {
      tag_ = 0;
    }
  }
  __device__ __forceinline__ void ClearCacheLine() { tag_ = 0; }
  __device__ __forceinline__ void ClearModify() { tag_ &= ~kModifiedMask; }
  static constexpr int kCacheSetSize      = 32;
  static constexpr int kLocalIDBits       = 14;
  static constexpr int kScaledCounterBits = 14;
  static constexpr uint32_t kValidMask    = (1U << 14);
  static constexpr uint32_t kModifiedMask = (1U << 15);
  static constexpr uint32_t kLocalIDMask  = (1U << 14) - 1;
  static constexpr uint32_t kCountMask    = (1U << 14) - 1;
  static constexpr uint32_t kScaleMask    = (1U << 14);
  uint32_t tag_;
  uint32_t lfu_count_;
};

template <typename NodeIDT>
class CacheSetUpdater {
 public:
  static constexpr int kTopKRegisterCount = 4;
  static constexpr int kCacheSetSize      = CacheLineInfo::kCacheSetSize;
  static constexpr int kScaledCounterBits = 14;
  using BlockTopK =
    wholegraph_ops::BlockRadixTopKRegister<int64_t, kCacheSetSize, kTopKRegisterCount, true, int>;
  struct TempStorage : BlockTopK::TempStorage {};
  /**
   * From all invalid CacheSet, recompute lids to cache, and update cache_line_info.
   * NOTE: data are not loaded, need to load after this function
   * @param temp_storage : temp_storage
   * @param cache_line_info : cache_line_info, will be updated.
   * @param memory_lfu_counter : lfu_counter in memory of this cache set
   * @param id_count : valid count in this cache set, most cases it is cache_set_coverage,
   * maybe smaller than kCacheSetSize tailing cache set.
   */
  __device__ __forceinline__ void ReComputeCache(TempStorage& temp_storage,
                                                 CacheLineInfo& cache_line_info,
                                                 int64_t* memory_lfu_counter,
                                                 int id_count)
  {
    if (id_count <= 0) return;
    assert(cache_line_info.IsInValid());
#pragma unroll
    for (int i = 0; i < kTopKRegisterCount; i++) {
      candidate_lfu_count_[i] = -1;
      candidate_local_id_[i]  = -1;
    }
    int base_idx    = 0;
    int valid_count = 0;
    FillCandidate<0, false>(
      nullptr, nullptr, memory_lfu_counter, base_idx, valid_count, 0, id_count, temp_storage, -1);
    while (base_idx < id_count) {
      FillCandidate<1, false>(
        nullptr, nullptr, memory_lfu_counter, base_idx, valid_count, 0, id_count, temp_storage, -1);
      FillCandidate<2, false>(
        nullptr, nullptr, memory_lfu_counter, base_idx, valid_count, 0, id_count, temp_storage, -1);
      FillCandidate<3, false>(
        nullptr, nullptr, memory_lfu_counter, base_idx, valid_count, 0, id_count, temp_storage, -1);
    }
    cache_line_info.ClearCacheLine();
    cache_line_info.SetLocalID(candidate_local_id_[0]);
    cache_line_info.SetScaleLfuCountSync(candidate_local_id_[0] >= 0 ? candidate_lfu_count_[0] : 0);
  }
  /**
   * Update cache set according to gids and inc_count
   * @tparam NeedOutputLoadIDs : If need to output IDs that should be loaded into cache
   * @tparam NeedOutputWriteBackIDs : If need to output IDs that should be write back to memory
   * @param temp_storage : Work space storage
   * @param cache_line_info : cache line info that already loaded, will be updated
   * @param memory_lfu_counter : counter pointer of IDs that current cache set covers.
   * @param gids : GIDs to update
   * @param inc_count : same length as gids, the count of each GIDs to added, if nullptr, each GID
   * add 1.
   * @param need_load_to_cache_ids : output of IDs that should be loaded into cache, if not needed,
   * use nullptr
   * @param need_write_back_ids : output of IDs that should be write back to memory, if not needed,
   * use nullptr
   * @param set_start_id : start GID of current cache set
   * @param id_count : count of GIDs
   */
  template <bool NeedOutputLoadIDs, bool NeedOutputWriteBackIDs>
  __device__ __forceinline__ void UpdateCache(TempStorage& temp_storage,
                                              CacheLineInfo& cache_line_info,
                                              int64_t* memory_lfu_counter,
                                              const NodeIDT* gids,
                                              const int* inc_count,
                                              NodeIDT* need_load_to_cache_ids,
                                              NodeIDT* need_write_back_ids,
                                              int64_t set_start_id,
                                              int id_count)
  {
    if (id_count <= 0) return;
#pragma unroll
    for (int i = 0; i < kTopKRegisterCount; i++) {
      candidate_lfu_count_[i] = -1;
      candidate_local_id_[i]  = -1;
    }
    int base_idx           = 0;
    int valid_count        = 0;
    int cached_local_id    = cache_line_info.LocalID();
    int has_local_id_count = 0;
    has_local_id_count += FillCandidate<0>(gids,
                                           inc_count,
                                           memory_lfu_counter,
                                           base_idx,
                                           valid_count,
                                           set_start_id,
                                           id_count,
                                           temp_storage,
                                           cached_local_id);
    while (base_idx < id_count) {
      has_local_id_count += FillCandidate<1>(gids,
                                             inc_count,
                                             memory_lfu_counter,
                                             base_idx,
                                             valid_count,
                                             set_start_id,
                                             id_count,
                                             temp_storage,
                                             cached_local_id);
      has_local_id_count += FillCandidate<2>(gids,
                                             inc_count,
                                             memory_lfu_counter,
                                             base_idx,
                                             valid_count,
                                             set_start_id,
                                             id_count,
                                             temp_storage,
                                             cached_local_id);
      has_local_id_count += FillCandidate<3>(gids,
                                             inc_count,
                                             memory_lfu_counter,
                                             base_idx,
                                             valid_count,
                                             set_start_id,
                                             id_count,
                                             temp_storage,
                                             cached_local_id);
    }
    // printf("[TopK init dump] threadIdx.x=%d, lfu_count=%ld, lid=%d\n", threadIdx.x,
    // candidate_lfu_count_[0], candidate_local_id_[0]);
    candidate_lfu_count_[1] = -1;
    candidate_local_id_[1]  = -1;
    unsigned int match_flag;
    // match_flag = WarpMatchLocalIDPairSync(candidate_local_id_[0], cached_local_id);
    int64_t estimated_lfu_count = cache_line_info.LfuCountSync();
    // Valid AND NOT exist in update list
    if (cached_local_id != -1 && has_local_id_count == 0) {
      // cached key not updated, use estimated lfu_count from cache
      candidate_lfu_count_[1] = estimated_lfu_count;
      candidate_local_id_[1]  = cached_local_id;
    }
    BlockTopK(temp_storage)
      .radixTopKToStriped(
        candidate_lfu_count_, candidate_local_id_, kCacheSetSize, kCacheSetSize * 2);
    // printf("[TopK merge dump] threadIdx.x=%d, lfu_count=%ld, lid=%d\n", threadIdx.x,
    // candidate_lfu_count_[0], candidate_local_id_[0]);
    match_flag     = WarpMatchLocalIDPairSync(candidate_local_id_[0], cached_local_id);
    int from_lane  = -1;
    bool has_match = (cached_local_id >= 0 && match_flag != 0);
    if (has_match) from_lane = __ffs(match_flag) - 1;
    unsigned int can_update_mask   = __ballot_sync(0xFFFFFFFF, !has_match);
    unsigned int lower_thread_mask = (1U << threadIdx.x) - 1;
    int updatable_cache_line_rank  = !has_match ? __popc(can_update_mask & lower_thread_mask) : -1;
    unsigned int new_match_flag = WarpMatchLocalIDPairSync(cached_local_id, candidate_local_id_[0]);
    // printf("tid=%d, cached_local_id=%d, candidate_local_id_=%d, new_match_flag=%x\n",
    //        threadIdx.x,
    //        cached_local_id,
    //        candidate_local_id_[0],
    //        new_match_flag);
    bool new_need_slot              = (candidate_local_id_[0] >= 0 && new_match_flag == 0);
    unsigned int need_new_slot_mask = __ballot_sync(0xFFFFFFFF, new_need_slot);
    int insert_data_rank = new_need_slot ? __popc(need_new_slot_mask & lower_thread_mask) : -1;
    // printf("tid=%d, updatable_cache_line_rank=%d, insert_data_rank=%d\n", threadIdx.x,
    // updatable_cache_line_rank, insert_data_rank);
    unsigned int rank_match_flag =
      WarpMatchLocalIDPairSync(insert_data_rank, updatable_cache_line_rank);
    if (updatable_cache_line_rank != -1 && rank_match_flag != 0) {
      from_lane = __ffs(rank_match_flag) - 1;
    }
    int src_lane_idx      = from_lane >= 0 ? from_lane : 0;
    int64_t new_lfu_count = __shfl_sync(0xFFFFFFFF, candidate_lfu_count_[0], src_lane_idx, 32);
    int new_local_id      = __shfl_sync(0xFFFFFFFF, candidate_local_id_[0], src_lane_idx, 32);
    if (from_lane == -1) {
      new_local_id  = -1;
      new_lfu_count = 0;
    }
    // printf("tid=%d, new_local_id=%d, new_lfu_count=%ld\n", threadIdx.x, new_local_id,
    // new_lfu_count);
    if (NeedOutputLoadIDs && need_load_to_cache_ids != nullptr) {
      int new_cached_lid = -1;
      if (new_need_slot) { new_cached_lid = candidate_local_id_[0]; }
      unsigned int load_cache_mask = __ballot_sync(0xFFFFFFFF, new_cached_lid >= 0);
      int output_idx               = __popc(load_cache_mask & ((1 << threadIdx.x) - 1));
      int total_load_count         = __popc(load_cache_mask);
      if (new_need_slot) {
        need_load_to_cache_ids[output_idx] = new_cached_lid + set_start_id;
        // printf("tid=%d, load_cache_mask=%x, NeedLoadGIDs[%d]=%ld\n", threadIdx.x,
        // load_cache_mask, output_idx, new_cached_lid + set_start_id);
      }
      if (threadIdx.x >= total_load_count && threadIdx.x < min(id_count, kCacheSetSize)) {
        need_load_to_cache_ids[threadIdx.x] = -1;
      }
    }

    if (NeedOutputWriteBackIDs && need_write_back_ids != nullptr) {
      int write_back_lid   = -1;
      bool need_write_back = cached_local_id >= 0 && !has_match && cache_line_info.IsModified();
      if (need_write_back) { write_back_lid = cache_line_info.LocalID(); }

      unsigned int write_back_mask = __ballot_sync(0xFFFFFFFF, write_back_lid >= 0);
      int output_idx               = __popc(write_back_mask & ((1 << threadIdx.x) - 1));
      if (need_write_back) {
        need_write_back_ids[output_idx] = write_back_lid + set_start_id;
        // printf("tid=%d, WriteBackGIDs[%d]=%ld\n", threadIdx.x, output_idx, write_back_lid +
        // set_start_id);
      }
      int total_write_back_count = __popc(write_back_mask);
      if (threadIdx.x >= total_write_back_count && threadIdx.x < min(id_count, kCacheSetSize)) {
        need_write_back_ids[threadIdx.x] = -1;
      }
    }
    cache_line_info.SetScaleLfuCountSync(new_lfu_count);
    cache_line_info.SetLocalID(new_local_id);
  }

 private:
  int64_t candidate_lfu_count_[kTopKRegisterCount];
  int candidate_local_id_[kTopKRegisterCount];
  template <int StrideIdx, bool IncCounter = true>
  __device__ __forceinline__ int FillCandidate(const NodeIDT* gids,
                                               const int* inc_freq_count,
                                               int64_t* cache_set_coverage_counter,
                                               int& base_idx,
                                               int& valid_count,
                                               int64_t cache_set_start_id,
                                               int id_count,
                                               TempStorage& temp_storage,
                                               int cached_local_id)
  {
    int const idx = base_idx + threadIdx.x;
    valid_count += min(kCacheSetSize, max(0, id_count - base_idx));
    int local_id = -1;
    if (idx < id_count) {
      local_id                        = gids != nullptr ? gids[idx] - cache_set_start_id : idx;
      candidate_lfu_count_[StrideIdx] = cache_set_coverage_counter[local_id];
      if (IncCounter) {
        int id_inc_count = inc_freq_count != nullptr ? inc_freq_count[idx] : 1;
        candidate_lfu_count_[StrideIdx] += id_inc_count;
        cache_set_coverage_counter[local_id] = candidate_lfu_count_[StrideIdx];
      }
      candidate_local_id_[StrideIdx] = local_id;
    }
    unsigned int local_id_match_mask = WarpMatchLocalIDPairSync(local_id, cached_local_id);
    int has_local_id_count           = (cached_local_id != -1) ? __popc(local_id_match_mask) : 0;
    if (StrideIdx == kTopKRegisterCount - 1) {
      BlockTopK(temp_storage)
        .radixTopKToStriped(
          candidate_lfu_count_, candidate_local_id_, min(kCacheSetSize, valid_count), valid_count);
      valid_count = min(valid_count, kCacheSetSize);
    }
    base_idx += kCacheSetSize;
    return has_local_id_count;
  }
};

}  // namespace wholememory_ops