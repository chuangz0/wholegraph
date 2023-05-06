#pragma once

#include <assert.h>

#include "global_reference.h"

namespace wholememory {

template <typename DataTypeT>
class device_reference {
 public:
  __device__ __forceinline__ explicit device_reference(const wholememory_gref_t& gref)
    : pointer_(static_cast<DataTypeT*>(gref.pointer)),
      typed_stride_(gref.stride / sizeof(DataTypeT))
  {
    assert(gref.stride % sizeof(DataTypeT) == 0);
  }
  __device__ device_reference() = delete;

  __device__ __forceinline__ DataTypeT& operator[](size_t index)
  {
    if (typed_stride_ == 0) { return pointer_[index]; }
    size_t rank = index / typed_stride_;
    return static_cast<DataTypeT**>(
      static_cast<void*>(pointer_))[rank][index - rank * typed_stride_];
  }

 private:
  DataTypeT* pointer_;
  size_t typed_stride_;
};

}  // namespace wholememory
