#include "initialize.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

#include "communicator.hpp"
#include "cuda_macros.hpp"
#include "error.hpp"
#include "logger.hpp"

namespace wholememory {

static std::mutex mu;
static bool is_wm_init = false;

static const std::string RAFT_NAME  = "wholememory";
static cudaDeviceProp* device_props = nullptr;

wholememory_error_code_t init(unsigned int flags) noexcept
{
  try {
    std::unique_lock<std::mutex> lock(mu);
    (void)flags;
    WHOLEMEMORY_EXPECTS(!is_wm_init, "WholeMemory has already been initialized.");
    WM_CU_CHECK(cuInit(0));
    int dev_count = 0;
    WM_CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0) {
      WHOLEMEMORY_ERROR("init failed, no CUDA device found");
      return WHOLEMEMORY_CUDA_ERROR;
    }
    device_props = new cudaDeviceProp[dev_count];
    for (int i = 0; i < dev_count; i++) {
      WM_CUDA_CHECK(cudaGetDeviceProperties(device_props + i, i));
    }
    is_wm_init = true;
    return WHOLEMEMORY_SUCCESS;
  } catch (raft::logic_error& logic_error) {
    WHOLEMEMORY_ERROR("init failed, logic_error=%s", logic_error.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("init failed, logic_error=%s", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("init failed, cuda_error=%s", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (...) {
    WHOLEMEMORY_ERROR("init failed, Unknown error.");
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
}

wholememory_error_code_t finalize() noexcept
{
  std::unique_lock<std::mutex> lock(mu);
  is_wm_init = false;
  WHOLEMEMORY_RETURN_ON_FAIL(destroy_all_communicators());
  delete[] device_props;
  device_props = nullptr;
  return WHOLEMEMORY_SUCCESS;
}

cudaDeviceProp* get_device_prop(int dev_id) noexcept
{
  try {
    if (dev_id == -1) { WM_CUDA_CHECK(cudaGetDevice(&dev_id)); }
    WHOLEMEMORY_CHECK(dev_id >= 0);
    return device_props + dev_id;
  } catch (...) {
    WHOLEMEMORY_ERROR("get_device_prop for dev_id=%d failed.", dev_id);
    return nullptr;
  }
}

}  // namespace wholememory