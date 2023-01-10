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

static const std::string RAFT_NAME = "wholememory";

wholememory_error_code_t init(unsigned int flags) noexcept
{
  try {
    std::unique_lock<std::mutex> lock(mu);
    (void)flags;
    WHOLEMEMORY_EXPECTS(!is_wm_init, "WholeMemory has already been initialized.");
    WM_CU_CHECK(cuInit(0));
    is_wm_init = true;
    return WHOLEMEMORY_SUCCESS;
  } catch (raft::logic_error& logic_error) {
    WHOLEMEMORY_ERROR("init failed, logic_error=%s", logic_error.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
}

wholememory_error_code_t finalize() noexcept
{
  std::unique_lock<std::mutex> lock(mu);
  is_wm_init = false;
  return destroy_all_communicators();
}

}  // namespace wholememory