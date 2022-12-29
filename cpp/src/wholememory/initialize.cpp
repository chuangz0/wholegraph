#include "initialize.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

#include "raft/core/cu_macros.hpp"

#include "error.hpp"
#include "logger.hpp"
#include "communicator.hpp"

namespace wholememory {

static std::mutex mu;
static bool is_wm_init = false;

static const std::string RAFT_NAME = "wholememory";

wholememory_error_code_t init(unsigned int flags) noexcept {
  try {
    std::unique_lock<std::mutex> lock(mu);
    (void) flags;
    WHOLEMEMORY_EXPECTS(!is_wm_init, "WholeMemory has already been initialized.");
    CU_CHECK(cuInit(0));
    is_wm_init = true;
    return WHOLEMEMORY_SUCCESS;
  } catch (raft::logic_error& logic_error) {
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
}

wholememory_error_code_t finalize() noexcept {
  return destroy_all_communicators();
}

}