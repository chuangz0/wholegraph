#pragma once

#include <raft/core/error.hpp>

#include <cuda.h>

#include <chrono>
#include <cstdio>
#include <execinfo.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>

namespace raft {

/**
 * @brief Exception thrown when a CUDA driver error is encountered.
 */
struct cu_error : public raft::exception {
  explicit cu_error(char const* const message) : raft::exception(message) {}
  explicit cu_error(std::string const& message) : raft::exception(message) {}
};

}  // namespace raft

/**
 * @brief Error checking macro for CUDA driver API functions.
 *
 * Invokes a CUDA driver API function call, if the call does not return
 * CUDA_SUCCESS, invokes cuGetErrorString() to clear the error and throws an
 * exception detailing the CU error that occurred
 *
 */
#define RAFT_CU_TRY(call)                          \
  do {                                             \
    CUresult const status = call;                  \
    if (status != CUDA_SUCCESS) {                  \
      const char *p_err_name = nullptr;            \
      cuGetErrorName(status, &p_err_name);                                     \
      const char *p_err_str = nullptr;                                         \
      if (cuGetErrorString(status, &p_err_str) == CUDA_ERROR_INVALID_VALUE) {  \
        p_err_str = "Unrecoginzed CU error num";                               \
      }                                                                        \
      std::string msg{};                           \
      SET_ERROR_MSG(msg,                           \
                    "CU error encountered at: ",   \
                    "call='%s', Reason=%s:%s",     \
                    #call,                         \
                    p_err_name,                    \
                    p_err_str);                    \
      throw raft::cu_error(msg);                   \
    }                                              \
  } while (0)

// FIXME: Remove after consumers rename
#ifndef CU_TRY
#define CU_TRY(call) RAFT_CU_TRY(call)
#endif

/** FIXME: remove after cuml rename */
#ifndef CU_CHECK
#define CU_CHECK(call) RAFT_CU_TRY(call)
#endif

// /**
//  * @brief check for cuda driver API errors but log error instead of raising
//  *        exception.
//  */
#define RAFT_CU_TRY_NO_THROW(call)                                 \
  do {                                                             \
    CUresult const status = call;                  \
    if (status != CUDA_SUCCESS) {                  \
      const char *p_err_str = nullptr;                                         \
      if (cuGetErrorString(status, &p_err_str) == CUDA_ERROR_INVALID_VALUE) {  \
        p_err_str = "Unrecoginzed CU error num";                               \
      }                                                                        \
      std::string msg{};                           \
      printf("CU call='%s' at file=%s line=%d failed with %s\n",   \
             #call,                                                \
             __FILE__,                                             \
             __LINE__,                                             \
             p_err_str;                                            \
    }                                                              \
  } while (0)

// FIXME: Remove after cuml rename
#ifndef CU_CHECK_NO_THROW
#define CU_CHECK_NO_THROW(call) RAFT_CU_TRY_NO_THROW(call)
#endif
