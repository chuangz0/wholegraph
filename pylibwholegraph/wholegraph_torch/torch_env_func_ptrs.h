#pragma once

#include <wholememory/env_func_ptrs.h>

namespace wholegraph_torch {

/**
 * @brief : PyTorch environment functions for memory allocation.
 *
 * @return : pointers to the functions of current CUDA device
 */
wholememory_env_func_t* get_pytorch_env_func();

}