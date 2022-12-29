#pragma once

#include <wholememory/wholememory.h>

namespace wholememory {

wholememory_error_code_t init(unsigned int flags) noexcept;

wholememory_error_code_t finalize() noexcept;

}