#pragma once

#include <wholememory/embedding.h>

#include <functional>
#include <map>
#include <string>

#include "embedding_cache.hpp"

#ifdef __cplusplus
extern "C" {
#endif

struct wholememory_embedding_optimizer_ {
  wholememory_optimizer_type_t optimizer_type;
};

#ifdef __cplusplus
}
#endif

namespace wholememory {

class embedding_optimizer_impl_base;

using optimizer_parameter_setter_fn_t = std::function<wholememory_error_code_t(const void*)>;

class optimizer_state {
 public:
  optimizer_state()  = default;
  ~optimizer_state() = default;
  struct state {
    // name of this state
    std::string name;
    // local tensor, should NOT be WholeMemory Tensor but be local tensor of the WholeMemory Tensor.
    wholememory_tensor_t local_raw_data_tensor         = nullptr;
    wholememory_tensor_t local_cache_entry_meta_tensor = nullptr;
    wholememory_tensor_t local_cache_entry_data_tensor = nullptr;
  };
  std::vector<state> states;
};

class embedding_optimizer_impl_base : public wholememory_embedding_optimizer_ {
 public:
  embedding_optimizer_impl_base();
  virtual ~embedding_optimizer_impl_base() = default;
  virtual wholememory_error_code_t set_parameter(const char* parameter_name, void* value) noexcept;
  /**
   * Apply gradients.
   * As trainable Embedding use READWRITE cache, Cache communicator is the same as Embedding
   * communicator. Gradients will be partitioned and each rank is only responsible for its own
   * partition.
   *
   * @param indices : bucketed indices that belongs to current rank.
   * @param grads : bucketed gradients that belongs to current rank.
   * @param indice_offset : start index offset of current rank.
   * @param states : optimizer states
   * @param lr : learning rate
   * @return : wholememory_error_code_t
   */
  virtual wholememory_error_code_t step(wholememory_tensor_t indices,
                                        wholememory_tensor_t grads,
                                        int64_t indice_offset,
                                        optimizer_state* states,
                                        float lr) noexcept
  {
    return WHOLEMEMORY_NOT_IMPLEMENTED;
  };

 protected:
  static optimizer_parameter_setter_fn_t get_float_setter(float* target_ptr);
  std::map<std::string, optimizer_parameter_setter_fn_t> setter_fns_;
  const char* name_ = nullptr;
};

wholememory_error_code_t create_embedding_optimizer(
  wholememory_embedding_optimizer_t* optimizer,
  wholememory_optimizer_type_t optimizer_type) noexcept;

wholememory_error_code_t optimizer_set_parameter(wholememory_embedding_optimizer_t optimizer,
                                                 const char* parameter_name,
                                                 void* value) noexcept;

void destroy_embedding_optimizer(wholememory_embedding_optimizer_t optimizer) noexcept;

}  // namespace wholememory
