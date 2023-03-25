#include "embedding_optimizer.hpp"

#include "logger.hpp"

namespace wholememory {

embedding_optimizer_impl_base::embedding_optimizer_impl_base() = default;

wholememory_error_code_t float_setter_fn(float* target_ptr, const void* data)
{
  const auto* float_data = static_cast<const float*>(data);
  *target_ptr            = *float_data;
  return WHOLEMEMORY_SUCCESS;
}

optimizer_parameter_setter_fn_t embedding_optimizer_impl_base::get_float_setter(float* target)
{
  return std::bind(float_setter_fn, target, std::placeholders::_1);
}

wholememory_error_code_t embedding_optimizer_impl_base::set_parameter(const char* parameter_name,
                                                                      void* value) noexcept
{
  std::string const parameter_name_str = parameter_name;
  auto it                              = setter_fns_.find(parameter_name_str);
  if (it == setter_fns_.end()) {
    WHOLEMEMORY_ERROR("parameter name %s is not valid for optimizer %s", parameter_name, name_);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  return it->second(value);
}

class SGDEmbeddingOptimizer : public embedding_optimizer_impl_base {
 public:
  SGDEmbeddingOptimizer();

 protected:
  float weight_decay = 0.0f;
};

SGDEmbeddingOptimizer::SGDEmbeddingOptimizer()
{
  name_ = "SGD";
  setter_fns_.emplace(std::pair<std::string, optimizer_parameter_setter_fn_t>(
    "weight_decay", get_float_setter(&weight_decay)));
}

class LazyAdamEmbeddingOptimizer : public embedding_optimizer_impl_base {
 public:
  LazyAdamEmbeddingOptimizer();

 protected:
  float weight_decay = 0.0f;
  float epsilon      = 1e-8;
  float beta1        = 0.9;
  float beta2        = 0.999;
};

LazyAdamEmbeddingOptimizer::LazyAdamEmbeddingOptimizer()
{
  name_ = "LazyAdam";
  setter_fns_.emplace(std::pair<std::string, optimizer_parameter_setter_fn_t>(
    "weight_decay", get_float_setter(&weight_decay)));
  setter_fns_.emplace(
    std::pair<std::string, optimizer_parameter_setter_fn_t>("epsilon", get_float_setter(&epsilon)));
  setter_fns_.emplace(
    std::pair<std::string, optimizer_parameter_setter_fn_t>("beta1", get_float_setter(&beta1)));
  setter_fns_.emplace(
    std::pair<std::string, optimizer_parameter_setter_fn_t>("beta2", get_float_setter(&beta2)));
}

class AdaGradEmbeddingOptimizer : public embedding_optimizer_impl_base {
 public:
  AdaGradEmbeddingOptimizer();

 protected:
  float weight_decay = 0.0f;
  float epsilon      = 1e-8;
};

AdaGradEmbeddingOptimizer::AdaGradEmbeddingOptimizer()
{
  name_ = "AdaGrad";
  setter_fns_.emplace(std::pair<std::string, optimizer_parameter_setter_fn_t>(
    "weight_decay", get_float_setter(&weight_decay)));
  setter_fns_.emplace(
    std::pair<std::string, optimizer_parameter_setter_fn_t>("epsilon", get_float_setter(&epsilon)));
}

class RMSPropEmbeddingOptimizer : public embedding_optimizer_impl_base {
 public:
  RMSPropEmbeddingOptimizer();

 protected:
  float weight_decay = 0.0f;
  float epsilon      = 1e-8;
  float alpha        = 0.99;
};

RMSPropEmbeddingOptimizer::RMSPropEmbeddingOptimizer()
{
  name_ = "RMSProp";
  setter_fns_.emplace(std::pair<std::string, optimizer_parameter_setter_fn_t>(
    "weight_decay", get_float_setter(&weight_decay)));
  setter_fns_.emplace(
    std::pair<std::string, optimizer_parameter_setter_fn_t>("epsilon", get_float_setter(&epsilon)));
  setter_fns_.emplace(
    std::pair<std::string, optimizer_parameter_setter_fn_t>("alpha", get_float_setter(&alpha)));
}

wholememory_error_code_t create_embedding_optimizer(
  wholememory_embedding_optimizer_t* optimizer,
  wholememory_optimizer_type_t optimizer_type) noexcept
{
  embedding_optimizer_impl_base* optimizer_impl = nullptr;
  try {
    switch (optimizer_type) {
      case WHOLEMEMORY_OPT_SGD: {
        optimizer_impl = new SGDEmbeddingOptimizer();
        break;
      }
      case WHOLEMEMORY_OPT_LAZY_ADAM: {
        optimizer_impl = new LazyAdamEmbeddingOptimizer();
        break;
      }
      case WHOLEMEMORY_OPT_ADAGRAD: {
        optimizer_impl = new AdaGradEmbeddingOptimizer();
        break;
      }
      case WHOLEMEMORY_OPT_RMSPROP: {
        optimizer_impl = new RMSPropEmbeddingOptimizer();
        break;
      }
      default: {
        return WHOLEMEMORY_NOT_IMPLEMENTED;
      }
    }
  } catch (...) {
    WHOLEMEMORY_ERROR("create optimizer failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  *optimizer = static_cast<wholememory_embedding_optimizer_t>(optimizer_impl);
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t optimizer_set_parameter(wholememory_embedding_optimizer_t optimizer,
                                                 const char* parameter_name,
                                                 void* value) noexcept
{
  if (optimizer == nullptr) {
    WHOLEMEMORY_ERROR("Input optimizer is nullptr.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto* optimizer_impl = static_cast<embedding_optimizer_impl_base*>(optimizer);
  return optimizer_impl->set_parameter(parameter_name, value);
}

void destroy_embedding_optimizer(wholememory_embedding_optimizer_t optimizer) noexcept
{
  auto* optimizer_impl = static_cast<embedding_optimizer_impl_base*>(optimizer);
  delete optimizer_impl;
}

}  // namespace wholememory