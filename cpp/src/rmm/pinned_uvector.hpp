/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/exec_check_disable.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>
#include <rmm/pinned_buffer.hpp>

#include <cstddef>
#include <vector>

namespace rmm {

/**
 * @brief An *uninitialized* vector of elements in pinned host memory.
 *
 * Similar to a `thrust::host_vector`, `pinned_uvector` is a random access container of elements
 * stored contiguously in host memory. However, unlike `thrust::host_vector`, `pinned_uvector`
 * does *not* default initialize the vector elements. And underlying memory is pinned.
 *
 * If initialization is desired, this must be done explicitly by the caller, e.g., with
 * `thrust::uninitialized_fill`.
 *
 * Example:
 * @code{c++}
 * rmm::mr::pinned_memory_resource * mr = new my_custom_resource();
 * rmm::cuda_stream_view s{};
 *
 * // Allocates *uninitialized* pinned memory on stream `s` sufficient for 100 ints using the
 * // supplied resource `mr`
 * rmm::pinned_uvector<int> uv(100, s, mr);
 *
 * // Initializes all elements to 0 on stream `s`
 * thrust::uninitialized_fill(thrust::cuda::par.on(s), uv.begin(), uv.end(), 0);
 * @endcode
 *
 * Avoiding default initialization improves performance by eliminating the kernel launch required to
 * default initialize the elements. This initialization is often unnecessary, e.g., when the vector
 * is created to hold some output from some operation.
 *
 * However, this restricts the element type `T` to only trivially copyable types. In short,
 * trivially copyable types can be safely copied with `memcpy`. For more information, see
 * https://en.cppreference.com/w/cpp/types/is_trivially_copyable.
 *
 * Another key difference over `thrust::pinned_vector` is that all operations that invoke
 * allocation, kernels, or memcpys take a CUDA stream parameter to indicate on which stream the
 * operation will be performed.
 *
 * @tparam TT Trivially copyable element type
 */
template <typename TT>
class pinned_uvector {
  static_assert(std::is_trivially_copyable<TT>::value,
                "pinned_uvector only supports types that are trivially copyable.");

 public:
  using value_type      = TT;
  using size_type       = std::size_t;
  using reference       = value_type&;
  using const_reference = value_type const&;
  using pointer         = value_type*;
  using const_pointer   = value_type const*;
  using iterator        = pointer;
  using const_iterator  = const_pointer;

  RMM_EXEC_CHECK_DISABLE
  ~pinned_uvector() = default;

  RMM_EXEC_CHECK_DISABLE
  pinned_uvector(pinned_uvector&&) noexcept = default;

  pinned_uvector& operator=(pinned_uvector&&) noexcept = default;

  /**
   * @brief Copy ctor is deleted as it doesn't allow a stream argument
   */
  pinned_uvector(pinned_uvector const&) = delete;

  /**
   * @brief Copy assignment is deleted as it doesn't allow a stream argument
   */
  pinned_uvector& operator=(pinned_uvector const&) = delete;

  /**
   * @brief Default constructor is deleted as it doesn't allow a stream argument
   */
  pinned_uvector() = delete;

  /**
   * @brief Construct a new `pinned_uvector` with sufficient uninitialized storage for `size`
   * elements.
   *
   * Elements are uninitialized. Reading an element before it is initialized results in undefined
   * behavior.
   *
   * @param size The number of elements to allocate storage for
   * @param stream The stream on which to perform the allocation
   * @param mr The resource used to allocate the pinned storage
   */
  explicit pinned_uvector(
      std::size_t size,
      cuda_stream_view stream,
      rmm::mr::pinned_memory_resource* mr = rmm::mr::get_pinned_resource())
      : _storage{elements_to_bytes(size), stream, mr}
  {
  }

  /**
   * @brief Construct a new pinned_uvector by deep copying the contents of another `pinned_uvector`.
   *
   * Elements are copied as if by `memcpy`, i.e., `T`'s copy constructor is not invoked.
   *
   * @param other The vector to copy from
   * @param stream The stream on which to perform the copy
   * @param mr The resource used to allocate pinned memory for the new vector
   */
  explicit pinned_uvector(
      pinned_uvector const& other,
      cuda_stream_view stream,
      rmm::mr::pinned_memory_resource* mr = rmm::mr::get_pinned_resource())
      : _storage{other._storage, stream, mr}
  {
  }

  /**
   * @brief Returns pointer to the specified element
   *
   * Behavior is undefined if `element_index >= size()`.
   *
   * @throws rmm::out_of_range exception if `element_index >= size()`
   *
   * @param element_index Index of the specified element.
   * @return T* Pointer to the desired element
   */
  [[nodiscard]] pointer element_ptr(std::size_t element_index) noexcept
  {
    assert(element_index < size());
    return data() + element_index;
  }

  /**
   * @brief Returns pointer to the specified element
   *
   * Behavior is undefined if `element_index >= size()`.
   *
   * @throws rmm::out_of_range exception if `element_index >= size()`
   *
   * @param element_index Index of the specified element.
   * @return T* Pointer to the desired element
   */
  [[nodiscard]] const_pointer element_ptr(std::size_t element_index) const noexcept
  {
    assert(element_index < size());
    return data() + element_index;
  }

  /**
   * @brief Performs a synchronous copy of `v` to the specified element in pinned memory.
   *
   * Because this function synchronizes the stream `s`, it is safe to destroy or modify the object
   * referenced by `v` after this function has returned.
   *
   * Example:
   * \code{cpp}
   * rmm::pinned_uvector<int32_t> vec(100, stream);
   *
   * int v{42};
   *
   * // Copies 42 to element 0 on `stream` and synchronizes the stream
   * vec.set_element(0, v, stream);
   *
   * // It is safe to destroy or modify `v`
   * v = 13;
   * \endcode
   *
   *
   * @throws rmm::out_of_range exception if `element_index >= size()`
   *
   * @param element_index Index of the target element
   * @param value The value to copy to the specified element
   * @param stream The stream on which to perform the copy
   */
  void set_element(std::size_t element_index, value_type const& value, cuda_stream_view stream)
  {
    RMM_EXPECTS(
        element_index < size(), rmm::out_of_range, "Attempt to access out of bounds element.");
    data()[element_index] = value;
  }

  /**
   * @brief Returns the specified element from pinned memory
   *
   * @throws rmm::out_of_range exception if `element_index >= size()`
   *
   * @param element_index Index of the desired element
   * @param stream The stream on which to perform the copy
   * @return The value of the specified element
   */
  [[nodiscard]] value_type element(std::size_t element_index, cuda_stream_view stream) const
  {
    RMM_EXPECTS(
        element_index < size(), rmm::out_of_range, "Attempt to access out of bounds element.");
    return data()[element_index];
  }

  /**
   * @brief Returns the first element.
   *
   * @note This function incurs a device-to-host memcpy and should be used sparingly.
   * @note This function synchronizes `stream`.
   *
   * @throws rmm::out_of_range exception if the vector is empty.
   *
   * @param stream The stream on which to perform the copy
   * @return The value of the first element
   */
  [[nodiscard]] value_type front_element(cuda_stream_view stream) const
  {
    return element(0, stream);
  }

  /**
   * @brief Returns the last element.
   *
   * @throws rmm::out_of_range exception if the vector is empty.
   *
   * @param stream The stream on which to perform the copy
   * @return The value of the last element
   */
  [[nodiscard]] value_type back_element(cuda_stream_view stream) const
  {
    return element(size() - 1, stream);
  }

  /**
   * @brief Increases the capacity of the vector to `new_capacity` elements.
   *
   * If `new_capacity <= capacity()`, no action is taken.
   *
   * If `new_capacity > capacity()`, a new allocation of size `new_capacity` is created, and the
   * first `size()` elements from the current allocation are copied there as if by memcpy. Finally,
   * the old allocation is freed and replaced by the new allocation.
   *
   * @param new_capacity The desired capacity (number of elements)
   * @param stream The stream on which to perform the allocation/copy (if any)
   */
  void reserve(std::size_t new_capacity, cuda_stream_view stream)
  {
    _storage.reserve(elements_to_bytes(new_capacity), stream);
  }

  /**
   * @brief Resizes the vector to contain `new_size` elements.
   *
   * If `new_size > size()`, the additional elements are uninitialized.
   *
   * If `new_size < capacity()`, no action is taken other than updating the value of `size()`. No
   * memory is allocated nor copied. `shrink_to_fit()` may be used to force deallocation of unused
   * memory.
   *
   * If `new_size > capacity()`, elements are copied as if by memcpy to a new allocation.
   *
   * The invariant `size() <= capacity()` holds.
   *
   * @param new_size The desired number of elements
   * @param stream The stream on which to perform the allocation/copy (if any)
   */
  void resize(std::size_t new_size, cuda_stream_view stream)
  {
    _storage.resize(elements_to_bytes(new_size), stream);
  }

  /**
   * @brief Forces deallocation of unused pinned memory.
   *
   * If `capacity() > size()`, reallocates and copies vector contents to eliminate unused memory.
   *
   * @param stream Stream on which to perform allocation and copy
   */
  void shrink_to_fit(cuda_stream_view stream) { _storage.shrink_to_fit(stream); }

  /**
   * @brief Release ownership of pinned memory storage.
   *
   * @return The `pinned_buffer` used to store the vector elements
   */
  pinned_buffer release() noexcept { return std::move(_storage); }

  /**
   * @brief Returns the number of elements that can be held in currently allocated storage.
   *
   * @return std::size_t The number of elements that can be stored without requiring a new
   * allocation.
   */
  [[nodiscard]] std::size_t capacity() const noexcept
  {
    return bytes_to_elements(_storage.capacity());
  }

  /**
   * @brief Returns pointer to underlying pinned storage.
   *
   * @note If `size() == 0` it is undefined behavior to deference the returned pointer. Furthermore,
   * the returned pointer may or may not be equal to `nullptr`.
   *
   * @return Raw pointer to element storage in pinned memory.
   */
  [[nodiscard]] pointer data() noexcept { return static_cast<pointer>(_storage.data()); }

  /**
   * @brief Returns const pointer to underlying pinned storage.
   *
   * @note If `size() == 0` it is undefined behavior to deference the returned pointer. Furthermore,
   * the returned pointer may or may not be equal to `nullptr`.
   *
   * @return const_pointer Raw const pointer to element storage in pinned memory.
   */
  [[nodiscard]] const_pointer data() const noexcept
  {
    return static_cast<const_pointer>(_storage.data());
  }

  /**
   * @brief Returns an iterator to the first element.
   *
   * If the vector is empty, then `begin() == end()`.
   *
   * @return Iterator to the first element.
   */
  [[nodiscard]] iterator begin() noexcept { return data(); }

  /**
   * @brief Returns a const_iterator to the first element.
   *
   * If the vector is empty, then `cbegin() == cend()`.
   *
   * @return Immutable iterator to the first element.
   */
  [[nodiscard]] const_iterator cbegin() const noexcept { return data(); }

  /**
   * @brief Returns a const_iterator to the first element.
   *
   * If the vector is empty, then `begin() == end()`.
   *
   * @return Immutable iterator to the first element.
   */
  [[nodiscard]] const_iterator begin() const noexcept { return cbegin(); }

  /**
   * @brief Returns an iterator to the element following the last element of the vector.
   *
   * The element referenced by `end()` is a placeholder and dereferencing it results in undefined
   * behavior.
   *
   * @return Iterator to one past the last element.
   */
  [[nodiscard]] iterator end() noexcept { return data() + size(); }

  /**
   * @brief Returns a const_iterator to the element following the last element of the vector.
   *
   * The element referenced by `end()` is a placeholder and dereferencing it results in undefined
   * behavior.
   *
   * @return Immutable iterator to one past the last element.
   */
  [[nodiscard]] const_iterator cend() const noexcept { return data() + size(); }

  /**
   * @brief Returns an iterator to the element following the last element of the vector.
   *
   * The element referenced by `end()` is a placeholder and dereferencing it results in undefined
   * behavior.
   *
   * @return Immutable iterator to one past the last element.
   */
  [[nodiscard]] const_iterator end() const noexcept { return cend(); }

  /**
   * @brief Returns the number of elements.
   */
  [[nodiscard]] std::size_t size() const noexcept { return bytes_to_elements(_storage.size()); }

  /**
   * @brief Returns the signed number of elements.
   */
  [[nodiscard]] std::int64_t ssize() const noexcept
  {
    assert(size() < static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) &&
        "Size overflows signed integer");
    return static_cast<int64_t>(size());
  }

  /**
   * @brief Returns true if the vector contains no elements, i.e., `size() == 0`.
   *
   * @return true The vector is empty
   * @return false The vector is not empty
   */
  [[nodiscard]] bool is_empty() const noexcept { return size() == 0; }

  /**
   * @brief Returns pointer to the resource used to allocate and deallocate the pinned storage.
   *
   * @return Pointer to underlying resource
   */
  [[nodiscard]] mr::pinned_memory_resource* memory_resource() const noexcept
  {
    return _storage.memory_resource();
  }

  /**
   * @brief Returns stream most recently specified for allocation/deallocation
   */
  [[nodiscard]] cuda_stream_view stream() const noexcept { return _storage.stream(); }

  /**
   * @brief Sets the stream to be used for deallocation
   *
   * If no other rmm::pinned_uvector method that allocates memory is called
   * after this call with a different stream argument, then @p stream
   * will be used for deallocation in the `rmm::pinned_uvector destructor.
   * However, if either of `resize()` or `shrink_to_fit()` is called after this,
   * the later stream parameter will be stored and used in the destructor.
   */
  void set_stream(cuda_stream_view stream) noexcept { _storage.set_stream(stream); }

 private:
  pinned_buffer _storage{};  ///< Pinned memory storage for vector elements

  [[nodiscard]] std::size_t constexpr elements_to_bytes(std::size_t num_elements) const noexcept
  {
    return num_elements * sizeof(value_type);
  }

  [[nodiscard]] std::size_t constexpr bytes_to_elements(std::size_t num_bytes) const noexcept
  {
    return num_bytes / sizeof(value_type);
  }
};
}  // namespace rmm
