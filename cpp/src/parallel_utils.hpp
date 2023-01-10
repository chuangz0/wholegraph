#pragma once

#include <unistd.h>

#include <functional>
#include <vector>

#include "logger.hpp"

/*!
 * Run f with size threads
 * @param size : thread count
 * @param f : thread function
 */
void MultiThreadRun(int size, std::function<void(int, int)> f);

/*!
 * Run f with size processes
 * @note when using gtest with MultiProcessRun, ::testing::Test::HasFailure()
 * need to be called before f return and modify exit code according to if has
 * gtest failures. See parallel_utils_tests.cpp for reference.
 * @param size : process count
 * @param f : process function
 */
void MultiProcessRun(int size, std::function<void(int, int)> f);

inline int CreatePipes(std::vector<std::array<int, 2>>* pipes, int nproc)
{
  pipes->resize(nproc);
  for (int i = 0; i < nproc; i++) {
    if (pipe((*pipes)[i].data()) == -1) {
      WHOLEMEMORY_ERROR("Create pipe failed.");
      return -1;
    }
  }
  return 0;
}

inline void ClosePipes(std::vector<std::array<int, 2>>* pipes)
{
  for (size_t i = 0; i < pipes->size(); i++) {
    WHOLEMEMORY_CHECK(close(pipes->at(i)[0]) == 0);
    WHOLEMEMORY_CHECK(close(pipes->at(i)[1]) == 0);
  }
  pipes->clear();
}

template <typename T>
inline void PipeBroadcast(
  int rank, int world_size, int root, const std::vector<std::array<int, 2>>& pipes, T* data)
{
  if (rank == root) {
    for (int i = 0; i < world_size; i++) {
      auto wret = write(pipes[i][1], data, sizeof(T));
      if (wret != sizeof(T)) { WHOLEMEMORY_FATAL("write to pipe failed."); }
    }
  }
  auto rret = read(pipes[rank][0], data, sizeof(T));
  if (rret != sizeof(T)) { WHOLEMEMORY_FATAL("read to pipe failed."); }
}

template <typename T>
inline void PipeGroupBroadcast(int rank,
                               int world_size,
                               int group_root,
                               int group_size,
                               const std::vector<std::array<int, 2>>& pipes,
                               T* data)
{
  WHOLEMEMORY_CHECK(world_size % group_size == 0);
  if (rank % group_size == group_root) {
    for (int i = rank - group_root; i < rank - group_root + group_size; i++) {
      auto wret = write(pipes[i][1], data, sizeof(T));
      if (wret != sizeof(T)) { WHOLEMEMORY_FATAL("write to pipe failed."); }
    }
  }
  auto rret = read(pipes[rank][0], data, sizeof(T));
  if (rret != sizeof(T)) { WHOLEMEMORY_FATAL("read to pipe failed."); }
}

int ForkGetDeviceCount();
