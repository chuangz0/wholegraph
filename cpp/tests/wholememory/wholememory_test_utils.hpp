#pragma once

#include <gtest/gtest.h>

#include "parallel_utils.hpp"
#include "wholememory/communicator.hpp"

wholememory_comm_t create_communicator_by_pipes(const std::vector<std::array<int, 2>>& pipes,
                                                int rank,
                                                int world_size)
{
  wholememory_unique_id_t unique_id;
  if (rank == 0) { EXPECT_EQ(wholememory::create_unique_id(&unique_id), WHOLEMEMORY_SUCCESS); }

  PipeBroadcast(rank, world_size, 0, pipes, &unique_id);

  wholememory_comm_t wm_comm;
  EXPECT_EQ(wholememory::create_communicator(&wm_comm, unique_id, rank, world_size),
            WHOLEMEMORY_SUCCESS);
  return wm_comm;
}

wholememory_comm_t create_group_communicator_by_pipes(const std::vector<std::array<int, 2>>& pipes,
                                                      int rank,
                                                      int world_size,
                                                      int group_count)
{
  EXPECT_EQ(world_size % group_count, 0);
  int group_size = world_size / group_count;
  int group_rank = rank % group_size;
  wholememory_unique_id_t unique_id;
  if (group_rank == 0) {
    EXPECT_EQ(wholememory::create_unique_id(&unique_id), WHOLEMEMORY_SUCCESS);
  }

  wholememory_unique_id_t comm_unique_id;
  for (int g = 0; g < group_count; g++) {
    if (g * group_size == rank) comm_unique_id = unique_id;
    PipeBroadcast(rank, world_size, g * group_size, pipes, &comm_unique_id);
    if (rank / group_size == g) unique_id = comm_unique_id;
  }

  wholememory_comm_t wm_comm;
  EXPECT_EQ(wholememory::create_communicator(&wm_comm, unique_id, group_rank, group_size),
            WHOLEMEMORY_SUCCESS);
  return wm_comm;
}
