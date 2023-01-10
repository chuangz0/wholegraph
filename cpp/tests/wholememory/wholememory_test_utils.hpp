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
