import multiprocessing as mp


def multiprocess_run(world_size: int, func, inline_single_process = False):
    assert world_size > 0
    # if world_size == 1:
    #    func(0, 1)
    #    return
    mp.set_start_method('spawn')
    if world_size == 1 and inline_single_process == True:
        func(0, 1)
        return
    process_array = [None] * world_size
    for i in range(world_size):
        process_array[i] = mp.Process(target=func, args=(i, world_size))
        process_array[i].start()
    for i in range(world_size):
        process_array[i].join()
    for i in range(world_size):
        assert process_array[i].exitcode == 0


