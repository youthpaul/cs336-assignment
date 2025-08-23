import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import pandas as pd
import os
from timeit import default_timer as timer
import matplotlib.pyplot as plt

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == 'nccl':
        torch.cuda.set_device(rank)

def all_reduce_bench(
    rank, world_size, data_size, backend,
    benchmark_traild, warmup_trails, queue
):
    setup(rank, world_size, backend)

    num_elements = data_size * 1024**2 // 4
    device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")

    # warmup
    for _ in range(warmup_trails):
        data = torch.randint(0, 10, (num_elements, ), dtype=torch.float32, device=device)
        dist.all_reduce(data, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()
    
    # benchmark
    time = []
    for _ in range(benchmark_traild):
        data = torch.randint(0, 10, (num_elements, ), dtype=torch.float32, device=device)

        t0 = timer()
        dist.all_reduce(data, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()
        t1 = timer()

        time.append((t1 - t0) * 1000)

    avg_time = sum(time) / len(time)

    gathered_time = [None] * world_size
    dist.all_gather_object(gathered_time, avg_time)

    if rank == 0:
        final_avg = sum(gathered_time) / len(gathered_time)
        print (f"average all reduce time is {final_avg:.2f} ms")
        if queue is not None:
            queue.put(final_avg)



if __name__ == '__main__':
    backend = 'gloo'
    # backend = 'nccl'

    num_procs = [2, 4, 6]
    data_sizes = [1, 10, 100, 1024] # MB
    warmup_trails = 5
    benchmark_trails = 10

    result = {}
    for num_proc in num_procs:
        for data_size in data_sizes:
            print (f"Running {backend} with {num_proc} processes and {data_size} MB data")
            ctx = mp.get_context('spawn')
            queue = ctx.Queue()

            mp.spawn(
                fn = all_reduce_bench,
                args=(
                    num_proc, data_size, backend, 
                    benchmark_trails, warmup_trails, queue
                ),
                nprocs=num_proc,
                join=True
            )

            avg_time = queue.get()

            result[(num_proc, data_size)] = avg_time
    
    for (num_proc, data_size), avg_time in result.items():
        print(f'({num_proc}, {data_size}) : {avg_time}')
    