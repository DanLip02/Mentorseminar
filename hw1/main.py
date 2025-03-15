from mpi4py import MPI
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# size = comm.Get_size()
size = 12

N = 10**8
dx = 1.0 / N
num_iterations = 10

if len(sys.argv) > 1:
    P = int(sys.argv[1])
else:
    P = size

P = min(P, size)
if rank >= P:
    exit()

total_time = 0.0

for _ in range(num_iterations):
    comm.Barrier()
    start_time = time.time()

    chunk = N // P
    start = rank * chunk
    end = (rank + 1) * chunk if rank != P - 1 else N

    local_sum = 0.0
    for i in range(start, end):
        x = (i + 0.5) * dx
        local_sum += 4.0 / (1.0 + x**2)

    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    end_time = time.time()
    iteration_time = end_time - start_time
    total_time += iteration_time

if rank == 0:
    avg_time = total_time / num_iterations
    pi = global_sum * dx
    print(f"P={P}, Avg Time: {avg_time:.6f} sec")
