
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import time
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt

kernel_code = """
__global__ void heat_step(double *u, double *u_new, double alpha, double dt, double dx2, double dy2, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < nx - 1 && j < ny - 1) {
        int idx = i * ny + j;
        int up = (i - 1) * ny + j;
        int down = (i + 1) * ny + j;
        int left = i * ny + (j - 1);
        int right = i * ny + (j + 1);

        u_new[idx] = u[idx] + alpha * dt * (
            (u[down] - 2.0 * u[idx] + u[up]) / dx2 +
            (u[right] - 2.0 * u[idx] + u[left]) / dy2
        );
    }
}
"""


def solve_heat_pycuda(nx=128, ny=128, nt=100, alpha=0.01, dx=1.0, dy=1.0, dt=0.1):
    dx2, dy2 = dx ** 2, dy ** 2
    N = nx * ny

    u = np.zeros(N, dtype=np.float64)
    u_new = np.zeros_like(u)

    # горячая точка по центру
    u[(nx // 2) * ny + (ny // 2)] = 100.0

    # компиляция ядра
    mod = SourceModule(kernel_code)
    heat_step = mod.get_function("heat_step")

    # выделение памяти на GPU
    u_gpu = drv.mem_alloc(u.nbytes)
    u_new_gpu = drv.mem_alloc(u.nbytes)

    drv.memcpy_htod(u_gpu, u)
    drv.memcpy_htod(u_new_gpu, u_new)

    block_size = (16, 16, 1)
    grid_size = ((nx - 2) // 16 + 1, (ny - 2) // 16 + 1)

    start = time.time()
    for _ in range(nt):
        heat_step(u_gpu, u_new_gpu,
                  np.float64(alpha), np.float64(dt),
                  np.float64(dx2), np.float64(dy2),
                  np.int32(nx), np.int32(ny),
                  block=block_size, grid=grid_size)
        u_gpu, u_new_gpu = u_new_gpu, u_gpu
    drv.Context.synchronize()
    end = time.time()

    # результат обратно на CPU
    drv.memcpy_dtoh(u, u_gpu)
    u = u.reshape((nx, ny))
    return u, end - start


if __name__ == "__main__":
    for n in [64, 128, 256]:
        print(f"\nGrid size: {n}x{n}")
        _, T = solve_heat_pycuda(nx=n, ny=n, nt=200)
        print(f"Time elapsed: {T:.4f} s")