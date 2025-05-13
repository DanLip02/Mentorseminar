import numpy as np
from numba import cuda
import time


@cuda.jit
def heat2d_kernel(u, u_new, n, alpha, dt, dx):
    i, j = cuda.grid(2)
    if 0 < i < n - 1 and 0 < j < n - 1:
        u_new[i, j] = u[i, j] + alpha * dt / (dx * dx) * (
            u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - 4 * u[i, j])


def is_valid_block_size(block_size, n):
    """Проверяет, не превышает ли общее число потоков на блок допустимое значение (1024),
    и укладывается ли сетка в размер массива."""
    tx, ty = block_size
    if tx * ty > 1024:
        return False
    blocks_x = (n + tx - 1) // tx
    blocks_y = (n + ty - 1) // ty
    return blocks_x > 0 and blocks_y > 0


def solve_heat2d_numba(n=100, steps=1000, dt=0.0001, block_size=(16, 16)):
    if not is_valid_block_size(block_size, n):
        print(f"[!] Block size {block_size} недопустим для n={n}. Пропускаем...")
        return None

    alpha = 1.0
    dx = 1.0 / (n - 1)

    # Initialize
    u = np.ones((n, n), dtype=np.float64)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    u_new = np.copy(u)

    d_u = cuda.to_device(u)
    d_u_new = cuda.to_device(u_new)

    threads_per_block = block_size
    blocks_per_grid_x = (n + block_size[0] - 1) // block_size[0]
    blocks_per_grid_y = (n + block_size[1] - 1) // block_size[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    try:
        start = time.time()
        for step in range(steps):
            heat2d_kernel[blocks_per_grid, threads_per_block](
                d_u, d_u_new, n, alpha, dt, dx)
            d_u, d_u_new = d_u_new, d_u

        cuda.synchronize()
        elapsed = time.time() - start
        print(f"Time (block_size={block_size}): {elapsed:.4f} s")
        return d_u.copy_to_host()

    except cuda.cudadrv.driver.CudaAPIError as e:
        print(f"[CUDA ERROR] Block size {block_size} вызвал ошибку: {e}")
        return None


if __name__ == "__main__":
    for block in [8, 16, 24, 32, 48, 64, 128]:
        print(f"\n>>> Пробуем block_size = ({block}, {block})")
        result = solve_heat2d_numba(n=512, steps=10000, dt=0.0000001, block_size=(block, block))
        if result is not None:
            print(result)
