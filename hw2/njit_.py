import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads
import time


@njit(parallel=True)
def heat2d_cpu_chunked(u, u_new, n, alpha, dt, dx):
    for i in prange(1, n - 1):
        for j in range(1, n - 1):
            u_new[i, j] = u[i, j] + alpha * dt / (dx * dx) * (
                u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - 4 * u[i, j]
            )


def solve_heat2d_cpu(n=128, steps=1000, dt=None, alpha=1.0, threads=4):
    set_num_threads(threads)
    print(f"🧵 Используется потоков: {get_num_threads()}")

    dx = 1.0 / (n - 1)
    dx2 = dx * dx

    max_dt = 0.25 * dx2 / alpha
    if dt is None or dt > max_dt:
        print(f"[!] Переопределён dt = {dt} → {max_dt:.2e} (для устойчивости)")
        dt = max_dt

    u = np.ones((n, n), dtype=np.float64)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    u_new = np.copy(u)

    start = time.time()

    for step in range(steps):
        heat2d_cpu_chunked(u, u_new, n, alpha, dt, dx)
        u, u_new = u_new, u

    elapsed = time.time() - start
    print(f"⏱️ Время (n={n}, steps={steps}): {elapsed:.4f} с")

    if np.isnan(u).any():
        print("⚠️ Предупреждение: найдены NaN в результатах!")

    return u


if __name__ == "__main__":
    result = solve_heat2d_cpu(n=256, steps=100000, dt=0.0001, threads=8)
    print(result)
