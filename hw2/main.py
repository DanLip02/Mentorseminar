import numpy as np
import matplotlib.pyplot as plt
import time
import numba
from numba import njit, prange
import os

def initialize_grid(n, L=1.0):
    dx = L / n
    u = np.zeros((n, n), dtype=np.float32)
    cx, cy = n // 2, n // 2
    u[cx-1:cx+2, cy-1:cy+2] = 0.0  # горячий центр
    return u, dx

@njit(parallel=True)
def solve_heat_eq_numba(u0, alpha, dt, dx, dy, steps):
    n = u0.shape[0]
    u = u0.copy()
    for step in range(steps):
        u_new = u.copy()
        for i in prange(1, n - 1):
            for j in range(1, n - 1):
                u_new[i, j] = u[i, j] + alpha * dt * (
                    (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx**2 +
                    (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy**2
                )
        u = u_new
    return u

def benchmark(n, steps, alpha, dt, threads):
    numba.set_num_threads(threads)
    u0, dx = initialize_grid(n)
    dy = dx
    start = time.time()
    solve_heat_eq_numba(u0, alpha, dt, dx, dy, steps)
    end = time.time()
    return end - start

def run_experiment(ns, ps, steps=1000, alpha=1.0, dt=1e-4):
    results = {}
    for n in ns:
        results[n] = {'Tp': [], 'Sp': [], 'Ep': []}
        print(f"🔧 Grid size: {n}x{n}")
        T1 = None
        for p in ps:
            t = benchmark(n, steps, alpha, dt, p)
            results[n]['Tp'].append(t)
            if p == 1:
                T1 = t
            Sp = T1 / t
            Ep = Sp / p
            results[n]['Sp'].append(Sp)
            results[n]['Ep'].append(Ep)
            print(f"  Threads: {p}  |  T_p: {t:.4f}s  |  S_p: {Sp:.2f}  |  E_p: {Ep:.2f}")
    return results

def plot_results(results, ps):
    for n, data in results.items():
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.plot(ps, data['Tp'], marker='o')
        plt.title(f'Time T_p for n={n}')
        plt.xlabel('Threads p')
        plt.ylabel('Time T_p [s]')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(ps, data['Sp'], marker='o')
        plt.title(f'Speedup S_p for n={n}')
        plt.xlabel('Threads p')
        plt.ylabel('Speedup S_p')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(ps, data['Ep'], marker='o')
        plt.title(f'Efficiency E_p for n={n}')
        plt.xlabel('Threads p')
        plt.ylabel('Efficiency E_p')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ns = [128, 256, 512, 1024]
    ps = [1, 2, 4, 8, 12]
    results = run_experiment(ns, ps)
    plot_results(results, ps)
