import time
import numba
import numpy as np
import sys

# P = int(sys.argv[1])
numba.set_num_threads(2)  # Принудительно 4 потока

@numba.njit(parallel=True)
def compute_pi_omp(N=10 ** 8):
    dx = 1.0 / N
    local_sum = 0.0

    # Параллельный цикл
    for i in numba.prange(N):
        x = (i + 0.5) * dx
        local_sum += 4.0 / (1.0 + x ** 2)

    return local_sum * dx


def main():
    N = 10 ** 8
    check = []
    for _ in range(100):
        start_time = time.time()
        pi = compute_pi_omp(N)
        end_time = time.time()
        check.append(end_time - start_time)

    print(f"Time: {sum(check) / len(check):.6f} sec, Pi ≈ {pi:.10f}")


if __name__ == "__main__":
    print(f"Numba detected {numba.get_num_threads()} threads")
    main()
