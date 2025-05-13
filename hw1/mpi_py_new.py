from mpi4py import MPI
import time
import sys

def get_pi(num_steps, rank, size):
    h = 1.0 / num_steps
    local_sum = 0.0
    for i in range(rank, num_steps, size):
        x = h * (i + 0.5)  # Метод средних прямоугольников
        local_sum += 4.0 / (1.0 + x**2)
    return local_sum

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    real_size = comm.Get_size()  # Реальное количество процессов
    size = 12  # Жестко заданное количество процессов

    # Если процессов больше, чем 12, игнорируем лишние
    if real_size > size:
        if rank >= size:
            sys.exit()  # Останавливаем лишние процессы, если их больше 12

    # Количество точек для интегрирования
    N = 10**8
    num_iterations = 1  # Количество итераций для усреднения времени

    # Поделим работу на P процессов
    P = min(size, real_size)

    total_time = 0.0
    global_sum = 0.0

    for _ in range(num_iterations):
        comm.Barrier()  # Синхронизация процессов перед измерением времени
        start_time = time.time()

        local_sum = get_pi(N, rank, P)

        # Суммируем результаты всех процессов
        global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

        end_time = time.time()
        iteration_time = end_time - start_time

        # Собираем общее время выполнения
        global_time = comm.reduce(iteration_time, op=MPI.SUM, root=0)

    if rank == 0:
        pi = global_sum / N  # Финальный расчет числа pi
        avg_time = global_time / size  # Усредняем по всем процессам
        print(f"P={size}, Avg Time: {avg_time:.6f} sec, Pi ≈ {pi:.10f}")

if __name__ == "__main__":
    main()
