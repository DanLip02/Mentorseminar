#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 100000000;
    double dx = 1.0 / N;
    double local_sum = 0.0;
    int chunk = N / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? N : start + chunk;

    double start_time = MPI_Wtime();
    for (int i = start; i < end; i++) {
        double x = (i + 0.5) * dx;
        local_sum += 4.0 / (1.0 + x * x);
    }
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double pi = global_sum * dx;
        double end_time = MPI_Wtime();
        printf("P: %d\tTime: %f sec\tPi: %.10f\n", size, end_time - start_time, pi);
    }

    MPI_Finalize();
    return 0;
}