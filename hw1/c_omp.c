#include <omp.h>
#include <stdio.h>

int main() {
    const int N = 100000000;
    const int P_list[] = {1, 2, 4, 8, 16};
    const int num_tests = sizeof(P_list) / sizeof(P_list[0]);

    for (int test = 0; test < num_tests; test++) {
        int P = P_list[test];
        omp_set_num_threads(P);

        double dx = 1.0 / N;
        double sum = 0.0;

        double start_time = omp_get_wtime();

        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < N; i++) {
            double x = (i + 0.5) * dx;
            sum += 4.0 / (1.0 + x * x);
        }
        double pi = sum * dx;
        double end_time = omp_get_wtime();

        printf("P=%d,\tTime: %f sec, \tPi ≈ %.10f\n", P, end_time - start_time, pi);
    }

    return 0;
}
