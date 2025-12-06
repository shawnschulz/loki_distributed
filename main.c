#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <mpi.h>

void launch_multiply(const float* a, float* b);

// make a modified version of this code that copies an input file path onto a path called "backup" on each host
int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("cpu initialized rank: %d", rank);
    float* a = (float *)malloc(10000 * sizeof(float));
    float* b = (float *)malloc(10000 * sizeof(float));
    for (int i = 0; i < 10000; i++) {
        a[i] = 3;
        b[i] = 5;
    }
    launch_multiply(a, b);

    float total = 0;
    for (int i = 0; i < 10000; i++) {
        total += b[i];
    }
    printf("total: %f \n", total);

    MPI_Finalize();
    return 0;
}
