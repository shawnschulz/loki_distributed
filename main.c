#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <mpi.h>
#include "data_handler.h"

void launch_multiply(const float* a, float* b);

// make a modified version of this code that copies an input file path onto a path called "backup" on each host
// GENERAL IDEA:
// Train the dataset by splitting the parquet files problem and answer into test training validation
// tokenize the test training validation
// input to the model to train and validate
// create a set of weight outputs (may need to compress these)
// expose an inference function somewhere that is also distributed, this inference function should take variable length
// at some point the vocabulary should get serialized somwhere so we can detokenize output tokens from the model
// byte characters as input, tokenize them, run the model inference to get output tokens, then detokenize into byte characters
// output to user, preferably streamed to command line as the tokens are generated
// Give this GPL-3 code out to people. hopefully it works okay to train models on modest hardware

int main(int argc, char** argv) {

    // Should loop over files and pause data loading as it is being aggregated when a memroy cap
    // is reached, then spread the data out until a memory cap is reached per host, then
    // digest data by running device kernel and serializing a shard

    // Require the data_handler shared object library at this point
    const float* answers = tokenize("data.parquet", &nrows);
    const float* problems = tokenize("data.parquet", &nrows);

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
