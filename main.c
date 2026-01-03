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

    // to-dos:
    // 1. need the token encoding to be loaded in here (and also correctly flattened)
    // 2. need the token encoding to be finally transformed into an embedding. this can be
    // partial and split up between ranks or complete
    // 3. try and debug running it thru a kernel of the right shape. we should check outputs and
    // use small dimension sizes to start

    // Require the data_handler shared object library at this point
    const float* answers = tokenize("data.parquet", &nrows);
    const float* problems = tokenize("data.parquet", &nrows);

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("cpu initialized rank: %d", rank);
    // For now allocate more than enough for 1 GB of token data
    // these memory allocations are wonky but we just want to check whether htis
    // works first
    // Also for now ignore splitting amongst ranks. eventually want each rank to
    // independently load a portion of the data nad perform activation only
    // for their poriton
    float* tokens = C_data_loader("", "");
    float* embedding = (float *)malloc(2.5e9 * sizeof(float));

    launch_transformer_activation(tokens, embedding, 512, 2.5e8);

    // For now, check first and last embedding
    float total = 0;
    for (int i = 0; i < 10000; i++) {
        total += b[i];
    }
    printf("total: %f \n", total);

    MPI_Finalize();
    return 0;
}
