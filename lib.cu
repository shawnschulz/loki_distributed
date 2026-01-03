#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 256
#define GRID_SIZE 1

// initialize the convolution kernel in constant memory
#define MASK_WIDTH 5
#define TITLE_SIZE 4
#define INPUT_SIZE 12

__constant__ float M[MASK_WIDTH];

__global__ void _multiply(const float *a, float *b) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    b[i] *= a[i];
}

// activation
__global__ void _relu_kernel(float *x, float *y, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (x[i] < N) {
        y[i] = fmaxf(0.0f, x[i]);
    }
}

// convolution forward
__global__ void _convolution_layer_smp_fp16(float *input_data, float *output_data, int width) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float input_data_shared[TITLE_SIZE];
    input_data_shared[threadIdx.x] = input_data[i];
}

// loss
__global__ void _kl_divergence(float *a, float *b, float *kl_matrix) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    *kl_matrix += a[i] * log(a[i] / b[i]);
}

// Make a stochastic gradient descent kernel
__global__ void _sgd(float a[BLOCK_SIZE][BLOCK_SIZE], float b[BLOCK_SIZE][BLOCK_SIZE]) {
}


__global__ void _mulithead_attention(int* tokenized_chars, int vocab_length, float* output_logits) {

}

// train an embedding model using self attention to transform tokens into single precision vectors of size 512
// the model is a shallow 2 layer unsupervised model.
__global__ void _train_input_embedding_model(int* tokenized_chars, float* output_weights, int vocab_size, int input_size, int dim_size = 512) {
    // perform self attention

}
// Encode embeddings as a positional embedding using sinf and cosf positional embedding functions
// input to model i think has (dim_1, n_positions, model_dimensionality) shape
__global__ void _positional_encoder(int* tokenized_chars, float* embedded_input, int dim_1, int n_positions, int model_dimensionality) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i % 2 == 0 ) {
        sinf()
    }

}

// Important to note that this converts are initial tokens into a 512 x n_tokens matrix so we can
// actually do multihead attention
__global__ void _softmax(float *a, float *b, float *sum, int embedding_size = 512) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // shouldn't all be i, i is the embedding dims but it gets it depending on token index
    *sum += exp(a[i % embedding_size]);
    b[i] = exp(a[i % embedding_size]);
    _sync_threads();
    b[i] = b[i] / *sum;
}

// Take the input tokens and produce an initialized embedding for input to multi-head attention
// I think we eventulaly need to set a shard_offset argument, so that we can softmax at arbitrary indices of the token size
extern "C" void launch_transformer_activation(const float*tokens, float*embedding, size_t vector_dim, size_t n_tokens) {
    float* tokens_gpu;
    float* embeddings_gpu;
    float gpu_sum;
    cudaMalloc(&tokens_gpu, n_tokens);
    cudaMemcpy(tokens_gpu, tokens, n_tokens, cudaMemcpyHostToDevice);
    cudaMemcpy(&gpu_sum, 0.0, 1, cudaMemcpyHostToDevice);
    cudaMalloc(&embeddings_gpu, vector_dim * n_tokens);

    int threadsPerBlock = 256;
    int blocksPerGrid =
            (10000 + threadsPerBlock - 1) / threadsPerBlock;
    _softmax <<< blocksPerGrid, threadsPerBlock >>>(tokens_gpu, embeddings_gpu, &gpu_sum, vector_dim);
    cudaMemcpy(embedding, embeddings_gpu, size, cudaMemcpyDeviceToHost);
}

extern "C" void launch_VAE_inference(const float*a, float*b) {
    size_t size = 10000 * sizeof(float);
    float* a_gpu;
    cudaMalloc(&a_gpu, size);
    cudaMemcpy(a_gpu, a, size, cudaMemcpyHostToDevice);
    float* b_gpu;
    cudaMalloc(&b_gpu,  size);
    cudaMemcpy(b_gpu, b, size, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    float kl;
    int blocksPerGrid =
            (10000 + threadsPerBlock - 1) / threadsPerBlock;
    _kl_divergence <<< blocksPerGrid, threadsPerBlock >>>(a_gpu, b_gpu, &kl);
    cudaMemcpy(b, b_gpu, size, cudaMemcpyDeviceToHost);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
}

// Take input gene expression matrix and produce initial activation embedding for longevity VAE model
extern "C" void launch_VAE_activation(const float*X, float*embedding, size_t n_cells, size_t n_genes) {
    float* X_gpu;
    float* embedding_gpu;
    cudaMalloc(&X_gpu, n_cells * n_genes);
    cudaMemcpy(X_gpu, X, n_cells * n_genes, cudaMemcpyHostToDevice);
}

extern "C" void launch_multiply(const float*a, float*b) {
    size_t size = 10000 * sizeof(float);
    float* a_gpu;
    cudaMalloc(&a_gpu, size);
    cudaMemcpy(a_gpu, a, size, cudaMemcpyHostToDevice);
    float* b_gpu;
    cudaMalloc(&b_gpu,  size);
    cudaMemcpy(b_gpu, b, size, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (10000 + threadsPerBlock - 1) / threadsPerBlock;
    _multiply <<< blocksPerGrid, threadsPerBlock >>>(a_gpu, b_gpu);
    cudaMemcpy(b, b_gpu, size, cudaMemcpyDeviceToHost);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
}
