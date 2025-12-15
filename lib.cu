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
__global__ void _relu_f16_kernel(float *x, float *y, int N) {
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
    kl_matrix += a[i] * log(a[i] / b[i]);
}

// Make a stochastic gradient descent kernel
__global__ void _sgd(float a[BLOCK_SIZE][BLOCK_SIZE], float b[BLOCK_SIZE][BLOCK_SIZE]) {
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
    _kl_divergence <<< blocksPerGrid, threadsPerBlock >>>(a_gpu, b_gpu, *kl);
    cudaMemcpy(b, b_gpu, size, cudaMemcpyDeviceToHost);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
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
