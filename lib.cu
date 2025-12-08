#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 256
#define GRID_SIZE 1

// initialize the convolution kernel in constant memory
#define MASK_WIDTH 5
__constant__ __half M[MASK_WIDTH];
cudaMemcpyToSymbol(M,h_M,MASK_WIDTH*sizeof(float));

__global__ void _multiply(const float *a, float *b) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    b[i] *= a[i];
}

__global__ void _relu_f16_kernel(__half *x, __half *y, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        y[i] = fmaxf(0.0f, x[i]);
    }
}

__global__ void _convolution_layer(__half *input_data, __half *output_data, int width) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
}

__global__ void _kl_divergence(__half *input_data, __half *output_data) {
}

// Make a stochastic gradient descent kernel
__global__ void _sgd(__half a[BLOCK_SIZE][BLOCK_SIZE], __half b[BLOCK_SIZE][BLOCK_SIZE]) {
}

extern "C" void launch_VAE_inference(const float*a, float*b) {
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
