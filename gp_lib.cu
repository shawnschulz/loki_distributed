#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
    

#define BLOCK_SIZE 256
#define GRID_SIZE 1

// initialize the convolution kernel in constant memory
#define MASK_WIDTH 5
#define TITLE_SIZE 4
#define INPUT_SIZE 12

// The RBF kernel calculater
// Radial basis function (RBF) kernel is the squared euclidean distance between two feature vectors modified by a free parameter.
// wew use it to measure covariance or
// for ease of use let's grab CUDNN to make this much more trivial. we still make use of Cuda by dictating CPU -> GPU movements
// to use our dual GPU setup
// 
// 0 0 0   1 
// 0 0 0   2 
// 0 0 0   3
//         4
//         5
//         6
// Note x should include both the input features and new features X_new in a flat array
// assume that X and X_new have same shape, so you can just append the arrays together
__global__ void _RBF_kernel(float *K, float *x, int n, int m, int  n_x, int m_x, float l) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // take the norm of row m subtracted into row n in x
    // each thread needs to work on the K[n][m]'th element of the flat array representation
    // and index into the x[n] and x[n] row of 
    K[r * m + c] += powf(expf(-sqrtf(powf(x[i + m] - x[i + n], 2.0))), 2.0) / (2 * powf(l, 2.0)))
}

// Use the rbf kernel result to calculate the mean using the dot product of sections of hte rbf kernel
__global__ void _gaussian_process_mean() {
} 
// predict standard using dot product of inverse of the kernel
__global__ void _gaussian_process_std() {
}

// Idk why the cublas api i can't seem to find a sqrt kernel, so this 97 year old programmer will just do it the old fashioned way
__global__ void _sqrt_kernel(float *K) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // hopefully we don't race condition here copium
    K[i] = sqrtf(K[i]);
}

__global__ void _scale_by_length(float *K, float l) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    K[i] = K[i] / 2 * powf(l, 2.0);
}

// lda = leading dimension of a
void cublas_rbf_kernel(float* A, float* K, int m, int phi, int lda, cublasHandle_t handle) {
    // initialize a phi x phi kernel matrix with 0s
    cudaMemset(K, 0, phi * phi * sizeof(float));
    const float alpha = 1.0f, beta = 0.0f;
    const float l = 1.0f;
    // A_T * A gives sum of squares, put that into K first
    // note that CUBLAS_OP_T transposes A
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, phi, phi, m, &alpha, A, lda, A, lda, &beta, K, n);
    // next sqrt everything
    _sqrt_kernel <<< blocksPerGrid, threadsPerBlock >>>(K);
    _scale_by_length <<< blocksPerGrid, threadsPerBlock >>>(K, l);
    thrust::multiplies<thrust::complex<float> > op;
    // blah blah set up all the sigma mat stuff here
    float* sigma_mat;
    thrust::transform(thrust::device, K, K + phi, sigma_mat, z, op);
}

void launch_gp_model() {
    // High level for rbf kernel:
    // 1. construct a single matrix X with phi rows of input
    // 2. for all pairwise vector row norms i, j in matrix X, update the K[i, j] value with the norm.
    //    then add sigma times the phi x phi identity
    // 3. both the mean and stdev predictions are a simple (but a bit tricky to get syntax right) dot product
    //    of our training observations y and parts of the kernel K. thus if we calculate the kernel the rest is possible 
    const int n = 5;
    const float alpha = 1.0f, beta = -1.0f;
    float X[] = {1.0, 3.0, 5.0, 7.0, 9.0};
    float X_new[] = {5.5};
    float A[] = {1.0, 3.0, 5.0, 7.0, 9.0, 5.5};
    float Y[] = {16.0, 4.0, 0.0, 4.0, 16.0};
    float y_new[5];
    float *K, *x_new_gpu, *x_gpu, *y_gpu;
    cudaMalloc(x_gpu, 5 * sizeof(float));
    cudaMalloc(y_gpu, 5 * sizeof(float));
    cudaMalloc(K_gpu, 5 * sizeof(float));
    cudaMalloc(y_new_gpu, 5 * sizeof(float));
    cudaMalloc(x_new_gpu, 1 * sizeof(float));
    cudaMemcpy(x_gpu, X, 5 * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(y_gpu, Y, 5 * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(x_new_gpu, X_new, 1 * sizeof(float), cudaMemcpyHostToDevice); 
    cublasHandle_t handle;
    cublasCreate(&handle);
    float norm;
    // Can use the saxpy vector by vector but could we do it on a big matrix with all of the feature vectors?
    cublasSaxpy(handle, n, &alpha, x_gpu, 1, 
    cublasSnrm2(handle, n, K_gpu, 1, &norm);
}

void main() {
    launch_gp_model();
}
