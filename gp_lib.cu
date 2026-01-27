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


void launch_gp_model() {
    // Since GP models have an analytical solution, its actually pretty straightforward with cublas to
    // program a basic rbf kernel. this is toy data, so we want to make a nice interface for data loading
    // VAF values from .vcf files and have an input for true FDR and VAF (that can just be uncompressed tsv's)
    float X[] = {1.0, 3.0, 5.0, 7.0, 9.0};
    float X_new[] = {5.5};
    flaot Y[] = {16.0, 4.0, 0.0, 4.0, 16.0};
    float *y_new, *K, *x_new_gpu, *x_gpu, *y_gpu;
    cudaMemcpy(x_gpu, X, 5 * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(y_gpu, Y, 5 * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(x_new_gpu, X_new, 1 * sizeof(float), cudaMemcpyHostToDevice); 
    cublasHandle_t handle;
    cublasCreate(&handle);
    float norm;
    cublasSnrm2(handle, n, d_result, 1, &norm);
}

void main() {
    launch_gp_model();
}
