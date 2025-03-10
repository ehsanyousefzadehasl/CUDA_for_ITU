#include <stdio.h>
#include <cuda_runtime.h>
#include <limits.h>

#define N 1024  // Size of the array
#define THREADS_PER_BLOCK 256

__global__ void maxReduceKernel(int *arr, int *maxVal, int n) {
    __shared__ int sharedData[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Load data into shared memory
    if (global_tid < n) {
        sharedData[tid] = arr[global_tid];
    } else {
        sharedData[tid] = INT_MIN; // Handle out-of-bounds case
    }
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] = max(sharedData[tid], sharedData[tid + stride]);
        }
        __syncthreads();
    }
    
    // Write block maximum to global memory
    if (tid == 0) {
        maxVal[blockIdx.x] = sharedData[0];
    }
}

__global__ void finalMaxReduceKernel(int *maxVal, int *finalMax, int numBlocks) {
    __shared__ int sharedData[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    
    if (tid < numBlocks) {
        sharedData[tid] = maxVal[tid];
    } else {
        sharedData[tid] = INT_MIN;
    }
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] = max(sharedData[tid], sharedData[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *finalMax = sharedData[0];
    }
}

// CPU function to find the maximum value
int findMaxCPU(int *arr, int n) {
    int maxVal = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > maxVal) {
            maxVal = arr[i];
        }
    }
    return maxVal;
}

int main() {
    int h_arr[N];
    for (int i = 0; i < N; i++) {
        h_arr[i] = rand() % 21000; // Fill array with random numbers
    }
    
    int *d_arr, *d_maxVal, *d_finalMax;
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int *h_maxVal = (int *)malloc(numBlocks * sizeof(int));
    int h_finalMax;
    
    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMalloc(&d_maxVal, numBlocks * sizeof(int));
    cudaMalloc(&d_finalMax, sizeof(int));
    
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    
    maxReduceKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_arr, d_maxVal, N);
    
    finalMaxReduceKernel<<<1, numBlocks>>>(d_maxVal, d_finalMax, numBlocks);
    cudaMemcpy(&h_finalMax, d_finalMax, sizeof(int), cudaMemcpyDeviceToHost);
    
    int cpuMax = findMaxCPU(h_arr, N);
    
    printf("GPU Maximum value: %d\n", h_finalMax);
    printf("CPU Maximum value: %d\n", cpuMax);
    
    cudaFree(d_arr);
    cudaFree(d_maxVal);
    cudaFree(d_finalMax);
    free(h_maxVal);
    
    return 0;
}
