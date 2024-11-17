#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256  // Number of threads per block
#define ARRAY_SIZE 1024  // Size of the input arrays

// Kernel to compute dot product using shared memory
__global__ void dotProductSharedMemory(const float *a, const float *b, float *result, int n) {
    __shared__ float partialSum[BLOCK_SIZE];  // Shared memory for block-wise partial sums

    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Global index
    int tx = threadIdx.x;                            // Local thread index within the block

    // Compute partial dot product
    if (idx < n) {
        partialSum[tx] = a[idx] * b[idx];
    } else {
        partialSum[tx] = 0.0f;  // Handle out-of-bounds threads
    }

    __syncthreads();  // Synchronize threads in the block

    // Perform reduction within shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            partialSum[tx] += partialSum[tx + stride];
        }
        __syncthreads();  // Synchronize threads after each reduction step
    }

    // Write the block's partial sum to global memory
    if (tx == 0) {
        atomicAdd(result, partialSum[0]);  // Atomic operation to safely accumulate the result
    }
}

int main() {
    // Host data allocation
    int size = ARRAY_SIZE * sizeof(float);
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_result = (float *)malloc(sizeof(float));

    // Initialize input arrays and result
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_a[i] = 1.0f;  // Example: Initialize all elements to 1.0
        h_b[i] = 2.0f;  // Example: Initialize all elements to 2.0
    }
    *h_result = 0.0f;

    // Device data allocation
    float *d_a, *d_b, *d_result;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_result, sizeof(float));

    // Copy input arrays to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    dotProductSharedMemory<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, ARRAY_SIZE);

    // Copy the result back to host
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Dot Product: %f\n", *h_result);

    // Free device and host memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    free(h_a);
    free(h_b);
    free(h_result);

    return 0;
}
