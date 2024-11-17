#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void squareArrayGPU(int *d_input, int *d_output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_output[idx] = d_input[idx] * d_input[idx];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]); // Get array size from command line
    if (size <= 0) {
        printf("Array size must be a positive integer.\n");
        return 1;
    }

    int *input = (int *)malloc(size * sizeof(int));
    int *output = (int *)malloc(size * sizeof(int));
    int *d_input, *d_output;
    int bytes = size * sizeof(int);

    // Initialize input array
    for (int i = 0; i < size; i++) {
        input[i] = i + 1;
    }

    // Start timing (CPU timing, including memory operations and kernel execution)
    clock_t start = clock();

    // Allocate device memory
    cudaMalloc((void **)&d_input, bytes);
    cudaMalloc((void **)&d_output, bytes);

    // Copy data from host to device
    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    squareArrayGPU<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);

    // Copy results back to host
    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Stop timing
    clock_t end = clock();
    double gpu_time = (double)(end - start) / CLOCKS_PER_SEC;

    // Print execution time
    printf("GPU Execution Time (including memory operations): %.6f seconds\n", gpu_time);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(input);
    free(output);

    return 0;
}

