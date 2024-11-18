#include <stdio.h>
#include <stdlib.h>
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

    int size = atoi(argv[1]);
    if (size <= 0) {
        printf("Array size must be a positive integer.\n");
        return 1;
    }

    int *input = (int *)malloc(size * sizeof(int));
    int *output = (int *)malloc(size * sizeof(int));
    int *d_input, *d_output;
    int bytes = size * sizeof(int);

    for (int i = 0; i < size; i++) {
        input[i] = i + 1;
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaMalloc((void **)&d_input, bytes);
    cudaMalloc((void **)&d_output, bytes);

    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    

    // Start GPU timing
    cudaEventRecord(start);

    // Launch the kernel
    squareArrayGPU<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
    

    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);



    // Stop GPU timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    printf("GPU Kernel Execution Time: %.6f ms\n", gpu_time);


    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);

    return 0;
}
