#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

// Matrix multiplication kernel
__global__ void matrixMulGPU(float *d_A, float *d_B, float *d_C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0;
        for (int k = 0; k < N; k++) {
            value += d_A[row * N + k] * d_B[k * N + col];
        }
        d_C[row * N + col] = value;
    }
}

// CPU matrix multiplication
void matrixMulCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float value = 0.0;
            for (int k = 0; k < N; k++) {
                value += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = value;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]); // Matrix size NxN
    if (N <= 0) {
        printf("Matrix size must be a positive integer.\n");
        return 1;
    }

    int size = N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_CPU = (float *)malloc(size);
    float *h_C_GPU = (float *)malloc(size);

    float *d_A, *d_B, *d_C;

    // Initialize matrices with random values
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 10; // Random values between 0 and 9
        h_B[i] = rand() % 10;
    }

    // GPU memory allocation
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Define thread block and grid dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // GPU computation timing (including data transfers)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Copy matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    matrixMulGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C_GPU, d_C, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // CPU computation timing
    clock_t cpu_start = clock();
    matrixMulCPU(h_A, h_B, h_C_CPU, N);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Compare GPU and CPU results
    int error = 0;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C_GPU[i] - h_C_CPU[i]) > 1e-5) {  // Adjust tolerance if needed
            error = 1;
            break;
        }
    }

    if (!error) {
        printf("Results match!\n");
    } else {
        printf("Results do NOT match!\n");
    }

    // Print timing results
    printf("GPU Execution Time (including data transfers): %.6f ms\n", gpu_time);
    printf("CPU Execution Time: %.6f seconds\n", cpu_time);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_CPU);
    free(h_C_GPU);

    return 0;
}
