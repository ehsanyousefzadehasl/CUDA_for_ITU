#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32  // Number of threads per block
#define RADIUS 3        // Stencil radius
#define ARRAY_SIZE 100   // Input array size (for demonstration)

// CUDA kernel: 1D Stencil
__global__ void stencil_1d(int *in, int *out, int n) {
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];  // Shared memory with halo

    int gindex = threadIdx.x + blockIdx.x * blockDim.x;  // Global index
    int lindex = threadIdx.x + RADIUS;                  // Local index in shared memory

    // Load input elements into shared memory
    temp[lindex] = (gindex < n) ? in[gindex] : 0;

    // Load the halo (left and right)
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = (gindex >= RADIUS) ? in[gindex - RADIUS] : 0;
        temp[lindex + BLOCK_SIZE] = (gindex + BLOCK_SIZE < n) ? in[gindex + BLOCK_SIZE] : 0;
    }

    // Synchronize (ensure all the data is available before computing)
    __syncthreads();

    // Apply the stencil
    if (gindex < n) {  // Ensure we're within bounds
        int result = 0;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            result += temp[lindex + offset];
        }
        // Store the result in the output array
        out[gindex] = result;
    }
}

int main() {
    // Input and output array sizes
    int n = ARRAY_SIZE;

    // Host memory allocation
    int *h_in = (int *)malloc(n * sizeof(int));
    int *h_out = (int *)malloc(n * sizeof(int));

    // Initialize the input array with some values
    printf("Input array:\n");
    for (int i = 0; i < n; i++) {
        h_in[i] = i + 1;  // Example: [1, 2, 3, ..., n]
        printf("%d ", h_in[i]);
    }
    printf("\n");

    // Device memory allocation
    int *d_in, *d_out;
    cudaMalloc((void **)&d_in, n * sizeof(int));
    cudaMalloc((void **)&d_out, n * sizeof(int));

    // Copy input array to device
    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the stencil kernel
    stencil_1d<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the output array
    printf("Output array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
