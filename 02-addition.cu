#include<iostream>
#include<cuda_runtime.h>

// kernel for adding two array
__global__ void add(int *A, int *B, int *C, int N){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	
	printf("\nthread ID: %d , block Id: %d", threadIdx.x, blockIdx.x);

	if(index < N) {
		C[index] = A[index] + B[index];
	}	
}

int main() {
	int N = 2000;

	int size = N * sizeof(int);

	// allocate host memory

	int *h_A = new int[N];
	int *h_B = new int[N];
	int *h_C = new int[N];


	// initializing the array
	for(int i=1; i <= N; ++i){
		h_A[i] = i;
		h_B[i] = i;
	}

	// allocating GPU memory
	int *d_A, *d_B, *d_C;
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);
	
	// copy data from CPU to GPU
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	// LAUNCHING THE GPU KERNEL
	int threadsPerBlock = 256;

	int remainder = N % threadsPerBlock;
	int blocksPerGrid = 0;

	if(remainder == 0){
		blocksPerGrid = N/ threadsPerBlock;
	} else {
		blocksPerGrid = N/ threadsPerBlock + 1;
	}
	add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

	
	// copy results from GPU to CPU
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	
	// printing results
	for(int i = 0; i < N; ++i){
		std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
	}


	// free memory for CPU
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	
	// free memory for GPU
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
