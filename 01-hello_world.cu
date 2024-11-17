#include<iostream>
#include<cuda_runtime.h>

// kernel (function) to be executed on the GPU
__global__ void helloWorld() {
	printf("Hello from thread %d from block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
	// launch the GPU kernel
	helloWorld<<<10, 100>>>();

	cudaDeviceSynchronize();


	return 0;
}
