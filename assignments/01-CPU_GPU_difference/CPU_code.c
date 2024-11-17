#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void squareArrayCPU(int *input, int *output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * input[i];
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

    // Initialize input array
    for (int i = 0; i < size; i++) {
        input[i] = i + 1;
    }

    // Start timing
    clock_t start = clock();

    // Calculate squared values using CPU
    squareArrayCPU(input, output, size);

    // Stop timing
    clock_t end = clock();
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;

    // Print execution time
    printf("CPU Execution Time: %.6f seconds\n", cpu_time);

    // Free memory
    free(input);
    free(output);

    return 0;
}

