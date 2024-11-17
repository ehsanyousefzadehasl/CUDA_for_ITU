#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define FILTER_RADIUS 1

__constant__ int SOBEL_X[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int SOBEL_Y[3][3] = {
    {-1, -2, -1},
    {0,  0,  0},
    {1,  2,  1}
};

__global__ void sobelEdgeDetection(const unsigned char *input, unsigned char *output, int width, int height) {
    __shared__ unsigned char tile[TILE_SIZE + 2 * FILTER_RADIUS][TILE_SIZE + 2 * FILTER_RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx - FILTER_RADIUS;
    int row = blockIdx.y * TILE_SIZE + ty - FILTER_RADIUS;

    if (row >= 0 && row < height && col >= 0 && col < width) {
        tile[ty][tx] = input[row * width + col];
    } else {
        tile[ty][tx] = 0;
    }
    __syncthreads();

    if (tx >= FILTER_RADIUS && tx < TILE_SIZE + FILTER_RADIUS &&
        ty >= FILTER_RADIUS && ty < TILE_SIZE + FILTER_RADIUS &&
        row < height && col < width) {
        int Gx = 0, Gy = 0;

        for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
            for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
                Gx += SOBEL_X[i + FILTER_RADIUS][j + FILTER_RADIUS] * tile[ty + i][tx + j];
                Gy += SOBEL_Y[i + FILTER_RADIUS][j + FILTER_RADIUS] * tile[ty + i][tx + j];
            }
        }

        int magnitude = abs(Gx) + abs(Gy);
        output[row * width + col] = (magnitude > 255) ? 255 : magnitude;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char *h_input = stbi_load(argv[1], &width, &height, &channels, 1); // Load as grayscale
    if (!h_input) {
        printf("Failed to load image: %s\n", argv[1]);
        return 1;
    }

    int imageSize = width * height;
    unsigned char *h_output = (unsigned char *)malloc(imageSize * sizeof(unsigned char));

    unsigned char *d_input, *d_output;
    cudaMalloc((void **)&d_input, imageSize * sizeof(unsigned char));
    cudaMalloc((void **)&d_output, imageSize * sizeof(unsigned char));

    cudaMemcpy(d_input, h_input, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE + 2 * FILTER_RADIUS, TILE_SIZE + 2 * FILTER_RADIUS);
    dim3 blocksPerGrid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    sobelEdgeDetection<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);

    cudaMemcpy(h_output, d_output, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_png(argv[2], width, height, 1, h_output, width); // Save result as PNG

    printf("Edge detection complete. Result saved to: %s\n", argv[2]);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
