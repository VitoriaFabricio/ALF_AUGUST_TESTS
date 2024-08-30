#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>


#define FILTER_WIDTH 3
#define FILTER_HEIGHT 3
#define TILE_SIZE 32

void checkCUDAError(cudaError_t cudaStatus, const char* errorMessage) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", errorMessage, cudaGetErrorString(cudaStatus));
        exit(1);
    }
}

__global__ void filter(const unsigned int* input_image, unsigned int* output_image_horizontal, unsigned int* output_image_vertical, 
                       unsigned int* output_image_diagonal1, unsigned int* output_image_diagonal2, unsigned int height, unsigned int width) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // x e y global

    // Indices shared_block
    int shared_x = threadIdx.x + 1;
    int shared_y = threadIdx.y + 1;
    

    // ThreadIdx.x and threadIdx.y  (0-31)
    // If threadIdx.x is '0' shared_x is '1'

    __shared__ unsigned int shared_block[(32+2) * (32+2)];
    __shared__ unsigned int shared_block_horizontal[(32+2) * (32+2)];
    __shared__ unsigned int shared_block_vertical[(32+2) * (32+2)];
    __shared__ unsigned int shared_block_diagonal0[(32+2) * (32+2)];
    __shared__ unsigned int shared_block_diagonal1[(32+2) * (32+2)];


    //Center 
    
    if (y < height) {

            shared_block[(shared_y) * (34) + shared_x] = input_image[y * 1920 + x];     
        }
    

    // Corners

    // Top-left corner
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if(x > 0 && y > 0){
              shared_block[0] = input_image[(y - 1) * width + (x - 1)];
        }
    }

    // Top-right corner
    if (threadIdx.x == TILE_SIZE - 1 && threadIdx.y == 0) {
        if(x < width - 1 && y > 0){
             shared_block[34 - 1] = input_image[(y - 1) * width + (x + 1)];
        }
    }

    // Bottom-left corner
    if (threadIdx.x == 0 && threadIdx.y == TILE_SIZE - 1) {
        if(x > 0 && y < height - 1){
            shared_block[(34 - 1) * 34] = input_image[(y + 1) * width + (x - 1)];
        }
    }

    //Bottom-right corner
    if (threadIdx.x == TILE_SIZE - 1 && threadIdx.y == TILE_SIZE - 1) {
        if(x < width - 1 && y < height - 1){
           shared_block[(34 - 1) * 34 + 34 - 1] = input_image[(y + 1) * width + (x + 1)];

        }
    }


    // Edges

    // Left edge
    if (threadIdx.x == 0 && x > 0) {
        shared_block[shared_y * 34] = input_image[y * width + (x - 1)];
    }

    // Right edge
    if (threadIdx.x == TILE_SIZE - 1 && x < width - 1) {
        shared_block[shared_y * 34 + 34 - 1] = input_image[y * width + (x + 1)];
    }

    // Top edge
    if (threadIdx.y == 0 && y > 0) {
        shared_block[shared_x] = input_image[(y - 1) * width + x];
    }
    
    //Bottom edge
    if (threadIdx.y == TILE_SIZE - 1 && y < height - 1) {
        shared_block[(34 - 1) * 34 + shared_x] = input_image[(y + 1) * width + x];
    }

    __syncthreads();

    if (x < width && y < height) {
        int horizontal_filter[FILTER_WIDTH][FILTER_HEIGHT] = {
            {0, 0, 0},
            {1, -2, 1},
            {0, 0, 0}
        };
        int vertical_filter[FILTER_WIDTH][FILTER_HEIGHT] = {
            {0, 1, 0},
            {0, -2, 0},
            {0, 1, 0}
        };
        int diagonal1_filter[FILTER_WIDTH][FILTER_HEIGHT] = {
            {1, 0, 0},
            {0, -2, 0},
            {0, 0, 1}
        };
        int diagonal2_filter[FILTER_WIDTH][FILTER_HEIGHT] = {
            {0, 0, 1},
            {0, -2, 0},
            {1, 0, 0}
        };

        int sum_horizontal = 0;
        int sum_vertical = 0;
        int sum_diagonal1 = 0;
        int sum_diagonal2 = 0;


        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                sum_horizontal += shared_block[(shared_y + i) * (TILE_SIZE + 2) + (shared_x + j)] * horizontal_filter[i + 1][j + 1];
                sum_vertical += shared_block[(shared_y + i) * (TILE_SIZE + 2) + (shared_x + j)] * vertical_filter[i + 1][j + 1];
                sum_diagonal1 += shared_block[(shared_y + i) * (TILE_SIZE + 2) + (shared_x + j)] * diagonal1_filter[i + 1][j + 1];
                sum_diagonal2 += shared_block[(shared_y + i) * (TILE_SIZE + 2) + (shared_x + j)] * diagonal2_filter[i + 1][j + 1];
            }
        }

        shared_block_horizontal[(shared_y ) * (TILE_SIZE + 2) + (shared_x)] = abs(sum_horizontal); 
        shared_block_vertical[(shared_y ) * (TILE_SIZE + 2) + (shared_x)] = abs(sum_vertical);
        shared_block_diagonal0[(shared_y ) * (TILE_SIZE + 2) + (shared_x)] = abs(sum_diagonal1);
        shared_block_diagonal1[(shared_y ) * (TILE_SIZE + 2) + (shared_x)] = abs(sum_diagonal2);
   }
       __syncthreads();


    output_image_horizontal[y * width + x] = shared_block_horizontal[(shared_y ) * (TILE_SIZE + 2) + (shared_x)];
    output_image_vertical[y * width + x] = shared_block_vertical[(shared_y ) * (TILE_SIZE + 2) + (shared_x)];
    output_image_diagonal1[y * width + x] = shared_block_diagonal0[(shared_y ) * (TILE_SIZE + 2) + (shared_x)];
    output_image_diagonal2[y * width + x] = shared_block_diagonal1[(shared_y ) * (TILE_SIZE + 2) + (shared_x)];


}

int main() {
    // Image dimensions
    unsigned int height = 1024;
    unsigned int width = 1920;
    size_t size = height * width * sizeof(unsigned int);

    // Allocate memory for images on the host
    unsigned int* h_input_image = (unsigned int*)malloc(size);
    unsigned int* h_output_image_0 = (unsigned int*)malloc(size);
    unsigned int* h_output_image_1 = (unsigned int*)malloc(size);
    unsigned int* h_output_image_2 = (unsigned int*)malloc(size);
    unsigned int* h_output_image_3 = (unsigned int*)malloc(size);

    if (h_input_image == NULL || h_output_image_0 == NULL || h_output_image_1 == NULL || h_output_image_2 == NULL || h_output_image_3 == NULL) {
        fprintf(stderr, "Failed to allocate memory on host.\n");
        exit(1);
    }

    // Initialize the input image from file
    FILE* file = fopen("original_0.csv", "r");
    if (file == NULL) {
        fprintf(stderr, "Failed to open input file.\n");
        exit(1);
    }

    char line[10240];
    unsigned int row = 0;

    while (fgets(line, sizeof(line), file) && row < height) {
        char* token;
        unsigned int col = 0;

        token = strtok(line, ",");
        while (token != NULL && col < width) {
            h_input_image[row * width + col] = atoi(token);  // Use atoi to convert string to int
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }

    fclose(file);

    // Initialize the output image
    memset(h_output_image_0, 0, size);
    memset(h_output_image_1, 0, size);
    memset(h_output_image_2, 0, size);
    memset(h_output_image_3, 0, size);

    // Allocate memory for images on the device
    unsigned int* d_input_image;
    unsigned int* d_output_image_0;
    unsigned int* d_output_image_1;
    unsigned int* d_output_image_2;
    unsigned int* d_output_image_3;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc(&d_input_image, size);
    checkCUDAError(cudaStatus, "cudaMalloc failed for input image");

    cudaStatus = cudaMalloc(&d_output_image_0, size);
    checkCUDAError(cudaStatus, "cudaMalloc failed for output image 0");

    cudaStatus = cudaMalloc(&d_output_image_1, size);
    checkCUDAError(cudaStatus, "cudaMalloc failed for output image 1");

    cudaStatus = cudaMalloc(&d_output_image_2, size);
    checkCUDAError(cudaStatus, "cudaMalloc failed for output image 2");

    cudaStatus = cudaMalloc(&d_output_image_3, size);
    checkCUDAError(cudaStatus, "cudaMalloc failed for output image 3");

    // Measure execution time
    cudaEvent_t start0, stop0;

    cudaEventCreate(&start0);
    cudaEventCreate(&stop0);

    cudaEventRecord(start0);
    
    cudaStatus = cudaMemcpy(d_input_image, h_input_image, size, cudaMemcpyHostToDevice);
    checkCUDAError(cudaStatus, "cudaMemcpy failed from host to device");

    cudaEventRecord(stop0);
    cudaEventSynchronize(stop0);


    float Memcpy1;
    cudaEventElapsedTime(&Memcpy1, start0, stop0);
    printf("Execution time - copiando dados pra memória: %f s\n", Memcpy1/1000);


    // Define block and grid sizes
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Start recording
    cudaEvent_t start1, stop1;

    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1);

    timeval time_classification_start, time_classification_end;
    double time;

    gettimeofday(&time_classification_start, NULL);

    // Launch the kernel
    filter<<<gridSize, blockSize>>>(d_input_image, d_output_image_0, d_output_image_1, d_output_image_2, d_output_image_3, height, width);

    gettimeofday(&time_classification_end, NULL);
    time = (double) (time_classification_end.tv_usec - time_classification_start.tv_usec)/1000000 + (double) (time_classification_end.tv_sec - time_classification_start.tv_sec);
    printf("Tempo de execucao gettime: %f\n", time);

    // Stop recording
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start1, stop1);

    printf("Execution time: tempo de lançar o kernel %f s\n", elapsedTime/1000);

    // Copy output image from device to host

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2);

    cudaStatus = cudaMemcpy(h_output_image_0, d_output_image_0, size, cudaMemcpyDeviceToHost);
    checkCUDAError(cudaStatus, "cudaMemcpy failed from device to host for output image 0");

    cudaStatus = cudaMemcpy(h_output_image_1, d_output_image_1, size, cudaMemcpyDeviceToHost);
    checkCUDAError(cudaStatus, "cudaMemcpy failed from device to host for output image 1");

    cudaStatus = cudaMemcpy(h_output_image_2, d_output_image_2, size, cudaMemcpyDeviceToHost);
    checkCUDAError(cudaStatus, "cudaMemcpy failed from device to host for output image 2");

    cudaStatus = cudaMemcpy(h_output_image_3, d_output_image_3, size, cudaMemcpyDeviceToHost);
    checkCUDAError(cudaStatus, "cudaMemcpy failed from device to host for output image 3");

    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    float memcpy2;
    cudaEventElapsedTime(&memcpy2, start2, stop2);
    printf("Execution time: copiando output image do device pro host%f s\n", memcpy2/1000);



    // Save the output image to file
    file = fopen("output_image_0.csv", "w");
    if (file == NULL) {
        fprintf(stderr, "Failed to open output file for output_image_0.\n");
        exit(1);
    }
    for (unsigned int i = 0; i < height; i++) {
        for (unsigned int j = 0; j < width; j++) {
            fprintf(file, "%u", h_output_image_0[i * width + j]);
            if (j < width - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }
    fclose(file);

    file = fopen("output_image_1.csv", "w");
    if (file == NULL) {
        fprintf(stderr, "Failed to open output file for output_image_1.\n");
        exit(1);
    }
    for (unsigned int i = 0; i < height; i++) {
        for (unsigned int j = 0; j < width; j++) {
            fprintf(file, "%u", h_output_image_1[i * width + j]);
            if (j < width - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }
    fclose(file);

    file = fopen("output_image_2.csv", "w");
    if (file == NULL) {
        fprintf(stderr, "Failed to open output file for output_image_2.\n");
        exit(1);
    }
    for (unsigned int i = 0; i < height; i++) {
        for (unsigned int j = 0; j < width; j++) {
            fprintf(file, "%u", h_output_image_2[i * width + j]);
            if (j < width - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }
    fclose(file);

    file = fopen("output_image_3.csv", "w");
    if (file == NULL) {
        fprintf(stderr, "Failed to open output file for output_image_3.\n");
        exit(1);
    }
    for (unsigned int i = 0; i < height; i++) {
        for (unsigned int j = 0; j < width; j++) {
            fprintf(file, "%u", h_output_image_3[i * width + j]);
            if (j < width - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }
    fclose(file);

    // Free device memory
    cudaFree(d_input_image);
    cudaFree(d_output_image_0);
    cudaFree(d_output_image_1);
    cudaFree(d_output_image_2);
    cudaFree(d_output_image_3);

    // Free host memory
    free(h_input_image);
    free(h_output_image_0);
    free(h_output_image_1);
    free(h_output_image_2);
    free(h_output_image_3);

    return 0;
}
