#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>

#define TILE_SIZE 4


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void computeDirection(const unsigned int* input_horizontal, const unsigned int* input_vertical,
                                 const unsigned int* input_diagonal1, const unsigned int* input_diagonal2,
                                 unsigned int* output_direction,unsigned int* debug, unsigned int height, unsigned int width) {

    // x  e y globais
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // x e y dentro do bloco 4x4 
    int x_inside_block = threadIdx.x;
    int y_inside_block = threadIdx.y ;

    // x e y do bloco 
    int x_block =  blockIdx.x*4;
    int y_block =  blockIdx.y*4;
    
    // Indices for shared memory
    int shared_x = threadIdx.x + 2;
    int shared_y = threadIdx.y + 2; //matar

    __shared__ unsigned int shared_horizontal[(TILE_SIZE + 4) * (TILE_SIZE + 4)];
    __shared__ unsigned int shared_vertical[(TILE_SIZE + 4) * (TILE_SIZE + 4)];
    __shared__ unsigned int shared_diagonal1[(TILE_SIZE + 4) * (TILE_SIZE + 4)];
    __shared__ unsigned int shared_diagonal2[(TILE_SIZE + 4) * (TILE_SIZE + 4)];


    // Carregar valores do centro
    if(y<height){
            shared_horizontal[(y_inside_block+2) *(TILE_SIZE + 4) + (x_inside_block+2)] = input_horizontal[y * width + x];
            shared_vertical[(y_inside_block+2) *(TILE_SIZE + 4) + (x_inside_block+2)] = input_vertical[y * width + x];
            shared_diagonal1[(y_inside_block+2) *(TILE_SIZE + 4) + (x_inside_block+2)] = input_diagonal1[y * width + x];
            shared_diagonal2[(y_inside_block+2) *(TILE_SIZE + 4) + (x_inside_block+2)] = input_diagonal2[y * width + x];      
    }
 
    // Carregar valores das bordas esquerda e direita
    if (x_inside_block<2) {

        if(x_block > 0){

            shared_horizontal[(y_inside_block+2) *(TILE_SIZE + 4) + (x_inside_block)] = input_horizontal[(y_inside_block+y_block) * width + (x_inside_block+x_block)-2];
            shared_vertical[(y_inside_block+2) *(TILE_SIZE + 4) + (x_inside_block)] = input_vertical[(y_inside_block+y_block) * width + (x_inside_block+x_block)-2];
            shared_diagonal1[(y_inside_block+2) *(TILE_SIZE + 4) + (x_inside_block)] = input_diagonal1[(y_inside_block+y_block) * width + (x_inside_block+x_block)-2];
            shared_diagonal2[(y_inside_block+2) *(TILE_SIZE + 4) + (x_inside_block)] = input_diagonal2[(y_inside_block+y_block) * width + (x_inside_block+x_block)-2];
        }
        
    }
    else {

        if(x_block< width-4){
            shared_horizontal[(y_inside_block+2) *(TILE_SIZE + 4) + (x_inside_block+4)] = input_horizontal[(y_inside_block+y_block) * width + (x_inside_block+x_block)+4];
            shared_vertical[(y_inside_block+2) *(TILE_SIZE + 4) + (x_inside_block+4)] = input_vertical[(y_inside_block+y_block) * width + (x_inside_block+x_block)+4];
            shared_diagonal1[(y_inside_block+2) *(TILE_SIZE + 4) + (x_inside_block+4)] = input_diagonal1[(y_inside_block+y_block) * width + (x_inside_block+x_block)+4];
            shared_diagonal2[(y_inside_block+2) *(TILE_SIZE + 4) + (x_inside_block+4)] = input_diagonal2[(y_inside_block+y_block) * width + (x_inside_block+x_block)+4];
        }
        
    }
    // Carregar valores dos cantos
    if (x_inside_block == 0 && y_inside_block == 0) {

        if(x_block > 0 && y_block >0) {

            // canto  superior esquerdo

            shared_horizontal[(0) *(TILE_SIZE + 4) + (0)] = input_horizontal[(y_block-2)*width+(x_block-2)];
            shared_horizontal[(0) *(TILE_SIZE + 4) + (1)] = input_horizontal[(y_block-2)*width+(x_block-1)];
            shared_horizontal[(1) *(TILE_SIZE + 4) + (0)] = input_horizontal[(y_block-1)*width+(x_block-2)];
            shared_horizontal[(1) *(TILE_SIZE + 4) + (1)] = input_horizontal[(y_block-1)*width+(x_block-1)];

            shared_vertical[(0) *(TILE_SIZE + 4) + (0)] = input_vertical[(y_block-2)*width+(x_block-2)];
            shared_vertical[(0) *(TILE_SIZE + 4) + (1)] = input_vertical[(y_block-2)*width+(x_block-1)];
            shared_vertical[(1) *(TILE_SIZE + 4) + (0)] = input_vertical[(y_block-1)*width+(x_block-2)];
            shared_vertical[(1) *(TILE_SIZE + 4) + (1)] = input_vertical[(y_block-1)*width+(x_block-1)];

            shared_diagonal1[(0) *(TILE_SIZE + 4) + (0)] = input_diagonal1[(y_block-2)*width+(x_block-2)];
            shared_diagonal1[(0) *(TILE_SIZE + 4) + (1)] = input_diagonal1[(y_block-2)*width+(x_block-1)];
            shared_diagonal1[(1) *(TILE_SIZE + 4) + (0)] = input_diagonal1[(y_block-1)*width+(x_block-2)];
            shared_diagonal1[(1) *(TILE_SIZE + 4) + (1)] = input_diagonal1[(y_block-1)*width+(x_block-1)];

            shared_diagonal2[(0) *(TILE_SIZE + 4) + (0)] = input_diagonal2[(y_block-2)*width+(x_block-2)];
            shared_diagonal2[(0) *(TILE_SIZE + 4) + (1)] = input_diagonal2[(y_block-2)*width+(x_block-1)];
            shared_diagonal2[(1) *(TILE_SIZE + 4) + (0)] = input_diagonal2[(y_block-1)*width+(x_block-2)];
            shared_diagonal2[(1) *(TILE_SIZE + 4) + (1)] = input_diagonal2[(y_block-1)*width+(x_block-1)];

        }


        if(x_block< width-4 && y_block > 0){

            // canto superior direito

            shared_horizontal[(0) *(TILE_SIZE + 4) + (6)] = input_horizontal[(y_block-2)*width+(x_block+4)];
            shared_horizontal[(0) *(TILE_SIZE + 4) + (7)] = input_horizontal[(y_block-2)*width+(x_block+5)];
            shared_horizontal[(1) *(TILE_SIZE + 4) + (6)] = input_horizontal[(y_block-1)*width+(x_block+4)];
            shared_horizontal[(1) *(TILE_SIZE + 4) + (7)] = input_horizontal[(y_block-1)*width+(x_block+5)];

            shared_vertical[(0) *(TILE_SIZE + 4) + (6)] = input_vertical[(y_block-2)*width+(x_block+4)];
            shared_vertical[(0) *(TILE_SIZE + 4) + (7)] = input_vertical[(y_block-2)*width+(x_block+5)];
            shared_vertical[(1) *(TILE_SIZE + 4) + (6)] = input_vertical[(y_block-1)*width+(x_block+4)];
            shared_vertical[(1) *(TILE_SIZE + 4) + (7)] = input_vertical[(y_block-1)*width+(x_block+5)];


            shared_diagonal1[(0) *(TILE_SIZE + 4) + (6)] = input_diagonal1[(y_block-2)*width+(x_block+4)];
            shared_diagonal1[(0) *(TILE_SIZE + 4) + (7)] = input_diagonal1[(y_block-2)*width+(x_block+5)];
            shared_diagonal1[(1) *(TILE_SIZE + 4) + (6)] = input_diagonal1[(y_block-1)*width+(x_block+4)];
            shared_diagonal1[(1) *(TILE_SIZE + 4) + (7)] = input_diagonal1[(y_block-1)*width+(x_block+5)];

            shared_diagonal2[(0) *(TILE_SIZE + 4) + (6)] = input_diagonal2[(y_block-2)*width+(x_block+4)];
            shared_diagonal2[(0) *(TILE_SIZE + 4) + (7)] = input_diagonal2[(y_block-2)*width+(x_block+5)];
            shared_diagonal2[(1) *(TILE_SIZE + 4) + (6)] = input_diagonal2[(y_block-1)*width+(x_block+4)];
            shared_diagonal2[(1) *(TILE_SIZE + 4) + (7)] = input_diagonal2[(y_block-1)*width+(x_block+5)];

        }

        if(x_block > 0 ){
           
            //canto inferior esquerdo

            shared_horizontal[(6) *(TILE_SIZE + 4) + (0)] = input_horizontal[(y_block+4)*width+(x_block-2)];
            shared_horizontal[(6) *(TILE_SIZE + 4) + (1)] = input_horizontal[(y_block+4)*width+(x_block-1)];
            shared_horizontal[(7) *(TILE_SIZE + 4) + (0)] = input_horizontal[(y_block+4)*width+(x_block-2)];
            shared_horizontal[(7) *(TILE_SIZE + 4) + (1)] = input_horizontal[(y_block+4)*width+(x_block-1)];

            shared_vertical[(6) *(TILE_SIZE + 4) + (0)] = input_vertical[(y_block+4)*width+(x_block-2)];
            shared_vertical[(6) *(TILE_SIZE + 4) + (1)] = input_vertical[(y_block+4)*width+(x_block-1)];
            shared_vertical[(7) *(TILE_SIZE + 4) + (0)] = input_vertical[(y_block+4)*width+(x_block-2)];
            shared_vertical[(7) *(TILE_SIZE + 4) + (1)] = input_vertical[(y_block+4)*width+(x_block-1)];

            shared_diagonal1[(6) *(TILE_SIZE + 4) + (0)] = input_diagonal1[(y_block+4)*width+(x_block-2)];
            shared_diagonal1[(6) *(TILE_SIZE + 4) + (1)] = input_diagonal1[(y_block+4)*width+(x_block-1)];
            shared_diagonal1[(7) *(TILE_SIZE + 4) + (0)] = input_diagonal1[(y_block+4)*width+(x_block-2)];
            shared_diagonal1[(7) *(TILE_SIZE + 4) + (1)] = input_diagonal1[(y_block+4)*width+(x_block-1)];

            shared_diagonal2[(6) *(TILE_SIZE + 4) + (0)] = input_diagonal2[(y_block+4)*width+(x_block-2)];
            shared_diagonal2[(6) *(TILE_SIZE + 4) + (1)] = input_diagonal2[(y_block+5)*width+(x_block-1)];
            shared_diagonal2[(7) *(TILE_SIZE + 4) + (0)] = input_diagonal2[(y_block+4)*width+(x_block-2)];
            shared_diagonal2[(7) *(TILE_SIZE + 4) + (1)] = input_diagonal2[(y_block+5)*width+(x_block-1)];
            }


        
        if (x_block + 4 < width && y_block + 4 < height) {

                // cantos inferior direito

            shared_horizontal[(6) * (TILE_SIZE + 4) + (6)] = input_horizontal[(y_block + 4) * width + (x_block + 4)];
            shared_horizontal[(6) * (TILE_SIZE + 4) + (7)] = input_horizontal[(y_block + 4) * width + (x_block + 5)];
            shared_horizontal[(7) * (TILE_SIZE + 4) + (6)] = input_horizontal[(y_block + 5) * width + (x_block + 4)];
            shared_horizontal[(7) * (TILE_SIZE + 4) + (7)] = input_horizontal[(y_block + 5) * width + (x_block + 5)];

            shared_vertical[(6) * (TILE_SIZE + 4) + (6)] = input_vertical[(y_block + 4) * width + (x_block + 4)];
            shared_vertical[(6) * (TILE_SIZE + 4) + (7)] = input_vertical[(y_block + 4) * width + (x_block + 5)];
            shared_vertical[(7) * (TILE_SIZE + 4) + (6)] = input_vertical[(y_block + 5) * width + (x_block + 4)];
            shared_vertical[(7) * (TILE_SIZE + 4) + (7)] = input_vertical[(y_block + 5) * width + (x_block + 5)];

            shared_diagonal1[(6) * (TILE_SIZE + 4) + (6)] = input_diagonal1[(y_block + 4) * width + (x_block + 4)];
            shared_diagonal1[(6) * (TILE_SIZE + 4) + (7)] = input_diagonal1[(y_block + 4) * width + (x_block + 5)];
            shared_diagonal1[(7) * (TILE_SIZE + 4) + (6)] = input_diagonal1[(y_block + 5) * width + (x_block + 4)];
            shared_diagonal1[(7) * (TILE_SIZE + 4) + (7)] = input_diagonal1[(y_block + 5) * width + (x_block + 5)];

            shared_diagonal2[(6) * (TILE_SIZE + 4) + (6)] = input_diagonal2[(y_block + 4) * width + (x_block + 4)];
            shared_diagonal2[(6) * (TILE_SIZE + 4) + (7)] = input_diagonal2[(y_block + 4) * width + (x_block + 5)];
            shared_diagonal2[(7) * (TILE_SIZE + 4) + (6)] = input_diagonal2[(y_block + 5) * width + (x_block + 4)];
            shared_diagonal2[(7) * (TILE_SIZE + 4) + (7)] = input_diagonal2[(y_block + 5) * width + (x_block + 5)];
    
     }

       
    }

         // Carregar valores da  borda superior
        if (y_inside_block < 2 && y_block > 0) {

            shared_horizontal[(y_inside_block) * (TILE_SIZE + 4) + (x_inside_block + 2)] = input_horizontal[(y_block - 2 + y_inside_block) * width + x];
            shared_vertical[(y_inside_block) * (TILE_SIZE + 4) + (x_inside_block + 2)] = input_vertical[(y_block - 2 + y_inside_block) * width + x];
            shared_diagonal1[(y_inside_block) * (TILE_SIZE + 4) + (x_inside_block + 2)] = input_diagonal1[(y_block - 2 + y_inside_block) * width + x];
            shared_diagonal2[(y_inside_block) * (TILE_SIZE + 4) + (x_inside_block + 2)] = input_diagonal2[(y_block - 2 + y_inside_block) * width + x];
        }

        // Carregar valores da borda inferior
        if (y_inside_block >= 2 && y_block < height - 4) {
            shared_horizontal[(y_inside_block + 4) * (TILE_SIZE + 4) + (x_inside_block + 2)] = input_horizontal[(y_block + 4 + y_inside_block) * width + x];
            shared_vertical[(y_inside_block + 4) * (TILE_SIZE + 4) + (x_inside_block + 2)] = input_vertical[(y_block + 4 + y_inside_block) * width + x];
            shared_diagonal1[(y_inside_block + 4) * (TILE_SIZE + 4) + (x_inside_block + 2)] = input_diagonal1[(y_block + 4 + y_inside_block) * width + x];
            shared_diagonal2[(y_inside_block + 4) * (TILE_SIZE + 4) + (x_inside_block + 2)] = input_diagonal2[(y_block + 4 + y_inside_block) * width + x];
        }

    
    __syncthreads();

  
    if(threadIdx.x == 0 && threadIdx.y==0 && x_block == 4 && y_block ==4){
        for(int n = 0; n <8; n++){
            for(int m = 0; m<8; m++){
                printf("%d,", shared_horizontal[(n*8+m)]);
            }
                printf("\n");

        }


        printf("\n");

    }

    
   int sum_h = 0, sum_v = 0, sum_d0 = 0, sum_d1 = 0;
   for (int i = 0; i <= 7; i++) {
        for (int j = 0; j <= 7; j++) {
            //"zeros"
            if(i%2== j%2 ){
                sum_h += shared_horizontal[(shared_y + i - 1) * (TILE_SIZE + 4) + (shared_x + j - 1)];
                sum_v += shared_vertical[(shared_y + i - 1) * (TILE_SIZE + 4) + (shared_x + j - 1)];
                sum_d0 += shared_diagonal1[(shared_y + i - 1) * (TILE_SIZE + 4) + (shared_x + j - 1)];
                sum_d1 += shared_diagonal2[(shared_y + i - 1) * (TILE_SIZE + 4) + (shared_x + j - 1)];
            }
            
           }
       }
       
    // //calculo atividade

    // static const int th[16] = { 0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4 };
    // const int maxActivity = 15;

    // int tempAct = sum_h + sum_h; //ghv
    // int activity = 0;

    // int vbCTUHeight = 128;
    // int vbPos =  vbCTUHeight - 4;

    // const int y = (i + blkDst.pos().y) & (vbCTUHeight - 1);
    // //printf("valor de y = %d", (i + blkDst.pos().y) & (vbCTUHeight - 1));
    // if (y == vbPos - 4 || y == vbPos){

    //     //activity = (Pel)Clip3<int>(0, maxActivity, (tempAct * 96) >> shift);
    //     //printf("VB do Activity\n");
    //     //std::cout<<"oiu " << activity<<std::endl;
    // }
    // else{
    //     //activity = (Pel)Clip3<int>(0, maxActivity, (tempAct * 64) >> shift);
    // }
    //   int classIdx = th[activity]; //Ã

       
    // classification block size
    const int clsSizeY = 4;
    const int clsSizeX = 4;

    for( int i = 0; i < height; i += clsSizeY ){
        for( int j = 0; j < width; j += clsSizeX ){

        int hv1, hv0, d1, d0, hvd1, hvd0;
        int mainDirection, dirTempHV, dirTempD;

        if( sum_v > sum_h ) // maxhv = V
        {
            hv1 = sum_v;
            hv0 = sum_h;
            dirTempHV = 1;
        }
        else // maxhv = H
        {
            hv1 = sum_h;
            hv0 = sum_v;
            dirTempHV = 3;
        }
        if( sum_d0 > sum_d1 ) // maxd = D0
        {
            d1 = sum_d0;
            d0 = sum_d1;
            dirTempD = 0;
        }
        else  // maxd = D1
        {
            d1 = sum_d1;
            d0 = sum_d0;
            dirTempD = 2;
        }
   
        if( (int)d1 * (int)hv0 > (int)hv1 * (int)d0 )    {
            hvd1 = d1;
            hvd0 = d0;
            mainDirection = dirTempD;
        }
        else
        {
            hvd1 = hv1;
            hvd0 = hv0;
            mainDirection = dirTempHV;
        }

        int directionStrength = 0;
        if( hvd1 > 2 * hvd0 )
        {
            directionStrength = 1;
        }
        if( hvd1 * 2 > 9 * hvd0 )
        {
            directionStrength = 2;
        }

        if( directionStrength )
        {
           output_direction[i * width + j] = ( ( ( mainDirection & 0x1 ) << 1 ) + directionStrength );
        }
            
        }
    }
        
}

int main() {
    unsigned int height = 1024;
    unsigned int width = 1920;
    size_t size = height * width * sizeof(unsigned int);

    unsigned int* h_output_image_0 = (unsigned int*)malloc(size);
    unsigned int* h_output_image_1 = (unsigned int*)malloc(size);
    unsigned int* h_output_image_2 = (unsigned int*)malloc(size);
    unsigned int* h_output_image_3 = (unsigned int*)malloc(size);
    unsigned int* h_output_direction = (unsigned int*)malloc(size);
    unsigned int* h_debug = (unsigned int*)malloc(size);


    if (h_output_image_0 == NULL || h_output_image_1 == NULL || h_output_image_2 == NULL || h_output_image_3 == NULL || h_output_direction == NULL || h_debug == NULL) {
        fprintf(stderr, "Failed to allocate memory on host.\n");
        exit(1);
    }

    // Read previously generated output images from files
    FILE* file = fopen("output_image_0.csv", "r");
    if (file == NULL) {
        fprintf(stderr, "Failed to open input file output_image_0.csv.\n");
        exit(1);
    }
    for (unsigned int i = 0; i < height; i++) {
        for (unsigned int j = 0; j < width; j++) {
            fscanf(file, "%u,", &h_output_image_0[i * width + j]);
        }
    }
    fclose(file);

    file = fopen("output_image_1.csv", "r");
    if (file == NULL) {
        fprintf(stderr, "Failed to open input file output_image_1.csv.\n");
        exit(1);
    }
    for (unsigned int i = 0; i < height; i++) {
        for (unsigned int j = 0; j < width; j++) {
            fscanf(file, "%u,", &h_output_image_1[i * width + j]);
        }
    }
    fclose(file);

    file = fopen("output_image_2.csv", "r");
    if (file == NULL) {
        fprintf(stderr, "Failed to open input file output_image_2.csv.\n");
        exit(1);
    }
    for (unsigned int i = 0; i < height; i++) {
        for (unsigned int j = 0; j < width; j++) {
            fscanf(file, "%u,", &h_output_image_2[i * width + j]);
        }
    }
    fclose(file);

    file = fopen("output_image_3.csv", "r");
    if (file == NULL) {
        fprintf(stderr, "Failed to open input file output_image_3.csv.\n");
        exit(1);
    }
    for (unsigned int i = 0; i < height; i++) {
        for (unsigned int j = 0; j < width; j++) {
            fscanf(file, "%u,", &h_output_image_3[i * width + j]);
        }
    }
    fclose(file);

    unsigned int* d_output_image_0;
    unsigned int* d_output_image_1;
    unsigned int* d_output_image_2;
    unsigned int* d_output_image_3;
    unsigned int* d_output_direction;
    unsigned int* d_debug;

    gpuErrchk(  cudaMalloc((void**)&d_output_image_0, size));
    gpuErrchk(  cudaMalloc((void**)&d_output_image_1, size));
    gpuErrchk(  cudaMalloc((void**)&d_output_image_2, size));
    gpuErrchk(  cudaMalloc((void**)&d_output_image_3, size));
    gpuErrchk( cudaMalloc((void**)&d_output_direction, size));
    gpuErrchk(  cudaMalloc((void**)&d_debug, size));


    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

     // Start recording
    cudaEventRecord(start1);

    cudaMemcpy(d_output_image_0, h_output_image_0, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_image_1, h_output_image_1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_image_2, h_output_image_2, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_image_3, h_output_image_3, size, cudaMemcpyHostToDevice);

    std::cout<<"Output "<<h_output_image_0[10]<<std::endl;

    // Stop recording
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start1, stop1);

    // Clean up events
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    printf("Execution time: tempo para memcpy o kernel  %f ms\n", elapsedTime);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    // Measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording
    cudaEventRecord(start);

    computeDirection<<<numBlocks, threadsPerBlock>>>(d_output_image_0, d_output_image_1, d_output_image_2, d_output_image_3,
                                                      d_output_direction, d_debug, height, width);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // Stop recording
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime1;
    cudaEventElapsedTime(&elapsedTime1, start, stop);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Execution time: tempo para lençar o kernel  %f ms\n", elapsedTime1);

    cudaMemcpy(h_output_direction, d_output_direction, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_debug, d_debug, size, cudaMemcpyDeviceToHost);
    std::cout<<"OutputD "<<h_debug[5]<<std::endl;


    std::cout<<"passou"<<std::endl;


    file = fopen("output_direction.csv", "w");
    if (file == NULL) {
        fprintf(stderr, "Failed to open output file output_direction.csv.\n");
        exit(1);
    }
    for (unsigned int i = 0; i < height; i++) {
        for (unsigned int j = 0; j < width; j++) {
            fprintf(file, "%u,", h_output_direction[i * width + j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);

    file = fopen("debug.csv", "w");
    if (file == NULL) {
        fprintf(stderr, "Failed to open output file debug.csv.\n");
        exit(1);
    }
    for (unsigned int i = 0; i < height; i++) {
        for (unsigned int j = 0; j < width; j++) {
            fprintf(file, "%u,", h_debug[i * width + j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);

    free(h_output_image_0);
    free(h_output_image_1);
    free(h_output_image_2);
    free(h_output_image_3);
    free(h_output_direction);
    free(h_debug);

    cudaFree(d_output_image_0);
    cudaFree(d_output_image_1);
    cudaFree(d_output_image_2);
    cudaFree(d_output_image_3);
    cudaFree(d_output_direction);
    cudaFree(h_debug);

    return 0;
}
