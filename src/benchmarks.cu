#include "benchmarks.h"
#include "commons.h"
#include "defines.h"
#include <cuda_runtime.h>
#include "kernels.h"

//instead of the normal initMatrix, each benchmark should be loaded with a matrix from ./matrices/

int block_benchmark(uint N){
    //give access to the gpu
    int mem_size = N * N * sizeof(float);
    float* mat = (float*) malloc(mem_size);
    float* mat_t = (float*) malloc(mem_size);
    memset(mat_t, 0, mem_size);
    initMatrix(mat, N);
    float *d_mat, *d_mat_t;
    cudaError_t err;
    //int threads, blocks = 0;
    PRINTF("Allocating memory\n");
    if((err = cudaMalloc((void**)&d_mat, mem_size)) != cudaSuccess){
        printf("Error allocating memory for d_a: %s\n", cudaGetErrorString(err));
        printf("Line: %d\n", __LINE__);
        return -1;
    }
    if((err = cudaMalloc((void**)&d_mat_t, mem_size)) != cudaSuccess){
        printf("Error allocating memory for d_mat_t: %s\n", cudaGetErrorString(err));
        printf("Line: %d\n", __LINE__);
        return -1;
    }
    //cudaMalloc((void**)&d_mat_t, mem_size);
    PRINTF("Memory allocated\n");
    //copy data to gpu
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if((err = cudaMemcpy(d_mat, mat, N * N * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess){
        printf("Error copying data to d_mat: %s\n", cudaGetErrorString(err));
        printf("Line: %d\n", __LINE__);
        return -1;
    }
    PRINTF("Data copied\n");
    //setup grid and block size
    dim3 DimGrid = {N/TILE_SIZE, N/TILE_SIZE, 1};
    dim3 DimBlock = {TILE_SIZE, BLOCK_ROWS, 1};
    
    //call kernel as many times as needed
    //first a dummy kernel
    block_transpose<<<DimGrid, DimBlock>>>(d_mat, d_mat_t);
    cudaEvent_t startK, stopK;
    cudaEventCreate(&startK);
    cudaEventCreate(&stopK);
    cudaEventRecord(startK);
    for(int i = 0; i < TRANSPOSITIONS; i++){
        block_transpose<<<DimGrid, DimBlock>>>(d_mat, d_mat_t);
    }
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);
    PRINTF("Kernel returned\n");

    //copy data back
    if((err = cudaMemcpy(mat_t, d_mat_t, mem_size, cudaMemcpyDeviceToHost)) != cudaSuccess){
        printf("Error copying data to mat_t: %s\n", cudaGetErrorString(err));
        printf("Line: %d\n", __LINE__);
        return -1;
    }
    //sync
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    
    float millisecondsK = 0;
    cudaEventElapsedTime(&millisecondsK, startK, stopK);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float ogbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / milliseconds;
    float kgbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / millisecondsK;
    PRINTF("Operation Time: %11.2f ms\n", milliseconds);
    PRINTF("Throughput in GB/s: %7.2f\n", ogbs);
    PRINTF("Kernel Time: %11.2f ms\n", millisecondsK);
    PRINTF("Throughput in GB/s: %7.2f\n", kgbs);
    printf("%f, %f, %f, %f\n", milliseconds, ogbs, millisecondsK, kgbs);
    cudaEventDestroy(startK);
    cudaEventDestroy(stopK);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    //results
    printMatrix(mat_t, N);

    //test if the matrix is transposed
    testTranspose(mat, mat_t, N);  

    //free gpu resources
    cudaFree(d_mat);
    cudaFree(d_mat_t);
    free(mat);
    free(mat_t);
    return 0;
}

int basic_benchmark(uint N){
    //give access to the gpu
    int mem_size = N * N * sizeof(float);
    float* mat = (float*) malloc(mem_size);
    float* mat_t = (float*) malloc(mem_size);
    memset(mat_t, 0, mem_size);
    initMatrix(mat, N);
    float *d_mat, *d_mat_t;
    cudaError_t err;
    //int threads, blocks = 0;
    PRINTF("Allocating memory\n");
    if((err = cudaMalloc((void**)&d_mat, mem_size)) != cudaSuccess){
        printf("Error allocating memory for d_a: %s\n", cudaGetErrorString(err));
        printf("Line: %d\n", __LINE__);
        return -1;
    }
    if((err = cudaMalloc((void**)&d_mat_t, mem_size)) != cudaSuccess){
        printf("Error allocating memory for d_mat_t: %s\n", cudaGetErrorString(err));
        printf("Line: %d\n", __LINE__);
        return -1;
    }
    //cudaMalloc((void**)&d_mat_t, mem_size);
    PRINTF("Memory allocated\n");

    //copy data to gpu
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if((err = cudaMemcpy(d_mat, mat, mem_size, cudaMemcpyHostToDevice)) != cudaSuccess){
        printf("Error copying data to d_mat: %s\n", cudaGetErrorString(err));
        printf("Line: %d\n", __LINE__);
        return -1;
    }
    PRINTF("Data copied\n");
    //setup grid and block size
    dim3 gridB(N / B_TILE, N / B_TILE);
    dim3 blockB(B_TILE, B_ROWS);
    
    //call kernel as many times as needed
    //first a dummy kernel
    basic_transpose<<<gridB, blockB>>>(d_mat, d_mat_t, N);
    cudaEvent_t startK, stopK;
    cudaEventCreate(&startK);
    cudaEventCreate(&stopK);
    cudaEventRecord(startK);
    for(int i = 0; i < TRANSPOSITIONS; i++){
        basic_transpose<<<gridB, blockB>>>(d_mat, d_mat_t, N);
    }
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);
    PRINTF("Kernel returned\n");

    //copy data back
    if((err = cudaMemcpy(mat_t, d_mat_t, mem_size, cudaMemcpyDeviceToHost)) != cudaSuccess){
        printf("Error copying data to mat_t: %s\n", cudaGetErrorString(err));
        printf("Line: %d\n", __LINE__);
        return -1;
    }
    //sync
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    
    float millisecondsK = 0;
    cudaEventElapsedTime(&millisecondsK, startK, stopK);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float ogbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / milliseconds;
    float kgbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / millisecondsK;
    PRINTF("Operation Time: %11.2f ms\n", milliseconds);
    PRINTF("Throughput in GB/s: %7.2f\n", ogbs);
    PRINTF("Kernel Time: %11.2f ms\n", millisecondsK);
    PRINTF("Throughput in GB/s: %7.2f\n", kgbs);
    printf("%f, %f, %f, %f, ", milliseconds, ogbs, millisecondsK, kgbs);

    cudaEventDestroy(startK);
    cudaEventDestroy(stopK);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //test if the matrix is transposed
    PRINTF("basic results:\n");
    printMatrix(mat_t, N);
    testTranspose(mat, mat_t, N);  

    //reset output matrix
    memset(mat_t, 0, mem_size);
    //cudaMemset(d_mat_t, 0, mem_size);
    cudaFree(d_mat);
    cudaFree(d_mat_t);
    free(mat);
    free(mat_t);
    return 0;
}

int conflict_benchmark(uint N){
    //give access to the gpu
    int mem_size = N * N * sizeof(float);
    float* mat = (float*) malloc(mem_size);
    float* mat_t = (float*) malloc(mem_size);
    memset(mat_t, 0, mem_size);
    initMatrix(mat, N);
    float *d_mat, *d_mat_t;
    cudaError_t err;
    //int threads, blocks = 0;
    PRINTF("Allocating memory\n");
    if((err = cudaMalloc((void**)&d_mat, mem_size)) != cudaSuccess){
        printf("Error allocating memory for d_a: %s\n", cudaGetErrorString(err));
        printf("Line: %d\n", __LINE__);
        return -1;
    }
    if((err = cudaMalloc((void**)&d_mat_t, mem_size)) != cudaSuccess){
        printf("Error allocating memory for d_mat_t: %s\n", cudaGetErrorString(err));
        printf("Line: %d\n", __LINE__);
        return -1;
    }
    //cudaMalloc((void**)&d_mat_t, mem_size);
    PRINTF("Memory allocated\n");
    //copy data to gpu
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if((err = cudaMemcpy(d_mat, mat, N * N * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess){
        printf("Error copying data to d_mat: %s\n", cudaGetErrorString(err));
        printf("Line: %d\n", __LINE__);
        return -1;
    }
    PRINTF("Data copied\n");
    //setup grid and block size
    dim3 DimGrid = {N/TILE_SIZE, N/TILE_SIZE, 1};
    dim3 DimBlock = {TILE_SIZE, BLOCK_ROWS, 1};
    
    //call kernel as many times as needed
    //first a dummy kernel
    conflict_transpose<<<DimGrid, DimBlock>>>(d_mat, d_mat_t);
    cudaEvent_t startK, stopK;
    cudaEventCreate(&startK);
    cudaEventCreate(&stopK);
    cudaEventRecord(startK);
    for(int i = 0; i < TRANSPOSITIONS; i++){
        conflict_transpose<<<DimGrid, DimBlock>>>(d_mat, d_mat_t);
    }
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);
    PRINTF("Kernel returned\n");

    //copy data back
    if((err = cudaMemcpy(mat_t, d_mat_t, mem_size, cudaMemcpyDeviceToHost)) != cudaSuccess){
        printf("Error copying data to mat_t: %s\n", cudaGetErrorString(err));
        printf("Line: %d\n", __LINE__);
        return -1;
    }
    //sync
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    
    float millisecondsK = 0;
    cudaEventElapsedTime(&millisecondsK, startK, stopK);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    PRINTF("Operation Time: %11.2f ms\n", milliseconds);
    float ogbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / milliseconds;
    float kgbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / millisecondsK;
    PRINTF("Throughput in GB/s: %7.2f\n", ogbs);
    PRINTF("Kernel Time: %11.2f ms\n", millisecondsK);
    PRINTF("Throughput in GB/s: %7.2f\n", kgbs);
    printf("%f, %f, %f, %f, ", milliseconds, ogbs, millisecondsK, kgbs);

    cudaEventDestroy(startK);
    cudaEventDestroy(stopK);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    //results
    printMatrix(mat_t, N);

    //test if the matrix is transposed
    testTranspose(mat, mat_t, N);  

    //free gpu resources
    cudaFree(d_mat);
    cudaFree(d_mat_t);
    free(mat);
    free(mat_t);
    return 0;
}