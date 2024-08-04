#include "../include/benchmarks.cuh"
#include "../include/commons.h"
#include "../include/kernels.cuh"
#include "../include/debug.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <fstream>
//complete run for each transposition algorithm callled by complete_benchmark in main

int coo_transposition(coo_matrix* coo){
    if ((cudaSetDevice(0)) != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        return 1;
    }
    
    coo_element* el = coo->el;
    coo_matrix* d_coo;
    coo_element* d_el;
    CHECK_CUDA(cudaMallocManaged((void**)&d_coo, sizeof(coo_matrix)));
    CHECK_CUDA(cudaMallocManaged((void**)&d_el, coo->nnz * sizeof(coo_element)));
    CHECK_CUDA(cudaMemcpy(d_coo, coo, sizeof(coo_matrix), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_el, el, coo->nnz * sizeof(coo_element), cudaMemcpyHostToDevice));
    PRINTF("Copied memory\n");
    PRINTF("Copied memory\n");
    d_coo->el = d_el;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    printf("Before transpose\n");
    print_coo_less(d_coo);
    CHECK_CUDA(cudaEventRecord(start));
    cuCOOt<<<coo->nnz,1>>>(d_coo->el, d_coo->nnz);
    CHECK_CUDA(cudaMemcpy(d_coo, d_coo, sizeof(coo_matrix), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop));
    PRINTF("After transpose\n");
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time for executing cuCOOt operation: %f ms\n", milliseconds);
    print_coo_less(d_coo);
    if (is_transpose(coo, d_coo)) {
        PRINTF("Transpose is correct\n");
    } else {
        PRINTF("Transpose is incorrect\n");
    }
    std::ofstream output;
    output.open ("logs/results.log", std::ios::out | std::ios_base::app);
    output << "N_mat, " << "COO, " << "OpTime, Op-GB/s, " << milliseconds << "K-GB/s\n";
    output.close();

    CHECK_CUDA(cudaFree(d_coo));
    CHECK_CUDA(cudaFree(d_el));
    delete[] coo->el;
    delete coo;
    return 0;
}
int csr_transposition(csr_matrix* csr, csr_matrix* csr_t) {
    if ((cudaSetDevice(0)) != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        return 1;
    }
    // Example CSR matrix
    //assign real values instead of random
    thrust::host_vector<int> h_values(8);// = {10, 20, 30, 40, 50, 60, 70, 80};
    h_values[0] = 10; h_values[1] = 20; h_values[2] = 30; h_values[3] = 40; //TEMP just to avoid compiler error
    h_values[4] = 50; h_values[5] = 60; h_values[6] = 70; h_values[7] = 80;
    thrust::host_vector<int> h_col_indices(8);// = {0, 2, 1, 0, 1, 2, 0, 1};
    h_col_indices[0] = 0; h_col_indices[1] = 2; h_col_indices[2] = 1; h_col_indices[3] = 0;
    h_col_indices[4] = 1; h_col_indices[5] = 2; h_col_indices[6] = 0; h_col_indices[7] = 1;
    thrust::host_vector<int> h_row_ptr(5);// = {0, 2, 4, 7, 8};
    h_row_ptr[0] = 0; h_row_ptr[1] = 2; h_row_ptr[2] = 4; h_row_ptr[3] = 7; h_row_ptr[4] = 8;
    int num_rows = 4;
    int num_cols = 3;

    // Device vectors for transposed matrix
    thrust::device_vector<int> d_t_values;
    thrust::device_vector<int> d_t_row_indices;
    thrust::device_vector<int> d_t_col_ptr;

    // Transpose the matrix
    transposeCSRToCSC(h_values, h_col_indices, h_row_ptr, num_rows, num_cols, d_t_values, d_t_row_indices, d_t_col_ptr);

    // Copy the results back to the host and print
    thrust::host_vector<int> h_t_values = d_t_values;
    thrust::host_vector<int> h_t_row_indices = d_t_row_indices;
    thrust::host_vector<int> h_t_col_ptr = d_t_col_ptr;

    printf("Original Matrix:\n");
    pretty_print_matrix(h_values, h_col_indices, h_row_ptr, num_rows, num_cols);

    printf("Transposed Matrix:\n");
    pretty_print_matrix(h_t_values, h_t_row_indices, h_t_col_ptr, num_rows, num_cols); // ! error invert num_rows and num_cols

    return 0;
}

int block_trasposition(float* mat, unsigned int N){
    if ((cudaSetDevice(0)) != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        return 1;
    }
    int mem_size = N * N * sizeof(float);
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
    //printf("%f, %f, %f, %f\n", milliseconds, ogbs, millisecondsK, kgbs);
    
    std::ofstream output;
    output.open ("logs/results.log", std::ios::out | std::ios_base::app);
    output << "N_mat, " << "block, " << "OpTime, Op-GB/s, " << milliseconds << "K-GB/s\n";
    output.close();

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
int conflict_transposition(float* mat, unsigned int N){
    if ((cudaSetDevice(0)) != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        return 1;
    }
    int mem_size = N * N * sizeof(float);
    float* mat_t = (float*) malloc(mem_size);
    memset(mat_t, 0, mem_size);
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
    
    std::ofstream output;
    output.open ("logs/results.log", std::ios::out | std::ios_base::app);
    output << "N_mat, " << "Conflict, " << "OpTime, Op-GB/s, " << milliseconds << "K-GB/s\n";
    output.close();

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

int transposeCSRToCSC(const thrust::host_vector<int>& h_values, const thrust::host_vector<int>& h_col_indices,
                       const thrust::host_vector<int>& h_row_ptr, int num_rows, int num_cols,
                       thrust::device_vector<int>& d_t_values, thrust::device_vector<int>& d_t_row_indices,
                       thrust::device_vector<int>& d_t_col_ptr) {
    int nnz = h_values.size();
    if ((cudaSetDevice(0)) != cudaSuccess) {
        PRINTF("Failed to set CUDA device\n");
        return 1;
    }
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Copy input CSR data to device
    thrust::device_vector<int> d_values = h_values;
    thrust::device_vector<int> d_col_indices = h_col_indices;
    thrust::device_vector<int> d_row_ptr = h_row_ptr;
    
    // Initialize device vectors for transposed matrix
    d_t_values.resize(nnz);
    d_t_row_indices.resize(nnz);
    d_t_col_ptr.resize(num_cols + 1);

    CHECK_CUDA(cudaEventRecord(start));
    // Device vector for column counts
    thrust::device_vector<int> d_col_counts(num_cols, 0);
    
    // Kernel to count non-zero entries per column
    countNNZPerColumn<<<((nnz + 255) / 256), 256>>>(thrust::raw_pointer_cast(d_col_indices.data()),
                                                  thrust::raw_pointer_cast(d_col_counts.data()), nnz);

    // Compute column pointers using exclusive scan
    thrust::exclusive_scan(d_col_counts.begin(), d_col_counts.end(), d_t_col_ptr.begin());
    cudaCheckError();

    // Copy the column pointers to create a correct offset for scattering
    thrust::device_vector<int> d_col_ptr_copy = d_t_col_ptr;

    // Kernel to scatter values and row indices to transposed matrix
    scatterToTransposed<<<(num_rows + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_values.data()),
                                                         thrust::raw_pointer_cast(d_col_indices.data()),
                                                         thrust::raw_pointer_cast(d_row_ptr.data()),
                                                         thrust::raw_pointer_cast(d_t_values.data()),
                                                         thrust::raw_pointer_cast(d_t_row_indices.data()),
                                                         thrust::raw_pointer_cast(d_col_ptr_copy.data()), num_rows);
    CHECK_CUDA(cudaEventRecord(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time for executing cuSPARSECSRt operation: %f ms\n", milliseconds);
    std::ofstream output;
    output.open ("logs/results.log", std::ios::out | std::ios_base::app);
    output << "N_mat, " << "CSR" << "OpTime, Op-GB/s, " << milliseconds << "K-GB/s\n";
    output.close();
    cudaCheckError();
    return 0;
}

int pretty_print_matrix(const thrust::host_vector<int>& values, const thrust::host_vector<int>& row_indices,
                        const thrust::host_vector<int>& col_ptr, int num_rows, int num_cols) {
    printf("Values: ");
    for (int v : values) printf("%d ", v);
    printf("\n");

    printf("Row Indices: ");
    for (int ri : row_indices) printf("%d ", ri);
    printf("\n");

    printf("Column Pointers: ");
    for (int cp : col_ptr) printf("%d ", cp);
    printf("\n");

    printf("Matrix:\n");

    printf("NumCols: %d\n", num_cols);
    printf("NumRows: %d\n", num_rows);

    for (int i = 0; i < num_cols; ++i) {
        for (int j = col_ptr[i]; j < col_ptr[i + 1]; ++j) {
            printf("(%d, %d, %d)\n", row_indices[j], i, values[j]);
        }
    }

    return 0;
}