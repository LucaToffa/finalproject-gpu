//complete run for each transposition algorithm callled by complete_benchmark in main
#include "../include/benchmarks.cuh"
#include "../include/commons.h"
#include "../include/kernels.cuh"
#include "../include/debug.h"
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <fstream>

// ! TODO: add for(int i = 0; i < TRANSPOSITIONS; i++){} to all methods

int coo_transposition(coo_matrix* coo) {
    PRINTF("--------------------\n");
    PRINTF("COO Transposition Method called: coo_transposition() -> cuCOOt().\n");
    if ((cudaSetDevice(0)) != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        return 1;
    }
    coo_element* el = coo->el;
    coo_matrix* d_coo;
    coo_element* d_el;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMallocManaged((void**)&d_coo, sizeof(coo_matrix)));
    CHECK_CUDA(cudaMallocManaged((void**)&d_el, coo->nnz * sizeof(coo_element)));
    CHECK_CUDA(cudaMemcpy(d_coo, coo, sizeof(coo_matrix), cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(d_el, el, coo->nnz * sizeof(coo_element), cudaMemcpyHostToDevice));
    PRINTF("Copied & Allocated Memory Succesfully\n");
    d_coo->el = d_el;

    cudaEvent_t startK, stopK;
    CHECK_CUDA(cudaEventCreate(&startK));
    CHECK_CUDA(cudaEventCreate(&stopK));
    
    #ifdef DEBUG
        printf("Pre-Transpose Matrix:\n");
        print_coo_less(coo);
    #endif
    dummy_kernel<<<1,1>>>();
    CHECK_CUDA(cudaEventRecord(startK));
    for (int i = 0; i < TRANSPOSITIONS; i++) {
        cudaMemcpy(d_el, el, coo->nnz * sizeof(coo_element), cudaMemcpyHostToDevice);
        cuCOOt<<<(coo->nnz + 255) / 256, 256>>>(d_coo->el, d_coo->nnz);
    }
    //cuCOOt<<<coo->nnz,1>>>(d_coo->el, d_coo->nnz);
    CHECK_CUDA(cudaEventRecord(stopK));
    CHECK_CUDA(cudaMemcpy(d_coo, d_coo, sizeof(coo_matrix), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // PRINTF("Transposition Completed Succesfully.\n");
    // float milliseconds = 0;
    // CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    // printf("Time for executing cuCOOt operation: %f ms\n", milliseconds);
    int ret = 0;
    #ifdef DEBUG
        printf("Post-Transpose Matrix:\n");
        print_coo_less(d_coo);
    #endif
    if (is_transpose(coo, d_coo)) {
        PRINTF("Transpose is correct.\n");
    } else {
        ret = -1;
        PRINTF("Transpose is incorrect.\n");
    }

    float millisecondsK = 0;
    CHECK_CUDA(cudaEventElapsedTime(&millisecondsK, startK, stopK));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    int N = coo->cols; 
    float ogbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / milliseconds;
    float kgbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / millisecondsK;
    PRINTF("Time for executing cuCOOt operation: %f ms\n", milliseconds);
    PRINTF("Operation Throughput in GB/s: %7.2f\n", ogbs);
    PRINTF("Kernel Time: %11.2f ms\n", millisecondsK);
    PRINTF("Kernel Throughput in GB/s: %7.2f\n", kgbs);

    std::ofstream output;
    output.open ("logs/results.log", std::ios::out | std::ios_base::app);
    //output << "COO, " << "OpTime, Op-GB/s, " << milliseconds << " K-GB/s\n";
    // algorithm, OpTime, Op-GB/s, KTime, K-GB/s
    output << "COO, " <<  milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n"; /* *** */
    output.close();

    CHECK_CUDA(cudaFree(d_coo));
    CHECK_CUDA(cudaFree(d_el));
    PRINTF("Freed Memory Succesfully.\n");
    PRINTF("--------------------\n");
    return ret;
}

int csr_transposition(csr_matrix* csr, csr_matrix* csr_t) {
    PRINTF("--------------------\n");
    PRINTF("CSR Transposition Method Called: csr_transposition() -> transposeCSRToCSC().\n");
    if ((cudaSetDevice(0)) != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        return 1;
    }
    // Example CSR matrix
    //assign real values instead of random
    // thrust::host_vector<float> h_values(8);// = {10, 20, 30, 40, 50, 60, 70, 80};
    // h_values[0] = 10; h_values[1] = 20; h_values[2] = 30; h_values[3] = 40; //TEMP just to avoid compiler error
    // h_values[4] = 50; h_values[5] = 60; h_values[6] = 70; h_values[7] = 80;
    // thrust::host_vector<int> h_col_indices(8);// = {0, 2, 1, 0, 1, 2, 0, 1};
    // h_col_indices[0] = 0; h_col_indices[1] = 2; h_col_indices[2] = 1; h_col_indices[3] = 0;
    // h_col_indices[4] = 1; h_col_indices[5] = 2; h_col_indices[6] = 0; h_col_indices[7] = 1;
    // thrust::host_vector<int> h_row_ptr(5);// = {0, 2, 4, 7, 8};
    // h_row_ptr[0] = 0; h_row_ptr[1] = 2; h_row_ptr[2] = 4; h_row_ptr[3] = 7; h_row_ptr[4] = 8;
    // int num_rows = 4;
    // int num_cols = 3;

    // // Device vectors for transposed matrix
    // thrust::device_vector<float> d_t_values;
    // thrust::device_vector<int> d_t_row_indices;
    // thrust::device_vector<int> d_t_col_ptr;

    // // Transpose the matrix
    // transposeCSRToCSC(h_values, h_col_indices, h_row_ptr, num_rows, num_cols, d_t_values, d_t_row_indices, d_t_col_ptr);

    // // Copy the results back to the host and print
    // thrust::host_vector<float> h_t_values = d_t_values;
    // thrust::host_vector<int> h_t_row_indices = d_t_row_indices;
    // thrust::host_vector<int> h_t_col_ptr = d_t_col_ptr;

    // printf("Original Matrix:\n");
    // pretty_print_matrix(h_values, h_col_indices, h_row_ptr, num_rows, num_cols);

    // printf("Transposed Matrix:\n");
    // pretty_print_matrix(h_t_values, h_t_row_indices, h_t_col_ptr, num_rows, num_cols); // ! error invert num_rows and num_cols

    
    // int ret = transposeCSRToCSC_cuda(csr, csr_t);
    assert(csr != NULL && csr_t != NULL);
    assert(csr->rows == csr_t->cols && csr->cols == csr_t->rows);
    PRINTF("Transpose CSR to CSC Cuda Method Called: transposeCSRToCSC_cuda().\n");
    if ((cudaSetDevice(0)) != cudaSuccess) {
        PRINTF("Failed to set CUDA device\n");
        return 1;
    }
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    // Copy input CSR data to device
    int *d_col_indices, *d_col_counts;
    CHECK_CUDA(cudaMalloc((void**)&d_col_indices, csr->nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_col_counts, csr->cols * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_col_indices, csr->col_indices, csr->nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_col_counts, 0, csr->cols * sizeof(int)));

    int shared_mem_size = 2*(csr->cols) * sizeof(int); //declare the size of the shared memory 
    int *last = new int[1];
    int *d_last;
    CHECK_CUDA(cudaMalloc((void**)&d_last, sizeof(int)));


    float *d_values, *d_t_values; //ordered values of trasposed matrix
    int *d_t_col_indices;
    int *d_row_offsets, *d_t_row_offsets;
    CHECK_CUDA(cudaMalloc((void**)&d_values, csr->nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_t_values, csr->nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_t_col_indices, csr->nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_row_offsets, (csr->rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_t_row_offsets, (csr->cols + 1) * sizeof(int)));

    int *col_ptr = new int[csr->cols +1];
    int *d_col_ptr;
    CHECK_CUDA(cudaMalloc((void**)&d_col_ptr, (csr->cols) * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_values, csr->values, csr->nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_t_col_indices, 0, csr->nnz * sizeof(int)));

    int *t_col_indices_ordered = new int[csr->nnz];
    int *d_t_col_indices_ordered;
    CHECK_CUDA(cudaMalloc((void**)&d_t_col_indices_ordered, csr->nnz * sizeof(int)));

    cudaEvent_t startK, stopK;
    CHECK_CUDA(cudaEventCreate(&startK));
    CHECK_CUDA(cudaEventCreate(&stopK));

    dummy_kernel<<<1,1>>>();
    CHECK_CUDA(cudaEventRecord(startK));

    int * zeroes = new int[csr->cols];
    memset(zeroes, 0, csr->cols * sizeof(int)); //copy slightly better than memset
    for(int i = 0; i < TRANSPOSITIONS; i++)
    {
        //use origin col indices
        CHECK_CUDA(cudaMemcpy(d_col_counts, zeroes, csr->cols * sizeof(int), cudaMemcpyHostToDevice)); //reset col counts to compute correctly
        countNNZPerColumn<<<((csr->nnz + 255) / 256), 256>>>(d_col_indices, d_col_counts, csr->nnz);
    
        prefix_scan<<<1, (csr->cols), shared_mem_size>>>(d_col_ptr, d_col_counts, csr->cols, d_last);
        cudaCheckError();
        CHECK_CUDA(cudaMemcpy(col_ptr, d_col_ptr, (csr->cols) * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(last, d_last, sizeof(int), cudaMemcpyDeviceToHost));
        // printf("Last: %d\n", last[0]);
        //correct last element missing
        // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda : Figure 39-4 
        // for(int i = 0; i < csr->cols; i++){
        //     col_ptr[csr->cols] += col_ptr[i];
        // } // (d_t_col_indices) /* *** */
        col_ptr[csr->cols] = last[0];
        // printf("Col Ptr: ");
        // for (int i = 0; i < csr->cols +1; i++) {
        //     printf("%d ", col_ptr[i]);
        // }
        // printf("\n");

        csr_t->row_offsets = col_ptr;
        
        //CHECK_CUDA(cudaMemcpy(d_row_offsets, csr->row_offsets, (csr->rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_row_offsets, col_ptr, (csr->rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        //CHECK_CUDA(cudaMemset(d_t_row_offsets, 0, (csr->cols + 1) * sizeof(int)));

        //compute row_offsets in cpu
        int count = 0;
        for(int i = 0; i < csr->cols; i++){
            int els = csr->row_offsets[i+1] - csr->row_offsets[i];
            //int els = col_ptr[i+1] - col_ptr[i];
            //memset els time for speed
            for(int j = 0; j < els; j++){
                csr_t->col_indices[count] = i; //col indices in crescent order
                count++;
            }
        }
        // printf("Col Indices Transposed: ");
        // for(int i = 0; i < csr->nnz; i++){
        //     printf("%d ", csr_t->col_indices[i]);
        // }
        // printf("\n");
        CHECK_CUDA(cudaMemcpy(d_t_col_indices, csr_t->col_indices, csr->nnz * sizeof(int), cudaMemcpyHostToDevice));

        order_by_column<<<(csr->cols + 15) /16, 16>>>(d_values, d_col_indices, d_t_values, d_col_ptr, d_col_counts, csr->cols, csr->nnz, d_t_col_indices, d_t_col_indices_ordered);
        //order_by_column<<<1, csr->cols>>>(d_values, d_col_indices, d_t_values, d_col_ptr, d_col_counts, csr->cols, csr->nnz, d_t_col_indices, d_t_col_indices_ordered);
        //cudaCheckError();
        cudaDeviceSynchronize();
    }
    CHECK_CUDA(cudaEventRecord(stopK));
    CHECK_CUDA(cudaEventSynchronize(stopK));

        //return ordered col indices
        CHECK_CUDA(cudaMemcpy(csr_t->col_indices, d_t_col_indices_ordered, csr->nnz * sizeof(int), cudaMemcpyDeviceToHost));
        // printf("Ordered Col Indices Transposed: ");
        // for(int i = 0; i < csr_t->nnz; i++){
        //     printf("%d ", csr_t->col_indices[i]);
        // }
        // printf("\n");


        CHECK_CUDA(cudaMemcpy(csr_t->values, d_t_values, csr->nnz * sizeof(float), cudaMemcpyDeviceToHost));
        // printf("Ordered Values Transposed: ");
        // for(int i = 0; i < csr_t->nnz; i++){
        //     printf("%f ", csr_t->values[i]);
        // }
        // printf("\n");
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float millisecondsK = 0;
    CHECK_CUDA(cudaEventElapsedTime(&millisecondsK, startK, stopK));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    int N = csr->cols;
    float ogbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / milliseconds; 
    float kgbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / millisecondsK;
    printf("Time for executing transpose operation: %f ms\n", milliseconds);
    PRINTF("Operation Time: %11.2f ms\n", milliseconds);
    PRINTF("Operation Throughput in GB/s: %7.2f\n", ogbs);
    PRINTF("Kernel Time: %11.2f ms\n", millisecondsK);
    PRINTF("Kernel Throughput in GB/s: %7.2f\n", kgbs);
    std::ofstream output;
    output.open ("logs/results.log", std::ios::out | std::ios_base::app);
    //output << "CSR" << "OpTime, Op-GB/s, " << milliseconds << " K-GB/s\n";
    // algorithm, OpTime, Op-GB/s, KTime, K-GB/s
    output << "CSRtoCSCcuda, " <<  milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n"; /* *** */
    output.close();
    cudaCheckError();

    CHECK_CUDA(cudaFree(d_col_indices));
    CHECK_CUDA(cudaFree(d_col_counts));
    CHECK_CUDA(cudaFree(d_col_ptr));
    CHECK_CUDA(cudaFree(d_last));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_t_values));
    CHECK_CUDA(cudaFree(d_t_col_indices));
    CHECK_CUDA(cudaFree(d_row_offsets));
    CHECK_CUDA(cudaFree(d_t_row_offsets));
    CHECK_CUDA(cudaFree(d_t_col_indices_ordered));

    PRINTF("Transpose Completed.\n");

    if (is_transpose(csr, csr_t)) {
        PRINTF("Transpose is correct.\n");
    } else {
        PRINTF("Transpose is incorrect.\n");
        std::ofstream errlogstream;
        errlogstream.open("logs/transpose_err.log", std::ios::out | std::ios::app);
        errlogstream << "Transpose Error: CSR to CSC\n";
        errlogstream << "Original Matrix:\n";
        pretty_print_csr_matrix(csr, errlogstream);
        errlogstream << "\n\nTranposed Matrix:\n";
        pretty_print_csr_matrix(csr_t, errlogstream);
        errlogstream.close();
        PRINTF("--------------------\n");
        return -1;
    }

    // if (ret == 0) {
    //     PRINTF("Transposition Completed Succesfully.\n");
    // } else {
    //     PRINTF("Transposition Completed Unsuccesfully.\n");
    // }
    PRINTF("--------------------\n");
    return 0;
}

int block_trasposition(float* mat, unsigned int N) {
    PRINTF("--------------------\n");
    PRINTF("Block Transposition Method Called: block_transposition() -> block_transpose().\n");
    if ((cudaSetDevice(0)) != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        return 1;
    }
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    int mem_size = N * N * sizeof(float);
    float* mat_t = (float*) malloc(mem_size);
    memset(mat_t, 0, mem_size);
    initMatrix(mat, N);
    float *d_mat, *d_mat_t;
    //int threads, blocks = 0;
    PRINTF("Allocating memory.\n");
    CHECK_CUDA(cudaMalloc((void**)&d_mat, mem_size));
    CHECK_CUDA(cudaMalloc((void**)&d_mat_t, mem_size));
    //cudaMalloc((void**)&d_mat_t, mem_size);
    PRINTF("Memory allocated.\n");
    //copy data to gpu

    // CHECK_CUDA(cudaMemcpy(d_mat, mat, mem_size, cudaMemcpyHostToDevice));
    PRINTF("Data copied.\n");
    //setup grid and block size
    dim3 DimGrid = {N/TILE_SIZE, N/TILE_SIZE, 1};
    dim3 DimBlock = {TILE_SIZE, BLOCK_ROWS, 1};
    
    //call kernel as many times as needed
    //first a dummy kernel -> others don't have this
    //block_transpose<<<DimGrid, DimBlock>>>(d_mat, d_mat_t, N);
    cudaEvent_t startK, stopK;
    CHECK_CUDA(cudaEventCreate(&startK));
    CHECK_CUDA(cudaEventCreate(&stopK));

    dummy_kernel<<<1,1>>>();
    CHECK_CUDA(cudaEventRecord(startK));
    for(int i = 0; i < TRANSPOSITIONS; i++){
        CHECK_CUDA(cudaMemcpy(d_mat, mat, mem_size, cudaMemcpyHostToDevice));
        block_transpose<<<DimGrid, DimBlock>>>(d_mat, d_mat_t, N);
    }
    CHECK_CUDA(cudaEventRecord(stopK));
    CHECK_CUDA(cudaEventSynchronize(stopK));
    PRINTF("Kernel returned\n");

    //copy data back
    CHECK_CUDA(cudaMemcpy(mat_t, d_mat_t, mem_size, cudaMemcpyDeviceToHost));
    //sync
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    float millisecondsK = 0;
    CHECK_CUDA(cudaEventElapsedTime(&millisecondsK, startK, stopK));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float ogbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / milliseconds;
    float kgbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / millisecondsK;
    PRINTF("Operation Time: %11.2f ms\n", milliseconds);
    PRINTF("Operation Throughput in GB/s: %7.2f\n", ogbs);
    PRINTF("Kernel Time: %11.2f ms\n", millisecondsK);
    PRINTF("Kernel Throughput in GB/s: %7.2f\n", kgbs);
    //printf("%f, %f, %f, %f\n", milliseconds, ogbs, millisecondsK, kgbs);
    
    std::ofstream output;
    output.open("logs/results.log", std::ios::out | std::ios_base::app);

    // algorithm, OpTime, Op-GB/s, KTime, K-GB/s
    output << "block, " <<  milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n";
    output.close();

    CHECK_CUDA(cudaEventDestroy(startK));
    CHECK_CUDA(cudaEventDestroy(stopK));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));


    //results
    // printMatrix(mat_t, N);

    //test if the matrix is transposed
    int ret = 0;
    if (testTranspose(mat, mat_t, N) != 0) {
        // save the matrix to a file
        std::ofstream output;
        output.open("logs/block_transpose_err.log", std::ios::out | std::ios_base::app);
        output << "Matrix: " << N << "x" << N << "\n";
        output << "Original Matrix:\n";
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                output << mat[i + j*N] << " ";
            }
            output << "\n";
        }
        output << "\n\nTransposed Matrix:\n";
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                output << mat_t[i + j*N] << " ";
            }
            output << "\n";
        }
        output.close();

        ret = -1;
    }
    PRINTF("--------------------\n");
    //free gpu resources
    CHECK_CUDA(cudaFree(d_mat));
    CHECK_CUDA(cudaFree(d_mat_t));
    free(mat_t);
    return ret;
}

int conflict_transposition(float* mat, unsigned int N) {
    PRINTF("--------------------\n");
    PRINTF("Conflict Transposition Method Called: conflict_transposition() -> conflict_transpose().\n");
    if ((cudaSetDevice(0)) != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        return 1;
    }
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    PRINTF("Cuda Events Created.\n");

    int mem_size = N * N * sizeof(float);
    float* mat_t = (float*) malloc(mem_size);
    memset(mat_t, 0, mem_size);
    float *d_mat, *d_mat_t;
    //int threads, blocks = 0;
    PRINTF("Allocating memory.\n");
    CHECK_CUDA(cudaMalloc((void**)&d_mat, mem_size));
    CHECK_CUDA(cudaMalloc((void**)&d_mat_t, mem_size));
    //cudaMalloc((void**)&d_mat_t, mem_size);
    PRINTF("Memory allocated.\n");
    //copy data to gpu


    PRINTF("Now copying data from host (mat) to device (d_mat). Exactly: %d Bytes\n", mem_size);
    // CHECK_CUDA(cudaMemcpy(d_mat, mat, mem_size, cudaMemcpyHostToDevice));
    PRINTF("Data copied.\n");
    //setup grid and block size
    dim3 DimGrid = {N/TILE_SIZE, N/TILE_SIZE, 1};
    dim3 DimBlock = {TILE_SIZE, BLOCK_ROWS, 1};
    
    //call kernel as many times as needed
    //first a dummy kernel
    // conflict_transpose<<<DimGrid, DimBlock>>>(d_mat, d_mat_t);
    cudaEvent_t startK, stopK;
    CHECK_CUDA(cudaEventCreate(&startK));
    CHECK_CUDA(cudaEventCreate(&stopK));

    dummy_kernel<<<1,1>>>();
    CHECK_CUDA(cudaEventRecord(startK));
    for(int i = 0; i < TRANSPOSITIONS; i++){
        CHECK_CUDA(cudaMemcpy(d_mat, mat, mem_size, cudaMemcpyHostToDevice));
        conflict_transpose<<<DimGrid, DimBlock>>>(d_mat, d_mat_t);
    }
    CHECK_CUDA(cudaEventRecord(stopK));
    CHECK_CUDA(cudaEventSynchronize(stopK));
    PRINTF("Kernel returned\n");

    //copy data back
    CHECK_CUDA(cudaMemcpy(mat_t, d_mat_t, mem_size, cudaMemcpyDeviceToHost));
    //sync
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    float millisecondsK = 0;
    CHECK_CUDA(cudaEventElapsedTime(&millisecondsK, startK, stopK));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    PRINTF("Operation Time: %11.2f ms\n", milliseconds);
    float ogbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / milliseconds;
    float kgbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / millisecondsK;
    PRINTF("Throughput in GB/s: %7.2f\n", ogbs);
    PRINTF("Kernel Time: %11.2f ms\n", millisecondsK);
    PRINTF("Throughput in GB/s: %7.2f\n", kgbs);
    //printf("%f, %f, %f, %f, ", milliseconds, ogbs, millisecondsK, kgbs);
    
    std::ofstream output;
    output.open ("logs/results.log", std::ios::out | std::ios_base::app);
    //output << "Conflict, " << "OpTime, Op-GB/s, " << milliseconds << " K-GB/s\n";
    // algorithm, OpTime, Op-GB/s, KTime, K-GB/s
    output << "Conflict, " <<  milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n";
    output.close();

    CHECK_CUDA(cudaEventDestroy(startK));
    CHECK_CUDA(cudaEventDestroy(stopK));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));


    //results
    //printMatrix(mat_t, N);

    //test if the matrix is transposed
    int ret =  testTranspose(mat, mat_t, N);
    PRINTF("--------------------\n");
    //free gpu resources
    CHECK_CUDA(cudaFree(d_mat));
    CHECK_CUDA(cudaFree(d_mat_t));
    free(mat_t);
    return ret;
}

// int transposeCSRToCSC_cuda(csr_matrix *csr, csr_matrix *csr_t) {
    // assert(csr != NULL && csr_t != NULL);
    // assert(csr->rows == csr_t->cols && csr->cols == csr_t->rows);
    // PRINTF("Transpose CSR to CSC Cuda Method Called: transposeCSRToCSC_cuda().\n");
    // if ((cudaSetDevice(0)) != cudaSuccess) {
    //     PRINTF("Failed to set CUDA device\n");
    //     return 1;
    // }
    // cudaEvent_t start, stop;
    // CHECK_CUDA(cudaEventCreate(&start));
    // CHECK_CUDA(cudaEventCreate(&stop));
    // CHECK_CUDA(cudaEventRecord(start));

    // // Copy input CSR data to device
    // int *d_col_indices, *d_col_counts;
    // CHECK_CUDA(cudaMalloc((void**)&d_col_indices, csr->nnz * sizeof(int)));
    // CHECK_CUDA(cudaMalloc((void**)&d_col_counts, csr->cols * sizeof(int)));
    // CHECK_CUDA(cudaMemcpy(d_col_indices, csr->col_indices, csr->nnz * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemset(d_col_counts, 0, csr->cols * sizeof(int)));

    // int shared_mem_size = 2*(csr->cols) * sizeof(int); //declare the size of the shared memory 
    // int *last = new int[1];
    // int *d_last;
    // CHECK_CUDA(cudaMalloc((void**)&d_last, sizeof(int)));


    // float *d_values, *d_t_values; //ordered values of trasposed matrix
    // int *d_t_col_indices;
    // int *d_row_offsets, *d_t_row_offsets;
    // CHECK_CUDA(cudaMalloc((void**)&d_values, csr->nnz * sizeof(float)));
    // CHECK_CUDA(cudaMalloc((void**)&d_t_values, csr->nnz * sizeof(float)));
    // CHECK_CUDA(cudaMalloc((void**)&d_t_col_indices, csr->nnz * sizeof(int)));
    // CHECK_CUDA(cudaMalloc((void**)&d_row_offsets, (csr->rows + 1) * sizeof(int)));
    // CHECK_CUDA(cudaMalloc((void**)&d_t_row_offsets, (csr->cols + 1) * sizeof(int)));

    // int *col_ptr = new int[csr->cols +1];
    // int *d_col_ptr;
    // CHECK_CUDA(cudaMalloc((void**)&d_col_ptr, (csr->cols) * sizeof(int)));
    // CHECK_CUDA(cudaMemcpy(d_values, csr->values, csr->nnz * sizeof(float), cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemset(d_t_col_indices, 0, csr->nnz * sizeof(int)));

    // int *t_col_indices_ordered = new int[csr->nnz];
    // int *d_t_col_indices_ordered;
    // CHECK_CUDA(cudaMalloc((void**)&d_t_col_indices_ordered, csr->nnz * sizeof(int)));

    // cudaEvent_t startK, stopK;
    // CHECK_CUDA(cudaEventCreate(&startK));
    // CHECK_CUDA(cudaEventCreate(&stopK));
    // CHECK_CUDA(cudaEventRecord(startK));
     
    // for(int i = 0; i < TRANSPOSITIONS; i++)
    // {
    //     //use origin col indices
    //     cudaMemset(d_col_counts, 0, csr->cols * sizeof(int)); //reset col counts to compute correctly
    //     countNNZPerColumn<<<((csr->nnz + 255) / 256), 256>>>(d_col_indices, d_col_counts, csr->nnz);
    
    //     prefix_scan<<<1, (csr->cols), shared_mem_size>>>(d_col_ptr, d_col_counts, csr->cols, d_last);
    //     cudaCheckError();
    //     CHECK_CUDA(cudaMemcpy(col_ptr, d_col_ptr, (csr->cols) * sizeof(int), cudaMemcpyDeviceToHost));
    //     CHECK_CUDA(cudaMemcpy(last, d_last, sizeof(int), cudaMemcpyDeviceToHost));
    //     // printf("Last: %d\n", last[0]);
    //     //correct last element missing
    //     // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda : Figure 39-4 
    //     // for(int i = 0; i < csr->cols; i++){
    //     //     col_ptr[csr->cols] += col_ptr[i];
    //     // } // (d_t_col_indices) /* *** */
    //     col_ptr[csr->cols] = last[0];
    //     // printf("Col Ptr: ");
    //     // for (int i = 0; i < csr->cols +1; i++) {
    //     //     printf("%d ", col_ptr[i]);
    //     // }
    //     // printf("\n");

    //     csr_t->row_offsets = col_ptr;
        
    //     //CHECK_CUDA(cudaMemcpy(d_row_offsets, csr->row_offsets, (csr->rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    //     CHECK_CUDA(cudaMemcpy(d_row_offsets, col_ptr, (csr->rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    //     CHECK_CUDA(cudaMemset(d_t_row_offsets, 0, (csr->cols + 1) * sizeof(int)));

    //     //compute row_offsets in cpu
    //     int count = 0;
    //     for(int i = 0; i < csr->cols; i++){
    //         int els = csr->row_offsets[i+1] - csr->row_offsets[i];
    //         //int els = col_ptr[i+1] - col_ptr[i];
    //         //memset els time for speed
    //         for(int j = 0; j < els; j++){
    //             csr_t->col_indices[count] = i; //col indices in crescent order
    //             count++;
    //         }
    //     }
    //     // printf("Col Indices Transposed: ");
    //     // for(int i = 0; i < csr->nnz; i++){
    //     //     printf("%d ", csr_t->col_indices[i]);
    //     // }
    //     // printf("\n");
    //     CHECK_CUDA(cudaMemcpy(d_t_col_indices, csr_t->col_indices, csr->nnz * sizeof(int), cudaMemcpyHostToDevice));

    //     order_by_column<<<(csr->cols + 15) /16, 16>>>(d_values, d_col_indices, d_t_values, d_col_ptr, d_col_counts, csr->cols, csr->nnz, d_t_col_indices, d_t_col_indices_ordered);
    //     //order_by_column<<<1, csr->cols>>>(d_values, d_col_indices, d_t_values, d_col_ptr, d_col_counts, csr->cols, csr->nnz, d_t_col_indices, d_t_col_indices_ordered);
    //     //cudaCheckError();
    //     cudaDeviceSynchronize();
    
    //     CHECK_CUDA(cudaEventRecord(stopK));
    //     CHECK_CUDA(cudaEventSynchronize(stopK));

    //     //return ordered col indices
    //     CHECK_CUDA(cudaMemcpy(csr_t->col_indices, d_t_col_indices_ordered, csr->nnz * sizeof(int), cudaMemcpyDeviceToHost));
    //     // printf("Ordered Col Indices Transposed: ");
    //     // for(int i = 0; i < csr_t->nnz; i++){
    //     //     printf("%d ", csr_t->col_indices[i]);
    //     // }
    //     // printf("\n");


    //     CHECK_CUDA(cudaMemcpy(csr_t->values, d_t_values, csr->nnz * sizeof(float), cudaMemcpyDeviceToHost));
    //     // printf("Ordered Values Transposed: ");
    //     // for(int i = 0; i < csr_t->nnz; i++){
    //     //     printf("%f ", csr_t->values[i]);
    //     // }
    //     // printf("\n");
    // }
    // CHECK_CUDA(cudaEventRecord(stop));
    // CHECK_CUDA(cudaEventSynchronize(stop));

    // float millisecondsK = 0;
    // CHECK_CUDA(cudaEventElapsedTime(&millisecondsK, startK, stopK));
    // float milliseconds = 0;
    // CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    // int N = csr->cols;
    // float ogbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / milliseconds; 
    // float kgbs = 2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS / millisecondsK;
    // printf("Time for executing transpose operation: %f ms\n", milliseconds);
    // PRINTF("Operation Time: %11.2f ms\n", milliseconds);
    // PRINTF("Operation Throughput in GB/s: %7.2f\n", ogbs);
    // PRINTF("Kernel Time: %11.2f ms\n", millisecondsK);
    // PRINTF("Kernel Throughput in GB/s: %7.2f\n", kgbs);
    // std::ofstream output;
    // output.open ("logs/results.log", std::ios::out | std::ios_base::app);
    // //output << "CSR" << "OpTime, Op-GB/s, " << milliseconds << " K-GB/s\n";
    // // algorithm, OpTime, Op-GB/s, KTime, K-GB/s
    // output << "CSRtoCSCcuda, " <<  milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n"; /* *** */
    // output.close();
    // cudaCheckError();

    // CHECK_CUDA(cudaFree(d_col_indices));
    // CHECK_CUDA(cudaFree(d_col_counts));
    // CHECK_CUDA(cudaFree(d_col_ptr));
    // CHECK_CUDA(cudaFree(d_last));
    // CHECK_CUDA(cudaFree(d_values));
    // CHECK_CUDA(cudaFree(d_t_values));
    // CHECK_CUDA(cudaFree(d_t_col_indices));
    // CHECK_CUDA(cudaFree(d_row_offsets));
    // CHECK_CUDA(cudaFree(d_t_row_offsets));
    // CHECK_CUDA(cudaFree(d_t_col_indices_ordered));

    // PRINTF("Transpose Completed.\n");

    // if (is_transpose(csr, csr_t)) {
    //     PRINTF("Transpose is correct.\n");
    // } else {
    //     PRINTF("Transpose is incorrect.\n");
    //     std::ofstream errlogstream;
    //     errlogstream.open("logs/transpose_err.log", std::ios::out | std::ios::app);
    //     errlogstream << "Transpose Error: CSR to CSC\n";
    //     errlogstream << "Original Matrix:\n";
    //     pretty_print_csr_matrix(csr, errlogstream);
    //     errlogstream << "\n\nTranposed Matrix:\n";
    //     pretty_print_csr_matrix(csr_t, errlogstream);
    //     errlogstream.close();
    //     return -1;
    // }

    // return 0;
// }

// int transposeCSRToCSC(const thrust::host_vector<float>& h_values, const thrust::host_vector<int>& h_col_indices,
//                        const thrust::host_vector<int>& h_row_ptr, int num_rows, int num_cols,
//                        thrust::device_vector<float>& d_t_values, thrust::device_vector<int>& d_t_row_indices,
//                        thrust::device_vector<int>& d_t_col_ptr) {
//     PRINTF("Transpose CSR to CSC Method Called: transposeCSRToCSC().\n");
//     int nnz = h_values.size();
//     if ((cudaSetDevice(0)) != cudaSuccess) {
//         PRINTF("Failed to set CUDA device\n");
//         return 1;
//     }
//     cudaEvent_t start, stop;
//     CHECK_CUDA(cudaEventCreate(&start));
//     CHECK_CUDA(cudaEventCreate(&stop));

//     // Copy input CSR data to device
//     thrust::device_vector<float> d_values = h_values;
//     thrust::device_vector<int> d_col_indices = h_col_indices;
//     thrust::device_vector<int> d_row_ptr = h_row_ptr;
    
//     // Initialize device vectors for transposed matrix
//     d_t_values.resize(nnz);
//     d_t_row_indices.resize(nnz);
//     d_t_col_ptr.resize(num_cols + 1);

//     CHECK_CUDA(cudaEventRecord(start));
//     // Device vector for column counts
//     thrust::device_vector<int> d_col_counts(num_cols, 0);
    
//     // Kernel to count non-zero entries per column
//     countNNZPerColumn<<<((nnz + 255) / 256), 256>>>(thrust::raw_pointer_cast(d_col_indices.data()),
//                                                   thrust::raw_pointer_cast(d_col_counts.data()), nnz);

//     // Compute column pointers using exclusive scan
//     thrust::exclusive_scan(d_col_counts.begin(), d_col_counts.end(), d_t_col_ptr.begin());
//     cudaCheckError();

//     // Copy the column pointers to create a correct offset for scattering
//     thrust::device_vector<int> d_col_ptr_copy = d_t_col_ptr;

//     // Kernel to scatter values and row indices to transposed matrix
//     scatterToTransposed<<<(num_rows + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_values.data()),
//                                                          thrust::raw_pointer_cast(d_col_indices.data()),
//                                                          thrust::raw_pointer_cast(d_row_ptr.data()),
//                                                          thrust::raw_pointer_cast(d_t_values.data()),
//                                                          thrust::raw_pointer_cast(d_t_row_indices.data()),
//                                                          thrust::raw_pointer_cast(d_col_ptr_copy.data()), num_rows);
//     CHECK_CUDA(cudaEventRecord(stop));
//     CHECK_CUDA(cudaEventSynchronize(stop));
//     float milliseconds = 0;
//     CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
//     printf("Time for executing transpose operation: %f ms\n", milliseconds);
//     std::ofstream output;
//     output.open ("logs/results.log", std::ios::out | std::ios_base::app);
//     output << "CSR" << "OpTime, Op-GB/s, " << milliseconds << " K-GB/s\n";
//     // algorithm, OpTime, Op-GB/s, KTime, K-GB/s
//     // output << "CSRtoCSC, " <<  milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n"; /* *** */
//     output.close();
//     cudaCheckError();
//     return 0;
// }

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