//complete run for each transposition algorithm callled by complete_benchmark in main
#include "../include/benchmarks.cuh"
#include "../include/commons.h"
#include "../include/kernels.cuh"
#include "../include/debug.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <fstream>

int coo_transposition(coo_matrix* coo, int matrix_size) {
    PRINTF("--------------------\n");
    PRINTF("COO Transposition Method called: coo_transposition() -> cuCOOt().\n");
    if ((cudaSetDevice(0)) != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        return 1;
    }
    coo_element* el = coo->el;
    coo_matrix* d_coo;
    coo_element* d_el;

    cudaEvent_t start, stop, startK, stopK;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&startK));
    CHECK_CUDA(cudaEventCreate(&stopK));

    dummy_kernel<<<1,1>>>(); // ? Warm up the GPU

    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUDA(cudaMallocManaged((void**)&d_coo, sizeof(coo_matrix)));
    CHECK_CUDA(cudaMallocManaged((void**)&d_el, coo->nnz * sizeof(coo_element)));
    CHECK_CUDA(cudaMemcpy(d_coo, coo, sizeof(coo_matrix), cudaMemcpyHostToDevice));
    PRINTF("Copied & Allocated Memory Succesfully\n");
    d_coo->el = d_el;

    #ifdef DEBUG
        printf("Pre-Transpose Matrix:\n");
        print_coo_less(coo);
    #endif
    
    CHECK_CUDA(cudaEventRecord(startK));
    for (int i = 0; i < TRANSPOSITIONS; i++) {
        cudaMemcpy(d_el, el, coo->nnz * sizeof(coo_element), cudaMemcpyHostToDevice);
        cuCOOt<<<(coo->nnz + 255) / 256, 256>>>(d_coo->el, d_coo->nnz);
    }
    //cuCOOt<<<coo->nnz,1>>>(d_coo->el, d_coo->nnz);
    CHECK_CUDA(cudaEventRecord(stopK));

    CHECK_CUDA(cudaMemcpy(d_coo, d_coo, sizeof(coo_matrix), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stopK));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaDeviceSynchronize());

    int ret = 0;
    #ifdef DEBUG
        printf("Post-Transpose Matrix:\n");
        print_coo_less(d_coo);
    #endif
    if (is_transpose(coo, d_coo)) {
        PRINTF("Transpose is correct.\n");
    } else {
        ret = -1;
        printf("coo_transposition) Transpose is incorrect.\n");
    }

    float millisecondsK = 0;
    CHECK_CUDA(cudaEventElapsedTime(&millisecondsK, startK, stopK));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    int N = coo->cols; 
    float ogbs = (float)(2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / milliseconds;
    float kgbs = (float)(2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / millisecondsK;
    milliseconds /= (float)TRANSPOSITIONS;
    millisecondsK /= (float)TRANSPOSITIONS;
    PRINTF("Time for executing cuCOOt operation: %f ms\n", milliseconds);
    PRINTF("Operation Throughput in GB/s: %7.2f\n", ogbs);
    PRINTF("Kernel Time: %11.2f ms\n", millisecondsK);
    PRINTF("Kernel Throughput in GB/s: %7.2f\n", kgbs);

    std::ofstream output;
    output.open ("logs/results.log", std::ios::out | std::ios_base::app);
    // algorithm, OpTime, Op-GB/s, KTime, K-GB/s
    output << "COO, " << matrix_size << "x" << matrix_size << ", " <<  milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n"; /* *** */
    output.close();

    CHECK_CUDA(cudaFree(d_coo));
    CHECK_CUDA(cudaFree(d_el));
    PRINTF("Freed Memory Succesfully.\n");
    PRINTF("--------------------\n");
    return ret;
}
int csr_transposition(csr_matrix* csr, csr_matrix* csr_t, int matrix_size) {
    cudaEvent_t start, stop;
    cudaEvent_t startK, stopK;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventCreate(&startK));
    CHECK_CUDA(cudaEventCreate(&stopK));
    dummy_kernel<<<1,1>>>(); // ? Warm up the GPU
    CHECK_CUDA(cudaEventRecord(start));

    int *d_row_ptr, *d_col_indices;
    float *d_values;
    CHECK_CUDA(cudaMalloc((void**)&d_row_ptr, (csr->rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_col_indices, csr->nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_values, csr->nnz * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_row_ptr, csr->row_offsets, (csr->rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_indices, csr->col_indices, csr->nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, csr->values, csr->nnz * sizeof(float), cudaMemcpyHostToDevice));
    int *d_t_row_ptr, *d_t_col_indices;
    float *d_t_values;
    CHECK_CUDA(cudaMalloc((void**)&d_t_row_ptr, (csr->cols + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_t_col_indices, csr->nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_t_values, csr->nnz * sizeof(float)));
    
    CHECK_CUDA(cudaMemset(d_t_row_ptr, 0, (csr->cols + 1) * sizeof(int))); // zero init not needed in theory
    CHECK_CUDA(cudaMemset(d_t_col_indices, 0, csr->nnz * sizeof(int))); // in practice we avoid propagation of artifacts
    CHECK_CUDA(cudaMemset(d_t_values, 0, csr->nnz * sizeof(float)));

    int block_size = 1024;
    int grid_size = std::min((csr->nnz + block_size - 1) / block_size, 1024);

    CHECK_CUDA(cudaEventRecord(startK));
    int* intra;
    CHECK_CUDA(cudaMalloc((void**)&intra, csr->nnz * sizeof(int)));
    int* inter;
    CHECK_CUDA(cudaMalloc((void**)&inter, csr->cols * sizeof(int)));
    int* RowIdx;
    CHECK_CUDA(cudaMalloc((void**)&RowIdx, csr->nnz * sizeof(int)));

    int row_thr = 32;
    int row_blk = (csr->rows + row_thr - 1) / row_thr;
    for(int i = 0; i < TRANSPOSITIONS; i++) {
        // reset values
        CHECK_CUDA(cudaMemset(inter, 0, csr->cols * sizeof(int)));
        getRowIdx<<<row_blk, row_thr>>>(RowIdx, d_row_ptr, csr->rows, csr->nnz);
        cudaCheckError();
        getIntraInter<<<grid_size, block_size>>>(intra, inter, csr->nnz, d_col_indices);
        cudaCheckError();
        getRowOffsets<<<1, csr->cols>>>(d_t_row_ptr, inter, csr->cols);
        cudaCheckError();
        assignValues<<<grid_size, block_size>>>(d_t_col_indices, d_t_values, d_col_indices, d_values, intra, inter, RowIdx, d_t_row_ptr, csr->nnz);
        cudaCheckError();
    }// end of transpositions
    
    CHECK_CUDA(cudaEventRecord(stopK));
    CHECK_CUDA(cudaMemcpy(csr_t->row_offsets, d_t_row_ptr, (csr->cols + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(csr_t->col_indices, d_t_col_indices, csr->nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(csr_t->values, d_t_values, csr->nnz * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stopK));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float milliseconds = 0;
    float millisecondsK = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    CHECK_CUDA(cudaEventElapsedTime(&millisecondsK, startK, stopK));
    int N = csr->cols;
    float ogbs = (float)(2.0 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / milliseconds;
    float kgbs = (float)(2.0 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / millisecondsK;
    milliseconds /= (float)TRANSPOSITIONS;
    millisecondsK /= (float)TRANSPOSITIONS;
    PRINTF("Time for executing cuCSRt operation: %f ms\n", milliseconds);
    PRINTF("Operation Throughput in GB/s: %7.2f\n", ogbs);

    std::ofstream output;
    output.open("logs/results.log", std::ios::out | std::ios_base::app);
    // algorithm, MatSize, OpTime, Op-GB/s, KTime, K-GB/s
    output << "CSR gpu, " << matrix_size << "x" << matrix_size << ", " <<  milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n"; /* *** */
    output.close();

    if(is_transpose(csr, csr_t)) {
        PRINTF("Transpose is correct.\n");
    } else {
        printf("csr_transposition_2) Transpose is incorrect.\n");
        std::ofstream errlogstream;
        errlogstream.open("logs/transpose_err.log", std::ios::out | std::ios::app);
        errlogstream << "Transpose Error: CSR\n";
        errlogstream << "Original Matrix:\n";
        pretty_print_csr_matrix(csr, errlogstream);
        errlogstream << "\n\nTranposed Matrix:\n";
        pretty_print_csr_matrix(csr_t, errlogstream);
        errlogstream.close();
        PRINTF("--------------------\n");
        return -1;
    }

    CHECK_CUDA(cudaFree(inter));
    CHECK_CUDA(cudaFree(intra));
    CHECK_CUDA(cudaFree(RowIdx));

    CHECK_CUDA(cudaFree(d_row_ptr));
    CHECK_CUDA(cudaFree(d_col_indices));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_t_row_ptr));
    CHECK_CUDA(cudaFree(d_t_col_indices));
    CHECK_CUDA(cudaFree(d_t_values));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(startK));
    CHECK_CUDA(cudaEventDestroy(stopK));
    
    return 0;
}

int block_trasposition(float* mat, unsigned int N, int matrix_size) {
    PRINTF("--------------------\n");
    PRINTF("Block Transposition Method Called: block_transposition() -> block_transpose().\n");
    if ((cudaSetDevice(0)) != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        return 1;
    }
    cudaEvent_t start, stop;
    cudaEvent_t startK, stopK;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventCreate(&startK));
    CHECK_CUDA(cudaEventCreate(&stopK));

    dummy_kernel<<<1,1>>>(); // ? Warm up the GPU

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

    PRINTF("Data copied.\n");
    //setup grid and block size
    dim3 DimGrid = {N/TILE_SIZE, N/TILE_SIZE, 1};
    dim3 DimBlock = {TILE_SIZE, BLOCK_ROWS, 1};

    CHECK_CUDA(cudaMemcpy(d_mat, mat, mem_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(startK));
    for(int i = 0; i < TRANSPOSITIONS; i++){
        block_transpose<<<DimGrid, DimBlock>>>(d_mat, d_mat_t, N);
    }
    CHECK_CUDA(cudaEventRecord(stopK));
    PRINTF("Kernel returned\n");

    //copy data back
    CHECK_CUDA(cudaMemcpy(mat_t, d_mat_t, mem_size, cudaMemcpyDeviceToHost));
    //sync
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stopK));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaDeviceSynchronize());

    float millisecondsK = 0;
    CHECK_CUDA(cudaEventElapsedTime(&millisecondsK, startK, stopK));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float ogbs = (float)(2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / milliseconds;
    float kgbs = (float)(2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / millisecondsK;
    milliseconds /= (float)TRANSPOSITIONS;
    millisecondsK /= (float)TRANSPOSITIONS;
    PRINTF("Operation Time: %11.2f ms\n", milliseconds);
    PRINTF("Operation Throughput in GB/s: %7.2f\n", ogbs);
    PRINTF("Kernel Time: %11.2f ms\n", millisecondsK);
    PRINTF("Kernel Throughput in GB/s: %7.2f\n", kgbs);

    std::ofstream output;
    output.open("logs/results.log", std::ios::out | std::ios_base::app);

    // algorithm, OpTime, Op-GB/s, KTime, K-GB/s
    output << "block, " << matrix_size << "x" << matrix_size << ", " << milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n";
    output.close();

    CHECK_CUDA(cudaEventDestroy(startK));
    CHECK_CUDA(cudaEventDestroy(stopK));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    //test if the matrix is transposed
    int ret = 0;
    if (testTranspose(mat, mat_t, N) != 0) {
        printf("block_transpose) Transpose is incorrect.\n");
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

int conflict_transposition(float* mat, unsigned int N, int matrix_size) {
    PRINTF("--------------------\n");
    PRINTF("Conflict Transposition Method Called: conflict_transposition() -> conflict_transpose().\n");
    if ((cudaSetDevice(0)) != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        return 1;
    }
    cudaEvent_t start, stop;
    cudaEvent_t startK, stopK;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventCreate(&startK));
    CHECK_CUDA(cudaEventCreate(&stopK));

    dummy_kernel<<<1,1>>>(); // ? Warm up the GPU

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
    PRINTF("Memory allocated.\n");
    PRINTF("Now copying data from host (mat) to device (d_mat). Exactly: %d Bytes\n", mem_size);
    PRINTF("Data copied.\n");
    //setup grid and block size
    dim3 DimGrid = {N/TILE_SIZE, N/TILE_SIZE, 1};
    dim3 DimBlock = {TILE_SIZE, BLOCK_ROWS, 1};
    
    CHECK_CUDA(cudaMemcpy(d_mat, mat, mem_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(startK));
    for(int i = 0; i < TRANSPOSITIONS; i++){
        conflict_transpose<<<DimGrid, DimBlock>>>(d_mat, d_mat_t);
    }
    CHECK_CUDA(cudaEventRecord(stopK));
    PRINTF("Kernel returned\n");

    //copy data back
    CHECK_CUDA(cudaMemcpy(mat_t, d_mat_t, mem_size, cudaMemcpyDeviceToHost));
    //sync
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stopK));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaDeviceSynchronize());
    
    float millisecondsK = 0;
    CHECK_CUDA(cudaEventElapsedTime(&millisecondsK, startK, stopK));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    PRINTF("Operation Time: %11.2f ms\n", milliseconds);
    float ogbs = (float)(2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / milliseconds;
    float kgbs = (float)(2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / millisecondsK;
    milliseconds /= (float)TRANSPOSITIONS;
    millisecondsK /= (float)TRANSPOSITIONS;
    PRINTF("Throughput in GB/s: %7.2f\n", ogbs);
    PRINTF("Kernel Time: %11.2f ms\n", millisecondsK);
    PRINTF("Throughput in GB/s: %7.2f\n", kgbs);

    std::ofstream output;
    output.open("logs/results.log", std::ios::out | std::ios_base::app);
    // algorithm, OpTime, Op-GB/s, KTime, K-GB/s
    output << "Conflict, " << matrix_size << "x" << matrix_size << ", " <<  milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n";
    output.close();

    CHECK_CUDA(cudaEventDestroy(startK));
    CHECK_CUDA(cudaEventDestroy(stopK));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    //test if the matrix is transposed
    int ret =  testTranspose(mat, mat_t, N);
    if (ret != 0) {
        printf("conflict_transpose) Transpose is incorrect.\n");
    }
    PRINTF("--------------------\n");
    //free gpu resources
    CHECK_CUDA(cudaFree(d_mat));
    CHECK_CUDA(cudaFree(d_mat_t));
    free(mat_t);
    return ret;
}
