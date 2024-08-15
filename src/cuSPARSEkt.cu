#include "../include/commons.h"
#include "../include/csr.h"
#include "../include/cuSPARSEkt.cuh"
#include <cassert>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <fstream>


int cuSparseCSRt(const csr_matrix* in, csr_matrix* out) {
    assert(in != NULL && out != NULL && in->row_offsets != NULL && in->col_indices != NULL && in->values != NULL && out->row_offsets != NULL && out->col_indices != NULL && out->values != NULL);
    assert(in->rows == out->cols);
    assert(in->cols == out->rows);
    if(cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA device\n");
        return 1;
    }
    printf("cuSparseCSRt\n");
    // ? Create cuSPARSE handle and matrix descriptor
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    // ? Set matrix type and index base
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    size_t bufferSize = 0;
    int* d_in_row_offsets, *d_in_cols, *d_out_row_offsets, *d_out_cols;
    float* d_in_values, *d_out_values;
    // ? Allocate memory on device for Input Matrix
    CHECK_CUDA(cudaMalloc((void**)&d_in_row_offsets, (in->rows + 1) * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_in_cols, in->nnz * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_in_values, in->nnz * sizeof(float)));
    // ? Allocate memory on device for Output Matrix
    CHECK_CUDA(cudaMalloc((void**)&d_out_row_offsets, (out->cols + 1) * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_out_cols, out->nnz * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_out_values, out->nnz * sizeof(float)));
    // ? Create cuda events to measure time
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    // ? Copy data from host to device for Input Matrix
    CHECK_CUDA(cudaMemcpy(d_in_row_offsets, in->row_offsets, (in->rows + 1) * sizeof(size_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_in_cols, in->col_indices, in->nnz * sizeof(size_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_in_values, in->values, in->nnz * sizeof(float), cudaMemcpyHostToDevice));
    // ? Find buffer size to perform the transpose
    cusparseCsr2cscEx2_bufferSize(
        handle,
        in->rows, in->cols, in->nnz,
        d_in_values, d_in_row_offsets, d_in_cols,
        d_out_values, d_out_row_offsets, d_out_cols,
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &bufferSize
    );
    printf("Buffer size: %lu\n", bufferSize);
    // ? Allocate memory on device for buffer
    void* dBuffer = NULL;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    // ? Record time before performing the transpose operation
    CHECK_CUDA(cudaEventRecord(start));
    // ? Perform the actual transpose operation on device
    cusparseCsr2cscEx2(
        handle,
        in->rows, in->cols, in->nnz,
        d_in_values, d_in_row_offsets, d_in_cols,
        d_out_values, d_out_row_offsets, d_out_cols,
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, dBuffer
    );
    // ? Record time after performing the transpose operation
    CHECK_CUDA(cudaEventRecord(stop));
    // ? Copy data from device to host for Output Matrix
    CHECK_CUDA(cudaMemcpy(out->row_offsets, d_out_row_offsets, (out->cols + 1) * sizeof(size_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out->col_indices, d_out_cols, out->nnz * sizeof(size_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out->values, d_out_values, out->nnz * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time for executing cuSPARSECSRt operation: %f ms\n", milliseconds);

    std::ofstream output;
    output.open ("logs/results.log", std::ios::out | std::ios_base::app);
    output << "N_mat, " << "Cusparse, " << "OpTime, Op-GB/s, " << milliseconds << "K-GB/s\n";
    output.close();

    // Check if transpose was successful
    printf("Now checking if transpose was successful\n");
    if (is_transpose(in, out)) {
        printf("Transpose is correct\n");
    } else {
        printf("Transpose is incorrect\n");
        // save to log file
        std::ofstream logstream;
        logstream.open("logs/cusparse_transpose_err.log", std::ios::out);
        logstream << "Original Matrix:\n";
        pretty_print_csr_matrix(in, logstream);
        logstream << "\n\nTransposed Matrix:\n";
        pretty_print_csr_matrix(out, logstream);
        logstream.close();
    }

    //TODO: correct output
    // ? Free memory on device
    CHECK_CUDA(cudaFree(d_in_row_offsets));
    CHECK_CUDA(cudaFree(d_in_cols));
    CHECK_CUDA(cudaFree(d_in_values));
    CHECK_CUDA(cudaFree(d_out_row_offsets));
    CHECK_CUDA(cudaFree(d_out_cols));
    CHECK_CUDA(cudaFree(d_out_values));
    CHECK_CUDA(cudaFree(dBuffer));
    // ? Destroy cuSPARSE handle and matrix descriptor
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
    return 0;
}
