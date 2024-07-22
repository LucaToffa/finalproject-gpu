#include "../include/commons.h"
#include "../include/csr.h"
#include "../include/cuSPARSEkt.cuh"
#include <cuda_runtime.h>
#include <cusparse.h>


int cuSparseCSRt(csr_matrix* in, csr_matrix* out) {
    printf("cuSparseCSRt\n");
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    size_t bufferSize = 0;
    int* d_in_row_offsets, *d_in_cols, *d_out_row_offsets, *d_out_cols;
    float* d_in_values, *d_out_values;
    // ? Allocate memory on device for Input Matrix
    printf("Now allocating %d bytes...\n", (in->rows + 1) * sizeof(int));
    CHECK_CUDA(cudaMalloc((void**)&d_in_row_offsets, (in->rows + 1) * sizeof(int)));
    printf("Now allocating %d bytes...\n", in->nnz * sizeof(int));
    CHECK_CUDA(cudaMalloc((void**)&d_in_cols, in->nnz * sizeof(int)));
    printf("Now allocating %d bytes...\n", in->nnz * sizeof(float));
    CHECK_CUDA(cudaMalloc((void**)&d_in_values, in->nnz * sizeof(float)));
    // ? Allocate memory on device for Output Matrix
    printf("Now allocating %d bytes...\n", (out->rows + 1) * sizeof(int));
    CHECK_CUDA(cudaMalloc((void**)&d_out_row_offsets, (out->rows + 1) * sizeof(int)));
    printf("Now allocating %d bytes...\n", out->nnz * sizeof(int));
    CHECK_CUDA(cudaMalloc((void**)&d_out_cols, out->nnz * sizeof(int)));
    printf("Now allocating %d bytes...\n", out->nnz * sizeof(float));
    CHECK_CUDA(cudaMalloc((void**)&d_out_values, out->nnz * sizeof(float)));
    // ? Copy data from host to device for Input Matrix
    CHECK_CUDA(cudaMemcpy(d_in_row_offsets, in->row_offsets, (in->rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_in_cols, in->col_indices, in->nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_in_values, in->values, in->nnz * sizeof(float), cudaMemcpyHostToDevice));
    // ? Find buffer size to perform the transpose
    cusparseCsr2cscEx2_bufferSize(
        handle,
        in->rows, in->cols, in->nnz,
        d_in_values, d_in_row_offsets, d_in_cols,
        d_out_values, d_out_row_offsets, d_out_cols,
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT, &bufferSize
    );
    printf("Buffer size: %ld\n", bufferSize);
    void* dBuffer = NULL;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    // ? Perform the actual transpose operation on device
    cusparseCsr2cscEx2(
        handle,
        in->rows, in->cols, in->nnz,
        d_in_values, d_in_row_offsets, d_in_cols,
        d_out_values, d_out_row_offsets, d_out_cols,
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT, dBuffer
    );
    // ? Copy data from device to host for Output Matrix
    CHECK_CUDA(cudaMemcpy(out->row_offsets, d_out_row_offsets, out->rows * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out->col_indices, d_out_cols, out->nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out->values, d_out_values, out->nnz * sizeof(float), cudaMemcpyDeviceToHost));

    // ? Free memory on device
    CHECK_CUDA(cudaFree(d_in_row_offsets));
    CHECK_CUDA(cudaFree(d_in_cols));
    CHECK_CUDA(cudaFree(d_in_values));
    CHECK_CUDA(cudaFree(d_out_row_offsets));
    CHECK_CUDA(cudaFree(d_out_cols));
    CHECK_CUDA(cudaFree(d_out_values));
    CHECK_CUDA(cudaFree(dBuffer));

    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
    return 0;
}
