#include "include/commons.h"
#include "include/coo.h"
#include "include/csr.h"
#include "include/debug.h"
#include "include/kernels.cuh"
#include <cuda_runtime.h>
#include <cusparse.h>


int testing();
int cuda_transpose_example();
// cuSPARSE Transpose CSR
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

int main(int argc, char** argv) {
#ifdef DEBUG
    if (argc > 1) {
        printf("argc = %d:\n", argc);
        for (int i = 0; i < argc; i++) {
            printf("arg %d : %s\n", i+1, argv[i]);
        }
        printf("\n");
    }
#endif
    //try coo kernel
    //cuda_transpose_example();

    csr_matrix* csr = load_csr_matrix("matrices/tests/mockcsr.mtx");
    csr_matrix* csr_t = new csr_matrix(csr->rows, csr->cols, csr->nnz);
    //pretty_print_csr_matrix(csr);
    cuSparseCSRt(csr, csr_t);
    //pretty_print_csr_matrix(csr_t);
    if (is_transpose(csr, csr_t)) {
        printf("Transpose is correct\n");
    } else {
        printf("Transpose is incorrect\n");
    }
    delete csr;
    delete csr_t;

    return 0;
}

int cuda_transpose_example() {
    coo_matrix* coo = load_coo_matrix("matrices/circuit204.mtx");
    coo_element* el = coo->el;
    coo_matrix* d_coo;
    coo_element* d_el;
    cudaMallocManaged((void**)&d_coo, sizeof(coo_matrix));
    cudaMallocManaged((void**)&d_el, coo->nnz * sizeof(coo_element));
    cudaMemcpy(d_coo, coo, sizeof(coo_matrix), cudaMemcpyHostToDevice);
    cudaMemcpy(d_el, el, coo->nnz * sizeof(coo_element), cudaMemcpyHostToDevice);
    PRINTF("Copied memory\n");
    d_coo->el = d_el;
    printf("Before transpose\n");
    print_coo_less(d_coo);
    coo_transpose<<<coo->nnz,1>>>(d_coo);
    cudaMemcpy(d_coo, d_coo, sizeof(coo_matrix), cudaMemcpyDeviceToHost);
    printf("After transpose\n");
    print_coo_less(d_coo);

    cudaFree(d_coo);
    cudaFree(d_el);
    delete[] coo->el;
    delete coo;
    return 0;
}

int testing() {
    coo_matrix* coo = load_coo_matrix("matrices/tests/mockcoo.mtx");
    PRINTF("--------------------\n");
    print_coo_matrix(coo);
    PRINTF("--------------------\n");
    delete coo;
    coo = load_coo_matrix("matrices/circuit204.mtx");
    print_coo_metadata(coo);
    int full_size = 4;
    float *mat = new float [full_size*full_size];
    sparseInitMatrix(mat, full_size);
    coo_matrix* coo2 = mat_to_coo(mat, full_size);
    PRINTF("--------------------\n");
    print_coo_matrix(coo2);
    printMatrix(mat, full_size);
    PRINTF("--------------------\n");

    delete[] mat;
    delete coo;
    delete coo2;

    PRINTF("CSR tests\n");
    //csr_matrix* csr = load_csr_matrix("matrices/tests/mockcsr.mtx");
    csr_matrix* csr = load_csr_matrix();
    PRINTF("--------------------\n");
    print_csr_matrix(csr);
    PRINTF("--------------------\n");
    return 0;
}
