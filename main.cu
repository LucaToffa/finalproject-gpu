#include "include/commons.h"
#include "include/coo.h"
#include "include/csr.h"
#include "include/debug.h"
#include "include/kernels.cuh"
#include "include/cuSPARSEkt.cuh"
#include <cuda_runtime.h>


int cuSparse_transpose_example();
int cuda_transpose_example();


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
    cuda_transpose_example();

    // cuSparse_transpose_example();

    return 0;
}

int cuSparse_transpose_example() {
    csr_matrix* csr = load_csr_matrix("matrices/circuit204.mtx");
    csr_matrix* csr_t = new_csr_matrix(csr->rows, csr->cols, csr->nnz, csr->row_offsets, csr->col_indices, csr->values);
    print_csr_matrix(csr);
    pretty_print_csr_matrix(csr);
    cuSparseCSRt(csr, csr_t);
    pretty_print_csr_matrix(csr_t);
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
    CHECK_CUDA(cudaMallocManaged((void**)&d_coo, sizeof(coo_matrix)));
    CHECK_CUDA(cudaMallocManaged((void**)&d_el, coo->nnz * sizeof(coo_element)));
    CHECK_CUDA(cudaMemcpy(d_coo, coo, sizeof(coo_matrix), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_el, el, coo->nnz * sizeof(coo_element), cudaMemcpyHostToDevice));
    PRINTF("Copied memory\n");
    d_coo->el = d_el;
    printf("Before transpose\n");
    print_coo_less(d_coo);
    cuCOOt<<<coo->nnz,1>>>(d_coo->el, d_coo->nnz);
    CHECK_CUDA(cudaMemcpy(d_coo, d_coo, sizeof(coo_matrix), cudaMemcpyDeviceToHost));
    printf("After transpose\n");
    print_coo_less(d_coo);
    if (is_transpose(coo, d_coo)) {
        printf("Transpose is correct\n");
    } else {
        printf("Transpose is incorrect\n");
    }

    CHECK_CUDA(cudaFree(d_coo));
    CHECK_CUDA(cudaFree(d_el));
    delete[] coo->el;
    delete coo;
    return 0;
}
