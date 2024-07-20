#include "include/commons.h"
#include "include/coo.h"
#include "include/debug.h"
#include "include/kernels.cuh"

int testing();

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
