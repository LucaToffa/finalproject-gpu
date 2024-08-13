#include "include/commons.h"
#include "include/coo.h"
#include "include/csr.h"
#include "include/debug.h"
#include "include/kernels.cuh"
#include "include/cuSPARSEkt.cuh"
#include <cuda_runtime.h>
#include "include/benchmarks.cuh"
#include <fstream>

#define DEBUG

const int matrix_number = 11;
const char* matrix_list[] = { //for some reason in the original order csr load broke ???
    "matrices/494_bus.mtx",
    "matrices/bcsstm01.mtx",
    "matrices/circuit204.mtx",
    "matrices/collins_15NN.mtx",
    "matrices/lowThrust_1.mtx",
    "matrices/orbitRaising_3.mtx",
    "matrices/spaceStation_5.mtx",
    "matrices/tomography.mtx",
    "matrices/umistfacesnorm_10NN.mtx",
    "matrices/Vehicle_10NN.mtx",
    "matrices/west0989.mtx"
};

int cuSparse_transpose_example();
int cuda_transpose_example();

int complete_benchmark(); //could remain here or return to benchmark.cu
//leaving everything in main is ugly as hell

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
    complete_benchmark();
    return 0;
}

int cuSparse_transpose_example() {
    csr_matrix* csr = load_csr_matrix("matrices/circuit204.mtx");
    csr_matrix* csr_t = new_csr_matrix(csr->rows, csr->cols, csr->nnz, csr->row_offsets, csr->col_indices, csr->values);
    print_csr_matrix(csr);
    pretty_print_csr_matrix(csr, std::cout);
    cuSparseCSRt(csr, csr_t);
    pretty_print_csr_matrix(csr_t, std::cout);
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

int complete_benchmark() {
    unsigned int size;
    PRINTF("main.cu) Now Entering Loop...\n");
    std::ofstream output;
    output.open("logs/results.log");
    output << "#N_mat, algorithm, OpTime, Op-GB/s, KTime, K-GB/s#\n";
    output.close();
    for (int i = 0; i < matrix_number; i++) {
        PRINTF("main.cu) Matrix: %s\n", matrix_list[i]);
        //load as coo
        coo_matrix* coo = load_coo_matrix(matrix_list[i]);
        //load as csr
        csr_matrix* csr = load_csr_matrix(matrix_list[i]);
        csr_matrix* csr_t = new_csr_matrix(csr->cols, csr->rows, csr->nnz);
        //load as uncompressed
        float* uncompressed = coo_to_mat(coo);
        assert(coo->rows == coo->cols);
        size = coo->rows;
        /*
        each kernel + any other necessary operations / checks
        save to output file: mat | algo | time | BW | error
        inside the called function
        */
        PRINTF("main.cu) calling cuCOOt kernel\n");
        if(coo_transposition(coo)) {
            printf("error in coo transpose, matrix #%d\n", i);
        }
        PRINTF("main.cu) calling cuCSRt kernel\n");
        if(csr_transposition(csr, csr_t)) {
            printf("error in csr transpose, matrix #%d\n", i);
        }
        PRINTF("main.cu) calling block kernel\n");
        if(block_trasposition(uncompressed, size)) {
            printf("error in block transpose, matrix #%d\n", i);
        }
        PRINTF("main.cu) calling conflict kernel\n");
        if(conflict_transposition(uncompressed, size)) {
            printf("error in conflict transpose, matrix #%d\n", i);
        }
        PRINTF("main.cu) calling cuSparseCSRt kernel\n");
        if(cuSparseCSRt(csr, csr_t)) {
            printf("error in cuSparse transpose, matrix #%d\n", i);
        }
        printf("main.cu) matrix #%d done\n", i);

        //delete everything
        delete[] coo->el;
        delete coo;
        delete[] csr->row_offsets;
        delete[] csr->col_indices;
        delete[] csr->values;
        delete csr;
        delete[] csr_t->row_offsets;
        delete[] csr_t->col_indices;
        delete[] csr_t->values;
        delete csr_t;
        delete[] uncompressed;
    }
    return 0;
}
