#include "include/commons.h"
#include "include/coo.h"
#include "include/csr.h"
#include "include/debug.h"
#include "include/kernels.cuh"
#include "include/cuSPARSEkt.cuh"
#include <cuda_runtime.h>

#include "include/benchmarks.cuh"
const int matrix_number = 11;
const char* matrix_list[] = { //for some reason in the original order csr load broke ???
    "matrices/spaceStation_5.mtx",
    "matrices/bcsstm01.mtx", "matrices/494_bus.mtx",
    "matrices/collins_15NN.mtx",
    "matrices/lowThrust_1.mtx",
    "matrices/umistfacesnorm_10NN.mtx",
    "matrices/orbitRaising_3.mtx", "matrices/Vehicle_10NN.mtx",
    "matrices/circuit204.mtx",
    "matrices/west0989.mtx",
    "matrices/tomography.mtx"
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
    //try coo kernel
    //cuda_transpose_example();

    // cuSparse_transpose_example();
    //test csr
    // for(int i = 0; i < matrix_mumber; i++){
    //     printf("%s: ", matrix_list[i]);
    //     csr_matrix* csr = load_csr_matrix(matrix_list[i]);
    //     print_csr_matrix(csr);
    //     delete[] csr->cols
    //     dele
    //     delete csr;
    // }


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

int complete_benchmark() {

    unsigned int size;
    PRINTF("enter loop\n");
    for (int i = 0; i < matrix_number; i++) {
        PRINTF("Matrix: %s\n", matrix_list[i]);
        //load as coo
        coo_matrix* coo = load_coo_matrix(matrix_list[i]);
        //load as csr
        csr_matrix* csr = load_csr_matrix(matrix_list[i]);
        csr_matrix* csr_t = new_csr_matrix(csr->rows, csr->cols, csr->nnz, csr->row_offsets, csr->col_indices, csr->values);
        //load as uncompressed
        float* uncompressed = coo_to_mat(coo);
        size = coo->rows;
        /*
        each kernel + any other necessary operations / checks
        save to output file: mat | algo | time | BW | error
        inside the called function
        */
        PRINTF("calling cuCOOt kernel\n");
        if(coo_transposition(coo)){//coo transpose
            printf("error in coo transpose, matrix #%d\n", i);
        }
        PRINTF("calling cuCSRt kernel\n");
        if(csr_transposition(csr, csr_t)){//csr transpose
            printf("error in csr transpose, matrix #%d\n", i);
        }
        PRINTF("calling block kernel\n");
        if(block_trasposition(uncompressed, size)){//block transpose
            printf("error in block transpose, matrix #%d\n", i);
        }
        PRINTF("calling conflict kernel\n");
        if(conflict_transposition(uncompressed, size)){//conflict transpose
            printf("error in conflict transpose, matrix #%d\n", i);
        }
        PRINTF("calling cuSparseCSRt kernel\n");
        if(cuSparseCSRt(csr, csr_t)){//cuSparse transpose
            printf("error in cuSparse transpose, matrix #%d\n", i);
        }
        printf("matrix #%d done\n", i);

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


