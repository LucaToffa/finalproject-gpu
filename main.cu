#include "include/commons.h"
#include "include/coo.h"
#include "include/csr.h"
#include "include/debug.h"
#include "include/cuSPARSEkt.cuh"
#include <cuda_runtime.h>
#include "include/benchmarks.cuh"
#include <fstream>

const int MATRICES_LEN = 13;
const char* MATRICES[] = {
    "matrices/08blocks.mtx",
    "matrices/GD01_Acap.mtx",
    "matrices/494_bus.mtx", 
    "matrices/circuit204.mtx",  
    "matrices/collins_15NN.mtx",
    "matrices/lowThrust_1.mtx", 
    "matrices/orbitRaising_3.mtx",
    "matrices/spaceStation_5.mtx",
    "matrices/umistfacesnorm_10NN.mtx",
    "matrices/west0989.mtx",
    "matrices/bcsstm01.mtx",
    "matrices/tomography.mtx",
    "matrices/Vehicle_10NN.mtx"
};

int complete_benchmark();

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
    //clear all error logs
    std::ofstream output;
    output.open("logs/transpose_err.log", std::ios::out | std::ios_base::trunc);
    output.close();
    std::ofstream csr_log_output;
    csr_log_output.open("logs/csr.log", std::ios::out | std::ios_base::trunc);
    csr_log_output.close();
    complete_benchmark();

    return 0;
}

int complete_benchmark() {
    unsigned int dense_mat_size;
    std::ofstream output;
    output.open("logs/results.log");
    output << "#algorithm, MatSize, OpTime, Op-GB/s, KTime, K-GB/s#\n";
    output.close();
    for (int i = 0; i < MATRICES_LEN; i++) {
        // ? Load COO and CSR Matrix Representations
        coo_matrix* coo = load_coo_matrix(MATRICES[i]); // ? Load COO Matrix
        csc_matrix* csc = load_csc_matrix(MATRICES[i]); // ? Load CSC Matrix
        csr_matrix* csr = csc_to_csr(csc->rows, csc->cols, csc->nnz, csc->values, csc->row_indices, csc->col_offsets); // ? Convert CSC to CSR
        delete_csc(csc); // ? Delete CSC
        printf("main.cu) Matrix[%d] -> (%d x %d): %s\n", i, coo->rows, coo->cols, MATRICES[i]);
        std::ofstream csr_log_output;
        csr_log_output.open("logs/csr.log", std::ios::out | std::ios_base::app);
        csr_log_output << "Matrix: " << MATRICES[i] << "\n";
        csr_log_output.close();

        csr_matrix* csr_t = new_csr_matrix(csr->cols, csr->rows, csr->nnz); // ? Create CSR Transpose
        float* uncompressed = coo_to_mat_padded(coo); // ? Convert COO to Uncompressed Matrix
        assert(coo->rows == coo->cols); // ? Assert Square Matrix
        dense_mat_size = next_power_of_2(std::max(coo->rows, coo->cols)); // ? Get Padded Size of Uncompressed Matrix
        // dense_mat_size = coo->rows;

        // ? Clear the results.log file
        output.open("logs/results.log", std::ios::out | std::ios_base::app);
        output.close();
        int matrix_size = coo->rows;
        PRINTF("main.cu) calling cuSparseCSRt kernel\n");
        if(cuSparseCSRt(csr, csr_t, matrix_size)) {
            printf("error in cuSparse transpose, matrix #%d\n", i);
        }
        PRINTF("main.cu) calling cuCOOt kernel\n");
        if(coo_transposition(coo, matrix_size)) {
            printf("error in coo transpose, matrix #%d\n", i); 
        }
        PRINTF("main.cu) calling cuCSRt (gpu) kernel\n");
        if(csr_transposition_3(csr, csr_t, matrix_size)) {
            printf("error in csr transpose, matrix #%d\n", i);
        }
        PRINTF("main.cu) calling block kernel\n");
        if(block_trasposition(uncompressed, dense_mat_size, matrix_size)) {
            printf("error in block transpose, matrix #%d\n", i);
        }
        PRINTF("main.cu) calling conflict kernel\n");
        if(conflict_transposition(uncompressed, dense_mat_size, matrix_size)) {
            printf("error in conflict transpose, matrix #%d\n", i);
        }
        PRINTF("main.cu) matrix #%d done\n", i);

        // Free All Memory Allocated for the next iteration
        delete_coo(coo);
        delete_csr(csr);
        delete_csr(csr_t);
        delete[] uncompressed;
    }
    return 0;
}
