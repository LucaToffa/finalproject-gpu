#ifndef __BENCHMARKS_CUH__
#define __BENCHMARKS_CUH__

#include "coo.h"
#include "csr.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


/**
    * @brief Transpose a COO matrix using a CUDA kernel
    * @param {coo_matrix *} coo - COO matrix to be transposed
    * @param {int} matrix_size - Size of the matrix
    * @return {int} 0 if successful, 1 otherwise
 */
int coo_transposition(coo_matrix* coo, int matrix_size);
/**
    * @brief Transpose a CSR matrix using a CUDA kernel
    * @param {csr_matrix *} csr - CSR matrix to be transposed
    * @param {csr_matrix *} csr_t - Pointer to memory where the transposed matrix will be stored
    * @param {int} matrix_size - Size of the matrix
    * @return {int} 0 if successful, 1 otherwise
 */
int csr_transposition(csr_matrix* csr, csr_matrix* csr_t, int matrix_size);
/**
    * @brief Transpose a dense matrix using a a block transpose CUDA kernel
    * @param {float *} mat - Block matrix to be transposed
    * @param {unsigned int} N - Size of the block matrix
    * @param {int} matrix_size - Size of the matrix
    * @return {int} 0 if successful, 1 otherwise
 */
int block_trasposition(float* mat, unsigned int N, int matrix_size);
/**
    * @brief Transpose a dense matrix using a conflict-free transpose CUDA kernel
    * @param {float *} mat - Block matrix to be transposed
    * @param {unsigned int} N - Size of the block matrix
    * @param {int} matrix_size - Size of the matrix
    * @return {int} 0 if successful, 1 otherwise
 */
int conflict_transposition(float* mat, unsigned int N, int matrix_size);

#endif
