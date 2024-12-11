#ifndef __COMMONS_H__
#define __COMMONS_H__

#include <cstring>
#include "csr.h"
#include "coo.h"

//just test one case
#ifndef TILE_SIZE
    #define TILE_SIZE 32 
#endif
#ifndef BLOCK_ROWS
    #define BLOCK_ROWS 8
#endif
#define DEFAULT_SIZE 32
#ifndef DEBUG
    #define TRANSPOSITIONS 100
#else
    #define TRANSPOSITIONS 1
#endif

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

//in some places CHECK_CUDA does not work, so we stil need to use this 
#define cudaCheckError() {                                                        \
    cudaError_t e=cudaGetLastError();                                             \
    if(e!=cudaSuccess) {                                                          \
        printf("CUDA Error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                                       \
    }                                                                             \
}

/**
    * @brief Initialize a dense square matrix with incrementing values given a size
    * @param {float *} mat - Pointer to the matrix
    * @param {int} size - Size of the matrix
    * @returns {int} - 0 if successful
 */
int initMatrix(float* mat, int size);
/**
    * @brief Initialize a dense square matrix with incrementing values given a size, with some elements set to 0
    * @param {float *} mat - Pointer to the matrix
    * @param {int} size - Size of the matrix
    * @returns {int} - 0 if successful
 */
int sparseInitMatrix(float* mat, int size);
/**
    * @brief Print a dense square matrix given a size, padded to console as a matrix
    * @param {float *} mat - Pointer to the matrix
    * @param {int} size - Size of the matrix
    * @returns {int} - 0 if successful
 */
int printMatrix(const float* mat, int size);
/**
    * @brief Test if a dense square matrix is the transpose of another dense square matrix and prints to console any discrepancies
    * @param {const float *} mat - Pointer to the first matrix
    * @param {const float *} mat_t - Pointer to the second matrix
    * @param {int} size - Size of the matrix
    * @returns {int} - 0 if the matrices are one the transpose of the other, -1 otherwise
 */
int testTranspose(const float* mat, const float* mat_t, int size);
/**
    * !Not Yet Implemented!
    * @brief Converts a COO matrix to a CSR matrix
    * @param {coo_matrix *} coo - Pointer to the COO matrix
    * @returns {csr_matrix} - CSR matrix
 */
csr_matrix coo_to_csr(const coo_matrix *coo);
/**
    * !Not Yet Implemented!
    * @brief Converts a CSR matrix to a COO matrix
    * @param {csr_matrix *} csr - Pointer to the CSR matrix
    * @returns {coo_matrix} - COO matrix
 */
coo_matrix csr_to_coo(const csr_matrix *csr);
/**
    * @brief Converts a CSR matrix to a dense square matrix
    * @param {csr_matrix *} csr - Pointer to the CSR matrix
    * @returns {float *} - Pointer to the dense square matrix
 */
float* csr_to_mat(const csr_matrix *csr);
/**
    * @brief Converts a COO matrix to a dense square matrix
    * @param {coo_matrix *} coo - Pointer to the COO matrix
    * @returns {float *} - Pointer to the dense square matrix
 */
float* coo_to_mat(const coo_matrix *coo);
/**
    * @brief Converts a COO matrix to a dense square matrix, padded to the next power of 2
    * @param {coo_matrix *} coo - Pointer to the COO matrix
    * @returns {float *} - Pointer to the padded dense square matrix
 */
float* coo_to_mat_padded(const coo_matrix *coo);
/**
    * @brief Given an integer, returns the next power of 2 greater than or equal to the integer
    * @param {int} n - The given integer
    * @returns {int} - The next power of 2 greater than or equal to the integer
 */
int next_power_of_2(int n);

csr_matrix* csc_to_csr(int num_rows, int num_cols, int nnz, float* csc_values, int* csc_row_indices, int* csc_col_pointers);
#endif
