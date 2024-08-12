#ifndef __BENCHMARKS_CUH__
#define __BENCHMARKS_CUH__

#include "coo.h"
#include "csr.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


/**
    * @brief Transpose a COO matrix using a CUDA kernel
    * @param {coo_matrix *} coo - COO matrix to be transposed
    * @return {int} 0 if successful, 1 otherwise
 */
int coo_transposition(coo_matrix* coo);
/**
    * @brief Transpose a CSR matrix using a CUDA kernel
    * @param {csr_matrix *} csr - CSR matrix to be transposed
    * #param {csr_matrix *} csr_t - Pointer to memory where the transposed matrix will be stored
    * @return {int} 0 if successful, 1 otherwise
 */
int csr_transposition(csr_matrix* csr, csr_matrix* csr_t);
/**
    * @brief Transpose a dense matrix using a a block transpose CUDA kernel
    * @param {float *} mat - Block matrix to be transposed
    * @param {unsigned int} N - Size of the block matrix
    * @return {int} 0 if successful, 1 otherwise
 */
int block_trasposition(float* mat, unsigned int N);
int conflict_transposition(float* mat, unsigned int N);
int transposeCSRToCSC(const thrust::host_vector<int>& h_values, const thrust::host_vector<int>& h_col_indices,
                       const thrust::host_vector<int>& h_row_ptr, int num_rows, int num_cols,
                       thrust::device_vector<int>& d_t_values, thrust::device_vector<int>& d_t_row_indices,
                       thrust::device_vector<int>& d_t_col_ptr);
//i'm leaving this version because i can't test any changes
int pretty_print_matrix(const thrust::host_vector<int>& values, const thrust::host_vector<int>& row_indices,
                        const thrust::host_vector<int>& col_ptr, int num_rows, int num_cols);

#endif
