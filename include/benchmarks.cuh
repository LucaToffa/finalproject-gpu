#ifndef __BENCHMARKS_CUH__
#define __BENCHMARKS_CUH__

#include "coo.h"
#include "csr.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int coo_transposition(coo_matrix* coo);
int csr_transposition(csr_matrix* csr, csr_matrix* csr_t);
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