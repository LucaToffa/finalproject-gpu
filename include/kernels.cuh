#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include "coo.h"
#include "csr.h"
#include <cuda_runtime.h>

__global__ void dummy_kernel();
__global__ void cuCOOtCopy(coo_element *in, coo_element *out, size_t nnz);
__global__ void cuCOOt(coo_element *in, size_t nnz);
__global__ void csr_transpose(csr_matrix *csr);
__global__ void block_transpose(float *input, float *output, int N);
__global__ void conflict_transpose(float *input, float *output);
__global__ void basic_transpose(float *input, float *output, int N);
__global__ void countNNZPerColumn(const int* col_indices, int* col_counts, int nnz);
// Kernel to scatter values and row indices to transposed matrix
__global__ void scatterToTransposed(const float* values, const int* col_indices, const int* row_ptr,
                                    float* t_values, int* t_row_indices, int* t_col_ptr, int num_rows);
__global__ void prefix_scan(int *g_odata, int *g_idata, int n, int *last);
__global__ void order_by_column(const float* values, const int* col_indices, //col_offset
                                float* d_t_values, int* t_col_indices, int *d_col_counts,
                                int num_cols, int nnz,
                                int *d_t_col_indices, int *d_t_col_indices_ordered);
__global__ void csr_matrix_transpose_kernel(
    const int num_rows,       // Number of rows in original matrix
    const int num_cols,       // Number of columns in original matrix
    const int nnz,            // Number of non-zero elements

    // Input CSR matrix components
    int* col_indices, // Column indices array input
    float* values,    // Values array input

    // Output transposed CSR matrix components
    int* t_row_ptr,         // Transposed row pointer array
    int* t_col_indices,     // Transposed column indices array
    float* t_values,         // Transposed values array

    int* tex_off,
    int* idxtemp
);
#endif
