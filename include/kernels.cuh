#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include "coo.h"
#include "csr.h"
#include <cuda_runtime.h>

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
__global__ void prefix_scan(int *g_odata, int *g_idata, int n);
#endif
