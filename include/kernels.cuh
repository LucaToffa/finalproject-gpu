#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include "coo.h"
#include "csr.h"
#include <cuda_runtime.h>

__global__ void cuCOOtCopy(coo_element *in, coo_element *out, size_t nnz);
__global__ void cuCOOt(coo_element *in, size_t nnz);
__global__ void csr_transpose(csr_matrix *csr);
__global__ void block_transpose(float *input, float *output);
__global__ void conflict_transpose(float *input, float *output);
__global__ void basic_transpose(float *input, float *output, int N);

#endif
