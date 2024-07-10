#pragma once

#include "cuda_runtime.h"
#include "coo.h"
#include "csr.h"

__global__ void coo_transpose(coo_matrix *coo);

//old version of block transpose algorithm to check against the new ones
__global__ void block_transpose(float *input, float *output);

//without the +1 the memory access conflicts cannot be avoided
__global__ void conflict_transpose(float *input, float *output);

#define B_TILE TILE_SIZE
#define B_ROWS BLOCK_ROWS
// implementation of basic transpose in gpu
// to avoid ifs in the kernel, check the matrix size and derive block/threads size
__global__ void basic_transpose(float *input, float *output, int N);