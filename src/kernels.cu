#include "../include/coo.h"
#include "../include/csr.h"
#include "../include/kernels.cuh"
#include "../include/commons.h"
#include <cuda_runtime.h>


/**
 * @brief Kernel to Transpose a COO Matrix out of place
    * @param[in] in - COO Matrix to be transposed
    * @param[out] out - Transposed COO Matrix
 */
__global__ void cuCOOtCopy(coo_element *in, coo_element *out, size_t nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < nnz) {
        out[i].row = in[i].col;
        out[i].col = in[i].row;
        out[i].val = in[i].val;
    }
}
/**
 * @brief Kernel to Transpose a COO Matrix in-place
    * @param[in] in - COO Matrix to be transposed
 */
__global__ void cuCOOt(coo_element *in, size_t nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < nnz) {
        size_t tmp = in[i].row;
        in[i].row = in[i].col;
        in[i].col = tmp;
    }
}

/**
 * @brief Kernel to Transpose a CSR Matrix out of place
    * @param[in] in
    * @param[out] out
*/
__global__ void csr_transpose(const csr_matrix *in, csr_matrix *out) {
    //TODO: implement, now is random garbage
    //int i = blockIdx.x * blockDim.x + threadIdx.x;
}
// Kernel to count the number of non-zero entries per column
__global__ void countNNZPerColumn(const int* col_indices, int* col_counts, int nnz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nnz) {
        atomicAdd(&col_counts[col_indices[tid]], 1);
    }
}

// Kernel to scatter values and row indices to transposed matrix
__global__ void scatterToTransposed(const int* values, const int* col_indices, const int* row_ptr,
                                    int* t_values, int* t_row_indices, int* t_col_ptr, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
            int col = col_indices[j];
            int dest = atomicAdd(&t_col_ptr[col], 1);
            t_values[dest] = values[j];
            t_row_indices[dest] = row;
        }
    }
}


//old version of block transpose algorithm to check against the new ones
__global__ void block_transpose(float *input, float *output, int N){
    __shared__ float tile[TILE_SIZE][TILE_SIZE+1];
    
    //input to shared offsets
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int w = gridDim.x * TILE_SIZE;

    for(int i = 0; i < TILE_SIZE; i += BLOCK_ROWS){
        tile[threadIdx.y+i][threadIdx.x] = input[(y+i) * w + x];
    }

    __syncthreads();

    //shared to output offsets
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    for(int j = 0; j < TILE_SIZE; j += BLOCK_ROWS){
        output[(y+j) * w + x] = tile[threadIdx.x][threadIdx.y+j];
    }
}

//without the +1 the memory access conflicts cannot be avoided
__global__ void conflict_transpose(float *input, float *output){
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    //input to shared offsets
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int w = gridDim.x * TILE_SIZE;

    for(int i = 0; i < TILE_SIZE; i += BLOCK_ROWS){
        tile[threadIdx.y+i][threadIdx.x] = input[(y+i) * w + x];
    }

    __syncthreads();

    //shared to output offsets
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    for(int j = 0; j < TILE_SIZE; j += BLOCK_ROWS){
        output[(y+j) * w + x] = tile[threadIdx.x][threadIdx.y+j];
    }

}

#define B_TILE TILE_SIZE
#define B_ROWS BLOCK_ROWS
// implementation of basic transpose in gpu
// to avoid ifs in the kernel, check the matrix size and derive block/threads size
__global__ void basic_transpose(float *input, float *output, int N){
    //matrix transpose that works for any size
    int x = blockIdx.x * B_TILE + threadIdx.x;
    int y = blockIdx.y * B_TILE + threadIdx.y;

    int index_in = x + N * y;
    int index_out = y + N * x;

    for (int i = 0; i < B_TILE; i += B_ROWS){
        output[index_out + i] = input[index_in + i * N];
    }
}
