#include "../include/coo.h"
#include "../include/kernels.cuh"
#include "../include/commons.h"
#include <cuda_runtime.h>
#include <cassert>

__global__ void dummy_kernel() {
    return;
}
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

// Kernel to count the number of non-zero entries per column
__global__ void countNNZPerColumn(const int* col_indices, int* col_counts, int nnz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nnz) {
        atomicAdd(&col_counts[col_indices[tid]], 1);
    }
}

// // Kernel to scatter values and row indices to transposed matrix
// __global__ void scatterToTransposed(const float* values, const int* col_indices, const int* row_ptr,
//                                     float* t_values, int* t_row_indices, int* t_col_ptr, int num_rows) {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row < num_rows) {
//         for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
//             int col = col_indices[j];
//             int dest = atomicAdd(&t_col_ptr[col], 1);
//             t_values[dest] = values[j];
//             t_row_indices[dest] = row;
//         }
//     }
// }

//1 thread per col, .append if col == thdx
//join the threads in order
__global__ void order_by_column(const float* values, const int* col_indices, //col_offset
                                float* d_t_values, int* t_col_indices, int *d_col_counts,
                                int num_cols, int nnz,
                                int *d_t_col_indices, int *d_t_col_indices_ordered){
    int col = threadIdx.x + blockIdx.x * blockDim.x; //current working column
    //how many values are in this column?
    int start_offset = t_col_indices[col]; //col_ptr 0 1 2 4 7 
    //int num_values = d_col_counts[col]; //1 1 2 3
    int pos = 0;
    if (col < num_cols) {
        for(int i = 0; i < nnz; i++){
            if(col == col_indices[i]){
                //append to the end of the array val
                d_t_values[start_offset + pos] = values[i]; 
                d_t_col_indices_ordered[start_offset + pos] = d_t_col_indices[i];
                pos++;
            }
        }
    }
}
/**
    01 __ global__ void transp (int *AT.idx, ……) {
    02   tid = blockIdx.x * blockDim.x + threadIdx.x ;
    03   while (tid < NNZ) {
    04        temp = csr_t->row_ptr[csr->col_indeces[tid]] + tex_off[tid];
    05        csr_t->col_indeces[temp] = idxtemp[tid];
    06        csr_t->values[temp] = csr->values[tid];
    07        tid + = THREADS;
    08   }
    09 }
 */
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
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < nnz) {
        int temp = t_row_ptr[col_indices[tid]] + tex_off[tid];
        t_col_indices[temp] = idxtemp[tid];
        t_values[temp] = values[tid];
        tid += blockDim.x * gridDim.x;
    }
    __syncthreads();
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

__global__ void prefix_scan(int *g_odata, int *g_idata, int n, int *last)
{   
    extern __shared__ int temp[]; // allocated on invocation
    int thid = threadIdx.x;// + blockIdx.x * blockDim.x;
    // if(thid >= 2*n) return;
    //printf("prefix_scan) n: %d, thr: %d\n", n, thid);
    int offset = 1;
    temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
    temp[2 * thid + 1] = g_idata[2 * thid + 1];
    //printf("prefix_scan) temp[2 * thid]: %d, temp[2 * thid + 1]: %d\n", temp[2 * thid], temp[2 * thid + 1]);
    for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        //printf("prefix_scan) d: %d in thr: %d\n", d, thid);
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (thid == 0)
    {
        //printf("prefix_scan) d: 0 in thr: %d\n", thid);
        last[0] = temp[n - 1]; // write the last element of the scan to the last element of the block
        temp[n - 1] = 0;
    }                              // clear the last element
    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
    g_odata[2 * thid + 1] = temp[2 * thid + 1];
}

