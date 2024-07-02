#include  <iostream>
#include <cuda_runtime.h>
#include "commons.h"

#if DEBUG
    #define PRINTF(...) printf(__VA_ARGS__)
#else
    #define PRINTF(...)
#endif
#ifndef TILE_SIZE 
    #define TILE_SIZE 32 
#endif
#ifndef BLOCK_ROWS
    #define BLOCK_ROWS 8 //works up to 16
#endif

__global__ void coo_transpose(coo_matrix *coo){
    //TODO: implement, now is random garbage
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < coo->nnz){
        int tmp = coo->el[i].row;
        coo->el[i].row = coo->el[i].col;
        coo->el[i].col = tmp;
    }
}
__global__ void csr_transpose(csr_matrix *csr){
    //TODO: implement, now is random garbage
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < csr->nnz){
        int tmp = csr->col_indices[i];
        csr->col_indices[i] = csr->row_offsets[i];
        csr->row_offsets[i] = tmp;
    
    }
}

//old version of block transpose algorithm to check against the new ones
__global__ void block_transpose(float *input, float *output){
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

int main(int argc, char** argv){   
    coo_matrix* coo = load_coo_matrix("matrices/tests/mockcoo.mtx");
    PRINTF("--------------------\n");
    print_coo_matrix(coo);
    PRINTF("--------------------\n");
    delete coo;
    coo = load_coo_matrix("matrices/circuit204/circuit204.mtx");
    print_coo_matrix(coo);
    scanf("%d", &argc);
    int full_size = 4;
    float *mat = new float [full_size*full_size];
    sparseInitMatrix(mat, full_size);
    coo_matrix* coo2 = mat_to_coo(mat, full_size);
    PRINTF("--------------------\n");
    print_coo_matrix(coo2);
    printMatrix(mat, full_size);
    PRINTF("--------------------\n");

    delete[] mat;
    delete coo;
    delete coo2;

    PRINTF("CSR tests\n");
    //csr_matrix* csr = load_csr_matrix("matrices/tests/mockcsr.mtx");
    csr_matrix* csr = load_csr_matrix();
    PRINTF("--------------------\n");
    print_csr_matrix(csr);
    PRINTF("--------------------\n");

    return 0;
}

