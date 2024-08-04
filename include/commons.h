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
#define TRANSPOSITIONS 100

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

int initMatrix(float* mat, int size);
int sparseInitMatrix(float* mat, int size);
int printMatrix(float* mat, int size);
int testTranspose(float* mat, float* mat_t, int size);
csr_matrix coo_to_csr(coo_matrix *coo);
coo_matrix csr_to_coo(csr_matrix *csr);
float* csr_to_mat(csr_matrix *csr);
float* coo_to_mat(coo_matrix *coo);

#endif
