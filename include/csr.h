#pragma once
#include <iostream>

// The pointers to the row offsets array of length number of rows + 1
// that represents the starting position of each row in the columns and
// values arrays.

// The pointers to the column indices array of length nnz that contains the
// column indices of the corresponding elements in the values array.

// The pointers to the values array of length nnz that holds all nonzero values of
// the matrix in row-major ordering.

typedef struct csr_matrix{
    int rows;
    int cols;
    int nnz;
    int *row_offsets;
    int *col_indices;
    float *values;

    // Constructor
    csr_matrix(int rows, int cols, int nnz);
    void csr_fill(int *row_offsets, int *col_indices, float *values);
    // Destructor
    ~csr_matrix();
   
}csr_matrix;


void csr_hello();
csr_matrix* load_csr_matrix(const char *filename);
csr_matrix* load_csr_matrix(void);
int print_csr_matrix(csr_matrix *csr);
