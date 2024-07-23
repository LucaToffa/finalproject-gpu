#ifndef __CSR_H__
#define __CSR_H__

#include <iostream>
#include <memory>

/**
    * @brief Struct to represent a Compressed Sparse Row matrix
    * @param {unsigned long} rows Number of rows in the matrix
    * @param {unsigned long} cols Number of columns in the matrix
    * @param {unsigned long} nnz Number of non-zero elements in the matrix
    * @param {unsigned long[]} row_offsets Array of size rows + 1 containing the offsets of the rows
    * @param {unsigned long[]} col_indices Array of size nnz containing the column indices of the non-zero elements
    * @param {float[]} values Array of size nnz containing the values of the non-zero elements
 */
struct csr_matrix {
    size_t rows;
    size_t cols;
    size_t nnz;
    size_t* row_offsets;
    size_t* col_indices;
    float* values;
};

csr_matrix* new_csr_matrix(size_t rows, size_t cols, size_t nnz, size_t *row_offsets, size_t *col_indices, float *values);
csr_matrix* load_csr_matrix(const char *filename);
csr_matrix* load_csr_matrix(void);
bool is_transpose(csr_matrix *csr, csr_matrix *csr_t);
int print_csr_matrix(csr_matrix *csr);
int pretty_print_csr_matrix(csr_matrix *csr);

#endif
