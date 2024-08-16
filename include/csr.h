#ifndef __CSR_H__
#define __CSR_H__

#include <iostream>
#include <memory>

/**
    * @brief Struct to represent a Compressed Sparse Row matrix
    * @param {int} rows Number of rows in the matrix
    * @param {int} cols Number of columns in the matrix
    * @param {int} nnz Number of non-zero elements in the matrix
    * @param {int[]} row_offsets Array of size rows + 1 containing the offsets of the rows
    * @param {int[]} col_indices Array of size nnz containing the column indices of the non-zero elements
    * @param {float[]} values Array of size nnz containing the values of the non-zero elements
 */
struct csr_matrix {
    int rows;
    int cols;
    int nnz;
    int* row_offsets; // Lenght: csr->rows + 1
    int* col_indices; // Lenght: csr->nnz
    float* values;       // Lenght: csr->nnz
};

/**
    * @brief Builds a new CSR matrix from the given data
    * @param {int} rows Number of rows in the matrix
    * @param {int} cols Number of columns in the matrix
    * @param {int} nnz Number of non-zero elements in the matrix
    * @param {int[]} row_offsets Array of size rows + 1 containing the offsets of the rows
    * @param {int[]} col_indices Array of size nnz containing the column indices of the non-zero elements
    * @param {float[]} values Array of size nnz containing the values of the non-zero elements
    * @returns {csr_matrix *} Pointer to the CSR matrix
 */
csr_matrix* new_csr_matrix(int rows, int cols, int nnz, int *row_offsets, int *col_indices, float *values);
/**
    * @brief Builds a new CSR matrix with empty arrays
    * @param {unsigned long} rows Number of rows in the matrix
    * @param {unsigned long} cols Number of columns in the matrix
    * @param {unsigned long} nnz Number of non-zero elements in the matrix
    * @returns {csr_matrix *} Pointer to the CSR matrix
 */
csr_matrix* new_csr_matrix(int rows, int cols, int nnz);
/**
    * @brief Load a CSR matrix from a file
    * @param {const char *} filename Path to the file
    * @returns {csr_matrix *} Pointer to the CSR matrix
 */
csr_matrix* load_csr_matrix(const char *filename);
/**
    * @brief Load a CSR matrix from a hardcoded matrix
    * @returns {csr_matrix *} Pointer to the CSR matrix
 */
csr_matrix* load_csr_matrix(void);
/**
    * @brief Check if a CSR matrix is the transpose of another CSR matrix
    * @param {csr_matrix *} csr Pointer to the first CSR matrix
    * @param {csr_matrix *} csr_t Pointer to the second CSR matrix
    * @returns {bool} True if the matrices are one the transpose of the other, False otherwise
 */
bool is_transpose(const csr_matrix * const csr, const csr_matrix * const csr_t);
/**
    * @brief Print the metadata and arrays composing a CSR matrix
    * @param {csr_matrix *} csr Pointer to the CSR matrix
    * @returns {int} 0 if successful
 */
int print_csr_matrix(const csr_matrix *csr);
/**
    * @brief Print a CSR matrix padded in console as a matrix, with the metadata at the start
    * @param {csr_matrix *} csr Pointer to the CSR matrix
    * @param {std::ostream &} out Output stream to print the matrix
    * @returns {int} 0 if successful
 */
int pretty_print_csr_matrix(const csr_matrix *csr, std::ostream &out);

#endif
