#ifndef __COO_H__
#define __COO_H__

#include <iostream>
#include <cstring>

/**
    * @brief Struct to represent a Coordinate list element
    * @param {int} row - Row index
    * @param {int} col - Column index
    * @param {float} val - Value of the element
 */
struct coo_element {
    int row;
    int col;
    float val;
};

/**
    * @brief Struct to represent a Coordinate list matrix
    * @param {int} rows - Number of rows
    * @param {int} cols - Number of columns
    * @param {int} nnz - Number of non-zero elements
    * @param {coo_element[]} el - Array of non-zero elements
    * -> @param {int} el[].row - Row index
    * -> @param {int} el[].col - Column index
    * -> @param {float} el[].val - Value of the element
**/
struct coo_matrix {
    int rows;
    int cols;
    int nnz;
    coo_element *el;
};

/**
    * @brief Load a COO matrix from a file
    * @param {const char *} filename - Path to the file
    * @returns {coo_matrix *} - Pointer to the COO matrix
 */
coo_matrix* load_coo_matrix(const char *filename);
/**
    * @brief Load a COO matrix from a hardcoded matrix
    * @returns {coo_matrix *} - Pointer to the COO matrix
 */
coo_matrix* load_coo_matrix(void);
/**
    * @brief Check if a COO matrix is the transpose of another COO matrix
    * @param {coo_matrix *} coo - Pointer to the first COO matrix
    * @param {coo_matrix *} coo_t - Pointer to the second COO matrix
    * @returns {bool} - True if the matrices are one the transpose of the other, False otherwise
 */
bool is_transpose(const coo_matrix *coo, const coo_matrix *coo_t);
/**
    * @brief Print a COO matrix
    * @param {coo_matrix *} coo - Pointer to the COO matrix
    * @returns {int} - 0 if successful
 */
int print_coo_matrix(const coo_matrix *coo);
/**
    * @brief Prints a COO matrix metadata, such as rows, cols and nnz
    * @param {coo_matrix *} coo - Pointer to the COO matrix
    * @returns {int} - 0 if successful
 */
int print_coo_metadata(const coo_matrix *coo);
/**
    * @brief Same as print_coo_matrix(), but only prints the first 10 elements of the matrix
    * @param {coo_matrix *} coo - Pointer to the COO matrix
    * @returns {int} - 0 if successful
 */
int print_coo_less(const coo_matrix *coo);

#endif
