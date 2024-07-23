#ifndef __COO_H__
#define __COO_H__

#include <iostream>
#include <cstring>

/**
    * @brief Struct to represent a Coordinate list element
    * @param {unsigned long} row - Row index
    * @param {unsigned long} col - Column index
    * @param {float} val - Value of the element
 */
struct coo_element {
    size_t row;
    size_t col;
    float val;
};

/**
    * @brief Struct to represent a Coordinate list matrix
    * @param {unsigned long} rows - Number of rows
    * @param {unsigned long} cols - Number of columns
    * @param {unsigned long} nnz - Number of non-zero elements
    * @param {coo_element[]} el - Array of non-zero elements
    *? @param {unsigned long} el[].row - Row index
    *? @param {unsigned long} el[].col - Column index
    *? @param {float} el[].val - Value of the element
**/
struct coo_matrix {
    size_t rows;
    size_t cols;
    size_t nnz;
    coo_element *el;
};

coo_matrix* load_coo_matrix(const char *filename);
coo_matrix* load_coo_matrix(void);
bool is_transpose(coo_matrix *coo, coo_matrix *coo_t);
int print_coo_matrix(coo_matrix *coo);
int print_coo_metadata(coo_matrix *coo);
int print_coo_less(coo_matrix *coo);

#endif
