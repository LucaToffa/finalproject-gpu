#pragma once

typedef struct coo_element{
    int row;
    int col;
    float val;
} coo_element;

typedef struct coo_matrix{
    int rows;
    int cols;
    int nnz;
    coo_element *el;
} coo_matrix;

// TODO: use struct costructors and return the coo / csr directly
void coo_hello();
coo_matrix* load_coo_matrix(const char *filename);
coo_matrix* load_coo_matrix(void);
int print_coo_matrix(coo_matrix *coo);
int print_coo_metadata(coo_matrix *coo);
