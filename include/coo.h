#ifndef __COO_H__
#define __COO_H__

#include <iostream>
#include <cstring>

void coo_hello();

struct coo_element{
    int row;
    int col;
    float val;
};

struct coo_matrix{
    int rows;
    int cols;
    int nnz;
    coo_element *el;
};

coo_matrix* load_coo_matrix(const char *filename);
coo_matrix* load_coo_matrix(void);
int print_coo_matrix(coo_matrix *coo);
int print_coo_metadata(coo_matrix *coo);


#endif
