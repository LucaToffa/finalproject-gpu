#ifndef __CSR_H__
#define __CSR_H__

#include <iostream>

void csr_hello();

class csr_matrix{
    public:
        int rows;
        int cols;
        int nnz;
        int *row_offsets;
        int *col_indices;
        float *values;
        csr_matrix(int rows, int cols, int nnz);
        void csr_fill(int *row_offsets, int *col_indices, float *values);
        ~csr_matrix();
};

csr_matrix* load_csr_matrix(const char *filename);
csr_matrix* load_csr_matrix(void);
int print_csr_matrix(csr_matrix *csr);

#endif
