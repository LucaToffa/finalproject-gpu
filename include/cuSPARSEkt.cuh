#ifndef __CUSPARSEKT_CUH__
#define __CUSPARSEKT_CUH__

#include "csr.h"

/**
    * @brief Transpose a CSR matrix using cuSPARSE
    * @param {csr_matrix *} in - Input CSR matrix
    * @param {csr_matrix *} out - Output CSR matrix
    * @param {int} matrix_size - Size of the matrix
    * @return {int} 0 if successful, 1 otherwise
 */
int cuSparseCSRt(const csr_matrix *in, csr_matrix *out, int matrix_size);

#endif
