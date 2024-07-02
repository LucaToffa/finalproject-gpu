#pragma once

#include "coo.h"
#include "csr.h"

int initMatrix(float* mat, int size);
int sparseInitMatrix(float* mat, int size);
int printMatrix(float* mat, int size);
int testTranspose(float* mat, float* mt, int size);
csr_matrix coo_to_csr(coo_matrix *coo);
coo_matrix csr_to_coo(csr_matrix *csr);
csr_matrix mat_to_csr(float *mat, int size);
coo_matrix* mat_to_coo(float *mat , int size);
float* csr_to_mat(csr_matrix *csr); 
float* coo_to_mat(coo_matrix *coo);

