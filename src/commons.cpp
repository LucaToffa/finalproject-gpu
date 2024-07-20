#include "../include/csr.h"
#include "../include/coo.h"
#include "../include/commons.h"
#include "../include/debug.h"
#include <cstring>


int initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            mat[i + j*size] = (i*2+j)%(100);
        }
    }
    return 0;

}

int sparseInitMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if ((i != j) && ((i*2+j)%(3) == 0)) { mat[i + j*size] = (i*2+j)%(100); }
            else { mat[i + j*size] = 0; }
        }
    }
    return 0;
}

int printMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%2.2f ", mat[i + j*size]);
        }
        printf("\n");
    }
    printf("\n");
    return 0;
}

int testTranspose(float* mat, float* mat_t, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (mat[i + j*size] != mat_t[j + i*size]) {
                printf("Error at mat[%d, %d]\n", i, j);
                return -1;
            }
        }
    }
    PRINTF("Matrix transposed without errors\n");
    return 0;
}

csr_matrix coo_to_csr(coo_matrix *coo) {
    csr_matrix csr(coo->rows, coo->cols, coo->nnz);
    return csr;
}

coo_matrix csr_to_coo(csr_matrix *csr) {
    coo_matrix coo;
    coo.rows = csr->rows;
    coo.cols = csr->cols;
    coo.nnz = csr->nnz;
    coo.el = new coo_element[coo.nnz];
    return coo;
}

csr_matrix mat_to_csr(float *mat, int size) {
    csr_matrix csr(0, 0, 0);
    return csr;
}

coo_matrix* mat_to_coo(float *mat, int size) {
    coo_matrix* coo = new coo_matrix;
    coo->rows = size; coo->cols = size;
    int nnz = 0;
    //TODO: avoid going through the matrix twice
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (mat[i*size + j] != 0) { nnz++; }
        }
    }
    coo->nnz = nnz;
    coo->el = new coo_element[nnz];
    int k = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (mat[i*size + j] != 0) {
                coo->el[k].row = i;
                coo->el[k].col = j;
                coo->el[k].val = mat[i*size + j];
                k++;
            }
        }
    }
    return coo;
}

float* csr_to_mat(csr_matrix *csr){
    float* mat = new float[csr->rows*csr->cols];
    memset(mat, 0, csr->rows*csr->cols*sizeof(float));
    //TODO: this is random gargabe
    for (int i = 0; i < csr->nnz; i++) {
        mat[csr->row_offsets[i]*csr->cols + csr->col_indices[i]] = csr->values[i];
    }
    return mat;
}
float* coo_to_mat(coo_matrix *coo) {
    float* mat = new float[coo->rows*coo->cols];
    memset(mat, 0, coo->rows*coo->cols*sizeof(float));
    for (int i = 0; i < coo->nnz; i++) { mat[coo->el[i].row*coo->cols + coo->el[i].col] = coo->el[i].val; }
    return mat;
}
