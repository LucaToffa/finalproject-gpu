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

int printMatrix(const float* mat, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%2.2f ", mat[i + j*size]);
        }
        printf("\n");
    }
    printf("\n");
    return 0;
}

int testTranspose(const float* mat, const float* mat_t, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (mat[i + j*size] != mat_t[j + i*size]) {
                printf("Error at mat[%d, %d]\n", i, j);
                PRINTF("Expected: %2.2f, Got: %2.2f\n", mat[i + j*size], mat_t[j + i*size]);
                PRINTF("Matrix transposed with errors!\n");
                return -1;
            }
        }
    }
    PRINTF("Matrix transposed without errors\n");
    return 0;
}

csr_matrix coo_to_csr(const coo_matrix *coo) {
    // TODO: Implement this function (if needed)
    std::cout << "Function coo_to_csr not implemented" << std::endl;
    std::cerr << "Function coo_to_csr not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
    csr_matrix csr = csr_matrix {
        coo->rows,
        coo->cols,
        coo->nnz,
        new int[coo->rows + 1],
        new int[coo->nnz],
        new float[coo->nnz]
    };
    return csr;
}

coo_matrix csr_to_coo(const csr_matrix *csr) {
    // TODO: Implement this function (if needed)
    std::cout << "Function csr_to_coo not implemented" << std::endl;
    std::cerr << "Function csr_to_coo not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
    coo_matrix coo;
    coo.rows = csr->rows;
    coo.cols = csr->cols;
    coo.nnz = csr->nnz;
    coo.el = new coo_element[coo.nnz];
    return coo;
}

float* csr_to_mat(const csr_matrix *csr) {
    std::cout << "Function csr_to_mat not checked yet" << std::endl;
    std::cerr << "Function csr_to_mat not checked yet" << std::endl;
    std::exit(EXIT_FAILURE);
    float* mat = new float[csr->rows*csr->cols];
    memset(mat, 0, csr->rows*csr->cols*sizeof(float));
    //TODO: check if this is correct
    for (int i = 0; i < csr->nnz; i++) {
        mat[csr->row_offsets[i]*csr->cols + csr->col_indices[i]] = csr->values[i];
    }
    return mat;
}

float* coo_to_mat(const coo_matrix *coo) {
    float* mat = new float[coo->rows * coo->cols];
    memset(mat, 0, (coo->rows * coo->cols) * sizeof(float));
    for (int i = 0; i < coo->nnz; i++) {
        int idx = (coo->el[i].row * coo->cols) + coo->el[i].col;
        if((idx > (coo->cols * coo->rows)) || (idx < 0)) {
            printf("Out of bound matrix index during conversion at %d, index is %d\n", i, idx);
            printf("COO cols: %d, rows: %d, el.row: %d, el.col: %d\n", coo->cols, coo->rows, coo->el[i].col, coo->el[i].row);
            delete[] mat;
            return NULL;
        }
        mat[idx] = coo->el[i].val; 
    }
    return mat;
}

float* coo_to_mat_padded(const coo_matrix *coo) {
    int N = next_power_of_2(std::max(coo->rows, coo->cols));
    float* mat = new float[N * N];
    memset(mat, 0, (N * N) * sizeof(float));
    for (int i = 0; i < coo->nnz; i++) {
        int idx = (coo->el[i].row * N) + coo->el[i].col;
        if((idx > (N * N)) || (idx < 0)) {
            printf("Out of bound matrix index during conversion at %d, index is %d\n", i, idx);
            printf("COO cols: %d, rows: %d, el.row: %d, el.col: %d\n", coo->cols, coo->rows, coo->el[i].col, coo->el[i].row);
            delete[] mat;
            return NULL;
        }
        mat[idx] = coo->el[i].val;
    }
    return mat;
}

int next_power_of_2(int n) {
    int p = 1;
    if (n && !(n & (n - 1))) {
        return n;
    }
    while (p < n) {
        p <<= 1;
    }
    return p;
}
