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
    //std::cerr << "Function csr_to_mat not checked yet" << std::endl;
    //std::exit(EXIT_FAILURE);
    float* mat = new float[csr->rows*csr->cols];
    //float* mat = (float*) malloc(csr->rows*csr->cols*sizeof(float));
    if(mat == NULL) printf("alloc failed");
    memset(mat, 0, csr->rows*csr->cols*sizeof(float));
    //TODO: check if this is correct
    // for (int i = 0; i < csr->nnz; i++) {
    //     mat[csr->row_offsets[i]*csr->cols + csr->col_indices[i]] = csr->values[i];
    // }

    for (int row = 0; row < csr->rows; row++) {
        for (int i = csr->row_offsets[row]; i < csr->row_offsets[row + 1]; i++) {
            mat[csr->col_indices[i] * csr->rows + row] = csr->values[i];
        }
    }


    // int a = 0;
    // int b = 0;
    // printf("start extraction\n");
    // for (int i = 0; i < csr->rows; i++) {
    //     printf(".");
    //     for (int j = 0; j < csr->cols; j++) {
    //         bool found = false;
    //         /*
    //         if (i+1 >= csr->rows) {
    //             printf("Error: i = %zu\n", i);
    //             continue;
    //         }*/
    //         for (int k = csr->row_offsets[i]; k < csr->row_offsets[i + 1]; k++) {
    //             /*
    //             if (k > csr->nnz) {
    //                 printf("Error: k = %zu\t|Row Offset[i]: %d\t|Row Offset[i+1]:%d\n", k, csr->row_offsets[i], csr->row_offsets[i + 1]);
    //                 continue;
    //             }*/
    //             if (csr->col_indices[k] == j) {
    //                 //out << csr->values[k] << "\t";
    //                 mat[a + csr->cols * b] = csr->values[k];
    //                 found = true;
    //                 break;
    //             }
    //         }
    //         a++;
    //         // if (!found) {
    //         //     out << "0\t";
    //         // }
    //     }
    //     b++;
    //     //out << std::endl;
    // }

    return mat;
}

float* coo_to_mat(const coo_matrix *coo) {
    float* mat = new float[coo->rows * coo->cols];
    memset(mat, 0, (coo->rows * (coo->cols)) * sizeof(float));
    for (int i = 0; i < coo->nnz; i++) {
        int idx = (coo->el[i].col * (coo->rows)) + coo->el[i].row;
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

csr_matrix* csc_to_csr(int num_rows, int num_cols, int nnz, float* csc_values, int* csc_row_indices, int* csc_col_pointers) {

    // Step 0: Initialize CSR
    csr_matrix* csr = new_csr_matrix(num_rows, num_cols, nnz);

    // Step 1: Initialize row counts
    int* row_counts = (int*)calloc(num_rows, sizeof(int));

    // Step 2: Count the number of elements in each row
    for (int col = 0; col < num_cols; col++) {
        for (int i = csc_col_pointers[col]; i < csc_col_pointers[col + 1]; i++) {
            int row = csc_row_indices[i];
            row_counts[row]++;
        }
    }

    // Step 3: Compute row pointers for CSR
    // csr_row_pointers[0] = 0;
    csr->row_offsets[0] = 0;
    for (int i = 0; i < num_rows; i++) {
        // csr_row_pointers[i + 1] = csr_row_pointers[i] + row_counts[i];
        csr->row_offsets[i + 1] = csr->row_offsets[i] + row_counts[i];
    }

    // Step 4: Fill the CSR values and column indices
    int* current_position = (int*)calloc(num_rows, sizeof(int)); // To track position in CSR values array
    for (int col = 0; col < num_cols; col++) {
        for (int i = csc_col_pointers[col]; i < csc_col_pointers[col + 1]; i++) {
            int row = csc_row_indices[i];
            // int pos = csr_row_pointers[row] + current_position[row];
            int pos = csr->row_offsets[row] + current_position[row];

            // csr_values[pos] = csc_values[i];
            // csr_col_indices[pos] = col;
            csr->values[pos] = csc_values[i];
            csr->col_indices[pos] = col;

            current_position[row]++;
        }
    }

    // Free allocated memory
    free(row_counts);
    free(current_position);
    return csr;
}
