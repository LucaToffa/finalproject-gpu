#include "../include/csr.h"
#include "../include/debug.h"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <cstring>


csr_matrix* new_csr_matrix(size_t rows, size_t cols, size_t nnz, size_t *row_offsets, size_t *col_indices, float *values) {
    csr_matrix *csr = new csr_matrix {
        .rows = rows,
        .cols = cols,
        .nnz = nnz,
        .row_offsets = new size_t[rows + 1],
        .col_indices = new size_t[nnz],
        .values = new float[nnz]
    };
    for(int i = 0; i < rows + 1; i++) {
        csr->row_offsets[i] = row_offsets[i];
    }
    for(int i = 0; i < nnz; i++) {
        csr->col_indices[i] = col_indices[i];
        csr->values[i] = values[i];
    }
    return csr;
}

csr_matrix* new_csr_matrix(size_t rows, size_t cols, size_t nnz) {
    csr_matrix *csr = new csr_matrix {
        .rows = rows,
        .cols = cols,
        .nnz = nnz,
        .row_offsets = new size_t[rows + 1],
        .col_indices = new size_t[nnz],
        .values = new float[nnz]
    };
    memset(csr->row_offsets, 0, (rows + 1) * sizeof(size_t));
    memset(csr->col_indices, 0, nnz * sizeof(size_t));
    memset(csr->values, 0, nnz * sizeof(float));
    return csr;
}

csr_matrix* load_csr_matrix(const char *filename) {
    PRINTF("--------------------\n");
    PRINTF("Loading CSR Matrix from file: %s\n", filename);
    FILE *f = fopen(filename, "r");
    if(f == NULL){
        std::cerr << "Error opening file: " << filename << std::endl;
        return NULL;
    }
    int rows, cols, nnz;
    char* line = new char[1024];
    do{
        line = fgets(line, 1024, f);
    }while (line[0] < '0' || line[0] > '9');
    fseek(f, -strlen(line), SEEK_CUR);
    fscanf(f, "%d %d %d", &rows, &cols, &nnz);
    PRINTF("Metadata: { Rows: %d\t, Cols: %d\t, NNZ: %d }\n", rows, cols, nnz);
    size_t *row_offsets = new size_t[rows + 1];
    size_t *col_indices = new size_t[nnz];
    float *values = new float[nnz];

    for (int i = 0; i < nnz; i++) {
        // every line is of the form: (int)row (int)col (signed int | float)value
        size_t row, col;
        float value;
        fscanf(f, "%lu %lu %f", &row, &col, &value);
        row--; col--; // 1-indexed to 0-indexed
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            std::cerr << "Error: Invalid row or column index at line " << i << std::endl;
            std::cerr << "row: " << row << "col: " << col << "value: " << value << std::endl;
            return NULL;
        }
        values[i] = value;
        col_indices[i] = col;
        row_offsets[row + 1]++;
    }

    for (int i = 0; i < rows; i++) {
        row_offsets[i + 1] += row_offsets[i];
    }
    PRINTF("CSR Matrix loaded succesfully.\n");
    PRINTF("--------------------\n");
    //for(int i = 0; i < rows + 1; i++){ fscanf(f, "%d", &row_offsets[i]); }
    //for(int i = 0; i < nnz; i++){ fscanf(f, "%d", &col_indices[i]); }
    //for(int i = 0; i < nnz; i++){ fscanf(f, "%f", &values[i]); }
    csr_matrix *csr = new_csr_matrix(rows, cols, nnz, row_offsets, col_indices, values);
    delete[] line;
    delete[] row_offsets;
    delete[] col_indices;
    delete[] values;
    return csr;
}

csr_matrix* load_csr_matrix(void) {
    size_t row_offsets[4] = {0, 1, 1, 2};
    size_t col_indices[2] = {1, 1};
    float values[2] = {0.1, 2.3};
    csr_matrix *csr = new_csr_matrix(3, 3, 2, row_offsets, col_indices, values);
    return csr;
}

bool is_transpose(const csr_matrix * const csr, const csr_matrix * const csr_t) {
    if (csr->rows != csr_t->cols || csr->cols != csr_t->rows || csr->nnz != csr_t->nnz) {
        printf("Matrices have different dimensions\n");
        return false;
    }
    for (size_t i = 0; i < csr->rows; i++) {
        for (size_t j = 0; j < csr->cols; j++) {
            bool found = false;
            for (size_t k = csr->row_offsets[i]; k < csr->row_offsets[i + 1]; k++) {
                if (csr->nnz < k) {
                    printf("Error: k = %zu\n", k);
                    return false;
                }
                if (csr->col_indices[k] == j) {
                    found = false;
                    for (size_t l = csr_t->row_offsets[j]; l < csr_t->row_offsets[j + 1]; l++) {
                        if (csr_t->nnz < l) {
                            printf("Error: l = %zu\n", l);
                            return false;
                        }
                        if (csr_t->col_indices[l] == i) {
                            // since they are floats we need to compare them with a tolerance
                            if (std::fabs(csr->values[k] - csr_t->values[l]) > 1e-3) {
                                printf("Unequal - Error at mat[%zu, %zu]\n", i, j);
                                return false;
                            }
                            found = true;
                            break;
                        }
                    }
                    if (!found && i != j && i != csr->cols - 1) {
                        printf("Not Found - Error at mat[%zu, %zu]\n", i, j);
                        return false;
                    }
                    break;
                }
            }
            if (!found) {
                for (size_t l = csr_t->row_offsets[j]; l < csr_t->row_offsets[j + 1]; l++) {
                    if (csr_t->col_indices[l] == i) {
                        printf("Extra - Error at mat[%zu, %zu]\n", i, j);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

int print_csr_matrix(const csr_matrix *csr) {
    std::cout << "Debug Print - CSR Matrix:" << std::endl;
    std::cout << "Rows: " << csr->rows << " Cols: " << csr->cols << " NNZ: " << csr->nnz << std::endl;
    std::cout << "Row offsets: ";
    for(size_t i = 0; i < csr->rows + 1; i++){ std::cout << csr->row_offsets[i] << " "; }
    std::cout << std::endl;
    std::cout << "Col indices: ";
    for(size_t i = 0; i < csr->nnz; i++){ std::cout << csr->col_indices[i] << " "; }
    std::cout << std::endl;
    std::cout << "Values: ";
    for(size_t i = 0; i < csr->nnz; i++){ std::cout << csr->values[i] << " "; }
    std::cout << std::endl;
    return 0;
}

int pretty_print_csr_matrix(const csr_matrix *csr, std::ostream &out) {
    // this method prints the matrix in a more human readable way
    // it is useful for debugging
    out << "Pretty Print - CSR Matrix:" << std::endl;
    out << "Rows: " << csr->rows << " Cols: " << csr->cols << " NNZ: " << csr->nnz << std::endl;
    for (size_t i = 0; i < csr->rows; i++) {
        for (size_t j = 0; j < csr->cols; j++) {
            bool found = false;
            for (size_t k = csr->row_offsets[i]; k < csr->row_offsets[i + 1]; k++) {
                if (csr->col_indices[k] == j) {
                    out << csr->values[k] << "\t";
                    found = true;
                    break;
                }
            }
            if (!found) {
                out << "0\t";
            }
        }
        out << std::endl;
    }
    return 0;
}
