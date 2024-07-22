#include "../include/csr.h"
#include "../include/debug.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cstring>

csr_matrix::csr_matrix(int rows, int cols, int nnz) {
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;
    this->row_offsets = new int[rows + 1];
    this->col_indices = new int[nnz];
    this->values = new float[nnz];
}
void csr_matrix::csr_fill(int *row_offsets, int *col_indices, float *values) {
    //TODO: just random things
    for(int i = 0; i < rows + 1; i++){
        this->row_offsets[i] = row_offsets[i];
    }
    for(int i = 0; i < nnz; i++){
        this->col_indices[i] = col_indices[i];
        this->values[i] = values[i];
    }
}

csr_matrix::~csr_matrix() {
    delete[] row_offsets;
    delete[] col_indices;
    delete[] values;
}

void csr_hello() { std::cout << "Hello world!! from CSR" << std::endl; }

csr_matrix* load_csr_matrix(const char *filename) {
    FILE *f = fopen(filename, "r");
    if(f == NULL){
        std::cerr << "Error opening file: " << filename << std::endl;
        return NULL;
    }
    int rows, cols, nnz;
    char* line = new char[1024];
    while (line[0] < '0' || line[0] > '9') { line = fgets(line, 1024, f); }
    fseek(f, -strlen(line), SEEK_CUR);
    fscanf(f, "%d %d %d", &rows, &cols, &nnz);
    csr_matrix *csr = new csr_matrix(rows, cols, nnz);
    int *row_offsets = new int[rows + 1];
    int *col_indices = new int[nnz];
    float *values = new float[nnz];

    for (int i = 0; i < nnz; i++) {
        // every line is of the form: (int)row (int)col (signed int | float)value
        int row, col;
        float value;
        fscanf(f, "%d %d %f", &row, &col, &value);
        row--; col--; // 1-indexed to 0-indexed
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            std::cerr << "Error: Invalid row or column index at line " << i << std::endl;
            return NULL;
        }
        values[i] = value;
        col_indices[i] = col;
        row_offsets[row + 1]++;        
    }

    for (int i = 0; i < rows; i++) {
        row_offsets[i + 1] += row_offsets[i];
    }

    //for(int i = 0; i < rows + 1; i++){ fscanf(f, "%d", &row_offsets[i]); }
    //for(int i = 0; i < nnz; i++){ fscanf(f, "%d", &col_indices[i]); }
    //for(int i = 0; i < nnz; i++){ fscanf(f, "%f", &values[i]); }
    
    csr->csr_fill(row_offsets, col_indices, values);
    delete[] row_offsets;
    delete[] col_indices;
    delete[] values;
    return csr;
}

csr_matrix* load_csr_matrix(void) {
    csr_matrix *csr = new csr_matrix(3, 3, 2);
    int row_offsets[4] = {0, 1, 1, 2};
    int col_indices[2] = {1, 1};
    float values[2] = {0.1, 2.3};
    csr->csr_fill(row_offsets, col_indices, values);
    return csr;
}

bool is_transpose(csr_matrix *csr, csr_matrix *csr_t) {
    if (csr->rows != csr_t->cols || csr->cols != csr_t->rows || csr->nnz != csr_t->nnz) {
        return false;
    }
    for (int i = 0; i < csr->rows; i++) {
        for (int j = 0; j < csr->cols; j++) {
            bool found = false;
            for (int k = csr->row_offsets[i]; k < csr->row_offsets[i + 1]; k++) {
                if (csr->col_indices[k] == j) {
                    found = false;
                    for (int l = csr_t->row_offsets[j]; l < csr_t->row_offsets[j + 1]; l++) {
                        if (csr_t->col_indices[l] == i) {
                            // since they are floats we need to compare them with a tolerance
                            if (std::fabs((float)csr->values[k] - (float)csr_t->values[l]) > 1e-3) {
                                printf("Unequal - Error at mat[%d, %d]\n", i, j);
                                return false;
                            }
                            found = true;
                            break;
                        }
                    }
                    if (!found && i != j && i != csr->cols - 1) {
                        printf("Not Found - Error at mat[%d, %d]\n", i, j);
                        return false;
                    }
                    break;
                }
            }
            if (!found) {
                for (int l = csr_t->row_offsets[j]; l < csr_t->row_offsets[j + 1]; l++) {
                    if (csr_t->col_indices[l] == i) {
                        printf("Extra - Error at mat[%d, %d]\n", i, j);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

int print_csr_matrix(csr_matrix *csr) {
    std::cout << "Debug Print - CSR Matrix:" << std::endl;
    std::cout << "Rows: " << csr->rows << " Cols: " << csr->cols << " NNZ: " << csr->nnz << std::endl;
    std::cout << "Row offsets: ";
    for(int i = 0; i < csr->rows + 1; i++){ std::cout << csr->row_offsets[i] << " "; }
    std::cout << std::endl;
    std::cout << "Col indices: ";
    for(int i = 0; i < csr->nnz; i++){ std::cout << csr->col_indices[i] << " "; }
    std::cout << std::endl;
    std::cout << "Values: ";
    for(int i = 0; i < csr->nnz; i++){ std::cout << csr->values[i] << " "; }
    std::cout << std::endl;
    return 0;
}

int pretty_print_csr_matrix(csr_matrix *csr) {
    // this method prints the matrix in a more human readable way
    // it is useful for debugging
    std::cout << "Pretty Print - CSR Matrix:" << std::endl;
    std::cout << "Rows: " << csr->rows << " Cols: " << csr->cols << " NNZ: " << csr->nnz << std::endl;
    for (int i = 0; i < csr->rows; i++) {
        for (int j = 0; j < csr->cols; j++) {
            bool found = false;
            for (int k = csr->row_offsets[i]; k < csr->row_offsets[i + 1]; k++) {
                if (csr->col_indices[k] == j) {
                    std::cout << csr->values[k] << "\t";
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cout << "0\t";
            }
        }
        std::cout << std::endl;
    }
    return 0;
}
