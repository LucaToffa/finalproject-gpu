#include "../include/csr.h"
#include "../include/debug.h"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <cstring>


csr_matrix* new_csr_matrix(int rows, int cols, int nnz, int *row_offsets, int *col_indices, float *values) {
    csr_matrix *csr = new csr_matrix {
        .rows = rows,
        .cols = cols,
        .nnz = nnz,
        .row_offsets = new int[rows + 1],
        .col_indices = new int[nnz],
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

csr_matrix* new_csr_matrix(int rows, int cols, int nnz) {
    csr_matrix *csr = new csr_matrix {
        .rows = rows,
        .cols = cols,
        .nnz = nnz,
        .row_offsets = new int[rows + 1],
        .col_indices = new int[nnz],
        .values = new float[nnz]
    };
    memset(csr->row_offsets, 0, (rows + 1) * sizeof(int));
    memset(csr->col_indices, 0, nnz * sizeof(int));
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
    csr_matrix *csr = new_csr_matrix(rows, cols, nnz);

    for (int i = 0; i < nnz; i++) {
        // every line is of the form: (int)row (int)col (signed int | float)value
        int row, col;
        float value;
        fscanf(f, "%d %d %f", &row, &col, &value);
        row--; col--; // 1-indexed to 0-indexed
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            std::cerr << "Error: Invalid row or column index at line " << i << std::endl;
            std::cerr << "row: " << row << "col: " << col << "value: " << value << std::endl;
            return NULL;
        }
        csr->values[i] = value;
        csr->col_indices[i] = col;
        csr->row_offsets[row + 1]++;
    }

    for (int i = 0; i < rows; i++) {
        csr->row_offsets[i + 1] += csr->row_offsets[i];
    }
    PRINTF("CSR Matrix loaded succesfully.\n");
    PRINTF("--------------------\n");
    //for(int i = 0; i < rows + 1; i++){ fscanf(f, "%d", &row_offsets[i]); }
    //for(int i = 0; i < nnz; i++){ fscanf(f, "%d", &col_indices[i]); }
    //for(int i = 0; i < nnz; i++){ fscanf(f, "%f", &values[i]); }
    delete[] line;
    return csr;
}

csr_matrix* load_csr_matrix(void) {
    int row_offsets[4] = {0, 1, 1, 2};
    int col_indices[2] = {1, 1};
    float values[2] = {0.1, 2.3};
    csr_matrix *csr = new_csr_matrix(3, 3, 2, row_offsets, col_indices, values);
    return csr;
}

bool is_transpose(const csr_matrix * const csr, const csr_matrix * const csr_t) {
    if (csr->rows != csr_t->cols || csr->cols != csr_t->rows || csr->nnz != csr_t->nnz) {
        printf("Matrices have different dimensions: Input: { R: %d, C: %d, NNZ: %d }, Output: { R: %d, C: %d, NNZ: %d }\n", csr->rows, csr->cols, csr->nnz, csr_t->rows, csr_t->cols, csr_t->nnz);
        return false;
    }
    for (int i = 0; i < csr->rows; i++) {
        for (int j = 0; j < csr->cols; j++) {
            bool found = false;
            for (int k = csr->row_offsets[i]; k < csr->row_offsets[i + 1]; k++) {
                if (csr->nnz < k) {
                    printf("Error: k = %d\n", k);
                    return false;
                }
                if (csr->col_indices[k] == j) {
                    found = false;
                    for (int l = csr_t->row_offsets[j]; l < csr_t->row_offsets[j + 1]; l++) {
                        if (csr_t->nnz < l) {
                            printf("Error: l = %d\n", l);
                            return false;
                        }
                        if (csr_t->col_indices[l] == i) {
                            // since they are floats we need to compare them with a tolerance
                            if (std::fabs(csr->values[k] - csr_t->values[l]) > 1e-3) {
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

int print_csr_matrix(const csr_matrix *csr) {
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

int pretty_print_csr_matrix(const csr_matrix *csr, std::ostream &out) {
    // this method prints the matrix in a more human readable way
    // it is useful for debugging
    out << "Pretty Print - CSR Matrix:" << std::endl;
    out << "Rows: " << csr->rows << " Cols: " << csr->cols << " NNZ: " << csr->nnz << std::endl;
    for (int i = 0; i < csr->rows; i++) {
        for (int j = 0; j < csr->cols; j++) {
            bool found = false;
            /*
            if (i+1 >= csr->rows) {
                printf("Error: i = %zu\n", i);
                continue;
            }*/
            for (int k = csr->row_offsets[i]; k < csr->row_offsets[i + 1]; k++) {
                /*
                if (k > csr->nnz) {
                    printf("Error: k = %zu\t|Row Offset[i]: %d\t|Row Offset[i+1]:%d\n", k, csr->row_offsets[i], csr->row_offsets[i + 1]);
                    continue;
                }*/
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
