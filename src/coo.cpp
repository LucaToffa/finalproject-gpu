#include "../include/coo.h"
#include "../include/debug.h"
#include <cmath>
#include <iostream>
#include <cstring>
#include "../include/commons.h"

coo_matrix* load_coo_matrix(const char *filename) {
    PRINTF("--------------------\n");
    PRINTF("Loading COO Matrix from file: %s\n", filename);
    coo_matrix* coo = new coo_matrix;
    FILE *f = fopen(filename, "r");
    if(f == NULL) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return NULL;
    }
    char* line = new char[1024];
    do {
        line = fgets(line, 1024, f); 
    } while (line[0] < '0' || line[0] > '9');
    fseek(f, -strlen(line), SEEK_CUR);
    fscanf(f, "%d %d %d", &coo->rows, &coo->cols, &coo->nnz);
    coo->rows = next_power_of_2(std::max(coo->rows, coo->cols));
    coo->cols = next_power_of_2(std::max(coo->rows, coo->cols));
    PRINTF("Metadata: { Rows: %d\t, Cols: %d\t, NNZ: %d }\n", coo->rows, coo->cols, coo->nnz);
    coo->el = new coo_element[coo->nnz];
    int row, col;
    for(int i = 0; i < coo->nnz; i++){ 
        fscanf(f, "%d %d %f", &row, &col, &coo->el[i].val); 
        row--; col--; // 1-indexed to 0-indexed
        coo->el[i].row = row;
        coo->el[i].col = col;
    }
    delete[] line;
    fclose(f);
    PRINTF("COO Matrix loaded succesfully.\n");
    PRINTF("--------------------\n");
    return coo;
}

coo_matrix* load_coo_matrix(void) {
    coo_matrix* coo = new coo_matrix;
    coo->rows = 3; coo->cols = 3; coo->nnz = 2;
    coo->el = new coo_element[coo->nnz];
    coo->el[0].row = 0; coo->el[0].col = 1; coo->el[0].val = 1.1;
    coo->el[1].row = 2; coo->el[1].col = 1; coo->el[1].val = 2.5;
    return coo;
}

void delete_coo(coo_matrix *coo) {
    delete[] coo->el;
    delete coo;
}

bool is_transpose(const coo_matrix *coo, const coo_matrix *coo_t) {
    if(coo->rows != coo_t->cols || coo->cols != coo_t->rows || coo->nnz != coo_t->nnz) {
        return false;
    }
    for(int i = 0; i < coo->nnz; i++) {
        if(coo->el[i].row != coo_t->el[i].col || coo->el[i].col != coo_t->el[i].row || fabs(coo->el[i].val - coo_t->el[i].val) > 1e-6) {
            return false;
        }
    }
    return true;
}

int print_coo_matrix(const coo_matrix *coo) {
    std::cout << "COO Matrix Full: " << std::endl;
    std::cout << "rows: " << coo->rows << " cols: " << coo->cols << " nnz: " << coo->nnz << std::endl;
    for(int i = 0; i < coo->nnz; i++) {
        std::cout << "row: " << coo->el[i].row << " col: " << coo->el[i].col << " val: " << coo->el[i].val << std::endl;
    }
    return 0;
}

int print_coo_metadata(const coo_matrix *coo) {
    std::cout << "COO Matrix Header: " << std::endl;
    std::cout << "rows: " << coo->rows << " cols: " << coo->cols << " nnz: " << coo->nnz << std::endl;
    return 0;
}

int print_coo_less(const coo_matrix *coo) {
    std::cout << "COO matrix: " << std::endl;
    std::cout << "rows: " << coo->rows << " cols: " << coo->cols << " nnz: " << coo->nnz << std::endl;
    for(int i = 0; i < coo->nnz && i < 10; i++){
        std::cout << "row: " << coo->el[i].row << " col: " << coo->el[i].col << " val: " << coo->el[i].val << std::endl;
    }
    return 0;
}
