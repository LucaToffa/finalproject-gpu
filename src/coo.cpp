#include "../include/coo.h"
#include "../include/debug.h"
#include <cmath>
#include <iostream>
#include <cstring>

coo_matrix* load_coo_matrix(const char *filename) {
    coo_matrix* coo = new coo_matrix;
    FILE *f = fopen(filename, "r");
    if(f == NULL){
        std::cerr << "Error opening file: " << filename << std::endl;
        return NULL;
    }
    char* line = new char[1024];
    while (line[0] < '0' || line[0] > '9') { line = fgets(line, 1024, f); }
    fseek(f, -strlen(line), SEEK_CUR);
    fscanf(f, "%lu %lu %lu", &coo->rows, &coo->cols, &coo->nnz);
    coo->el = new coo_element[coo->nnz];
    for(int i = 0; i < coo->nnz; i++){ fscanf(f, "%lu %lu %f", &coo->el[i].row, &coo->el[i].col, &coo->el[i].val); }
    delete[] line;
    fclose(f);
    PRINTF("COO Matrix loaded from file: %s\n", filename);
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

bool is_transpose(coo_matrix *coo, coo_matrix *coo_t) {
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

int print_coo_matrix(coo_matrix *coo) {
    std::cout << "COO Matrix Full: " << std::endl;
    std::cout << "rows: " << coo->rows << " cols: " << coo->cols << " nnz: " << coo->nnz << std::endl;
    for(int i = 0; i < coo->nnz; i++) {
        std::cout << "row: " << coo->el[i].row << " col: " << coo->el[i].col << " val: " << coo->el[i].val << std::endl;
    }
    return 0;
}

int print_coo_metadata(coo_matrix *coo) {
    std::cout << "COO Matrix Header: " << std::endl;
    std::cout << "rows: " << coo->rows << " cols: " << coo->cols << " nnz: " << coo->nnz << std::endl;
    return 0;
}

//printing everything is too much, just print the first 10 elements
int print_coo_less(coo_matrix *coo) {
    std::cout << "COO matrix: " << std::endl;
    std::cout << "rows: " << coo->rows << " cols: " << coo->cols << " nnz: " << coo->nnz << std::endl;
    for(int i = 0; i < coo->nnz && i < 10; i++){
        std::cout << "row: " << coo->el[i].row << " col: " << coo->el[i].col << " val: " << coo->el[i].val << std::endl;
    }
    return 0;
}
