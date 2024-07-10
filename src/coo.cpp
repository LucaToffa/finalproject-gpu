/*
utilities to work with COO matrix format
*/

#include <iostream>
#include "coo.h"
#include <cstring>

void coo_hello(){
    std::cout << "Hello world!! from COO" << std::endl;
}

coo_matrix* load_coo_matrix(const char *filename){
    coo_matrix* coo = new coo_matrix;
    FILE *f = fopen(filename, "r");
    if(f == NULL){
        std::cerr << "Error opening file: " << filename << std::endl;
        return NULL;
    }
    //go to first row that begins with a number
    char* line = new char[1024];
    while(line[0] < '0' || line[0] > '9'){
        line = fgets(line, 1024, f);
    }
    printf("%s\n", line);
    //go back to previous line
    fseek(f, -strlen(line), SEEK_CUR);
    //get the matrix dimensions
    fscanf(f, "%d %d %d", &coo->rows, &coo->cols, &coo->nnz);
    //allocate memory for the el
    coo->el = new coo_element[coo->nnz];
    //read the elements
    for(int i = 0; i < coo->nnz; i++){
        fscanf(f, "%d %d %f", &coo->el[i].row, &coo->el[i].col, &coo->el[i].val);
    }
    return coo;
}
coo_matrix* load_coo_matrix(void){
    coo_matrix* coo = new coo_matrix;
    coo->rows = 3; coo->cols = 3; coo->nnz = 2;

    coo->el = new coo_element[coo->nnz];
    coo->el[0].row = 0; coo->el[0].col = 1; coo->el[0].val = 1.1;
    coo->el[1].row = 2; coo->el[1].col = 1; coo->el[1].val = 2.5;
    return coo;
}

int print_coo_matrix(coo_matrix *coo){
    std::cout << "COO matrix: " << std::endl;
    std::cout << "rows: " << coo->rows << " cols: " << coo->cols << " nnz: " << coo->nnz << std::endl;
    for(int i = 0; i < coo->nnz; i++){
        std::cout << "row: " << coo->el[i].row << " col: " << coo->el[i].col << " val: " << coo->el[i].val << std::endl;
    }
    return 0;
}

int print_coo_metadata(coo_matrix *coo){
    std::cout << "COO matrix: " << std::endl;
    std::cout << "rows: " << coo->rows << " cols: " << coo->cols << " nnz: " << coo->nnz << std::endl;
    return 0;}