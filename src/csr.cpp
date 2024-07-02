/*
utilities to work with CSR matrix format
*/

#include <iostream>
#include "csr.h"

csr_matrix::csr_matrix(int rows, int cols, int nnz){
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;
    this->row_offsets = new int[rows + 1];
    this->col_indices = new int[nnz];
    this->values = new float[nnz];
}
void csr_matrix::csr_fill(int *row_offsets, int *col_indices, float *values){
    //TODO: just random things
    for(int i = 0; i < rows + 1; i++){
        this->row_offsets[i] = row_offsets[i];
    }
    for(int i = 0; i < nnz; i++){
        this->col_indices[i] = col_indices[i];
        this->values[i] = values[i];
    }

}
    
csr_matrix::~csr_matrix(){
    delete[] row_offsets;
    delete[] col_indices;
    delete[] values;
}

void csr_hello(){
    std::cout << "Hello world!! from CSR" << std::endl;
}

csr_matrix* load_csr_matrix(const char *filename){
    FILE *f = fopen(filename, "r");
    if(f == NULL){
        std::cerr << "Error opening file: " << filename << std::endl;
        return NULL;
    }
    //go to fisrt row that begins with a number
    char c;
    do{
        c = fgetc(f);
    }while(c < '0' || c > '9');
    ungetc(c, f);
    //get the matrix dimensions
    int rows, cols, nnz;
    fscanf(f, "%d %d %d", &rows, &cols, &nnz);
    //allocate memory for the el
    csr_matrix *csr = new csr_matrix(rows, cols, nnz);
    //read the elements
    int *row_offsets = new int[rows + 1];
    int *col_indices = new int[nnz];
    float *values = new float[nnz];
    for(int i = 0; i < rows + 1; i++){
        fscanf(f, "%d", &row_offsets[i]);
    }
    for(int i = 0; i < nnz; i++){
        fscanf(f, "%d", &col_indices[i]);
    }
    for(int i = 0; i < nnz; i++){
        fscanf(f, "%f", &values[i]);
    }
    csr->csr_fill(row_offsets, col_indices, values);
    delete[] row_offsets;
    delete[] col_indices;
    delete[] values;
    return csr;
}
csr_matrix* load_csr_matrix(void){
    csr_matrix *csr = new csr_matrix(3, 3, 2);
    int row_offsets[4] = {0, 1, 1, 2};
    int col_indices[2] = {1, 1};
    float values[2] = {0.1, 2.3};
    csr->csr_fill(row_offsets, col_indices, values);
    return csr;

}
int print_csr_matrix(csr_matrix *csr){
    std::cout << "CSR matrix" << std::endl;
    std::cout << "Rows: " << csr->rows << " Cols: " << csr->cols << " NNZ: " << csr->nnz << std::endl;
    std::cout << "Row offsets: ";
    for(int i = 0; i < csr->rows + 1; i++){
        std::cout << csr->row_offsets[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Col indices: ";
    for(int i = 0; i < csr->nnz; i++){
        std::cout << csr->col_indices[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Values: ";
    for(int i = 0; i < csr->nnz; i++){
        std::cout << csr->values[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}