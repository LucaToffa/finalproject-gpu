#include  <iostream>
#include <cuda_runtime.h>
#include "commons.h"

#include "defines.h"
#include "benchmarks.h"

int main(int argc, char** argv){   
#ifdef DEBUG
    if (argc > 1){
        printf("argc = %d:\n", argc);
        for(int i = 0; i < argc; i++){
            printf("arg %d : %s\n", i+1, argv[i]);
        }
        printf("\n");
    }
    
#endif
    bool swipe = false;
    uint N = DEFAULT_SIZE;
    if(argc >= 2){
        N = (1<<atoi(argv[1]));
        PRINTF("N changed: %d\n", N);
        printf("N: %d, T: %d, B: %d \n", N, TILE_SIZE, BLOCK_ROWS);
    }else{
        swipe = true;
        //log shape of data
        printf("#T: %d, B: %d#\n", TILE_SIZE, BLOCK_ROWS);
        printf("#N, OpTime, Op-GB/s, KTime, K-GB/s (basic, conflcit, block)#\n");
    }
    if(BLOCK_ROWS > TILE_SIZE){
        printf("Error: BLOCK_ROWS > TILE_SIZE\n");
        return -1;

    }
    do{ //swipe across all N if none is given
        PRINTF("N: ");
        printf("%d, ", N);
        basic_benchmark(N);
        conflict_benchmark(N);
        block_benchmark(N);
        N *= 2;
    }while(swipe && N < (2<<13)); //2<<14 = 16384

    PRINTF("\n");

    return 0;
}

int testing(){
    coo_matrix* coo = load_coo_matrix("matrices/tests/mockcoo.mtx");
    PRINTF("--------------------\n");
    print_coo_matrix(coo);
    PRINTF("--------------------\n");
    delete coo;
    coo = load_coo_matrix("matrices/circuit204.mtx");
    print_coo_metadata(coo);
    int full_size = 4;
    float *mat = new float [full_size*full_size];
    sparseInitMatrix(mat, full_size);
    coo_matrix* coo2 = mat_to_coo(mat, full_size);
    PRINTF("--------------------\n");
    print_coo_matrix(coo2);
    printMatrix(mat, full_size);
    PRINTF("--------------------\n");

    delete[] mat;
    delete coo;
    delete coo2;

    PRINTF("CSR tests\n");
    //csr_matrix* csr = load_csr_matrix("matrices/tests/mockcsr.mtx");
    csr_matrix* csr = load_csr_matrix();
    PRINTF("--------------------\n");
    print_csr_matrix(csr);
    PRINTF("--------------------\n");
    return 0;
}

