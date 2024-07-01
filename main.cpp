#include  <iostream>
#include <cuda_runtime.h>
#include "coo.h"
#include "csr.h"

int main(int argc, char** argv){
    coo_hello();
    csr_hello();    
    return 0;
}

