//complete run for each transposition algorithm callled by complete_benchmark in main
#include "../include/benchmarks.cuh"
#include "../include/commons.h"
#include "../include/kernels.cuh"
#include "../include/debug.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <fstream>

int coo_transposition(coo_matrix* coo, int matrix_size) {
    PRINTF("--------------------\n");
    PRINTF("COO Transposition Method called: coo_transposition() -> cuCOOt().\n");
    if ((cudaSetDevice(0)) != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        return 1;
    }
    coo_element* el = coo->el;
    coo_matrix* d_coo;
    coo_element* d_el;

    cudaEvent_t start, stop, startK, stopK;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&startK));
    CHECK_CUDA(cudaEventCreate(&stopK));

    dummy_kernel<<<1,1>>>(); // ? Warm up the GPU

    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUDA(cudaMallocManaged((void**)&d_coo, sizeof(coo_matrix)));
    CHECK_CUDA(cudaMallocManaged((void**)&d_el, coo->nnz * sizeof(coo_element)));
    CHECK_CUDA(cudaMemcpy(d_coo, coo, sizeof(coo_matrix), cudaMemcpyHostToDevice));
    PRINTF("Copied & Allocated Memory Succesfully\n");
    d_coo->el = d_el;

    #ifdef DEBUG
        printf("Pre-Transpose Matrix:\n");
        print_coo_less(coo);
    #endif
    
    CHECK_CUDA(cudaEventRecord(startK));
    for (int i = 0; i < TRANSPOSITIONS; i++) {
        cudaMemcpy(d_el, el, coo->nnz * sizeof(coo_element), cudaMemcpyHostToDevice);
        cuCOOt<<<(coo->nnz + 255) / 256, 256>>>(d_coo->el, d_coo->nnz);
    }
    //cuCOOt<<<coo->nnz,1>>>(d_coo->el, d_coo->nnz);
    CHECK_CUDA(cudaEventRecord(stopK));

    CHECK_CUDA(cudaMemcpy(d_coo, d_coo, sizeof(coo_matrix), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stopK));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaDeviceSynchronize());

    int ret = 0;
    #ifdef DEBUG
        printf("Post-Transpose Matrix:\n");
        print_coo_less(d_coo);
    #endif
    if (is_transpose(coo, d_coo)) {
        PRINTF("Transpose is correct.\n");
    } else {
        ret = -1;
        printf("coo_transposition) Transpose is incorrect.\n");
    }

    float millisecondsK = 0;
    CHECK_CUDA(cudaEventElapsedTime(&millisecondsK, startK, stopK));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    int N = coo->cols; 
    float ogbs = (float)(2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / milliseconds;
    float kgbs = (float)(2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / millisecondsK;
    milliseconds /= (float)TRANSPOSITIONS;
    millisecondsK /= (float)TRANSPOSITIONS;
    PRINTF("Time for executing cuCOOt operation: %f ms\n", milliseconds);
    PRINTF("Operation Throughput in GB/s: %7.2f\n", ogbs);
    PRINTF("Kernel Time: %11.2f ms\n", millisecondsK);
    PRINTF("Kernel Throughput in GB/s: %7.2f\n", kgbs);

    std::ofstream output;
    output.open ("logs/results.log", std::ios::out | std::ios_base::app);
    // algorithm, OpTime, Op-GB/s, KTime, K-GB/s
    output << "COO, " << matrix_size << "x" << matrix_size << ", " <<  milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n"; /* *** */
    output.close();

    CHECK_CUDA(cudaFree(d_coo));
    CHECK_CUDA(cudaFree(d_el));
    PRINTF("Freed Memory Succesfully.\n");
    PRINTF("--------------------\n");
    return ret;
}
int csr_transposition_3(csr_matrix* csr, csr_matrix* csr_t, int matrix_size) {
    cudaEvent_t start, stop;
    cudaEvent_t startK, stopK;
    size_t free_gpu_memory, total_gpu_memory;
    cudaMemGetInfo(&free_gpu_memory, &total_gpu_memory);
    PRINTF("Free GPU Memory: %zu, Total GPU Memory: %zu\n", free_gpu_memory, total_gpu_memory);
    CHECK_CUDA(cudaEventCreate(&start));
    cudaMemGetInfo(&free_gpu_memory, &total_gpu_memory);
    PRINTF("Free GPU Memory: %zu, Total GPU Memory: %zu\n", free_gpu_memory, total_gpu_memory);
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventCreate(&startK));
    CHECK_CUDA(cudaEventCreate(&stopK));
    dummy_kernel<<<1,1>>>(); // ? Warm up the GPU
    CHECK_CUDA(cudaEventRecord(start));

    int *d_row_ptr, *d_col_indices;
    float *d_values;
    CHECK_CUDA(cudaMalloc((void**)&d_row_ptr, (csr->rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_col_indices, csr->nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_values, csr->nnz * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_row_ptr, csr->row_offsets, (csr->rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_indices, csr->col_indices, csr->nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, csr->values, csr->nnz * sizeof(float), cudaMemcpyHostToDevice));
    PRINTF("hello\n");
    int *d_t_row_ptr, *d_t_col_indices;
    float *d_t_values;
    CHECK_CUDA(cudaMalloc((void**)&d_t_row_ptr, (csr->cols + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_t_col_indices, csr->nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_t_values, csr->nnz * sizeof(float)));

    int block_size = 1024;
    int grid_size = std::min((csr->nnz + block_size - 1) / block_size, 1024);

    CHECK_CUDA(cudaEventRecord(startK));
    PRINTF("hello response\n");
    // ? Implement everything in GPU
    int* intra;
    CHECK_CUDA(cudaMalloc((void**)&intra, csr->nnz * sizeof(int)));
    int* inter;
    CHECK_CUDA(cudaMalloc((void**)&inter, csr->cols * sizeof(int)));
    int* RowIdx;
    CHECK_CUDA(cudaMallocManaged((void**)&RowIdx, csr->nnz * sizeof(int)));

    PRINTF("rowidx: %d\n", csr->nnz);

    for(int i = 0; i < TRANSPOSITIONS; i++) {
        // reset values
        CHECK_CUDA(cudaMemset(inter, 0, csr->cols * sizeof(int)));
        // ? Run this on GPU over i = thread_id = threadIdx.x + blockIdx.x * blockDim.x < csr->rows
        getRowIdx<<<1, csr->rows>>>(RowIdx, d_row_ptr, csr->rows, csr->nnz);
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free_gpu_memory, &total_gpu_memory);
        PRINTF("(freed rowidx)Free GPU Memory: %zu, Total GPU Memory: %zu\n", free_gpu_memory, total_gpu_memory);
        
// #ifdef DEBUG
//         
//         //copy back to host RowIdx
//         int* h_RowIdx = new int[csr->nnz];
//         CHECK_CUDA(cudaMemcpy(h_RowIdx, RowIdx, csr->nnz * sizeof(int), cudaMemcpyDeviceToHost));
//         printf("RowIdx: ");
//         for(int i = 0; i < csr->nnz; i++) {
//             printf("%d ", h_RowIdx[i]);
//         }
//         delete[] h_RowIdx;
// #endif
        // ? Run this on GPU over i = thread_id = threadIdx.x + blockIdx.x * blockDim.x < csr->nnz
        //re reversed the logic now that everything is correct
        // getIntraInter<<<grid_size, block_size>>>(intra, inter, csr->nnz, d_col_indices);
        getIntraInter<<<1, 1>>>(intra, inter, csr->nnz, d_col_indices);
        cudaDeviceSynchronize();
        
// #ifdef DEBUG
//         //copy back to host intra and inter
//         
//         int* h_intra = new int[csr->nnz];
//         int* h_inter = new int[csr->cols];
//         CHECK_CUDA(cudaMemcpy(h_intra, intra, csr->nnz * sizeof(int), cudaMemcpyDeviceToHost));
//         CHECK_CUDA(cudaMemcpy(h_inter, inter, csr->cols * sizeof(int), cudaMemcpyDeviceToHost));
//         printf("\nIntra: ");
//         for(int i = 0; i < csr->nnz; i++) {
//             printf("%d ", h_intra[i]);
//         }
//         printf("\nInter: ");
//         for(int i = 0; i < csr->cols; i++) {
//             printf("%d ", h_inter[i]);
//         }
//         delete[] h_intra;
//         delete[] h_inter;
// #endif
        //all in one
        getRowOffsets<<<1, csr->cols>>>(d_t_row_ptr, inter, csr->cols);
        cudaDeviceSynchronize();
        
// #ifdef DEBUG
//         //copy back to host row offsets
//         int* h_t_row_ptr = new int[csr->cols + 1];
//         CHECK_CUDA(cudaMemcpy(h_t_row_ptr, d_t_row_ptr, (csr->cols + 1) * sizeof(int), cudaMemcpyDeviceToHost));
//         printf("\nRow Offsets: ");
//         for(int i = 0; i < csr->cols + 1; i++) {
//             printf("%d ", h_t_row_ptr[i]);
//         }
// #endif
        assignValues<<<grid_size, block_size>>>(d_t_col_indices, d_t_values, d_col_indices, d_values, intra, inter, RowIdx, d_t_row_ptr, csr->nnz);
        cudaDeviceSynchronize();
// #ifdef DEBUG
        
//         //copy back to host col indices and values
//         int* h_t_col_indices = new int[csr->nnz];
//         float* h_t_values = new float[csr->nnz];
//         CHECK_CUDA(cudaMemcpy(h_t_col_indices, d_t_col_indices, csr->nnz * sizeof(int), cudaMemcpyDeviceToHost));
//         CHECK_CUDA(cudaMemcpy(h_t_values, d_t_values, csr->nnz * sizeof(float), cudaMemcpyDeviceToHost));
//         printf("\nCol Indices: ");
//         for(int i = 0; i < csr->nnz; i++) {
//             printf("%d ", h_t_col_indices[i]);
//         }
//         printf("\nValues: ");
//         for(int i = 0; i < csr->nnz; i++) {
//             printf("%f ", h_t_values[i]);
//         }
// #endif

// #ifdef DEBUG
//         delete[] h_t_row_ptr;
//         delete[] h_t_col_indices;
//         delete[] h_t_values;

// #endif 
    }// end of transpositions
    cudaMemGetInfo(&free_gpu_memory, &total_gpu_memory);
    PRINTF("(before copy back) Free GPU Memory: %zu, Total GPU Memory: %zu\n", free_gpu_memory, total_gpu_memory);
    
    CHECK_CUDA(cudaEventRecord(stopK));

    CHECK_CUDA(cudaMemcpy(csr_t->row_offsets, d_t_row_ptr, (csr->cols + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(csr_t->col_indices, d_t_col_indices, csr->nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(csr_t->values, d_t_values, csr->nnz * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stopK));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float milliseconds = 0;
    float millisecondsK = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    CHECK_CUDA(cudaEventElapsedTime(&millisecondsK, startK, stopK));
    int N = csr->cols;
    float ogbs = (float)(2.0 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / milliseconds;
    float kgbs = (float)(2.0 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / millisecondsK;
    milliseconds /= (float)TRANSPOSITIONS;
    millisecondsK /= (float)TRANSPOSITIONS;
    PRINTF("Time for executing cuCOOt operation: %f ms\n", milliseconds);
    PRINTF("Operation Throughput in GB/s: %7.2f\n", ogbs);

    std::ofstream output;
    output.open("logs/results.log", std::ios::out | std::ios_base::app);
    // algorithm, MatSize, OpTime, Op-GB/s, KTime, K-GB/s
    output << "CSR gpu, " << matrix_size << "x" << matrix_size << ", " <<  milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n"; /* *** */
    output.close();

    if(is_transpose(csr, csr_t)) {
        PRINTF("Transpose is correct.\n");
    } else {
        printf("csr_transposition_2) Transpose is incorrect.\n");
        std::ofstream errlogstream;
        errlogstream.open("logs/transpose_err.log", std::ios::out | std::ios::app);
        errlogstream << "Transpose Error: CSR\n";
        errlogstream << "Original Matrix:\n";
        pretty_print_csr_matrix(csr, errlogstream);
        errlogstream << "\n\nTranposed Matrix:\n";
        pretty_print_csr_matrix(csr_t, errlogstream);
        errlogstream.close();
        PRINTF("--------------------\n");
        return -1;
    }

    CHECK_CUDA(cudaFree(inter));
    CHECK_CUDA(cudaFree(intra));
    CHECK_CUDA(cudaFree(RowIdx));

    CHECK_CUDA(cudaFree(d_row_ptr));
    CHECK_CUDA(cudaFree(d_col_indices));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_t_row_ptr));
    CHECK_CUDA(cudaFree(d_t_col_indices));
    CHECK_CUDA(cudaFree(d_t_values));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(startK));
    CHECK_CUDA(cudaEventDestroy(stopK));
    

    cudaMemGetInfo(&free_gpu_memory, &total_gpu_memory);
    // printf("(end of transposition) Free GPU Memory: %zu, Total GPU Memory: %zu\n", free_gpu_memory, total_gpu_memory);
    return 0;
}

int csr_transposition_2(csr_matrix* csr, csr_matrix* csr_t, int matrix_size) {

    csr_matrix* new_csr_t = new_csr_matrix(csr->cols, csr->rows, csr->nnz);
    new_csr_t->row_offsets = new int[csr->cols + 1];
    new_csr_t->col_indices = new int[csr->nnz];
    new_csr_t->values = new float[csr->nnz];
    memset(new_csr_t->row_offsets, 0, (csr->cols + 1) * sizeof(int));
    memset(new_csr_t->col_indices, 0, csr->nnz * sizeof(int));
    memset(new_csr_t->values, 0, csr->nnz * sizeof(float));
    new_csr_t->rows = csr->cols;
    new_csr_t->cols = csr->rows;
    new_csr_t->nnz = csr->nnz;

    cudaEvent_t start, stop;
    cudaEvent_t startK, stopK;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventCreate(&startK));
    CHECK_CUDA(cudaEventCreate(&stopK));
    dummy_kernel<<<1,1>>>(); // ? Warm up the GPU
    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUDA(cudaEventRecord(startK));
    //for(int i = 0; i < TRANSPOSITIONS; i++) {

    // ? Implement everything in CPU for now
    int nthread = 0;
    int nthreads = 1;
    int* intra = new int[csr->nnz];
    int* inter = new int[(nthreads + 1) * csr->cols];
    memset(inter, 0, (nthreads + 1) * csr->cols * sizeof(int));
    int* csrRowIdx = new int[csr->nnz];

    // ? Run this on GPU over i = thread_id = threadIdx.x + blockIdx.x * blockDim.x < csr->rows
    for(int i = 0; i < csr->rows; i++) {
        for(int j = csr->row_offsets[i]; j < csr->row_offsets[i+1]; j++) {
            csrRowIdx[j] = i;
        }
    }
#ifdef DEBUG
    printf("csrRowIdx: ");
    for(int i = 0; i < csr->nnz; i++) {
        printf("%d ", csrRowIdx[i]);
    }
    printf("\n");
#endif

    for(int i = 0 ; i < csr->nnz; i++) {
        intra[i] = inter[(nthread + 1) * csr->col_indices[i]]++;
    }
#ifdef DEBUG
    printf("intra: ");
    for(int i = 0; i < csr->nnz; i++) {
        printf("%d ", intra[i]);
    }
    printf("\n");

    printf("inter: "); //ok
    for(int i = 0; i < csr->cols; i++) {
        //for(int j = 0; j < nthreads + 1; j++) {
            printf("%d ", inter[i// + csr->cols * j
            ]);
        //}
    }
    printf("\n");
#endif
    for(int i = 0; i < csr->cols; i++) { //ok 
        new_csr_t->row_offsets[i + 1] = inter[csr->cols * (nthread) + i];
         new_csr_t->row_offsets[i + 1] += new_csr_t->row_offsets[i];
    }
#ifdef DEBUG
    printf("row_offsets: ");
    for(int i = 0; i < csr->cols + 1; i++) {
        printf("%d ", new_csr_t->row_offsets[i]);
    }
    printf("\n");
    printf("row_offsets after prefix sum: ");
    for(int i = 0; i < csr->cols + 1; i++) {
        printf("%d ", new_csr_t->row_offsets[i]);
    }
    printf("\n");
#endif
    // int *loc_list = new int[csr->nnz];
    // int *loc_list_intra = new int[csr->nnz];
    for(int i = 0; i < csr->nnz; i++) {
        int loc = new_csr_t->row_offsets[csr->col_indices[i]] + inter[nthread * csr->cols + csr->col_indices[i]];
        // loc_list[i] = loc;
        loc = loc - intra[i];
        // loc_list_intra[i] = loc;
        // if (loc < 1) {
        //     printf("loc: %d, i: %d\n", loc, i); // segfault negative index
        //     continue;
        // }
        new_csr_t->col_indices[loc-1] = csrRowIdx[i];
        new_csr_t->values[loc-1] = csr->values[i];
    }
    // printf("Loc_list: \n");
    // for(int i = 0; i < csr->nnz; i++) {
    //     printf("%d ", loc_list[i]);
    // }
    // printf("\n");
    // printf("loc list intra: \n");
    // for(int i = 0; i < csr->nnz; i++) {
    //     printf("%d ", loc_list_intra[i]);
    // }
    // printf("\n");
    // delete[] loc_list;
    // delete[] loc_list_intra;
#ifdef DEBUG
    printf("original: ");
    print_csr_matrix(csr);
    printf("\n");
    printf("data: ");
    print_csr_matrix(new_csr_t);
    printf("\n");
#endif
    delete[] intra;
    delete[] inter;
    delete[] csrRowIdx;

    //}
    CHECK_CUDA(cudaEventRecord(stopK));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stopK));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float milliseconds = 0;
    float millisecondsK = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    CHECK_CUDA(cudaEventElapsedTime(&millisecondsK, startK, stopK));
    int N = csr->cols;
    float ogbs = (float)(2.0 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / milliseconds;
    float kgbs = (float)(2.0 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / millisecondsK;
    milliseconds /= (float)TRANSPOSITIONS;
    millisecondsK /= (float)TRANSPOSITIONS;
    PRINTF("Time for executing cuCOOt operation: %f ms\n", milliseconds);
    PRINTF("Operation Throughput in GB/s: %7.2f\n", ogbs);

    std::ofstream output;
    output.open("logs/results.log", std::ios::out | std::ios_base::app);
    // algorithm, MatSize, OpTime, Op-GB/s, KTime, K-GB/s
    output << "CSR, " << matrix_size << "x" << matrix_size << ", " <<  milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n"; /* *** */
    output.close();

    if(is_transpose(csr, new_csr_t)) {
        PRINTF("Transpose is correct.\n");
    } else {
        printf("csr_transposition_2) Transpose is incorrect.\n");
        std::ofstream errlogstream;
        errlogstream.open("logs/transpose_err.log", std::ios::out | std::ios::app);
        errlogstream << "Transpose Error: CSR\n";
        errlogstream << "Original Matrix:\n";
        pretty_print_csr_matrix(csr, errlogstream);
        errlogstream << "\n\nTranposed Matrix:\n";
        pretty_print_csr_matrix(csr_t, errlogstream);
        errlogstream.close();
        PRINTF("--------------------\n");
        return -1;
    }

    delete_csr(new_csr_t);
    
    return 0;
}

int csr_transposition(csr_matrix* csr, csr_matrix* csr_t, int matrix_size) {
    PRINTF("--------------------\n");
    PRINTF("CSR Transposition Method Called: csr_transposition() -> transposeCSRToCSC().\n");
    if ((cudaSetDevice(0)) != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        return 1;
    }
    assert(csr != NULL && csr_t != NULL);
    assert(csr->rows == csr_t->cols && csr->cols == csr_t->rows);
    PRINTF("Transpose CSR to CSC Cuda Method Called: transposeCSRToCSC_cuda().\n");
    if ((cudaSetDevice(0)) != cudaSuccess) {
        PRINTF("Failed to set CUDA device\n");
        return 1;
    }
    cudaEvent_t start, stop;
    cudaEvent_t startK1, stopK1;
    cudaEvent_t startK2, stopK2;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventCreate(&startK1));
    CHECK_CUDA(cudaEventCreate(&stopK1));

    CHECK_CUDA(cudaEventCreate(&startK2));
    CHECK_CUDA(cudaEventCreate(&stopK2));

    dummy_kernel<<<1,1>>>(); // ? Warm up the GPU

    CHECK_CUDA(cudaEventRecord(start));

    // Copy input CSR data to device
    int *d_col_indices, *d_col_counts;
    CHECK_CUDA(cudaMalloc((void**)&d_col_indices, csr->nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_col_counts, csr->cols * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_col_indices, csr->col_indices, csr->nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_col_counts, 0, csr->cols * sizeof(int)));

    int shared_mem_size = 2*(csr->cols) * sizeof(int); //declare the size of the shared memory 
    int *last = new int[1];
    int *d_last;
    CHECK_CUDA(cudaMalloc((void**)&d_last, sizeof(int)));

    float *d_values, *d_t_values; //ordered values of trasposed matrix
    int *d_t_col_indices;
    int *d_row_offsets, *d_t_row_offsets;
    CHECK_CUDA(cudaMalloc((void**)&d_values, csr->nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_t_values, csr->nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_t_col_indices, csr->nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_row_offsets, (csr->rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_t_row_offsets, (csr->cols + 1) * sizeof(int)));

    int *col_ptr = new int[csr->cols +1];
    int *d_col_ptr;
    CHECK_CUDA(cudaMalloc((void**)&d_col_ptr, (csr->cols) * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_values, csr->values, csr->nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_t_col_indices, 0, csr->nnz * sizeof(int)));

    //int *t_col_indices_ordered = new int[csr->nnz];
    int *d_t_col_indices_ordered;
    CHECK_CUDA(cudaMalloc((void**)&d_t_col_indices_ordered, csr->nnz * sizeof(int)));

    float millisecondsK = 0;
    int * zeroes = new int[csr->cols];
    memset(zeroes, 0, csr->cols * sizeof(int)); //copy slightly better than memset
    for(int i = 0; i < TRANSPOSITIONS; i++) {
        CHECK_CUDA(cudaMemcpy(d_col_counts, zeroes, csr->cols * sizeof(int), cudaMemcpyHostToDevice)); //reset col counts to compute correctly

        float millisecondsK1 = 0;
        CHECK_CUDA(cudaEventRecord(startK1));
        countNNZPerColumn<<<((csr->nnz + 255) / 256), 256>>>(d_col_indices, d_col_counts, csr->nnz);
        prefix_scan<<<1, (csr->cols), shared_mem_size>>>(d_col_ptr, d_col_counts, csr->cols, d_last);
        CHECK_CUDA(cudaEventRecord(stopK1));
        CHECK_CUDA(cudaEventSynchronize(stopK1));
        CHECK_CUDA(cudaEventElapsedTime(&millisecondsK1, startK1, stopK1));
        //printf("csr) countNNZ + prefix_scan: %f ms\n", millisecondsK1);
        cudaCheckError();
        CHECK_CUDA(cudaMemcpy(col_ptr, d_col_ptr, (csr->cols) * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(last, d_last, sizeof(int), cudaMemcpyDeviceToHost));
        // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda : Figure 39-4 
        col_ptr[csr->cols] = last[0];
        csr_t->row_offsets = col_ptr;
        CHECK_CUDA(cudaMemcpy(d_row_offsets, col_ptr, (csr->rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        //compute row_offsets in cpu (inclusive_scan)

        int count = 0;
        for(int i = 0; i < csr->cols; i++) {
            int els = csr->row_offsets[i+1] - csr->row_offsets[i];
            for(int j = 0; j < els; j++) {
                csr_t->col_indices[count] = i; //col indices in crescent order
                count++;
            }
        }
         
        CHECK_CUDA(cudaMemcpy(d_t_col_indices, csr_t->col_indices, csr->nnz * sizeof(int), cudaMemcpyHostToDevice));
        float millisecondsK2 = 0;
        CHECK_CUDA(cudaEventRecord(startK2));
        order_by_column<<<(csr->cols + 15) /16, 16>>>(d_values, d_col_indices, d_t_values, d_col_ptr, d_col_counts, csr->cols, csr->nnz, d_t_col_indices, d_t_col_indices_ordered);
        //order_by_column<<<(csr->cols + 3) /4, 4>>>(d_values, d_col_indices, d_t_values, d_col_ptr, d_col_counts, csr->cols, csr->nnz, d_t_col_indices, d_t_col_indices_ordered);
        CHECK_CUDA(cudaEventRecord(stopK2));
        CHECK_CUDA(cudaEventSynchronize(stopK2));
        CHECK_CUDA(cudaEventElapsedTime(&millisecondsK2, startK2, stopK2));
        //printf("csr) order_by_column: %f ms\n", millisecondsK2);
        millisecondsK += millisecondsK1 + millisecondsK2;
    }
    //return ordered col indices
    CHECK_CUDA(cudaMemcpy(csr_t->col_indices, d_t_col_indices_ordered, csr->nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(csr_t->values, d_t_values, csr->nnz * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stopK1));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    int N = csr->cols;
    float ogbs = (float)(2.0 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / milliseconds; 
    float kgbs = (float)(2.0 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / millisecondsK;
    milliseconds /= (float)TRANSPOSITIONS;
    millisecondsK /= (float)TRANSPOSITIONS;
    PRINTF("Time for executing transpose operation: %f ms\n", milliseconds);
    PRINTF("Operation Time: %11.2f ms\n", milliseconds);
    PRINTF("Operation Throughput in GB/s: %7.2f\n", ogbs);
    PRINTF("Kernel Time: %11.2f ms\n", millisecondsK);
    PRINTF("Kernel Throughput in GB/s: %7.2f\n", kgbs);
    std::ofstream output;
    output.open ("logs/results.log", std::ios::out | std::ios_base::app);
    // algorithm, OpTime, Op-GB/s, KTime, K-GB/s
    output << "CSRtoCSCcuda, " << matrix_size << "x" << matrix_size << ", " << milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n"; /* *** */
    output.close();
    cudaCheckError();

    CHECK_CUDA(cudaFree(d_col_indices));
    CHECK_CUDA(cudaFree(d_col_counts));
    CHECK_CUDA(cudaFree(d_col_ptr));
    CHECK_CUDA(cudaFree(d_last));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_t_values));
    CHECK_CUDA(cudaFree(d_t_col_indices));
    CHECK_CUDA(cudaFree(d_row_offsets));
    CHECK_CUDA(cudaFree(d_t_row_offsets));
    CHECK_CUDA(cudaFree(d_t_col_indices_ordered));

    delete last;
    delete[] zeroes;

    PRINTF("Transpose Completed.\n");

    if (is_transpose(csr, csr_t)) {
        PRINTF("Transpose is correct.\n");
    } else {
        printf("csr_transposition) Transpose is incorrect.\n");
        std::ofstream errlogstream;
        errlogstream.open("logs/transpose_err.log", std::ios::out | std::ios::app);
        errlogstream << "Transpose Error: CSR to CSC\n";
        errlogstream << "Original Matrix:\n";
        pretty_print_csr_matrix(csr, errlogstream);
        errlogstream << "\n\nTranposed Matrix:\n";
        pretty_print_csr_matrix(csr_t, errlogstream);
        errlogstream.close();
        PRINTF("--------------------\n");
        return -1;
    }
    PRINTF("--------------------\n");
    return 0;
}

int block_trasposition(float* mat, unsigned int N, int matrix_size) {
    PRINTF("--------------------\n");
    PRINTF("Block Transposition Method Called: block_transposition() -> block_transpose().\n");
    if ((cudaSetDevice(0)) != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        return 1;
    }
    cudaEvent_t start, stop;
    cudaEvent_t startK, stopK;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventCreate(&startK));
    CHECK_CUDA(cudaEventCreate(&stopK));

    dummy_kernel<<<1,1>>>(); // ? Warm up the GPU

    CHECK_CUDA(cudaEventRecord(start));

    int mem_size = N * N * sizeof(float);
    float* mat_t = (float*) malloc(mem_size);
    memset(mat_t, 0, mem_size);
    initMatrix(mat, N);
    float *d_mat, *d_mat_t;
    //int threads, blocks = 0;
    PRINTF("Allocating memory.\n");
    CHECK_CUDA(cudaMalloc((void**)&d_mat, mem_size));
    CHECK_CUDA(cudaMalloc((void**)&d_mat_t, mem_size));
    //cudaMalloc((void**)&d_mat_t, mem_size);
    PRINTF("Memory allocated.\n");
    //copy data to gpu

    PRINTF("Data copied.\n");
    //setup grid and block size
    dim3 DimGrid = {N/TILE_SIZE, N/TILE_SIZE, 1};
    dim3 DimBlock = {TILE_SIZE, BLOCK_ROWS, 1};

    CHECK_CUDA(cudaMemcpy(d_mat, mat, mem_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(startK));
    for(int i = 0; i < TRANSPOSITIONS; i++){
        block_transpose<<<DimGrid, DimBlock>>>(d_mat, d_mat_t, N);
    }
    CHECK_CUDA(cudaEventRecord(stopK));
    PRINTF("Kernel returned\n");

    //copy data back
    CHECK_CUDA(cudaMemcpy(mat_t, d_mat_t, mem_size, cudaMemcpyDeviceToHost));
    //sync
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stopK));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaDeviceSynchronize());

    float millisecondsK = 0;
    CHECK_CUDA(cudaEventElapsedTime(&millisecondsK, startK, stopK));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float ogbs = (float)(2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / milliseconds;
    float kgbs = (float)(2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / millisecondsK;
    milliseconds /= (float)TRANSPOSITIONS;
    millisecondsK /= (float)TRANSPOSITIONS;
    PRINTF("Operation Time: %11.2f ms\n", milliseconds);
    PRINTF("Operation Throughput in GB/s: %7.2f\n", ogbs);
    PRINTF("Kernel Time: %11.2f ms\n", millisecondsK);
    PRINTF("Kernel Throughput in GB/s: %7.2f\n", kgbs);

    std::ofstream output;
    output.open("logs/results.log", std::ios::out | std::ios_base::app);

    // algorithm, OpTime, Op-GB/s, KTime, K-GB/s
    output << "block, " << matrix_size << "x" << matrix_size << ", " << milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n";
    output.close();

    CHECK_CUDA(cudaEventDestroy(startK));
    CHECK_CUDA(cudaEventDestroy(stopK));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    //test if the matrix is transposed
    int ret = 0;
    if (testTranspose(mat, mat_t, N) != 0) {
        printf("block_transpose) Transpose is incorrect.\n");
        // save the matrix to a file
        std::ofstream output;
        output.open("logs/block_transpose_err.log", std::ios::out | std::ios_base::app);
        output << "Matrix: " << N << "x" << N << "\n";
        output << "Original Matrix:\n";
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                output << mat[i + j*N] << " ";
            }
            output << "\n";
        }
        output << "\n\nTransposed Matrix:\n";
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                output << mat_t[i + j*N] << " ";
            }
            output << "\n";
        }
        output.close();
        ret = -1;
    }
    PRINTF("--------------------\n");
    //free gpu resources
    CHECK_CUDA(cudaFree(d_mat));
    CHECK_CUDA(cudaFree(d_mat_t));
    free(mat_t);
    return ret;
}

int conflict_transposition(float* mat, unsigned int N, int matrix_size) {
    PRINTF("--------------------\n");
    PRINTF("Conflict Transposition Method Called: conflict_transposition() -> conflict_transpose().\n");
    if ((cudaSetDevice(0)) != cudaSuccess) {
        printf("Failed to set CUDA device\n");
        return 1;
    }
    cudaEvent_t start, stop;
    cudaEvent_t startK, stopK;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventCreate(&startK));
    CHECK_CUDA(cudaEventCreate(&stopK));

    dummy_kernel<<<1,1>>>(); // ? Warm up the GPU

    CHECK_CUDA(cudaEventRecord(start));
    PRINTF("Cuda Events Created.\n");

    int mem_size = N * N * sizeof(float);
    float* mat_t = (float*) malloc(mem_size);
    memset(mat_t, 0, mem_size);
    float *d_mat, *d_mat_t;
    //int threads, blocks = 0;
    PRINTF("Allocating memory.\n");
    CHECK_CUDA(cudaMalloc((void**)&d_mat, mem_size));
    CHECK_CUDA(cudaMalloc((void**)&d_mat_t, mem_size));
    PRINTF("Memory allocated.\n");
    PRINTF("Now copying data from host (mat) to device (d_mat). Exactly: %d Bytes\n", mem_size);
    PRINTF("Data copied.\n");
    //setup grid and block size
    dim3 DimGrid = {N/TILE_SIZE, N/TILE_SIZE, 1};
    dim3 DimBlock = {TILE_SIZE, BLOCK_ROWS, 1};
    
    CHECK_CUDA(cudaMemcpy(d_mat, mat, mem_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(startK));
    for(int i = 0; i < TRANSPOSITIONS; i++){
        conflict_transpose<<<DimGrid, DimBlock>>>(d_mat, d_mat_t);
    }
    CHECK_CUDA(cudaEventRecord(stopK));
    PRINTF("Kernel returned\n");

    //copy data back
    CHECK_CUDA(cudaMemcpy(mat_t, d_mat_t, mem_size, cudaMemcpyDeviceToHost));
    //sync
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stopK));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaDeviceSynchronize());
    
    float millisecondsK = 0;
    CHECK_CUDA(cudaEventElapsedTime(&millisecondsK, startK, stopK));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    PRINTF("Operation Time: %11.2f ms\n", milliseconds);
    float ogbs = (float)(2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / milliseconds;
    float kgbs = (float)(2 * N * N * sizeof(float) * 1e-6 * TRANSPOSITIONS) / millisecondsK;
    milliseconds /= (float)TRANSPOSITIONS;
    millisecondsK /= (float)TRANSPOSITIONS;
    PRINTF("Throughput in GB/s: %7.2f\n", ogbs);
    PRINTF("Kernel Time: %11.2f ms\n", millisecondsK);
    PRINTF("Throughput in GB/s: %7.2f\n", kgbs);

    std::ofstream output;
    output.open("logs/results.log", std::ios::out | std::ios_base::app);
    // algorithm, OpTime, Op-GB/s, KTime, K-GB/s
    output << "Conflict, " << matrix_size << "x" << matrix_size << ", " <<  milliseconds << ", "<< ogbs << ", " << millisecondsK << ", " << kgbs << "\n";
    output.close();

    CHECK_CUDA(cudaEventDestroy(startK));
    CHECK_CUDA(cudaEventDestroy(stopK));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    //test if the matrix is transposed
    int ret =  testTranspose(mat, mat_t, N);
    if (ret != 0) {
        printf("conflict_transpose) Transpose is incorrect.\n");
    }
    PRINTF("--------------------\n");
    //free gpu resources
    CHECK_CUDA(cudaFree(d_mat));
    CHECK_CUDA(cudaFree(d_mat_t));
    free(mat_t);
    return ret;
}
