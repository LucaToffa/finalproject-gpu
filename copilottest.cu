#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

// CUDA error checking macro
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                \
    if(e!=cudaSuccess) {                                             \
        printf("CUDA Error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
}
//we will use this everywhere
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

// Kernel to count the number of non-zero entries per column
__global__ void countNNZPerColumn(const int* col_indices, int* col_counts, int nnz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nnz) {
        atomicAdd(&col_counts[col_indices[tid]], 1);
    }
}

// Kernel to scatter values and row indices to transposed matrix
__global__ void scatterToTransposed(const int* values, const int* col_indices, const int* row_ptr,
                                    int* t_values, int* t_row_indices, int* t_col_ptr, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
            int col = col_indices[j];
            int dest = atomicAdd(&t_col_ptr[col], 1);
            t_values[dest] = values[j];
            t_row_indices[dest] = row;
        }
    }
}

int transposeCSRToCSC(const thrust::host_vector<int>& h_values, const thrust::host_vector<int>& h_col_indices,
                       const thrust::host_vector<int>& h_row_ptr, int num_rows, int num_cols,
                       thrust::device_vector<int>& d_t_values, thrust::device_vector<int>& d_t_row_indices,
                       thrust::device_vector<int>& d_t_col_ptr) {
    int nnz = h_values.size();

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Copy input CSR data to device
    thrust::device_vector<int> d_values = h_values;
    thrust::device_vector<int> d_col_indices = h_col_indices;
    thrust::device_vector<int> d_row_ptr = h_row_ptr;
    
    // Initialize device vectors for transposed matrix
    d_t_values.resize(nnz);
    d_t_row_indices.resize(nnz);
    d_t_col_ptr.resize(num_cols + 1);

    CHECK_CUDA(cudaEventRecord(start));
    // Device vector for column counts
    thrust::device_vector<int> d_col_counts(num_cols, 0);
    
    // Kernel to count non-zero entries per column
    countNNZPerColumn<<<(nnz + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_col_indices.data()),
                                                  thrust::raw_pointer_cast(d_col_counts.data()), nnz);
    cudaCheckError();

    // Compute column pointers using exclusive scan
    thrust::exclusive_scan(d_col_counts.begin(), d_col_counts.end(), d_t_col_ptr.begin());
    cudaCheckError();

    // Copy the column pointers to create a correct offset for scattering
    thrust::device_vector<int> d_col_ptr_copy = d_t_col_ptr;

    // Kernel to scatter values and row indices to transposed matrix
    scatterToTransposed<<<(num_rows + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_values.data()),
                                                         thrust::raw_pointer_cast(d_col_indices.data()),
                                                         thrust::raw_pointer_cast(d_row_ptr.data()),
                                                         thrust::raw_pointer_cast(d_t_values.data()),
                                                         thrust::raw_pointer_cast(d_t_row_indices.data()),
                                                         thrust::raw_pointer_cast(d_col_ptr_copy.data()), num_rows);
    CHECK_CUDA(cudaEventRecord(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time for executing cuSPARSECSRt operation: %f ms\n", milliseconds);
    cudaCheckError();
    return 0;
}

int pretty_print_matrix(const thrust::host_vector<int>& values, const thrust::host_vector<int>& row_indices,
                        const thrust::host_vector<int>& col_ptr, int num_rows, int num_cols) {
    printf("Values: ");
    for (int v : values) printf("%d ", v);
    printf("\n");

    printf("Row Indices: ");
    for (int ri : row_indices) printf("%d ", ri);
    printf("\n");

    printf("Column Pointers: ");
    for (int cp : col_ptr) printf("%d ", cp);
    printf("\n");

    printf("Matrix:\n");

    printf("NumCols: %d\n", num_cols);
    printf("NumRows: %d\n", num_rows);

    for (int i = 0; i < num_cols; ++i) {
        for (int j = col_ptr[i]; j < col_ptr[i + 1]; ++j) {
            printf("(%d, %d, %d)\n", row_indices[j], i, values[j]);
        }
    }

    return 0;
}

int main() {
    // Example CSR matrix
    thrust::host_vector<int> h_values(8);// = {10, 20, 30, 40, 50, 60, 70, 80};
    h_values[0] = 10; h_values[1] = 20; h_values[2] = 30; h_values[3] = 40; //TEMP just to avoid compiler error
    h_values[4] = 50; h_values[5] = 60; h_values[6] = 70; h_values[7] = 80;
    thrust::host_vector<int> h_col_indices(8);// = {0, 2, 1, 0, 1, 2, 0, 1};
    h_col_indices[0] = 0; h_col_indices[1] = 2; h_col_indices[2] = 1; h_col_indices[3] = 0;
    h_col_indices[4] = 1; h_col_indices[5] = 2; h_col_indices[6] = 0; h_col_indices[7] = 1;
    thrust::host_vector<int> h_row_ptr(5);// = {0, 2, 4, 7, 8};
    h_row_ptr[0] = 0; h_row_ptr[1] = 2; h_row_ptr[2] = 4; h_row_ptr[3] = 7; h_row_ptr[4] = 8;
    int num_rows = 4;
    int num_cols = 3;

    // Device vectors for transposed matrix
    thrust::device_vector<int> d_t_values;
    thrust::device_vector<int> d_t_row_indices;
    thrust::device_vector<int> d_t_col_ptr;

    // Transpose the matrix
    transposeCSRToCSC(h_values, h_col_indices, h_row_ptr, num_rows, num_cols, d_t_values, d_t_row_indices, d_t_col_ptr);

    // Copy the results back to the host and print
    thrust::host_vector<int> h_t_values = d_t_values;
    thrust::host_vector<int> h_t_row_indices = d_t_row_indices;
    thrust::host_vector<int> h_t_col_ptr = d_t_col_ptr;

    printf("Original Matrix:\n");
    pretty_print_matrix(h_values, h_col_indices, h_row_ptr, num_rows, num_cols);

    printf("Transposed Matrix:\n");
    pretty_print_matrix(h_t_values, h_t_row_indices, h_t_col_ptr, num_rows, num_cols); // ! error invert num_rows and num_cols

    return 0;
}
