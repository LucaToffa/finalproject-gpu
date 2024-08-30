# Matrix transposition in CUDA
## Final project - GPU computing course 

for the project on sparse computation please select two different data-structure to store the sparse matrix. 
The possibilities are
Compressed sparse row (CSR).
Coordinate format (COO).

Blocked compressed sparse row (BCSR)

Please select 10 matrices with different degree and pattern of sparsity .
The matrices must be no symmetric! 



This project requires to design an efficient algorithm to transpose a sparse matrix. Specifically the
matrix should be highly sparse, namely the number of zero element is more that 75% of the whole
(n ×n) elements. The implementation should emphasize:
•storage format for storing sparse matrices (for example, compressed sparse row);
•the implementation to perform the transposition;
•a comparison against vendors’ library (e.g., cuSPARSE);
•dataset for the benchmark (compare all the implementation presented by selecting at least 10
matrices from suite sparse matrix collection https://sparse.tamu.edu/);
As usual, the metric to consider is the effective bandwidth



### useful sites
https://sparse.tamu.edu/?per_page=All (The dataset to use)

https://docs.nvidia.com/cuda/cusparse/ (cuSPARSE documentation)


### running on cluster
```bash
module load cuda/12.1
make clean
make all
sbatch sbatch.sh
```

#### csr data organization
```c
//get the data in this format : (val, col, row)
//order by col 
//1 thread per col, .append if col == thdx
//join the threads in order
//ordered values = loop list[i][0] (d_t_values)  /* *** */
// -> call order_by_column, result in d_t_values
//row_ptr = loop list[i][2] (d_t_row_offsets)  /* *** */
// -> call order_by_column, result in d_t_row_offsets
//return csr_t with /* *** */ values
```