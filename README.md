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
make setup
make main
sbatch sbatch.sh
```

#### summing up row pointers

1 2 0 1 3 => 1 3 3 4 7
in gpu??
    1 | 2 0 1 3 _
    1    
        3
            1
                3
second iteration      
    1  3 | 0 1 3 3 _
    1  3
           3
               4
                   3
third iteration
    1  3  3 | 1 3 3 4 _
    1  3  3
              1
                  3
                      4

      
      