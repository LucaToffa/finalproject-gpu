# Matrix transposition in CUDA
## Final project - GPU computing course 

```
for the project on sparse computation please select **two** different data-structure to store the sparse matrix. 

The possibilities are:

Compressed sparse row (CSR).
Coordinate format (COO).
Blocked compressed sparse row (BCSR)

Please select 10 matrices with different degree and pattern of sparsity .
The matrices must be no symmetric! 
```

This project requires to design an efficient algorithm to transpose a sparse matrix. Specifically the
matrix should be highly sparse, namely the number of zero element is more that 75% of the whole
(n ×n) elements. The implementation should emphasize:

* storage format for storing sparse matrices (for example, compressed sparse row);

* the implementation to perform the transposition;

* a comparison against vendors’ library (e.g., cuSPARSE);

As usual, the metric to consider is the effective bandwidth

#### dependencies:
- python3, mathplotlib, numpy, pandas
- nvidia-cuda-toolkit
- g++

### running on cluster

```bash
module load cuda/12.1
make clean
make setup
make main
sbatch sbatch.sh
```

### data visualization

The performance of the different algorithms can be visualized by running the following commands:

```bash
python3 src/plot_data.py logs/results.log # data grouped by matrix size, approximated to nearest power of 2
python3 src/plot_data.py logs/results.log -u # ungrouped data for each matrix
```
  
### useful sites and resources
- https://sparse.tamu.edu/?per_page=All (The dataset to use for the benchmark)
- https://docs.nvidia.com/cuda/cusparse/ (cuSPARSE documentation)
- https://docs.nvidia.com/cuda/cusparse/index.html#coordinate-coo (COO Format Definition by CUDA)
- https://docs.nvidia.com/cuda/cusparse/index.html#compressed-sparse-row-csr (CSR Format Definition by CUDA)
- https://sparse.tamu.edu/?per_page=All (Dataset for our benchmark)
- https://docs.nvidia.com/cuda/cusparse/ (CuSPARSE documentation)
- https://developer.download.nvidia.com/compute/DevZone/C/html_x64/6_Advanced/transpose/doc/MatrixTranspose.pdf (Block Transpose implementation by CUDA Developers)
- https://dl.acm.org/doi/pdf/10.1145/2925426.2926291 (Parallel Transposition of Sparse Data Structures)