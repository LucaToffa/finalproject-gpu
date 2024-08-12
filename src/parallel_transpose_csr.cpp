#include <memory>
#include <vector>
#include <Openmp/omp-tools.h>
#include <omp.h>
#include <numeric>

template <typename SIZE, typename R, typename C, typename V>
auto ParallelTranspose(const SIZE rows, const SIZE cols, const SIZE nnz,
                      const SIZE base, const R &ai, const C &aj, const V &av) {
  using ROWTYPE = typename std::decay<decltype(ai[0])>::type;
  using COLTYPE = typename std::decay<decltype(aj[0])>::type;
  using VALTYPE = typename std::decay<decltype(av[0])>::type;
  const SIZE cols_transpose = rows;
  const SIZE rows_transpose = cols;
  std::shared_ptr<ROWTYPE[]> ai_transpose(new ROWTYPE[rows_transpose + 1]);
  std::shared_ptr<COLTYPE[]> aj_transpose(new COLTYPE[nnz]);
  std::shared_ptr<VALTYPE[]> av_transpose(new VALTYPE[nnz]);

  ai_transpose[0] = base;

  std::vector<std::unique_ptr<ROWTYPE[]>> threadPrefixSum(
      omp_get_max_threads());

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();

    threadPrefixSum[tid].reset(new ROWTYPE[rows_transpose]());

#pragma omp for
    for (SIZE i = 0; i < rows; i++) {
      for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
        threadPrefixSum[tid][aj[j] - base]++;
      }
    }

#pragma omp barrier
#pragma omp for
    for (SIZE rowID = 0; rowID < rows_transpose; rowID++) {
      ai_transpose[rowID + 1] = 0;
      for (int t = 0; t < nthreads; t++) {
        ai_transpose[rowID + 1] += threadPrefixSum[t][rowID];
      }
    }

#pragma omp barrier

// may be optimized by a parallel scan
#pragma omp master
    {

      std::inclusive_scan(ai_transpose.get(),
                          ai_transpose.get() + rows_transpose + 1,
                          ai_transpose.get());
    }

#pragma omp barrier
#pragma omp for
    for (SIZE rowID = 0; rowID < rows_transpose; rowID++) {
      ROWTYPE tmp = threadPrefixSum[0][rowID];
      threadPrefixSum[0][rowID] = ai_transpose[rowID];
      for (int t = 1; t < nthreads; t++) {
        std::swap(threadPrefixSum[t][rowID], tmp);
        threadPrefixSum[t][rowID] += threadPrefixSum[t - 1][rowID];
      }
    }

#pragma omp barrier

#pragma omp for
    for (SIZE i = 0; i < rows; i++) {
      for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
        const COLTYPE idx = threadPrefixSum[tid][aj[j] - base]++ - base;
        aj_transpose[idx] = i + base;
        av_transpose[idx] = av[j];
      }
    }
  }
  return std::make_tuple(ai_transpose, aj_transpose, av_transpose);
}
