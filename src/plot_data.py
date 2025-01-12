import matplotlib.pyplot as plt
import pandas as pd
import math
# default log file
RESULTS_FILE = 'logs/results.log'
#else get the file from the arguments
import sys
if len(sys.argv) > 1:
    RESULTS_FILE = sys.argv[1]
    if len(sys.argv) > 2 and (sys.argv[2] == '-u' or sys.argv[2] == '--ungrouped'):
        grouped = False
    else:
        grouped = True

df = pd.read_csv(RESULTS_FILE, comment='#', names=['algorithm', 'MatSize', 'OpTime', 'Op-GB/s', 'KTime', 'K-GB/s'])

df.columns = df.columns.str.strip()

''' print(df) output:
       algorithm     MatSize     OpTime   Op-GB/s      KTime     K-GB/s
0       CUsparse     512x512   17.26990  12.14340   17.10370   12.26140
1            COO     512x512   10.46880  20.03230    9.92246   21.13540
2   CSRtoCSCcuda     512x512   32.08280   6.53669   25.81890    8.12253
3          block     512x512    5.23597  40.05280    1.01178  207.27400
4       Conflict     512x512    2.85469  73.46340    2.11789   99.02090
..           ...         ...        ...       ...        ...        ...
60      CUsparse   1024x1024   11.07600  75.73700   10.91410   76.86000
61           COO   1024x1024   11.20780  74.84590   11.01310   76.16970
62  CSRtoCSCcuda   1024x1024  210.11600   3.99238  201.42600    4.16462
63         block   1024x1024   23.07360  36.35590    6.03792  138.93200
64      Conflict   1024x1024    9.02467  92.95200    6.93286  120.99800
'''

# ? Write here the plotting function...
def plot_gpu_performance(df):
    
    """
    Create two plots to visualize GPU performance:
    1. Throughput (GB/s) by Matrix Size for different algorithms
    2. Execution Time by Matrix Size for different algorithms
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # Get the unique matrices sizes and group by them
    mat_sizes = df['MatSize'].unique()
    mat_sizes = sorted(mat_sizes, key=lambda x: int(x.split('x')[0]))
    df['MatSize'] = pd.Categorical(df['MatSize'], categories=mat_sizes, ordered=True)
    df = df.sort_values('MatSize')

    # ? We will have multiple values for each matrix size, take the mean of them and plot that
    # ? Also take the max and min values to show the range of values

    if (grouped):
        df = df.groupby(['algorithm', 'MatSize']).agg({'OpTime': 'mean', 'Op-GB/s': 'mean'}).reset_index()
    
        # Plot 1: Throughput (GB/s) by Matrix Size for different algorithms
        for algo, group in df.groupby('algorithm'):
            ax1.plot(group['MatSize'], group['Op-GB/s'], marker='o', label=algo)
        ax1.set_title('Throughput (GB/s) by Matrix Size')
        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('Throughput (GB/s)')
        ax1.legend()

        # Plot 2: Execution Time by Matrix Size for different algorithms
        # ? Make it log scale on y axis
        for algo, group in df.groupby('algorithm'):
            ax2.plot(group['MatSize'], group['OpTime'], marker='o', label=algo)
        ax2.set_title('Execution Time by Matrix Size')
        ax2.set_xlabel('Matrix Size')
        ax2.set_ylabel('Execution Time (ms)')
        ax2.set_yscale('log')
        ax2.legend()
        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()
    else:
        num_algos = 5
        df = df.reset_index()
        df['idx'] = df.index / num_algos
        for i in range(df.shape[0]):
            df['idx'][i] = math.floor(df['idx'][i])

        # Plot 1: Throughput (GB/s) by Matrix Size for different algorithms
        for algo, group in df.groupby('algorithm'):
            ax1.plot(group['idx'], group['Op-GB/s'], marker='o', label=algo)
        ax1.set_title('Throughput (GB/s) by Matrix Size')
        ax1.set_xlabel('Matrix')
        ax1.set_ylabel('Throughput (GB/s)')
        ax1.legend()

        # Plot 2: Execution Time by Matrix Size for different algorithms
        # ? Make it log scale on y axis
        for algo, group in df.groupby('algorithm'):
            ax2.plot(group['idx'], group['OpTime'], marker='o', label=algo)
        ax2.set_title('Execution Time by Matrix Size')
        ax2.set_xlabel('Matrix')
        ax2.set_ylabel('Execution Time (ms)')
        ax2.set_yscale('log')
        ax2.legend()
        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()        

plot_gpu_performance(df)
