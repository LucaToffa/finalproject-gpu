#!/bin/bash
#SBATCH --job-name=transpose
#SBATCH --output=output-%j.out
#SBATCH --error=output-%j.err
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

module load cuda/12.1
make main
srun ./main