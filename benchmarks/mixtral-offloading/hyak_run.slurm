#!/bin/bash

#SBATCH --job-name=mixtral-offload-rtx6k
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yilegu@cs.washington.edu 

#SBATCH --account=ml4ml
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=100
#SBATCH --gpus=1
#SBATCH --time=12:00:00

#SBATCH --chdir=/mmfs1/home/yilegu/fiddler/benchmarks/mixtral-offloading
#SBATCH --export=all
#SBATCH --output=feb_7_mixtral_offload_rtx6k_stdout
#SBATCH --error=feb_7_mixtral_offload_rtx6k_stderr

# Your programs to run.
./run.sh