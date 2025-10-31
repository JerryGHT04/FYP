#!/bin/bash
#PBS -lwalltime=72:00:00
#PBS -l select=1:ncpus=8:mem=24gb:ngpus=1:gpu_type=A100
module restore warpxGCC13_3
export AMREX_CUDA_ARCH=8.0 #optimized for A100
cd $PBS_O_WORKDIR

source warpx_env/bin/activate
cd 2DBenchmark_Final

python3 -u run_2D_Benchmark_final.py
