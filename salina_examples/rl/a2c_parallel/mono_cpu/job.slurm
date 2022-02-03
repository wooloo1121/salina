#!/bin/bash --login
#$ -cwd
#$ -pe smp.pe 4

module load libs/cuda/11.2.0
module load libs/nccl/2.8.3

conda activate test

python /mnt/iusers01/fatpou01/compsci01/q12038kw/scratch/salina/salina_examples/rl/a2c/mono_cpu/main.py 
