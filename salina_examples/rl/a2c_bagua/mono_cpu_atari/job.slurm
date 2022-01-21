#!/bin/bash --login
#$ -cwd
#$ -l nvidia_v100=2
#$ -pe smp.pe 8

module load libs/cuda/11.2.0
module load libs/nccl/2.8.3

conda activate test
export OMP_NUM_THREADS=$NSLOTS

python -m bagua.distributed.launch --nproc_per_node=2 /mnt/iusers01/fatpou01/compsci01/q12038kw/scratch/salina/salina_examples/rl/a2c_bagua/mono_cpu_atari/main.py 
