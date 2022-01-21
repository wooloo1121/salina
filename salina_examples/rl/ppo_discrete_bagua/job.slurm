#!/bin/bash --login
#$ -cwd
#$ -l nvidia_v100=2
#$ -pe smp.pe 4

module load libs/cuda/11.2.0
module load libs/nccl/2.8.3

conda activate test

python -m bagua.distributed.launch --nproc_per_node=2 /mnt/iusers01/fatpou01/compsci01/q12038kw/scratch/salina/salina_examples/rl/ppo_discrete_bagua/ppo.py 
