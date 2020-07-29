#!/bin/bash
#SBATCH -c 12
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py3-teaser
OMP_NUM_THREADS=12 python ./eval_teaser_threedmatch.py 

