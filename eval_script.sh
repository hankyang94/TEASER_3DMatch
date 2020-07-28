#!/bin/bash
#SBATCH -c 16
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py3-teaser
python ./eval_teaser_threedmatch.py 

