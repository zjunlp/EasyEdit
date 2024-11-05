#!/bin/sh

#SBATCH --job-name=mend_zsre
#SBATCH --out="la_scripts/log_test_mend0.txt"
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:tesla_a100_80g:1
#SBATCH --cpus-per-gpu=4

conda init
conda activate myenv397
source /home/smg/v-lauraom/venvs/EasyEdit/bin/activate 

python la_scripts/run_mend.py
