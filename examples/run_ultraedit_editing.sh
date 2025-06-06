#!/bin/bash

#SBATCH --partition=i64m1tga40u

#SBATCH --nodes=1

#SBATCH --gres=gpu:1

# unset CUDA_VISIBLE_DEVICES


python run_ultraedit_editing.py \
    --editing_method=UltraEdit \
    --hparams_dir=../hparams/ULTRAEDIT/gpt-j-6B.yaml \
    --data_dir=../data/ultraedit \
    --ds_size=20000 \
    --batch_size=100 \
    --data_type=zsre \
    --sequential_edit

# python run_ultraedit_editing.py \
#     --editing_method=UltraEdit \
#     --hparams_dir=../hparams/ULTRAEDIT/gpt-j-6B.yaml \
#     --data_dir=../data/ultraedit \
#     --ds_size=10 \
#     --batch_size=10 \
#     --data_type=zsre \
#     --sequential_edit