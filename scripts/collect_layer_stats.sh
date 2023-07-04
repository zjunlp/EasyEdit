#!/bin/bash

# Start from parent directory of script
SCRIPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
cd "$(dirname ${SCRIPT_DIR})"

run_gpu_0() {
  CUDA_VISIBLE_DEVICES=0 python -m rome.layer_stats --layers=$(seq -s, 0 1 27) --sample_size 100000 --model_name=EleutherAI/gpt-j-6B
}

run_gpu_1() {
  CUDA_VISIBLE_DEVICES=1 python -m rome.layer_stats --layers=$(seq -s, 0 1 27) --sample_size 100000 --model_name=EleutherAI/gpt-j-6B
}

# run_gpu_0 &>stats0.out&
run_gpu_1 &>stats1.out&
