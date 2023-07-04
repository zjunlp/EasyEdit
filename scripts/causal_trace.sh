#!/bin/bash

# Start from parent directory of script
SCRIPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
cd "$(dirname ${SCRIPT_DIR})"

python -m experiments.causal_trace --model_name "EleutherAI/gpt-j-6B" --noise_level 0.025
python -m experiments.causal_trace --model_name "gpt2-xl" --noise_level 0.1
python -m experiments.causal_trace --model_name "EleutherAI/gpt-neox-20b" --noise_level 0.03
