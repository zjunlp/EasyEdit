#!/bin/bash
##########################################
################# PRISM ##################
##########################################

# This script provides examples for running experiments

# Set default values
DEVICE=${DEVICE:-"cuda:0"}

##########################################
############### AxBench ##################
##########################################

############# Steer Examples #############
# RePS Method - Vector for Powerseeking
# python run_PRISM.py \
#     --dataset axbench \
#     --method all \
#     --model_name gemma-2-9b-it \
#     --intervention_method all \
#     --mode both \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# SFT Method - LoRA for Powerseeking
# python run_PRISM.py \
#     --dataset axbench \
#     --method all \
#     --model_name qwen2.5-7b-instruct \
#     --intervention_method all \
#     --mode both \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

##########################################
############### Psychopathy ##############
##########################################

############# Steer Examples #############
# RePS Method - Vector for Powerseeking
# python run_PRISM.py \
#     --dataset psychopathy \
#     --method all \
#     --model_name gemma-2-9b-it \
#     --intervention_method all \
#     --mode both \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# SFT Method - LoRA for Powerseeking
# python run_PRISM.py \
#     --dataset psychopathy \
#     --method all \
#     --model_name qwen2.5-7b-instruct \
#     --intervention_method all \
#     --mode both \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

##########################################
############## Powerseeking ##############
##########################################

############# Steer Examples #############
# RePS Method - Vector for Powerseeking
# python run_PRISM.py \
#     --dataset powerseeking \
#     --method all \
#     --model_name gemma-2-9b-it \
#     --intervention_method all \
#     --mode both \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# SFT Method - LoRA for Powerseeking
# python run_PRISM.py \
#     --dataset powerseeking \
#     --method all \
#     --model_name qwen2.5-7b-instruct \
#     --intervention_method all \
#     --mode both \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

##########################################
############## Quick Start ###############
##########################################

# Uncomment and modify the following line to run yprism experiment:

python run_PRISM.py \
    --dataset psychopathy \
    --method prism \
    --model_name gemma-2-9b-it \
    --intervention_method local_weight \
    --mode generate \
    --multipliers 1.0 \
    --device cuda:0 \
    --base_dir .
