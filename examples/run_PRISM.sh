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

### Steer Generation Examples ###

# CAA Method (only supports vector intervention)
# python run_PRISM.py \
#     --dataset axbench \
#     --method caa \
#     --model_name gemma-2-9b-it \
#     --intervention_method vector \
#     --mode generate \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# RePS Method - Vector
# python run_PRISM.py \
#     --dataset axbench \
#     --method reps \
#     --model_name gemma-2-9b-it \
#     --intervention_method vector \
#     --mode generate \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# RePS Method - LoRA
# python run_PRISM.py \
#     --dataset axbench \
#     --method reps \
#     --model_name gemma-2-9b-it \
#     --intervention_method lora \
#     --mode generate \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# RePS Method - Local Weight
# python run_PRISM.py \
#     --dataset axbench \
#     --method reps \
#     --model_name gemma-2-9b-it \
#     --intervention_method local_weight \
#     --mode generate \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# SFT Method - Vector
# python run_PRISM.py \
#     --dataset axbench \
#     --method sft \
#     --model_name gemma-2-9b-it \
#     --intervention_method vector \
#     --mode generate \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# SFT Method - LoRA
# python run_PRISM.py \
#     --dataset axbench \
#     --method sft \
#     --model_name gemma-2-9b-it \
#     --intervention_method lora \
#     --mode generate \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# SFT Method - Local Weight
# python run_PRISM.py \
#     --dataset axbench \
#     --method sft \
#     --model_name gemma-2-9b-it \
#     --intervention_method local_weight \
#     --mode generate \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# prism Method - Vector
# python run_PRISM.py \
#     --dataset axbench \
#     --method prism \
#     --model_name gemma-2-9b-it \
#     --intervention_method vector \
#     --mode generate \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# prism Method - LoRA
# python run_PRISM.py \
#     --dataset axbench \
#     --method prism \
#     --model_name gemma-2-9b-it \
#     --intervention_method lora \
#     --mode generate \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# prism Method - Local Weight
# python run_PRISM.py \
#     --dataset axbench \
#     --method prism \
#     --model_name gemma-2-9b-it \
#     --intervention_method local_weight \
#     --mode generate \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

### Steer Apply Examples ###

# SFT Method - Vector with multiplier 1.0
# python run_PRISM.py \
#     --dataset axbench \
#     --method sft \
#     --model_name gemma-2-9b-it \
#     --intervention_method vector \
#     --mode apply \
#     --multipliers 1.0 \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# SFT Method - Vector with multiple multipliers
# python run_PRISM.py \
#     --dataset axbench \
#     --method sft \
#     --model_name gemma-2-9b-it \
#     --intervention_method vector \
#     --mode apply \
#     --multipliers 1.0 2.0 3.0 \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# RePS Method - LoRA with multiplier 1.0
# python run_PRISM.py \
#     --dataset axbench \
#     --method reps \
#     --model_name gemma-2-9b-it \
#     --intervention_method lora \
#     --mode apply \
#     --multipliers 1.0 \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

### Both Generation and Application ###

# Run both generation and application in one command
# python run_PRISM.py \
#     --dataset axbench \
#     --method sft \
#     --model_name gemma-2-9b-it \
#     --intervention_method vector \
#     --mode both \
#     --multipliers 1.0 \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

##########################################
############### Psychopathy ##############
##########################################

### Steer Generation Examples ###

# RePS Method - Vector for Psychopathy
# python run_PRISM.py \
#     --dataset psychopathy \
#     --method reps \
#     --model_name gemma-2-9b-it \
#     --intervention_method vector \
#     --mode generate \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# SFT Method - LoRA for Psychopathy
# python run_PRISM.py \
#     --dataset psychopathy \
#     --method sft \
#     --model_name gemma-2-9b-it \
#     --intervention_method lora \
#     --mode generate \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

### Steer Apply Examples ###

# SFT Method - Vector for Psychopathy
# python run_PRISM.py \
#     --dataset psychopathy \
#     --method sft \
#     --model_name gemma-2-9b-it \
#     --intervention_method vector \
#     --mode apply \
#     --multipliers 1.0 \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

##########################################
############## Powerseeking ##############
##########################################

### Steer Generation Examples ###

# RePS Method - Vector for Powerseeking
# python run_PRISM.py \
#     --dataset powerseeking \
#     --method reps \
#     --model_name gemma-2-9b-it \
#     --intervention_method vector \
#     --mode generate \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# SFT Method - LoRA for Powerseeking
# python run_PRISM.py \
#     --dataset powerseeking \
#     --method sft \
#     --model_name gemma-2-9b-it \
#     --intervention_method lora \
#     --mode generate \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

### Steer Apply Examples ###

# SFT Method - Vector for Powerseeking
# python run_PRISM.py \
#     --dataset powerseeking \
#     --method sft \
#     --model_name gemma-2-9b-it \
#     --intervention_method vector \
#     --mode apply \
#     --multipliers 1.0 \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

##########################################
############## Quick Start ###############
##########################################

# Uncomment and modify the following line to run yprism experiment:
python run_prism.py \
    --dataset psychopathy \
    --method all \
    --model_name gemma-2-9b-it \
    --intervention_method local_weight \
    --mode generate \
    --multipliers 1.0 \
    --device cuda:0 \
    --base_dir .
