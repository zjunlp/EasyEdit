#!/bin/bash
##########################################
################# SPLIT ##################
##########################################

# This script provides examples for running experiments

# Set default values
DEVICE=${DEVICE:-"cuda:0"}

##########################################
############### AxBench ##################
##########################################

############# Steer Examples #############
# SPLIT Method - All interventions for AxBench
# python run_SPLIT.py \
#     --dataset axbench \
#     --method all \
#     --model_name gemma-2-9b-it \
#     --intervention_method all \
#     --mode both \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# SPLIT Method - All interventions for AxBench (Qwen model)
# python run_SPLIT.py \
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
# SPLIT Method - All interventions for Psychopathy
# python run_SPLIT.py \
#     --dataset psychopathy \
#     --method all \
#     --model_name gemma-2-9b-it \
#     --intervention_method all \
#     --mode both \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# SPLIT Method - All interventions for Psychopathy (Qwen model)
# python run_SPLIT.py \
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
# SPLIT Method - All interventions for Powerseeking
# python run_SPLIT.py \
#     --dataset powerseeking \
#     --method all \
#     --model_name gemma-2-9b-it \
#     --intervention_method all \
#     --mode both \
#     --device ${DEVICE} \
#     --base_dir ${BASE_DIR}

# SPLIT Method - All interventions for Powerseeking (Qwen model)
# python run_SPLIT.py \
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

# Uncomment and modify the following line to run SPLIT experiment:

python ./examples/run_SPLIT.py \
    --dataset axbench \
    --method SPLIT \
    --model_name gemma-2-9b-it \
    --intervention_method local_weight \
    --mode generate \
    --multipliers 1.0 \
    --device cuda:0 \
    --base_dir .