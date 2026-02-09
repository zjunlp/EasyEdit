#!/bin/bash

gpu=1
method="reps_vector"
datasets_list=("language_features" "personality" "reasoning_patterns" "sentiment")
multipliers_list=( 5 6 7 8)

# model=qwen-2.5-7b-it
# layers=14
model=llama-3.1-8b-it
layers=12

# search with different multipliers
for dataset in "${datasets_list[@]}"; do
    # generate vector for all datasets
    ./steer_eval.sh --gpu=0  --model=${model} --method=reps_vector --dataset=steer_eval/${dataset}          --data_path=gpt41_q_gpt41mini_a   --generate_vector=true   --gen_out_path=f_steer_eval  --generate_response=true --generate_orig_output=false --evaluate=false --layers=$layers --multipliers=0  --exp=valid
    for multipliers in "${multipliers_list[@]}"; do
        ./steer_eval.sh   --gpu=${gpu} \
                        --model=${model} \
                        --method=${method} \
                        --dataset=steer_eval/${dataset}  \
                        --data_path=gpt41_q_gpt41mini_a   \
                        --generate_vector=false  \
                        --gen_out_path=f_version_5   \
                        --generate_response=true  \
                        --generate_orig_output=false \
                        --evaluate=false \
                        --layers=${layers} \
                        --multipliers=${multipliers}  \
                        --exp=valid
    done
    # evaluate with the best multipliers
    ./steer_eval.sh --gpu=0 --model=${model} --method=reps_vector --dataset=steer_eval/${dataset}     --data_path=gpt41_q_gpt41mini_a   --generate_vector=false  --gen_out_path=f_steer_eval --generate_response=true    --generate_orig_output=false   --evaluate=false --layers=${layers} --use_best_multip=true  --exp=test
done
