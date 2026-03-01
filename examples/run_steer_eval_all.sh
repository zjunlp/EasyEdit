#!/bin/bash

gpu=0
datasets_list=("language_features" "personality" "reasoning_patterns" "sentiment")
multipliers_list=(1 2 3 4 5 6 7 8)
model=llama-3.1-8b-it
layers=12

method_list=("caa" "reps_vector")
for method in "${method_list[@]}"; do
    # search with different multipliers
    for dataset in "${datasets_list[@]}"; do
        ./steer_eval.sh --gpu=${gpu}  --model=${model}  --method=${method} --dataset=steer_eval/${dataset}         --data_path=gpt41_q_gpt41mini_a   --generate_vector=true   --gen_out_path=f_steer_eval --generate_response=false   --generate_orig_output=false  --evaluate=false --layers=$layers --multipliers=0  --exp=test
        for multipliers in "${multipliers_list[@]}"; do
            ./steer_eval.sh   --gpu=${gpu} \
                            --model=${model} \
                            --method=${method} \
                            --dataset=steer_eval/${dataset}  \
                            --data_path=gpt41_q_gpt41mini_a   \
                            --generate_vector=false  \
                            --gen_out_path=f_steer_eval   \
                            --generate_response=true  \
                            --generate_orig_output=false \
                            --evaluate=false \
                            --layers=${layers} \
                            --multipliers=${multipliers}  \
                            --exp=valid
        done
        ./steer_eval.sh --gpu=${gpu} --model=${model} --method=${method} --dataset=steer_eval/${dataset}           --data_path=gpt41_q_gpt41mini_a   --generate_vector=false  --gen_out_path=f_steer_eval --generate_response=true    --generate_orig_output=false   --evaluate=false --layers=${layers} --use_best_multip=true  --exp=test
    done
done

# PCA
for dataset in "${datasets_list[@]}"; do
    for multipliers in "${multipliers_list[@]}"; do
        ./steer_eval.sh   --gpu=${gpu} \
                        --model=${model} \
                        --method=caa \
                        --use_pca=true \
                        --dataset=steer_eval/${dataset}  \
                        --data_path=gpt41_q_gpt41mini_a   \
                        --generate_vector=false  \
                        --gen_out_path=f_steer_eval   \
                        --generate_response=true  \
                        --generate_orig_output=false \
                        --evaluate=false \
                        --layers=${layers} \
                        --multipliers=${multipliers}  \
                        --exp=valid
    done
    # evaluate with the best multipliers
    ./steer_eval.sh --gpu=${gpu} --model=$model --method=caa --use_pca=true --dataset=steer_eval/${dataset}     --data_path=gpt41_q_gpt41mini_a   --generate_vector=false  --gen_out_path=f_steer_eval --generate_response=true    --generate_orig_output=false   --evaluate=false --layers=$layers --use_best_multip=true  --exp=test
done

# Prompt
method="prompt"
shot_list=(0 1 2 3 4 8 16)
for dataset in "${datasets_list[@]}"; do
    ./steer_eval.sh --gpu=1 --model=${model}  --method=prompt --dataset=steer_eval/${dataset}           --data_path=gpt41_q_gpt41mini_a   --generate_vector=true   --gen_out_path=f_steer_eval --generate_response=false    --generate_orig_output=false   --evaluate=false --layers=$layers --multipliers=${multipliers}  --exp=test
    for shot in "${shot_list[@]}"; do
        ./steer_eval.sh   --gpu=${gpu} \
                        --model=${model} \
                        --method=${method} \
                        --dataset=steer_eval/${dataset}  \
                        --data_path=gpt41_q_gpt41mini_a   \
                        --generate_vector=false  \
                        --gen_out_path=f_steer_eval   \
                        --generate_response=true  \
                        --generate_orig_output=false \
                        --evaluate=false \
                        --layers=${layers} \
                        --multipliers=${shot}  \
                        --exp=test_shots
    done
done