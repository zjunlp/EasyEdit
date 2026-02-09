#!/bin/bash

gpu=1
method="caa"
datasets_list=("language_features" )
multipliers_list=(1)

# model=qwen-2.5-7b-it
# layers=14
model=llama-3.1-8b-it
layers=12


# search with different multipliers
for dataset in "${datasets_list[@]}"; do

    ./steer_eval.sh --gpu=${gpu}  --model=${model}  --method=caa --dataset=steer_eval/${dataset}         --data_path=gpt41_q_gpt41mini_a   --generate_vector=true   --gen_out_path=f_steer_eval --generate_response=false   --generate_orig_output=false  --evaluate=false --layers=$layers --multipliers=0  --exp=test

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

    ./steer_eval.sh --gpu=${gpu} --model=${model} --method=caa --dataset=steer_eval/${dataset}           --data_path=gpt41_q_gpt41mini_a   --generate_vector=false  --gen_out_path=f_steer_eval --generate_response=true    --generate_orig_output=false   --evaluate=false --layers=${layers} --use_best_multip=true  --exp=test

done

