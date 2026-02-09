#!/bin/bash
model=qwen-2.5-7b-it
layers=14
multipliers=0

# model=gemma-2-9b-it
# layers=20

# model=llama-3.1-8b-it
# layers=12
# multipliers=0

# Generate Prompt
./steer_eval.sh --gpu=1 --model=${model}  --method=prompt --dataset=steer_eval/personality           --data_path=gpt41_q_gpt41mini_a   --generate_vector=true   --gen_out_path=f_steer_eval --generate_response=false    --generate_orig_output=false   --evaluate=false --layers=$layers --multipliers=${multipliers}  --exp=test
./steer_eval.sh --gpu=2 --model=${model}  --method=prompt --dataset=steer_eval/language_features     --data_path=gpt41_q_gpt41mini_a   --generate_vector=true   --gen_out_path=f_steer_eval --generate_response=false    --generate_orig_output=false   --evaluate=false --layers=$layers --multipliers=${multipliers}  --exp=test
./steer_eval.sh --gpu=2 --model=${model}  --method=prompt --dataset=steer_eval/reasoning_patterns    --data_path=gpt41_q_gpt41mini_a   --generate_vector=true   --gen_out_path=f_steer_eval --generate_response=false    --generate_orig_output=false   --evaluate=false --layers=$layers --multipliers=${multipliers}  --exp=test
./steer_eval.sh --gpu=2 --model=${model}  --method=prompt --dataset=steer_eval/sentiment             --data_path=gpt41_q_gpt41mini_a   --generate_vector=true   --gen_out_path=f_steer_eval --generate_response=false    --generate_orig_output=false   --evaluate=false --layers=$layers --multipliers=${multipliers}  --exp=test

# Generate Response
./steer_eval.sh --gpu=1 --model=${model}  --method=prompt --dataset=steer_eval/personality           --data_path=gpt41_q_gpt41mini_a   --generate_vector=false    --gen_out_path=f_steer_eval --generate_response=true    --generate_orig_output=false   --evaluate=false --layers=$layers --multipliers=${multipliers}  --exp=test
./steer_eval.sh --gpu=2 --model=${model}  --method=prompt --dataset=steer_eval/language_features     --data_path=gpt41_q_gpt41mini_a   --generate_vector=false    --gen_out_path=f_steer_eval --generate_response=true    --generate_orig_output=false   --evaluate=false --layers=$layers --multipliers=${multipliers}  --exp=test
./steer_eval.sh --gpu=2 --model=${model}  --method=prompt --dataset=steer_eval/reasoning_patterns    --data_path=gpt41_q_gpt41mini_a   --generate_vector=false    --gen_out_path=f_steer_eval --generate_response=true    --generate_orig_output=false   --evaluate=false --layers=$layers --multipliers=${multipliers}  --exp=test
./steer_eval.sh --gpu=2 --model=${model}  --method=prompt --dataset=steer_eval/sentiment             --data_path=gpt41_q_gpt41mini_a   --generate_vector=false    --gen_out_path=f_steer_eval --generate_response=true    --generate_orig_output=false   --evaluate=false --layers=$layers --multipliers=${multipliers}  --exp=test


# few-shot prompt
gpu=0
method="prompt"
datasets_list=("language_features" "personality")
shot_list=(0 1 2 3 4 8 16)

for dataset in "${datasets_list[@]}"; do
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