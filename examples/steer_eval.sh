

#!/bin/bash

# ===========================
# Default Values
# ===========================
gpu=0
vllm_enable=false

model=gemma-2-9b-it
method=reps_vector
use_pca=false
dataset=p
data_path=final_2

generate_vector=true
gen_out_path=final_2-4

generate_response=true
generate_orig_output=false
evaluate=false

layers=20
multipliers=3

use_best_multip=false

exp=test

# ===========================
# Argument Parsing
# ===========================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu=*) gpu="${1#*=}"; shift ;;
        --gpu)   gpu="$2"; shift; shift ;;

        --vllm_enable=*) vllm_enable="${1#*=}"; shift ;;
        --vllm_enable)   vllm_enable="$2"; shift; shift ;;

        --model=*) model="${1#*=}"; shift ;;
        --model)   model="$2"; shift; shift ;;

        --method=*) method="${1#*=}"; shift ;;
        --method)   method="$2"; shift; shift ;;

        --use_pca=*) use_pca="${1#*=}"; shift ;;
        --use_pca)   use_pca="$2"; shift; shift ;;

        --dataset=*) dataset="${1#*=}"; shift ;;
        --dataset)   dataset="$2"; shift; shift ;;

        --data_path=*) data_path="${1#*=}"; shift ;;
        --data_path)   data_path="$2"; shift; shift ;;

        --generate_vector=*) generate_vector="${1#*=}"; shift ;;
        --generate_vector)   generate_vector="$2"; shift; shift ;;

        --gen_out_path=*) gen_out_path="${1#*=}"; shift ;;
        --gen_out_path)   gen_out_path="$2"; shift; shift ;;
        
        --generate_response=*) generate_response="${1#*=}"; shift ;;
        --generate_response)   generate_response="$2"; shift; shift ;;

        --generate_orig_output=*) generate_orig_output="${1#*=}"; shift ;;
        --generate_orig_output)   generate_orig_output="$2"; shift; shift ;;

        --evaluate=*) evaluate="${1#*=}"; shift ;;
        --evaluate)   evaluate="$2"; shift; shift ;;
        
        --layers=*) layers="${1#*=}"; shift ;;
        --layers)   layers="$2"; shift; shift ;;

        --multipliers=*) multipliers="${1#*=}"; shift ;;
        --multipliers)   multipliers="$2"; shift; shift ;;

        --use_best_multip=*) use_best_multip="${1#*=}"; shift ;;
        --use_best_multip)   use_best_multip="$2"; shift; shift ;;

        --exp=*) exp="${1#*=}"; shift ;;
        --exp)   exp="$2"; shift; shift ;;
        *) echo "unknown: $1"; exit 1 ;;
    esac
done

# ===========================
# Verify Inputs
# ===========================

echo "--------------------------------"
echo "GPU ID:          $gpu"
echo "vLLM Enable:     $vllm_enable"

echo "Model:           $model"
echo "Method:          $method"
echo "use PCA:         $use_pca"

echo "Dataset:         $dataset"
echo "Data Path:       $data_path"

echo "Generate Vector: $generate_vector"
echo "Output Path:     $gen_out_path"

echo "Generate Resp:   $generate_response"
echo "Orig Output:     $generate_orig_output"

echo "evaluate:        $evaluate"

echo "Layers:          $layers"
echo "Multipliers:     $multipliers"
echo "Best Multip:     $use_best_multip"

echo "Experiment:      $exp"
echo "--------------------------------"

best_multip_path=None
if [ "$use_best_multip" = true ] ; then
    if [ "$use_pca" = true ] ; then
        echo "PCA is enabled for CAA method."
        best_multip_path="/mnt/40t/xkw/Bench/generation/$model/${dataset}/${gen_out_path}/pca/best_multipliers.json"

    else
        best_multip_path="/mnt/40t/xkw/Bench/generation/$model/${dataset}/${gen_out_path}/${method}/best_multipliers.json"
    fi
    echo "Using best multipliers from: $best_multip_path"
    multipliers=0
fi

model_name_or_path="../hf_models/${model}"

steer_train_hparam_paths="[hparams/Steer/experiment_hparams/steer_eval/$model/${method}/generate_${method}.yaml]"
apply_steer_hparam_paths="[hparams/Steer/experiment_hparams/steer_eval/$model/${method}/apply_${method}.yaml]"

steer_vector_output_dirs="[vectors/$model/${dataset}/${gen_out_path}]"

steer_vector_load_dir="[vectors/$model/${dataset}/${gen_out_path}]"


if [ "$use_pca" = true ] ; then
    echo "PCA is enabled for CAA method."
    generation_output_dir=generation/$model/${dataset}/${gen_out_path}/pca/layer_${layers}_multip_${multipliers}

else
    generation_output_dir=generation/$model/${dataset}/${gen_out_path}/${method}/layer_${layers}_multip_${multipliers}
fi

logdir=logs/${model}/${dataset}/${gen_out_path}/${method}/layer_${layers}_multip_${multipliers}.log 
mkdir -p logs/${model}/${dataset}/${gen_out_path}/${method}
CUDA_VISIBLE_DEVICES=$gpu python steer_eval.py \
    model_name_or_path=${model_name_or_path} \
    +method=${method} \
    +use_pca=${use_pca} \
    +dataset=${dataset} \
    +data_path=${data_path} \
    steer_train_hparam_paths=$steer_train_hparam_paths \
    apply_steer_hparam_paths=$apply_steer_hparam_paths \
    +generate_vector=${generate_vector} \
    steer_vector_output_dirs="$steer_vector_output_dirs" \
    +generate_response=$generate_response \
    steer_vector_load_dir=$steer_vector_load_dir \
    generation_output_dir=$generation_output_dir \
    generate_orig_output=$generate_orig_output \
    +evaluate=$evaluate \
    +vllm_enable=$vllm_enable \
    +layers=[$layers] \
    +multipliers=[$multipliers] \
    +best_multip_path=$best_multip_path \
    +exp=$exp \
    2>&1 | tee $logdir

    

# Example Command:
#  personality reasoning_patterns sentiment language_features
# ./steer_eval.sh --gpu=1  --method=reps_vector --dataset=version2/personality  --data_path=gpt41_q_qwenmax_a   --generate_vector=true   --gen_out_path=f_version_2 --generate_response=false --generate_orig_output=false --evaluate=false --layers=20 --multipliers=10  --exp=test
# ./steer_eval.sh --gpu=2  --method=caa --dataset=version2/personality  --data_path=gpt41_q_qwenmax_a   --generate_vector=true   --gen_out_path=f_version_2 --generate_response=false --generate_orig_output=false --evaluate=false --layers=20 --multipliers=10  --exp=test
# ./steer_eval.sh --gpu=1  --method=reps_vector --dataset=version1/reasoning_patterns  --data_path=gpt41_q_qwenmax_a    --generate_vector=true    --gen_out_path=f_version_1_batch_4 --generate_response=false  --generate_orig_output=false --evaluate=false --layers=20 --multipliers=10  --exp=test