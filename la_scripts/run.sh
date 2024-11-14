#!/bin/sh

#SBATCH --job-name=mend_safeedit
# 1 "la_scripts/log_test_mend0.txt"
# 2 "la_scripts/log_test_safeedit_DINM0.txt" 
# 3 "la_scripts/log_test_safeedit_DINM_wo_bias.txt" 
#SBATCH --out="la_scripts/log_test_safeedit_DINM_all.txt" 
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:tesla_a100_80g:1
#SBATCH --cpus-per-gpu=4

conda init
conda activate myenv397
source /home/smg/v-lauraom/venvs/EasyEdit/bin/activate 

# 1 python la_scripts/run_mend_zsre_llama-7b.py
# 2 python examples/run_safety_editing.py --editing_method=MEND --edited_model=llama-7b --hparams_dir=./hparams/MEND/llama-7b --safety_classifier_dir=zjunlp/SafeEdit-Safety-Classifier --metrics_save_dir=./safety_results
# 3 python examples/run_safety_editing.py --editing_method=DINM --edited_model=llama-7b --hparams_dir=./hparams/DINM/llama-7b --data_path=./data/SafeEdit_test_ALL.json --safety_classifier_dir=zjunlp/SafeEdit-Safety-Classifier --metrics_save_dir=./safety_results/DINM/all --ckpt_save_dir=safety_results/models/llama-7b-DIMN-all
python examples/run_safety_editing.py --editing_method=DINM --edited_model=llama-7b --hparams_dir=./hparams/DINM/llama-7b --data_path=./data/SafeEdit_test_wo_bias.json --safety_classifier_dir=zjunlp/SafeEdit-Safety-Classifier --metrics_save_dir=./safety_results/DINM/wo-bias --ckpt_save_dir=safety_results/models/llama-7b-DIMN-wo-bias

# sbatch -p qgpu-debug la_scripts/run.sh
