python run_wise_editing.py \
  --editing_method=GRACE \
  --hparams_dir=../hparams/GRACE/llama-7B \
  --data_dir=../data/wise \
  --ds_size=3 \
  --data_type=ZsRE \
  --sequential_edit



#python run_wise_editing.py \
#  --editing_method=WISE \
#  --hparams_dir=../hparams/WISE/llama-7b \
#  --data_dir=../data/wise \
#  --ds_size=3 \
#  --data_type=hallucination
##  --sequential_edit