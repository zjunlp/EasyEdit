# Example: SimIE with ROME on ZsRE dataset
CUDA_VISIBLE_DEVICES=4 nohup python run_SimIE_editing.py \
    --editing_method SimIE \
    --hparams_dir ../hparams/SimIE/llama3-8b.yaml \
    --data_dir ../data \
    --data_type ZsRE \
    --ds_size 3 \
    --sequential_edit \
    --output_dir ./outputs > test_SimIE_1.log 2>&1 &
