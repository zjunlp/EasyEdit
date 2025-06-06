python run_ultraedit_editing.py \
    --editing_method=UltraEdit \
    --hparams_dir=../hparams/ULTRAEDIT/mistral-7b.yaml \
    --data_dir=../data/ultraedit \
    --ds_size=20000 \
    --batch_size=100 \
    --data_type=zsre \
    --sequential_edit
