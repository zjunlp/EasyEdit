nohup python test_InstructEdit.py \
    --editing_method=InstructEdit \
    --hparams_dir=hparams/MEND/gpt2-xl-instruct.yaml \
    --data_dir=data \
    --data_type=zsre \
    --data=zsre_mend_eval_portability_gpt4.json \
    >logs/run_inst_zsre.out 2>&1 &

nohup python test_InstructEdit.py \
    --editing_method=InstructEdit \
    --hparams_dir=hparams/MEND/gpt2-xl-instruct.yaml \
    --data_dir=data \
    --data_type=wikirecent \
    --data=recent_test.json \
    >logs/run_inst_wikirecent.out 2>&1 &

nohup python test_InstructEdit.py \
    --editing_method=InstructEdit \
    --hparams_dir=hparams/MEND/gpt2-xl-instruct.yaml \
    --data_dir=data \
    --data_type=counterfact \
    --data=test_cf.json \
    >logs/run_inst_counterfact.out 2>&1 &

nohup python test_InstructEdit.py \
    --editing_method=InstructEdit \
    --hparams_dir=hparams/MEND/gpt2-xl-instruct.yaml \
    --data_dir=data \
    --data_type=convsent \
    --data=convsent_test_reconstruct.json \
    >logs/run_inst_convsent.out 2>&1 &
