#!/bin/bash

python data_sampler.py \
    --input-file /mnt/20t/wsx/data/sentiment/sst2_500.json \
    --output-file /mnt/20t/wsx/steer_experiment_data/sentiment/sst2_train_1000.json \
    --sample-size 1000 \
    --seed 42


python sample.py \
    --hf-dataset SetFit/sst2 \
    --hf-mirror \
    --hf-split train \
    --output-file /mnt/20t/wsx/steer_experiment_data/sentiment/sst2_train_1000.json \
    --sample-size 1000 \
    --seed 42 \
    --convert-format
