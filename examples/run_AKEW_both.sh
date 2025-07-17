# if you wanna run uns_data evaluate, don't forget to set --use_unstructured_data and choose --model_path all-MiniLM-L6-v2 to calculate bert_score or rouge
# here is an example using ft_uns 
# CUDA_VISIBLE_DEVICES=4 nohup python run_AKEW_both.py \
#     --editing_method FT_uns  \
#     --hparams_dir ../hparams/FT_uns/Qwen2.5-7B-Instruct.json \
#     --data_dir /data3/xubuqiang/QiQi/EasyEdit/data \
#     --data_type mquake \
#     --ds_size 2 \
#     --metrics_save_dir ./results/AKEW_mquake-cf_FT_uns_test \
#     --use_unstructured_data \
#     --batch_size 1 \
#     --model_path "/data3/xubuqiang/QiQi/all-MiniLM-L6-v2"  \
#     --device 0 \
#     > 2025_7_10_AKEW_mquake_FT_uns.log 2>&1 &
# also we support method to calculate Reliability、Generalization、Locality、Portability of structed data
# here is an example using ft
# CUDA_VISIBLE_DEVICES=4 nohup python run_AKEW_both.py \
#     --editing_method FT \
#     --hparams_dir ../hparams/FT/qwen2.5-7b.yaml \
#     --data_dir /data3/xubuqiang/QiQi/EasyEdit/data/MQuAKE-CF_with_locality.json \
#     --data_type mquake \
#     --ds_size 10 \
#     --metrics_save_dir ./results/AKEW_mquake_FT_structured_test \
#     > 2025_7_8_AKEW_mquake_FT_structured.log 2>&1 &



