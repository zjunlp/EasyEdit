import os
import argparse
import torch
from tqdm import tqdm
from .generate_reps_vector_hparams import RePSHyperParams
<<<<<<< HEAD
from .utils import get_grouped_data_by_concept_id, get_prefix_length, load_state, prepare_groups, save_state
from steer.trainer.RepsVectorTrainer import RepsVectorTrainer
=======
from .utils import get_grouped_data_by_concept_id, get_prefix_length, load_state
from ...trainer.BaseModelTrainer import BaseModelTrainer


# def generate_reps_vectors(hparams:RePSHyperParams, dataset, model = None):
#     from ...models.get_model import get_model
#     from ...datasets.caa_data import get_tokens_for_caa
#     # Whether to delete the model after the function
#     del_model = True
#     args = hparams
    
    
#     metadata_path = os.path.join(args.data_dir, 'metadata.jsonl')
#     metadata = load_metadata(metadata_path)
    


#     pos_activations = dict([(layer, []) for layer in args.layers])
#     neg_activations = dict([(layer, []) for layer in args.layers])


#     pos_tokens_list, neg_tokens_list = get_tokens_for_caa(dataset, tokenizer, hparams)

#     for p_tokens_dict, n_tokens_dict in tqdm(
#         zip(pos_tokens_list, neg_tokens_list),
#         total=len(pos_tokens_list),
#         desc="Processing prompts",
#     ):
#         p_tokens = p_tokens_dict["pos_tokens"]
#         n_tokens = n_tokens_dict["neg_tokens"]
#         ques_tokens_len = p_tokens_dict["ques_tokens_len"]
#         model.reset_all()
#         model.get_logits(p_tokens)

#         for layer in args.layers:
#             p_activations = model.get_last_activations(layer)
#             # mean the activation over all answer tokens
#             if args.multiple_choice == True:
#                 p_activations = p_activations[0, -2, :].detach().cpu()
#             else:
#                 p_activations = p_activations[0, ques_tokens_len:, :].mean(0).detach().cpu()
#             pos_activations[layer].append(p_activations)

#         model.reset_all()
#         model.get_logits(n_tokens)
        
#         ques_tokens_len = n_tokens_dict["ques_tokens_len"]
#         for layer in args.layers:
#             n_activations = model.get_last_activations(layer)
#             if args.multiple_choice == True:
#                 n_activations = n_activations[0, -2, :].detach().cpu()
#             else:
#                 n_activations = n_activations[0, ques_tokens_len:, :].mean(0).detach().cpu()
#             neg_activations[layer].append(n_activations)
            
#     if hparams.save_vectors is True:
#         if args.multiple_choice == True:
#             steer_vector_output_dir = os.path.join(
#                 args.steer_vector_output_dir, "caa_vector_multiple_choice"
#             )
#         else:
#             steer_vector_output_dir = os.path.join(
#                 args.steer_vector_output_dir, "caa_vector"
#             )

#         if not os.path.exists(steer_vector_output_dir):
#             os.makedirs(steer_vector_output_dir)

#     try:
#         vectors = {}
#         for layer in args.layers:
#             all_pos_layer = torch.stack(pos_activations[layer])
#             all_neg_layer = torch.stack(neg_activations[layer])
#             vec = (all_pos_layer - all_neg_layer).mean(dim=0)
#             if hparams.save_vectors is True:
#                 torch.save(
#                     vec,
#                     os.path.join(steer_vector_output_dir, f"layer_{layer}.pt"),
#                 )
#             vectors[f"layer_{layer}"] = vec
#     finally:
#         if del_model:
#             model.model.to('cpu')
#             del model
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#         return vectors


import os
import argparse
import yaml
import json
import glob
import pickle
import torch
import shutil
import requests
import datetime
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from pathlib import Path


# all supported methods
import axbench

>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)

def generate_reps_vectors(args:RePSHyperParams, dataset, model = None):
    from ...models.get_model import get_model
    # Whether to delete the model after the function
    del_model = True

    # Load grouped data by concept_id, key is concept_id, value is da
    concept_grouped_data = get_grouped_data_by_concept_id(dataset)
<<<<<<< HEAD
    
    
    print(f"[INFO] Total number of concept df loaded: {len(concept_grouped_data)}")
    if args.max_concepts:
        print(f"[WARNING] Only processing {args.max_concepts} concepts")
        concept_grouped_data = concept_grouped_data[:args.max_concepts]

=======

    print(f"[INFO] Total number of concept df loaded: {len(concept_grouped_data)}")
    if args.max_concepts:
        print(f"[WARNING] All ranks only processing {args.max_concepts} concepts")
        concept_grouped_data = concept_grouped_data[:args.max_concepts]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
    tokenizer.padding_side = "right"

    # Load model instance onto device
    if args.use_bf16:
        print(f"[WARNING] Using bfloat16 for model {args.model_name}")
        

>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
    if model is None:
        model, tokenizer = get_model(args)
    else:
        del_model = False
        model, tokenizer = model, model.tokenizer
        model.hparams = args
<<<<<<< HEAD
    
    # Load model instance onto device
    if model.torch_dtype == torch.bfloat16:
        print(f"[WARNING] Using bfloat16 for model {args.model_name_or_path}")
        
    model.tokenizer.padding_side = "right"
    model.tokenizer.model_max_length = 512
=======
        
>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
    model.model = model.model.eval()
    model.model.to(args.device)
    # is_chat_model = True if args.model_name in CHAT_MODELS else False
    
   
<<<<<<< HEAD
    if model.tokenizer.unk_token == None and model.tokenizer.pad_token == None:
        # raw llama3
        print("[INFO] adding a special padding token...")
        model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
=======
    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("[INFO] adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
        need_resize = True
    else:
        need_resize = False
    if need_resize:
<<<<<<< HEAD
        model.model.resize_token_embeddings(len(model.tokenizer))
=======
        model.model.resize_token_embeddings(len(tokenizer))
>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)

    prefix_length = 1 # prefix is default to 1 for all models due to theBOS token.
    if args.use_chat_template:
        prefix_length = get_prefix_length(tokenizer)
        print(f"[WARNING] Chat model prefix length: {prefix_length}")

    state = load_state(args.steer_vector_output_dir)
    last_concept_id = state.get("last_concept_id", None) if state else None
    print(f"[WARNING] last concept_id processed: {last_concept_id}")

<<<<<<< HEAD
<<<<<<< HEAD
    vec = None
=======
>>>>>>> 2a46a1c5 (Forward Still Bug)
    for concept_id, concept_groups in concept_grouped_data:
        concept_id = int(concept_id) 

        # if the concept has been processed, skip it
=======
 
    # 遍历待处理的概念列表，每个元素包含一个 concept_id 和对应的数据 (concept_df)
    for concept_id, concept_groups in concept_grouped_data:
        concept_id = int(concept_id)  # 将 concept_id 转换为整数

        # 如果这个概念已经被处理过（用于从中断处恢复），则跳过
>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
        if last_concept_id is not None and concept_id <= last_concept_id:
            print(f"[WARNING] Skip concept_id {concept_id}, because it has been processed")
            continue
        
        print(f"[WARNING] Start to train RePS for concept_id {concept_id}")

<<<<<<< HEAD
        concept = concept_groups[0]["concept"]
        print(f"[WARNING] Using concept '{concept}' to train RePS")
        
        # Init the trainer
        benchmark_model = RepsVectorTrainer(
            model=model,
            layers=args.layers,
            hparams=args 
        )
        # get the low rank dimension,  set to 1 as a vector
        low_rank_dimension = args.low_rank_dimension if args.low_rank_dimension else 1
        
        # incorporate the intervention to the model
        benchmark_model.make_model(
            mode="train",
            embed_dim=model.model.config.hidden_size,
            low_rank_dimension=low_rank_dimension,
            dtype=model.torch_dtype,
            intervention_type=args.intervention_type, 
            concept_id=concept_id,
            dump_dir=args.steer_vector_output_dir,
            model_params=args,
=======

        # 从元数据中获取当前概念的名称
        concept = concept_groups[0]["concept"]
        print(f"[WARNING] Using concept '{concept}' to train RePS")
        
        # --- 模型初始化 ---
        # 使用 getattr 动态地从 axbench 模块获取并实例化对应的模型训练器类
        benchmark_model = sdf(
            model=model,
            layer=args.layer,
            hparams=args # 传入该模型专属的训练参数
        )
        # 获取低秩维度，如果未指定则默认为 1
        low_rank_dimension = args.low_rank_dimension if args.low_rank_dimension else 1
        
        # --- 创建可干预模型 ---
        # 调用 make_model 方法配置并构建用于训练的可干预模型
        benchmark_model.make_model(
            mode="train",
            embed_dim=model_instance.config.hidden_size,
            low_rank_dimension=low_rank_dimension,
            dtype=torch.bfloat16 if args.use_bf16 else None, # 如果启用，则使用 bfloat16 类型
            intervention_type=args.intervention_type, # 干预类型
            # ... 其他模型构建参数
            concept_id=concept_id,
            metadata_path=metadata_path,
            dump_dir=dump_dir,
            model_params=args.models[model_name],
>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
            dropout=args.dropout,
            intervention_positions_dropout=args.intervention_positions_dropout,
            preference_pairs=args.preference_pairs,
        )

<<<<<<< HEAD
        benchmark_model.model.steer_vector.to(model.torch_dtype)
        
        # prepare the training parameters
        training_kwargs = {
            "prefix_length": prefix_length,
            "positions": args.intervention_positions, 
            "exclude_bos": args.exclude_bos,
=======
        # 如果使用 bfloat16，并且模型不是特定类型，则手动转换干预模块的数据类型
        if args.use_bf16:
            if isinstance(benchmark_model.ax, list):
                for ax in benchmark_model.ax:
                    ax.to(torch.bfloat16)
            else:
                benchmark_model.ax.to(torch.bfloat16)
        
        # --- 准备训练参数 ---
        # 将所有需要传递给 train 方法的参数打包到 kwargs 字典中
        kwargs = {
            "prefix_length": prefix_length,
            "positions": args.intervention_positions, # 干预位置
            "exclude_bos": args.exclude_bos,
            "use_dpo_loss": args.use_dpo_loss, # 是否使用DPO损失
            "logging_metadata": { # 用于 wandb 日志记录的元数据
                "concept_id": concept_id,
                "model_name": model_name,
                "layer": args.layer,
            },
            "wandb_project": args.wandb_project,
            "wandb_name": args.wandb_name,
            # ... 其他训练相关参数
            "negative_only": args.negative_only,
>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
            "preference_pairs": args.preference_pairs,
            "steering_prompt_type": args.steering_prompt_type,
            "substraction_type": args.substraction_type,
        }
        
<<<<<<< HEAD
        # prepare the data
        prepared_groups = prepare_groups(
            concept_groups, 
            concept, 
            model.tokenizer, 
            use_chat_template=args.use_chat_template,
            model_name_or_path=args.model_name_or_path,
            max_num_of_examples=args.max_num_of_examples,
            steering_prompt_type=args.steering_prompt_type,
        )

        benchmark_model.train(prepared_groups, **training_kwargs)

        # get the trained vectors
<<<<<<< HEAD
        vec = benchmark_model.model.steer_vector.proj.weight.data.clone().cpu()
=======
        vectors = {}
        for layer in args.layers:
            vec = benchmark_model.model.steer_vector.proj.weight.data.clone().cpu()
            vectors[f"layer_{layer}"] = vec
>>>>>>> 2a46a1c5 (Forward Still Bug)
        
        # save the current state
        current_state = {'last_concept_id': concept_id}
        save_state(args.steer_vector_output_dir, current_state)
        
        # save the vectors and the model
        if args.save_vectors:
            concept_dir = os.path.join(args.steer_vector_output_dir, f"concept_{concept}")
            os.makedirs(concept_dir, exist_ok=True)
<<<<<<< HEAD
            torch.save(vec, os.path.join(concept_dir, f"concept_{concept}.pt"))
            # benchmark_model.save(concept_dir, model_name=f"{args.model_name_or_path}")
=======
            torch.save(vectors, os.path.join(concept_dir, f"layer_{args.layers}.pt"))
            benchmark_model.save(concept_dir, model_name=f"{args.model_name_or_path}")
>>>>>>> 2a46a1c5 (Forward Still Bug)
        
        # clean the memory
        del benchmark_model
        torch.cuda.empty_cache()

    if del_model:
        model.model.to('cpu')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
<<<<<<< HEAD
    if vec:
        return vec
=======
    
    return vectors
>>>>>>> 2a46a1c5 (Forward Still Bug)
=======
        # --- 数据预处理 ---
        prepared_df = concept_df.copy()
        # 调用 prepare_df 函数对数据进行格式化和预处理，以适应模型训练的需要
        prepared_df = prepare_df(
            prepared_df, negative_df, concept, metadata[concept_id], tokenizer, 
            binarize=args.models[model_name].binarize_dataset, 
            train_on_negative=args.models[model_name].train_on_negative,
            use_dpo_loss=args.use_dpo_loss,
            is_chat_model=is_chat_model,
            output_length=int(args.output_length),
            model_name=args.model_name,
            max_num_of_examples=args.max_num_of_examples,
            steering_prompt_type=args.models[model_name].steering_prompt_type,
            keep_orig_axbench_format=generate_args.keep_orig_axbench_format,
        )

        # --- 执行训练和保存 ---
        benchmark_model.train(prepared_df, **kwargs)
        # 将当前进程训练好的模型权重保存到磁盘
        benchmark_model.save(dump_dir, model_name=f"rank_{rank}_{model_name}")

  

        print(f"[WARNING] 已为模型 {model_name} 在进程 {rank} 上保存权重")
        
        # --- 清理内存 ---
        del benchmark_model
        torch.cuda.empty_cache()

        # --- 保存当前处理状态 ---
        # 在处理完一个 concept_id 的所有模型后，保存状态，以便任务中断后可以恢复
        current_state = {'last_concept_id': concept_id}
        save_state(args.steer_vector_output_dir, current_state, metadata[concept_id])





>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
