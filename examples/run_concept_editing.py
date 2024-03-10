import os.path
import sys
sys.path.append('..')
import json
import random
from easyeditor import FTHyperParams, MEMITHyperParams, ROMEHyperParams, HyperParams
from easyeditor import  ConceptEditor
import numpy as np 
import torch
import math


import argparse

models_implement = ['mistral','llama2chat','gpt2','gptj']
model_names = ['./hugging_cache/Mistral-7B-v0.1','./hugging_cache/llama2-7b-chat','./hugging_cache/gpt2-xl','./hugging_cache/gpt-j-6B']

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--edited_model', required=True, type=str)
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--metrics_save_dir', default='./final_result_upload', type=str)
    parser.add_argument('--inter', action='store_true')

    args = parser.parse_args()

    if os.path.exists(args.metrics_save_dir) is False:
        os.makedirs(args.metrics_save_dir)

    if args.edited_model not in models_implement:
        raise NotImplementedError

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'PROMPT':
        editing_hparams = HyperParams 
    else:
        raise NotImplementedError
    
    if args.inter:
        module = "inter"
    else:
        module = "intra"

    test_data = json.load(open(os.path.join(args.data_dir, f"final_{args.edited_model}_{module}.json"), 'r', encoding='utf-8'))


    # 设置随机数种子
    setup_seed(42)

    # test_data= test_data[:3]

    prompts = [test_data_['prompt'] for test_data_ in test_data]
    rephrase_prompts = [edit_data_['phrase_prompt'] for edit_data_ in test_data]
    target_new = [edit_data_['target_new_desc'] for edit_data_ in test_data]
    entity_prompts = [edit_data_['instance_prompt'] for edit_data_ in test_data]
    in_locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in test_data]
    in_locality_ans = [edit_data_['locality_answer'] for edit_data_ in test_data]

    locality_inputs = {
        'neighborhood':{
            'prompt': in_locality_prompts,
            'ground_truth': in_locality_ans
        }
    }
    instance_inputs = {
        'instance':{
            'prompt': entity_prompts
        },
    }
    
    subject = [edit_data_['label'] for edit_data_ in test_data]
    train_ds = None



    if args.editing_method == 'PROMPT':
        prompt_hparams = {'model_name': model_names[models_implement.index(args.edited_model)], 'device': 0}
        hparams = None
        editor = ConceptEditor.from_hparams(hparams, prompt_hparams)

    else:
        hparams = editing_hparams.from_hparams(args.hparams_dir)
        editor = ConceptEditor.from_hparams(hparams)

    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subject,
        train_ds=train_ds,
        locality_inputs=locality_inputs,
        instance_inputs=instance_inputs,
        # concept_consistency = True,
        keep_original_weight=True
    )

    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_results_{args.edited_model}_{module}.json'), 'w'), indent=4)


    rewrite_acc = 0
    rephrase_acc = 0
    locality = 0
    loc_list = []    
    instance = 0
    port_list = []
    with open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_results_{args.edited_model}_{module}.json'), "r") as f:
        result = json.load(f)

    for i, item in enumerate(result):
        
        case = item["post"]
        # print(case)
        if not math.isnan(case["rewrite_acc"][0]):
            rewrite_acc = ((rewrite_acc * i) + np.mean(case["rewrite_acc"][0])) / (i + 1)
        else:
            print(f'{i}: {case}')
        if not math.isnan(case["rephrase_acc"][0]):
            rephrase_acc = ((rephrase_acc * i) + np.mean(case["rephrase_acc"][0])) / (i + 1)
        else:
            print(f'{i}: {case}')

        locality_ = 0
        instance_ = 0
        if "locality" in case.keys() and case["locality"]:
            if "neighborhood_acc" in case["locality"].keys():
                locality_ += np.mean(case["locality"]["neighborhood_acc"])
            if not math.isnan(locality_):
                loc_list.append(locality_)
            
        if "instance" in case.keys() and case["instance"]:
            if "instance_change" in case["instance"].keys():
                if case["instance"]["instance_change"] == -1:
                    case["instance"]["instance_change"] = 1
                instance_ += np.mean(case["instance"]["instance_change"])
            if not math.isnan(instance_):
                port_list.append(instance_)
    locality = np.mean(loc_list) if loc_list else 0
    instance = np.mean(port_list) if port_list else 0
    sub1 = instance
    
    print(f'dir: {args.metrics_save_dir}\npost\nReliability: {rewrite_acc*100}\nGeneralization: {rephrase_acc*100}\nLocality: {locality*100}')

    port_list = []

    for i, item in enumerate(result):
        case = item["pre"]
        instance_ = 0
        if "instance" in case.keys() and case["instance"]:
            if "instance_change" in case["instance"].keys():
                if case["instance"]["instance_change"] == -1:
                    case["instance"]["instance_change"] = 1
                instance_ += np.mean(case["instance"]["instance_change"])
            if not math.isnan(instance_):
                port_list.append(instance_)
    instance = np.mean(port_list) if port_list else 0
    sub2 =instance

    print('instance_change: ',end='')
    print((sub2-sub1)*100)