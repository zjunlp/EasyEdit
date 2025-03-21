import os
import os.path
import sys
import json
import random
sys.path.append('..')
from easyeditor import (
    AlphaEditHyperParams,
    FTHyperParams,  
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    GraceHyperParams
    )
from easyeditor import BaseEditor
from easyeditor import WikiBigEditDataset

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--sequential_edit', action="store_true")
    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'AlphaEdit':
        editing_hparams = AlphaEditHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    else:
        raise NotImplementedError
    

    datas = WikiBigEditDataset(args.data_dir,size=args.ds_size)
    prompts=[data['prompt'] for data in datas]
    target_new = [data['target_new'] for data in datas]
    ground_truth = [data['ground_truth'] for data in datas]
    subject = [data['subject'] for data in datas]
    rephrase_prompts = [data['rephrase'] for data in datas]
    portability_personas_prompts = [[data['portability_personas']] if isinstance(data['portability_personas'], str) else None for data in datas]
    portability_personas_answers = [[data['target_new']] for data in datas]
    portability_hop_prompts = [[data['portability_hop']] if isinstance(data['portability_hop'], str) else None for data in datas]
    portability_hop_answers = [[data['portability_hop_ans']] if isinstance(data['portability_hop_ans'], str) else None for data in datas]
    locality_prompts = [[data['locality']] for data in datas]
    locality_answers = [[data['locality_ans']] for data in datas]

    assert len(prompts)==len(portability_personas_prompts)==len(portability_personas_answers)==len(portability_hop_prompts)==len(portability_hop_answers)

    assert len(prompts)==len(locality_prompts)==len(locality_answers)

    locality_inputs = {}
    portability_inputs = {}
    locality_inputs = {
        'locality':{
            'prompt': locality_prompts,
            'ground_truth': locality_answers
        }
    }
    portability_inputs = {
        'personas':{ 
            'prompt': portability_personas_prompts,
            'ground_truth': portability_personas_answers  
        },
        'mhop':{
            'prompt': portability_hop_prompts,
            'ground_truth': portability_hop_answers
        }
    }

    hparams = editing_hparams.from_hparams(args.hparams_dir)
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        ground_truth=ground_truth,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        subject = subject,
        keep_original_weight=True,
        sequential_edit=args.sequential_edit
    )
    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_{hparams.model_name.split("/")[-1]}_WikiBigEdit_results.json'), 'w'), indent=4)
