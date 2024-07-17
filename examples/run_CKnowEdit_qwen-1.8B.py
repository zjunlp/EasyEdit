import os
import os.path
import sys
import json
import random
sys.path.append('..')
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    GraceHyperParams,
    MENDHyperParams,
    SERACHparams
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import CKnowEditDataset

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--pre_file', default='./seq_pre.json', type=str)
    parser.add_argument('--chinese_ds_type', required=True, type=str)
    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
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
    

    datas = CKnowEditDataset(args.data_dir,size=args.ds_size)
    prompts=[data['prompt'] for data in datas]
    target_new = [data['target_new'] for data in datas]
    ground_truth = [data['target_old'] for data in datas]
    subject = [data['subject'] for data in datas]
    rephrase_prompts = [data['rephrase'] for data in datas]
    portability_data =[data['portability'] for data in datas]
    locality_data = [data['locality'] for data in datas]

    portability_prompts=[]
    portability_answers=[]
    for item in portability_data:
        if item is None:
            portability_prompts.append(None)
            portability_answers.append(None)
        else:
            temp_prompts = []
            temp_answers = []
            for pr in item:
                prompt=pr['prompt']
                an=pr['answer']
                temp_prompts.append(prompt)
                temp_answers.append(an)
            portability_prompts.append(temp_prompts)
            portability_answers.append(temp_answers)
    assert len(prompts)==len(portability_prompts)==len(portability_answers)

    locality_prompts=[]
    locality_answers=[]
    for item in locality_data:
        if item is None:
            locality_prompts.append(None)
            locality_answers.append(None)
        else:
            temp_prompts = []
            temp_answers = []
            for pr in item:
                if 'prompt' in pr.keys():
                    prompt=pr["prompt"]
                    an=pr["answer"]
                    temp_prompts.append(prompt)
                    temp_answers.append(an)
            locality_prompts.append(temp_prompts)
            locality_answers.append(temp_answers)
    assert len(prompts)==len(locality_prompts)==len(locality_answers)

    locality_inputs = {}
    portability_inputs = {}
    locality_inputs = {
        'loc_hop':{
            'prompt': locality_prompts,
            'ground_truth': locality_answers
        }
    }
    portability_inputs = {
        'por_hop':{
            'prompt': portability_prompts,
            'ground_truth': portability_answers  
        }       
    }

    hparams = editing_hparams.from_hparams(args.hparams_dir)
    args.pre_file = f"./{args.editing_method}_{hparams.model_name.split('/')[-1]}_{args.datatype}_{args.chinese_ds_type}_pre_edit.json"
    print(args.pre_file)
    if args.pre_file is not None and os.path.exists(args.pre_file):
        pre_edit = json.load(open(args.pre_file,'r'))
        assert len(pre_edit) == len(prompts)
    else:
        pre_edit = None
    if args.editing_method == 'IKE':
        train_ds = CKnowEditDataset(args.train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.generate_edit(
        prompts=prompts,
        target_new=target_new,
        ground_truth=ground_truth,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        subject = subject,
        train_ds=train_ds,
        keep_original_weight=True,
        pre_file=args.pre_file,
        pre_edit = pre_edit,
        test_generation=True,
        sequential_edit = False
    )
    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_{args.datatype}_{hparams.model_name.split("/")[-1]}_{args.chinese_ds_type}_results.json'), 'w'), indent=4)