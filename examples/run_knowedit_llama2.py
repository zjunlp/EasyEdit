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
    MENDHyperParams,
    SERACHparams
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import KnowEditDataset

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--datatype',default=None,type=str)

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
    else:
        raise NotImplementedError
    

    datas = KnowEditDataset(args.data_dir)
    if args.ds_size is not None:
        datas = random.sample(datas, args.ds_size)

    if args.datatype == 'counterfact':
        prompts=[data['prompt'] for data in datas]
        subjects=[data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]
        portability_r =[data['portability_r'] for data in datas]
        portability_s =[data['portability_s'] for data in datas]
        locality_rs = [data['locality_rs'] for data in datas]
        locality_f = [data['locality_f'] for data in datas]

        portability_reasoning_prompts=[]
        portability_reasoning_ans=[]
        for pr in portability_r[0]:
            prompt=pr["prompt"]
            an=pr["ground_truth"][0][0]
            portability_reasoning_prompts.append(prompt)
            portability_reasoning_ans.append(an)
    
        portability_Subject_Aliasing_prompts=[]
        portability_Subject_Aliasing_ans=[]           
        for ps in portability_s[0]:
            prompt=ps["prompt"]
            an=ps["ground_truth"][0][0]
            portability_Subject_Aliasing_prompts.append(prompt)
            portability_Subject_Aliasing_ans.append(an)

        locality_Relation_Specificity_prompts=[]
        locality_Relation_Specificity_ans=[]
        for lr in locality_rs[0]:
            prompt=lr["prompt"]
            an=lr["ground_truth"][0][0]
            locality_Relation_Specificity_prompts.append(prompt)
            locality_Relation_Specificity_ans.append(an)

        locality_Forgetfulness_prompts=[]        
        locality_Forgetfulness_ans=[]
        for lf in locality_f[0]:
            prompt = lf["prompt"]
            an = lf["ground_truth"][0][0]
            locality_Forgetfulness_prompts.append(prompt)
            locality_Forgetfulness_ans.append(an)

    if args.datatype == 'recent':
        prompts=[data['prompt'] for data in datas]
        subjects=[data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]
        portability_r =[data['portability_r'] for data in datas]
        portability_s =[data['portability_s'] for data in datas]
        locality_rs = [data['locality_rs'] for data in datas]
        locality_f = [data['locality_f'] for data in datas]

        portability_reasoning_prompts=[]
        portability_reasoning_ans=[]
        for pr in portability_r[0]:
            prompt=pr["prompt"]
            an=pr["ground_truth"][0][0]
            portability_reasoning_prompts.append(prompt)
            portability_reasoning_ans.append(an)
    
        portability_Subject_Aliasing_prompts=[]
        portability_Subject_Aliasing_ans=[]           
        for ps in portability_s[0]:
            prompt=ps["prompt"]
            an=ps["ground_truth"][0][0]
            portability_Subject_Aliasing_prompts.append(prompt)
            portability_Subject_Aliasing_ans.append(an)

        locality_Relation_Specificity_prompts=[]
        locality_Relation_Specificity_ans=[]
        for lr in locality_rs[0]:
            prompt=lr["prompt"]
            an=lr["ground_truth"][0][0]
            locality_Relation_Specificity_prompts.append(prompt)
            locality_Relation_Specificity_ans.append(an)

        locality_Forgetfulness_prompts=[]        
        locality_Forgetfulness_ans=[]
        if isinstance(locality_f[0] , list):
            for lf in locality_f[0]:
                prompt = lf["prompt"]
                an = lf["ground_truth"][0][0]
                locality_Forgetfulness_prompts.append(prompt)
                locality_Forgetfulness_ans.append(an)

    if args.datatype == 'wikibio':
        prompts=[data['prompt'] for data in datas]
        subjects=[data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]
        portability_r =[data['portability_r'] for data in datas]
        portability_s =[data['portability_s'] for data in datas]
        locality_rs = [data['locality_rs'] for data in datas]
        locality_f = [data['locality_f'] for data in datas]

        portability_reasoning_prompts=[]
        portability_reasoning_ans=[]
    
        portability_Subject_Aliasing_prompts=[]
        portability_Subject_Aliasing_ans=[]           

        locality_Relation_Specificity_prompts=[]
        locality_Relation_Specificity_ans=[]
        for lr in locality_rs[0]:
            prompt=lr["prompt"]
            an=lr["ground_truth"][0]
            locality_Relation_Specificity_prompts.append(prompt)
            locality_Relation_Specificity_ans.append(an)

        locality_Forgetfulness_prompts=[]        
        locality_Forgetfulness_ans=[]

    if args.datatype == 'zsre':
        prompts=[data['prompt'] for data in datas]
        subjects=[data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]
        portability_r =[data['portability_r'] for data in datas]
        portability_s =[data['portability_s'] for data in datas]
        locality_rs = [data['locality_rs'] for data in datas]
        locality_f = [data['locality_f'] for data in datas]

        portability_reasoning_prompts=[]
        portability_reasoning_ans=[]
        for pr in portability_r[0]:
            prompt=pr["prompt"]
            an=pr["ground_truth"][0]
            portability_reasoning_prompts.append(prompt)
            portability_reasoning_ans.append(an)
    
        portability_Subject_Aliasing_prompts=[]
        portability_Subject_Aliasing_ans=[]

        if isinstance(portability_s[0] , list):   
            for ps in portability_s[0]:
                prompt=ps["prompt"]
                an=ps["ground_truth"][0]
                portability_Subject_Aliasing_prompts.append(prompt)
                portability_Subject_Aliasing_ans.append(an)

        locality_Relation_Specificity_prompts=[]
        locality_Relation_Specificity_ans=[]
        for lr in locality_rs[0]:
            prompt=lr["prompt"]
            an=lr["ground_truth"][0]
            locality_Relation_Specificity_prompts.append(prompt)
            locality_Relation_Specificity_ans.append(an)

        locality_Forgetfulness_prompts=[]        
        locality_Forgetfulness_ans=[]
        if isinstance(locality_f[0] , list):   
            for lf in locality_f[0]:
                prompt = lf["prompt"]
                an = lf["ground_truth"][0]
                locality_Forgetfulness_prompts.append(prompt)
                locality_Forgetfulness_ans.append(an)




    locality_inputs = {
        'Relation_Specificity':{
            'prompt': locality_Relation_Specificity_prompts,
            'ground_truth': locality_Relation_Specificity_ans
        },
        'Forgetfulness':{
            'prompt':locality_Forgetfulness_prompts,
            'ground_truth':locality_Forgetfulness_ans
        }
    }
    portability_inputs = {
        'Subject_Aliasing':{
            'prompt': portability_Subject_Aliasing_prompts,
            'ground_truth': portability_Subject_Aliasing_ans
        },
        'reasoning':{
            'prompt': portability_reasoning_prompts,
            'ground_truth': portability_reasoning_ans           
        }
    }

    hparams = editing_hparams.from_hparams(args.hparams_dir)


    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        subject=subjects,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True
    )

    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_results.json'), 'w'), indent=4)
